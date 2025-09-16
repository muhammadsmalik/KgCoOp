import os.path as osp
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class BasePromptLearner(nn.Module):
    """Reusable prompt learner logic based on KgCoOp with config hooks."""

    def __init__(self, cfg, cfg_node, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.cfg_node = cfg_node
        n_cls = len(classnames)
        n_ctx = cfg_node.N_CTX
        ctx_init = cfg_node.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            template = getattr(cfg_node, 'TEMPLATE', 'a photo of a')
            template = template.replace('_', ' ')
            n_ctx = len(template.split(' '))
            prompt = clip.tokenize(template)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = template
        else:
            if getattr(cfg_node, 'CSC', False):
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = ' '.join(['X'] * n_ctx)

        print(f"{self.__class__.__name__} init prefix: '{prompt_prefix}' (tokens={n_ctx})")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace('_', ' ') for name in classnames]
        prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer('token_prefix', embedding[:, :1, :])
        self.register_buffer('token_suffix', embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        return prompts


class ContentPromptLearner(BasePromptLearner):
    pass


class StylePromptLearner(BasePromptLearner):
    def __init__(self, cfg, cfg_node, classnames, clip_model):
        super().__init__(cfg, cfg_node, classnames, clip_model)
        dropout = getattr(cfg_node, 'DROPOUT', 0.0)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.domain_embed = None
        if getattr(cfg_node, 'USE_DOMAIN_ID', False):
            emb_dim = cfg_node.DOMAIN_EMB_DIM
            num_domains = len(cfg.DATASET.SOURCE_DOMAINS)
            self.domain_embed = nn.Embedding(num_domains, emb_dim)
            ctx_dim = self.ctx.shape[-1]
            self.domain_proj = nn.Linear(emb_dim, ctx_dim)

    def forward(self, domain_ids: Optional[torch.Tensor] = None):
        prompts = super().forward()
        if self.domain_embed is not None and domain_ids is not None:
            emb = self.domain_embed(domain_ids)
            bias = self.domain_proj(emb)
            bias = bias.unsqueeze(1).expand(-1, self.n_ctx, -1)
            prompts = prompts + bias
        if self.dropout is not None:
            prompts = self.dropout(prompts)
        return prompts


class GateModule(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        vis_dim = clip_model.visual.output_dim
        hidden_ratio = cfg.TRAINER.CSDG.GATE.HIDDEN_RATIO
        hidden = max(1, vis_dim // hidden_ratio)
        layers = [nn.Linear(vis_dim, hidden)]
        if cfg.TRAINER.CSDG.GATE.DROPOUT > 0:
            layers.append(nn.Dropout(cfg.TRAINER.CSDG.GATE.DROPOUT))
        layers.extend([nn.ReLU(inplace=True), nn.Linear(hidden, 1)])
        self.mlp = nn.Sequential(*layers)
        self.temperature = cfg.TRAINER.CSDG.GATE.SIGMOID_TEMPERATURE
        init_bias = cfg.TRAINER.CSDG.GATE.INIT_BIAS
        with torch.no_grad():
            self.mlp[-1].bias.fill_(init_bias)

    def forward(self, image_feats):
        gate_logit = self.mlp(image_feats)
        alpha = torch.sigmoid(gate_logit / self.temperature)
        return alpha.squeeze(-1), gate_logit.squeeze(-1)


class CSDGModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.content_prompt = ContentPromptLearner(cfg, cfg.TRAINER.CSDG.CONTENT, classnames, clip_model)
        self.style_prompt = StylePromptLearner(cfg, cfg.TRAINER.CSDG.STYLE, classnames, clip_model)
        self.tokenized_prompts = self.content_prompt.tokenized_prompts

        self.gate = GateModule(cfg, clip_model)

    def encode_text(self, prompts):
        return self.text_encoder(prompts, self.tokenized_prompts)

    def forward(self, images, domain_ids=None):
        images = images.type(self.dtype)
        image_feats = self.image_encoder(images)
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

        # Content stream
        content_prompts = self.content_prompt()
        content_features = self.encode_text(content_prompts)
        content_features = content_features / content_features.norm(dim=-1, keepdim=True)
        content_logits = self.logit_scale.exp() * image_feats @ content_features.t()

        # Style stream
        style_prompts = self.style_prompt(domain_ids)
        style_features = self.encode_text(style_prompts)
        style_features = style_features / style_features.norm(dim=-1, keepdim=True)
        style_logits = self.logit_scale.exp() * image_feats @ style_features.t()

        alpha, gate_logits = self.gate(image_feats)
        fused_logits = content_logits  # TODO: apply gating fusion once losses ready

        return {
            'logits': fused_logits,
            'content_logits': content_logits,
            'style_logits': style_logits,
            'alpha': alpha,
            'gate_logits': gate_logits,
            'content_features': content_features,
            'style_features': style_features,
            'image_features': image_feats
        }


@TRAINER_REGISTRY.register()
class CSDG(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.CSDG.PREC in ['fp16', 'fp32', 'amp']
        if cfg.TRAINER.CSDG.LOSS.FUSE_MODE not in ['sigmoid']:
            raise ValueError(f"Unsupported fuse mode: {cfg.TRAINER.CSDG.LOSS.FUSE_MODE}")

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.CSDG.PREC in ['fp32', 'amp']:
            clip_model.float()

        print("Building CSDG model")
        self.model = CSDGModel(cfg, classnames, clip_model)

        print("Freezing CLIP backbone/text encoders")
        for name, param in self.model.named_parameters():
            if name.startswith('image_encoder') or name.startswith('text_encoder'):
                param.requires_grad_(False)

        self.model.to(self.device)

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optim = build_optimizer(params, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('csdg', self.model, self.optim, self.sched)

    def forward_backward(self, batch):
        images = batch['img'].to(self.device)
        labels = batch['label'].to(self.device)
        domain = batch.get('domain')
        if domain is not None:
            domain = domain.to(self.device)

        outputs = self.model(images, domain)
        logits = outputs['logits']
        loss = F.cross_entropy(logits, labels)

        self.model_backward_and_update(loss, names='csdg')

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(logits, labels)[0].item(),
            'alpha_mean': outputs['alpha'].mean().item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.sched.step()
        return loss_summary

    def parse_batch_train(self, batch):
        return batch['img'], batch['label'], batch.get('domain')

    def model_inference(self, images):
        outputs = self.model(images)
        return outputs['logits']
