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

from .kgcoop import CUSTOM_TEMPLATES as KGCOOP_TEMPLATES

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
    def __init__(self, cfg, cfg_node, classnames, clip_model):
        super().__init__(cfg, cfg_node, classnames, clip_model)

        template_key = cfg.DATASET.NAME
        anchor_template = KGCOOP_TEMPLATES.get(template_key, 'a photo of a {}.')
        classnames_clean = [name.replace('_', ' ') for name in classnames]
        prompts = [anchor_template.format(name) for name in classnames_clean]
        tokenized = torch.cat([clip.tokenize(p) for p in prompts])
        device = clip_model.token_embedding.weight.device
        with torch.no_grad():
            tokenized = tokenized.to(device)
            text_features = clip_model.encode_text(tokenized)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.register_buffer('zeroshot_features', text_features.type(clip_model.dtype))


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
            self.domain_proj = nn.Linear(emb_dim, self.n_ctx * ctx_dim)

    def forward(self, domain_ids: Optional[torch.Tensor] = None):
        shared_prompts = super().forward()
        if self.domain_embed is None or domain_ids is None:
            if self.dropout is not None:
                shared_prompts = self.dropout(shared_prompts)
            return {
                'shared_prompts': shared_prompts,
                'per_domain_prompts': None,
                'domain_indices': None
            }

        max_domain = self.domain_embed.num_embeddings
        if domain_ids.max().item() >= max_domain:
            if self.dropout is not None:
                shared_prompts = self.dropout(shared_prompts)
            return {
                'shared_prompts': shared_prompts,
                'per_domain_prompts': None,
                'domain_indices': None
            }

        device = shared_prompts.device
        domain_ids = domain_ids.to(device)
        unique_domains, inverse = domain_ids.unique(sorted=True, return_inverse=True)

        per_domain_prompts = []
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        for dom_id in unique_domains:
            emb = self.domain_embed(dom_id)
            bias = self.domain_proj(emb)
            bias = bias.view(self.n_ctx, -1)
            ctx_shifted = ctx + bias.unsqueeze(0)
            prompts = torch.cat([self.token_prefix, ctx_shifted, self.token_suffix], dim=1)
            per_domain_prompts.append(prompts)

        per_domain_prompts = torch.stack(per_domain_prompts)
        if self.dropout is not None:
            shared_prompts = self.dropout(shared_prompts)
            per_domain_prompts = self.dropout(per_domain_prompts)

        return {
            'shared_prompts': shared_prompts,
            'per_domain_prompts': per_domain_prompts,
            'domain_indices': inverse
        }


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
        # Align feature dtype with the gate's parameters to avoid fp16/fp32 mismatches
        mlp_dtype = self.mlp[0].weight.dtype
        gate_logit = self.mlp(image_feats.to(mlp_dtype))
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
        style_out = self.style_prompt(domain_ids)
        if style_out['per_domain_prompts'] is None:
            style_prompts = style_out['shared_prompts']
            style_features = self.encode_text(style_prompts)
            style_features = style_features / style_features.norm(dim=-1, keepdim=True)
            style_logits = self.logit_scale.exp() * image_feats @ style_features.t()
            style_features_batch = style_features.unsqueeze(0).expand(image_feats.size(0), -1, -1)
        else:
            per_domain_prompts = style_out['per_domain_prompts']
            domain_indices = style_out['domain_indices']
            unique_feats = []
            for idx in range(per_domain_prompts.size(0)):
                prompts = per_domain_prompts[idx]
                feats = self.encode_text(prompts)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                unique_feats.append(feats)
            unique_feats = torch.stack(unique_feats)
            logit_list = []
            style_feat_per_sample = []
            for i, dom_idx in enumerate(domain_indices):
                feats = unique_feats[dom_idx]
                logits_i = self.logit_scale.exp() * image_feats[i] @ feats.t()
                logit_list.append(logits_i)
                style_feat_per_sample.append(feats)
            style_logits = torch.stack(logit_list)
            style_features_batch = torch.stack(style_feat_per_sample)
            style_features = style_features_batch.mean(dim=0)

        alpha, gate_logits = self.gate(image_feats)
        gate_weight = (1 - alpha).unsqueeze(-1)
        style_logits_for_fusion = style_logits
        if self.cfg.TRAINER.CSDG.GATE.DETACH_STYLE_GRAD:
            style_logits_for_fusion = style_logits_for_fusion.detach()
        fused_logits = content_logits + gate_weight * style_logits_for_fusion

        zeroshot_features = self.content_prompt.zeroshot_features
        zeroshot_logits = self.logit_scale.exp() * image_feats @ zeroshot_features.t()

        return {
            'logits': fused_logits,
            'content_logits': content_logits,
            'style_logits': style_logits,
            'alpha': alpha,
            'gate_logits': gate_logits,
            'content_features': content_features,
            'style_features': style_features,
            'style_features_batch': style_features_batch,
            'image_features': image_feats,
            'zeroshot_features': zeroshot_features,
            'zeroshot_logits': zeroshot_logits
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
        losses = OrderedDict()
        ce_loss = F.cross_entropy(logits, labels)
        losses['ce'] = ce_loss

        anchor_weight = self.cfg.TRAINER.CSDG.CONTENT.ANCHOR_WEIGHT
        if anchor_weight > 0:
            content_feat = outputs['content_features']
            zeroshot_feat = outputs['zeroshot_features']
            anchor_loss = 1 - (content_feat * zeroshot_feat).sum(dim=-1).mean()
            losses['anchor'] = anchor_weight * anchor_loss

        decor_weight = self.cfg.TRAINER.CSDG.LOSS.STYLE_DECORR_WEIGHT
        if decor_weight > 0:
            content_feat = outputs['content_features']
            style_batch = outputs['style_features_batch'].mean(dim=0)
            content_centered = content_feat - content_feat.mean(dim=0, keepdim=True)
            style_centered = style_batch - style_batch.mean(dim=0, keepdim=True)
            denom = max(content_centered.size(0) - 1, 1)
            cross_cov = content_centered.t() @ style_centered / denom
            decor_loss = cross_cov.pow(2).mean()
            losses['decor'] = decor_weight * decor_loss

        gate_ent_weight = self.cfg.TRAINER.CSDG.LOSS.GATE_ENT_WEIGHT
        if gate_ent_weight > 0:
            alpha = outputs['alpha']
            alpha = alpha.clamp(1e-6, 1 - 1e-6)
            entropy = -(alpha * alpha.log() + (1 - alpha) * (1 - alpha).log()).mean()
            gate_loss = -entropy
            losses['gate'] = gate_ent_weight * gate_loss

        style_ce_weight = self.cfg.TRAINER.CSDG.LOSS.STYLE_CE_WEIGHT
        if style_ce_weight > 0:
            style_ce = F.cross_entropy(outputs['style_logits'], labels)
            losses['style_ce'] = style_ce_weight * style_ce

        zs_kl_weight = self.cfg.TRAINER.CSDG.LOSS.ZERO_SHOT_KL_WEIGHT
        if zs_kl_weight > 0:
            teacher = F.softmax(outputs['zeroshot_logits'].detach(), dim=1)
            student_log = F.log_softmax(logits, dim=1)
            kl = F.kl_div(student_log, teacher, reduction='batchmean')
            losses['zero_shot_kl'] = zs_kl_weight * kl

        total_loss = sum(losses.values())
        self.model_backward_and_update(total_loss, names='csdg')

        with torch.no_grad():
            alpha = outputs['alpha']
            loss_summary = {
                'loss': total_loss.item(),
                'loss_ce': ce_loss.item(),
                'acc': compute_accuracy(logits, labels)[0].item(),
                'alpha_mean': alpha.mean().item(),
                'alpha_std': alpha.std(unbiased=False).item()
            }

            if anchor_weight > 0:
                loss_summary['loss_anchor'] = (anchor_loss * anchor_weight).item()
            if decor_weight > 0:
                loss_summary['loss_decor'] = (decor_loss * decor_weight).item()
            if gate_ent_weight > 0:
                loss_summary['loss_gate'] = (gate_loss * gate_ent_weight).item()
                loss_summary['gate_entropy'] = entropy.item()
            if style_ce_weight > 0:
                loss_summary['loss_style_ce'] = (style_ce * style_ce_weight).item()
            if zs_kl_weight > 0:
                loss_summary['loss_zero_shot_kl'] = (kl * zs_kl_weight).item()

        if (self.batch_idx + 1) == self.num_batches:
            self.sched.step()
        return loss_summary

    def parse_batch_train(self, batch):
        return batch['img'], batch['label'], batch.get('domain')

    def model_inference(self, images):
        outputs = self.model(images)
        return outputs['logits']
