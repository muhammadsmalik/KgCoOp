"""CSDG trainer scaffold rebuilt with maximal reuse of KgCoOp/CoCoOp components."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from yacs.config import CfgNode

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler

from .kgcoop import (
    load_clip_to_cpu,
    TextEncoder as KgCoOpTextEncoder,
    PromptLearner as KgCoOpPromptLearner,
)
from .cocoop import PromptLearner as CoCoOpPromptLearner


def _adapt_prompt_cfg(cfg: CfgNode, prompt_node: CfgNode) -> CfgNode:
    """Clone the main cfg and map CSDG prompt settings onto COOP defaults."""

    cfg_copy = cfg.clone()
    cfg_copy.defrost()

    coop_cfg = cfg_copy.TRAINER.COOP
    coop_cfg.N_CTX = prompt_node.N_CTX
    coop_cfg.CTX_INIT = prompt_node.CTX_INIT
    coop_cfg.CSC = getattr(prompt_node, "CSC", False)
    coop_cfg.PREC = cfg.TRAINER.CSDG.PREC

    cocoop_cfg = cfg_copy.TRAINER.COCOOP
    cocoop_cfg.N_CTX = prompt_node.N_CTX
    cocoop_cfg.CTX_INIT = prompt_node.CTX_INIT
    cocoop_cfg.PREC = cfg.TRAINER.CSDG.PREC

    cfg_copy.freeze()
    return cfg_copy


class ContentPromptLearner(KgCoOpPromptLearner):
    """Content stream prompt learner that keeps KgCoOp behaviour."""

    def __init__(self, cfg: CfgNode, cfg_node: CfgNode, classnames, clip_model):
        prompt_cfg = _adapt_prompt_cfg(cfg, cfg_node)
        super().__init__(prompt_cfg, classnames, clip_model)

        # Cache zeroshot features for the anchor loss.
        zeroshot = self.text_features.detach().clone().to(dtype=clip_model.dtype)
        self.register_buffer("zeroshot_features", zeroshot)


class CSDGTextEncoder(KgCoOpTextEncoder):
    """Thin wrapper to emphasise reuse of KgCoOp text encoder."""

    pass


class StylePromptLearner(CoCoOpPromptLearner):
    """Style stream prompt learner reusing CoCoOp construction helpers."""

    def __init__(self, cfg: CfgNode, cfg_node: CfgNode, classnames, clip_model):
        prompt_cfg = _adapt_prompt_cfg(cfg, cfg_node)
        super().__init__(prompt_cfg, classnames, clip_model)

        # Disable CoCoOp's meta net; gating happens outside this module.
        self.meta_net = None

        dropout = getattr(cfg_node, "DROPOUT", 0.0)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

        self.domain_embed = None
        self.domain_proj: Optional[nn.Linear] = None
        self._domain_debug_printed = False

        if getattr(cfg_node, "USE_DOMAIN_ID", False):
            emb_dim = cfg_node.DOMAIN_EMB_DIM
            num_domains = len(cfg.DATASET.SOURCE_DOMAINS)
            self.domain_embed = nn.Embedding(num_domains, emb_dim)
            ctx_dim = self.ctx.shape[-1]
            self.domain_proj = nn.Linear(emb_dim, self.n_ctx * ctx_dim)

    def _base_prompts(self) -> torch.Tensor:
        """Construct shared prompts without domain conditioning."""

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)
        if self.dropout_layer is not None:
            prompts = self.dropout_layer(prompts)
        return prompts

    def forward(self, domain_ids: Optional[torch.Tensor] = None) -> dict:
        if self.domain_embed is None or domain_ids is None:
            shared = self._base_prompts()
            return {
                "shared_prompts": shared,
                "per_domain_prompts": None,
                "domain_indices": None,
            }

        max_domain = self.domain_embed.num_embeddings
        max_seen = int(domain_ids.max().item())
        if max_seen >= max_domain:
            if not hasattr(self, "_domain_overflow_warned"):
                print(
                    f"[CSDG] domain id {max_seen} exceeds source range (0-{max_domain - 1}); "
                    "falling back to shared style prompts."
                )
                self._domain_overflow_warned = True
            shared = self._base_prompts()
            return {
                "shared_prompts": shared,
                "per_domain_prompts": None,
                "domain_indices": None,
            }

        device = self.ctx.device
        domain_ids = domain_ids.to(device)
        unique_domains, inverse = domain_ids.unique(sorted=True, return_inverse=True)

        if self.training and not self._domain_debug_printed:
            print(f"[CSDG] unique domain ids in batch: {unique_domains.tolist()}")
            self._domain_debug_printed = True

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        per_domain = []
        for dom_id in unique_domains:
            emb = self.domain_embed(dom_id)
            shift = self.domain_proj(emb).to(ctx.dtype)
            shift = shift.view(1, self.n_ctx, -1)
            shift = shift.expand(self.n_cls, -1, -1)
            ctx_shifted = ctx + shift
            prompts_dom = self.construct_prompts(
                ctx_shifted, self.token_prefix, self.token_suffix
            )
            per_domain.append(prompts_dom)

        per_domain_prompts = torch.stack(per_domain)
        shared = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)

        if self.dropout_layer is not None:
            shared = self.dropout_layer(shared)
            per_domain_prompts = self.dropout_layer(per_domain_prompts)

        return {
            "shared_prompts": shared,
            "per_domain_prompts": per_domain_prompts,
            "domain_indices": inverse,
        }


class GateModule(nn.Module):
    """Scalar gate reusing CoCoOp meta-net dimensions."""

    def __init__(self, cfg: CfgNode, clip_model):
        super().__init__()
        vis_dim = clip_model.visual.output_dim
        hidden_ratio = cfg.TRAINER.CSDG.GATE.HIDDEN_RATIO
        hidden_dim = max(1, vis_dim // hidden_ratio)

        layers = [nn.Linear(vis_dim, hidden_dim), nn.ReLU(inplace=True)]
        if cfg.TRAINER.CSDG.GATE.DROPOUT > 0:
            layers.insert(1, nn.Dropout(cfg.TRAINER.CSDG.GATE.DROPOUT))
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self.temperature = cfg.TRAINER.CSDG.GATE.SIGMOID_TEMPERATURE

        init_bias = cfg.TRAINER.CSDG.GATE.INIT_BIAS
        with torch.no_grad():
            self.mlp[-1].bias.fill_(init_bias)

    def forward(self, image_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = self.mlp[0].weight.dtype
        logits = self.mlp(image_feats.to(dtype)).squeeze(-1)
        alpha = torch.sigmoid(logits / self.temperature)
        return alpha, logits


class CSDGModel(nn.Module):
    """Dual-stream model wiring reused prompt learners and gate."""

    def __init__(self, cfg: CfgNode, classnames, clip_model):
        super().__init__()
        self.cfg = cfg

        self.image_encoder = clip_model.visual
        self.text_encoder = CSDGTextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.content_prompt = ContentPromptLearner(
            cfg, cfg.TRAINER.CSDG.CONTENT, classnames, clip_model
        )
        self.style_prompt = StylePromptLearner(
            cfg, cfg.TRAINER.CSDG.STYLE, classnames, clip_model
        )
        self.tokenized_prompts = self.content_prompt.tokenized_prompts

        self.gate = GateModule(cfg, clip_model)

    def encode_text(self, prompts: torch.Tensor) -> torch.Tensor:
        feats = self.text_encoder(prompts, self.tokenized_prompts)
        return F.normalize(feats, dim=-1)

    def forward(self, images: torch.Tensor, domain_ids: Optional[torch.Tensor] = None) -> dict:
        images = images.type(self.dtype)
        image_feats = self.image_encoder(images)
        image_feats = F.normalize(image_feats, dim=-1)

        # Content stream
        content_prompts = self.content_prompt()
        content_features = self.encode_text(content_prompts)
        content_logits = self.logit_scale.exp() * image_feats @ content_features.t()

        # Style stream
        style_out = self.style_prompt(domain_ids)
        if style_out["per_domain_prompts"] is None:
            style_prompts = style_out["shared_prompts"]
            style_features = self.encode_text(style_prompts)
            style_logits = self.logit_scale.exp() * image_feats @ style_features.t()
            style_features_batch = style_features.unsqueeze(0).expand(image_feats.size(0), -1, -1)
        else:
            per_domain_prompts = style_out["per_domain_prompts"]
            domain_indices = style_out["domain_indices"]
            domain_feats = []
            for prompts in per_domain_prompts:
                feats = self.encode_text(prompts)
                domain_feats.append(feats)
            domain_feats = torch.stack(domain_feats)

            logits_per_sample = []
            feats_per_sample = []
            logit_scale = self.logit_scale.exp()
            for idx, dom_idx in enumerate(domain_indices):
                feats = domain_feats[dom_idx]
                logits_i = logit_scale * image_feats[idx] @ feats.t()
                logits_per_sample.append(logits_i)
                feats_per_sample.append(feats)

            style_logits = torch.stack(logits_per_sample)
            style_features_batch = torch.stack(feats_per_sample)
            style_features = style_features_batch.mean(dim=0)

        alpha, gate_logits = self.gate(image_feats)
        gate_weight = (1 - alpha).unsqueeze(-1)
        style_logits_for_fusion = style_logits
        if self.cfg.TRAINER.CSDG.GATE.DETACH_STYLE_GRAD:
            style_logits_for_fusion = style_logits_for_fusion.detach()
        fused_logits = content_logits + gate_weight * style_logits_for_fusion

        zeroshot_features = self.content_prompt.zeroshot_features
        zeroshot_logits = self.logit_scale.exp() * image_feats @ zeroshot_features.t()

        content_expanded = content_features.unsqueeze(0)
        style_content_sim = (style_features_batch * content_expanded).sum(dim=-1)

        return {
            "logits": fused_logits,
            "content_logits": content_logits,
            "style_logits": style_logits,
            "alpha": alpha,
            "gate_logits": gate_logits,
            "content_features": content_features,
            "style_features": style_features,
            "style_features_batch": style_features_batch,
            "style_content_sim": style_content_sim,
            "image_features": image_feats,
            "zeroshot_features": zeroshot_features,
            "zeroshot_logits": zeroshot_logits,
        }
