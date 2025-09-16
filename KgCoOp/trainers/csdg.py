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
