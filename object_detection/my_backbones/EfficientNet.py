#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

try:
    # timm >= 0.9
    from timm import create_model
except Exception:
    # timm <= 0.8 fallback
    from timm.models import create_model

from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS as MODELS_MMDET
from mmseg.registry import MODELS as MODELS_MMSEG


# -------------------------------------------------------
# Utils
# -------------------------------------------------------

class LayerNorm2d(nn.LayerNorm):
    """Channel-first LayerNorm for (N, C, H, W)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def _make_norm(norm: str, num_channels: int) -> nn.Module:
    """factory for norm layers used on 2D feature maps"""
    norm = norm.lower()
    if norm == 'ln2d':
        return LayerNorm2d(num_channels)
    elif norm == 'ln':
        return nn.LayerNorm(num_channels)
    elif norm == 'bn':
        return nn.BatchNorm2d(num_channels)
    else:
        # default to ln2d
        return LayerNorm2d(num_channels)


# -------------------------------------------------------
# EfficientNet backbone (features_only) + channel adapters
# -------------------------------------------------------

class _EfficientNetFeatures(nn.Module):
    """
    Wrap timm EfficientNet with features_only=True and expose 4 pyramid features.
    Automatically builds 1x1 adapters to target dims (e.g., [80, 160, 320, 640]).
    """
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        
        timm_pretrained: bool = False,
        out_indices: Tuple[int, int, int, int] = (0, 1, 2, 3),
        target_dims: Tuple[int, int, int, int] = (80, 160, 320, 640),
    ):
        super().__init__()
       
        self.backbone = create_model(
            model_name,
            pretrained=timm_pretrained, 
            features_only=True,
            out_indices=None  # 
        )

        # Detect available feature channels from timm
        feat_chs: List[int] = list(self.backbone.feature_info.channels())
       
        if len(feat_chs) >= 4:
            self.pick = list(range(len(feat_chs) - 4, len(feat_chs)))
        else:
            self.pick = list(range(len(feat_chs)))

        chosen_chs = [feat_chs[i] for i in self.pick]
        if len(chosen_chs) != 4:
            while len(chosen_chs) < 4:
                chosen_chs.append(chosen_chs[-1])
            if len(chosen_chs) > 4:
                chosen_chs = chosen_chs[-4:]

        assert len(target_dims) == 4, "target_dims must have 4 integers"
        self.adapters = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            for in_ch, out_ch in zip(chosen_chs, target_dims)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = self.backbone(x)  # list of feature maps (low->high)
        picked = [feats[i] for i in self.pick]
        if len(picked) != 4:
            while len(picked) < 4:
                picked.append(picked[-1])
            if len(picked) > 4:
                picked = picked[-4:]

        outs = [adp(f) for adp, f in zip(self.adapters, picked)]
        return outs  # [C2, C3, C4, C5]-like with channels = target_dims


# -------------------------------------------------------
# MMDet/MMseg compatible backbone
# -------------------------------------------------------

@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class MM_EfficientNet(nn.Module):
    """
    MMDetection/MMSegmentation-compatible EfficientNet backbone that outputs
    4 feature maps with channels [dim, 2*dim, 4*dim, 8*dim] (default dim=80).
   
    """
    def __init__(
        self,
        dim: int = 80,
        out_indices: Tuple[int, int, int, int] = (0, 1, 2, 3),
        pretrained: Optional[str] = None,          
        norm_layer: str = "ln2d",
        model_name: str = "efficientnet_b0",
      
        depths: Tuple[int, int, int, int] = (1, 3, 8, 4),
        num_heads: Tuple[int, int, int, int] = (2, 4, 8, 16),
        window_size: Tuple[int, int, int, int] = (8, 8, 8, 8),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.2,
        use_checkpoint: bool = False,
        enable_pcs: bool = True,
        enable_sac: bool = True,
        enable_sl_bridge: bool = False,
      
        timm_checkpoint: Optional[str] = None,
      
        timm_pretrained: bool = False,
    ):
        super().__init__()
        self.out_indices = out_indices
        self.dims = [int(dim * (2 ** i)) for i in range(4)]  # [80, 160, 320, 640] if dim=80

       
        self.feat = _EfficientNetFeatures(
            model_name=model_name,
            timm_pretrained=bool(timm_pretrained),  
            target_dims=tuple(self.dims)
        )

     
        if isinstance(timm_checkpoint, str):
            try:
                sd = torch.load(timm_checkpoint, map_location='cpu')
               
                if isinstance(sd, dict) and 'state_dict' in sd:
                    sd = sd['state_dict']
                missing, unexpected = self.feat.backbone.load_state_dict(sd, strict=False)
                print(f"[MM_EfficientNet] Loaded timm_checkpoint with missing={len(missing)}, unexpected={len(unexpected)}")
            except Exception as e:
                print(f"[MM_EfficientNet] Failed to load timm_checkpoint '{timm_checkpoint}': {e}")

        # Per-stage normalization
        self.outnorms = nn.ModuleList([_make_norm(norm_layer, c) for c in self.dims])

       
        self.channel_first = True

      
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return set()

    def init_weights(self, pretrained: Optional[str] = None):
        """Kept for mmengine compatibility."""
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.feat(x)  # list of 4 tensors with channels self.dims
        outs = []
        for i, f in enumerate(feats):
            if i in self.out_indices:
                outs.append(self.outnorms[i](f).contiguous())
        if len(self.out_indices) == 0:
            # Edge case
            return feats[-1]
        return outs

    def load_pretrained(self, ckpt: Optional[str] = None, key: str = "state_dict"):
        if ckpt is None:
            return
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location="cpu")
            print(f"[MM_EfficientNet] Successfully load ckpt {ckpt}")
            incompatible = self.load_state_dict(_ckpt.get(key, _ckpt), strict=False)
            print(incompatible)
        except Exception as e:
            print(f"[MM_EfficientNet] Failed loading checkpoint from {ckpt}: {e}")
