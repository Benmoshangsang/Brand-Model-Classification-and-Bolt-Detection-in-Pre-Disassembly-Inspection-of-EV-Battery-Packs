#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025
# Note: This file replaces the original MambaVision backbone with a ResNet-50 backbone
# and provides MMDet / MMSeg adapters.
# Dependencies: torch, torchvision, mmengine, mmdet, mmseg

import os
import math
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast

# ---- timm is only imported for compatibility
# (e.g., if other modules in the project require it).
# This implementation does not depend on timm models. ----
try:
    from timm.models.registry import register_model  # noqa: F401
    from timm.models.layers import trunc_normal_, DropPath  # noqa: F401
except Exception:
    # Provide minimal fallback implementations to make this file usable independently
    def register_model(fn):
        return fn
    def trunc_normal_(tensor, std=0.02):
        nn.init.trunc_normal_(tensor, std=std)
    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0): super().__init__()
        def forward(self, x): return x

# ---- torchvision ResNet imports ----
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.models import resnet50

# ---- MM engine / registries ----
from mmengine.runner import load_checkpoint
from mmengine.model import BaseModule
from mmdet.registry import MODELS as MODELS_MMDET
from mmseg.registry import MODELS as MODELS_MMSEG

# -------------------------------------------------------
# Utility functions (weight loading, keeping the same interface as the original file)
# -------------------------------------------------------

def _load_state_dict(module, state_dict, strict=False, logger=None):
    """Custom weight loading function for handling partially mismatched state_dict."""
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(mod, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        mod._load_from_state_dict(
            state_dict, prefix, local_metadata, True,
            all_missing_keys, unexpected_keys, err_msg
        )
        for name, child in mod._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [k for k in all_missing_keys if 'num_batches_tracked' not in k]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(model,
                     filename,
                     map_location='cpu',
                     strict=False,
                     logger=None):
    """Load a checkpoint file and process the state_dict inside it."""
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    # Remove common prefixes
    def strip_prefix(sd, prefix):
        if any(k.startswith(prefix) for k in sd.keys()):
            return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
        return sd
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Compatible with several common saving formats
    for pfx in ['backbone.', 'model.', 'encoder.', 'resnet.', 'net.']:
        state_dict = strip_prefix(state_dict, pfx)

    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint

# -------------------------------------------------------
# Basic normalization (same interface as the original file, for adapter layer selection)
# -------------------------------------------------------

class LayerNorm2d(nn.LayerNorm):
    """2D version of LayerNorm (applies LN over the channel dimension)."""
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return x

# -------------------------------------------------------
# ResNet-50 backbone (optional classification head)
# -------------------------------------------------------

class ResNet50Backbone(nn.Module):
    """
    Standard ResNet-50 backbone:
      - forward_features: returns the final global feature vector
      - forward: returns logits (if num_classes > 0), otherwise returns features
      - extract_features: returns [C2, C3, C4, C5]
    Channel settings: C2=256, C3=512, C4=1024, C5=2048
    """
    def __init__(self,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 norm_eval: bool = False,
                 frozen_stages: int = -1):
        super().__init__()

        # —— Option A: do not keep self.resnet, only use a local variable to extract required layers ——
        res: ResNet = resnet50(weights=None)

        # Adapt input channels
        if in_chans != 3:
            conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
            res.conv1 = conv1

        self.num_classes = num_classes
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages

        # Only attach the required modules to the current class
        self.stem = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
        self.layer1 = res.layer1  # 256
        self.layer2 = res.layer2  # 512
        self.layer3 = res.layer3  # 1024
        self.layer4 = res.layer4  # 2048

        # Classification head
        self.avgpool = res.avgpool
        self.fc = nn.Linear(2048, num_classes) if num_classes > 0 else nn.Identity()

        # Initialization (consistent with torchvision)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.zeros_(m.bn3.weight)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for p in self.stem.parameters():
                p.requires_grad = False
        if self.frozen_stages >= 1:
            for p in self.layer1.parameters():
                p.requires_grad = False
        if self.frozen_stages >= 2:
            for p in self.layer2.parameters():
                p.requires_grad = False
        if self.frozen_stages >= 3:
            for p in self.layer3.parameters():
                p.requires_grad = False
        if self.frozen_stages >= 4:
            for p in self.layer4.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def extract_features(self, x) -> List[torch.Tensor]:
        """
        Return multi-scale features [C2, C3, C4, C5]
        """
        x = self.stem(x)
        c2 = self.layer1(x)          # 256
        c3 = self.layer2(c2)         # 512
        c4 = self.layer3(c3)         # 1024
        c5 = self.layer4(c4)         # 2048
        return [c2, c3, c4, c5]

    def forward_features(self, x):
        """
        Return the final global feature vector (before classification)
        """
        feats = self.extract_features(x)
        c5 = feats[-1]
        x = self.avgpool(c5)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        """
        If num_classes > 0, return logits; otherwise return the global feature vector
        """
        x = self.forward_features(x)
        x = self.fc(x)
        return x

    def _load_state_dict(self, pretrained, strict: bool = False):
        _load_checkpoint(self, pretrained, strict=strict)

# -------------------------------------------------------
# MMDet / MMseg adapter backbone
# -------------------------------------------------------

@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class MM_resnet50(ResNet50Backbone):
    """
    ResNet-50 backbone adapted for MMDetection / MMSegmentation.
      - out_indices specifies outputs among [C2, C3, C4, C5]
      - optional output normalization: 'ln2d' / 'bn' / 'ln'
    """
    def __init__(self,
                 in_chans: int = 3,
                 num_classes: int = 0,
                 zero_init_residual: bool = False,
                 norm_eval: bool = False,
                 frozen_stages: int = -1,
                 out_indices: Tuple[int, ...] = (0, 1, 2, 3),
                 pretrained: Optional[str] = None,
                 norm_layer: str = "ln2d",
                 **kwargs):
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            norm_eval=norm_eval,
            frozen_stages=frozen_stages,
        )
        self.dims = [256, 512, 1024, 2048]
        self.out_indices = out_indices

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer_mod: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        if norm_layer_mod is None:
            raise ValueError(f"Unsupported norm_layer: {norm_layer}. Choose from 'ln2d', 'bn', 'ln'.")

        for i in out_indices:
            c = self.dims[i]
            layer_name = f'outnorm{i}'
            layer = norm_layer_mod(c)
            self.add_module(layer_name, layer)

        self.init_weights(pretrained)

    def load_pretrained(self, ckpt=None, key="state_dict"):
        if ckpt is None:
            return
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatible = self.load_state_dict(_ckpt.get(key, _ckpt), strict=False)
            print(incompatible)
        except Exception as e:
            print(f"Failed loading checkpoint from {ckpt}: {e}")

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x: torch.Tensor):
        # stem + stage1-4
        x = self.stem(x)
        c2 = self.layer1(x)   # 256
        c3 = self.layer2(c2)  # 512
        c4 = self.layer3(c3)  # 1024
        c5 = self.layer4(c4)  # 2048

        feats = [c2, c3, c4, c5]
        outs = []
        for i in self.out_indices:
            norm_layer = getattr(self, f'outnorm{i}')
            f = feats[i]
            # Important: check LayerNorm2d first to avoid double transpose
            if isinstance(norm_layer, LayerNorm2d):
                f = norm_layer(f)
            elif isinstance(norm_layer, nn.LayerNorm):
                # Pure nn.LayerNorm requires channels to be moved to the last dimension
                f = f.permute(0, 2, 3, 1)  # (B,C,H,W)->(B,H,W,C)
                f = norm_layer(f)
                f = f.permute(0, 3, 1, 2)  # (B,H,W,C)->(B,C,H,W)
            else:
                # Use directly for BN and similar layers
                f = norm_layer(f)
            outs.append(f.contiguous())
        return outs if len(self.out_indices) > 0 else c5

# -------------------------------------------------------
# (Optional) classification export: consistent with timm-style register
# -------------------------------------------------------

@register_model
def resnet50_cls(pretrained=False, **kwargs):
    """
    Return a ResNet-50 for classification using the implementation in this file,
    not the built-in timm version.
    """
    model = ResNet50Backbone(
        in_chans=kwargs.pop('in_chans', 3),
        num_classes=kwargs.pop('num_classes', 1000),
        zero_init_residual=kwargs.pop('zero_init_residual', False),
        norm_eval=kwargs.pop('norm_eval', False),
        frozen_stages=kwargs.pop('frozen_stages', -1),
    )
    if isinstance(pretrained, str):
        model._load_state_dict(pretrained, strict=False)
    return model

# -------------------------------------------------------
# Usage examples (comments):
# 1) As a general classification backbone:
#    model = ResNet50Backbone(in_chans=3, num_classes=1000)
#    logits = model(torch.randn(2,3,224,224))
#
# 2) As an MMDet / MMSeg backbone:
#    backbone = MM_resnet50(in_chans=3, out_indices=(0,1,2,3), norm_layer='ln2d', pretrained=None)
#    c2, c3, c4, c5 = backbone(torch.randn(1,3,640,640))
# -------------------------------------------------------
