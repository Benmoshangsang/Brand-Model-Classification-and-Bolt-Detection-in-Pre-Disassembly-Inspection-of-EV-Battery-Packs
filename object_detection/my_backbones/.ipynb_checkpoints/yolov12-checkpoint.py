#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import os
from pathlib import Path
from functools import partial
from typing import Optional, Tuple, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint

from mmdet.registry import MODELS as MODELS_MMDET
from mmseg.registry import MODELS as MODELS_MMSEG

# -------------------------------------------------------
# 基础配置（占位，保持与原工程兼容）
# -------------------------------------------------------

def _cfg(url='', **kwargs):
    return {'url': url,
            'num_classes': 1000,
            'input_size': (3, 224, 224),
            'pool_size': None,
            'crop_pct': 0.875,
            'interpolation': 'bicubic',
            'fixed_input_size': True,
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            **kwargs
            }

default_cfgs = {
    'yolov12_T': _cfg(url='', input_size=(3, 224, 224)),
    'yolov12_S': _cfg(url='', input_size=(3, 224, 224)),
    'yolov12_B': _cfg(url='', input_size=(3, 256, 256)),
    'yolov12_L': _cfg(url='', input_size=(3, 320, 320)),
}

# -------------------------------------------------------
# 权重加载工具（与原逻辑一致）
# -------------------------------------------------------

def _load_state_dict(module, state_dict, strict=False, logger=None):
    """自定义的权重加载函数，用于处理不完全匹配的 state_dict。"""
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys, err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None
    missing_keys = [key for key in all_missing_keys if 'num_batches_tracked' not in key]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
    if len(err_msg) > 0:
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
    """加载检查点文件并处理其中的 state_dict。"""
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint

# -------------------------------------------------------
# 基础层：归一化、下采样
# -------------------------------------------------------

class LayerNorm2d(nn.LayerNorm):
    """2D 版本的 LayerNorm。"""
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Downsample(nn.Module):
    """下采样模块，使用步长为2的卷积实现。"""
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        dim_out = dim if keep_dim else 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim_out, eps=1e-4),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.reduction(x)

# -------------------------------------------------------
# YOLO 家族常见模块（ConvBNAct / Bottleneck / C2f）
# -------------------------------------------------------

class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        p = (k // 2) if p is None else p
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out, eps=1e-4)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """标准 CSP 风格瓶颈，带残差"""
    def __init__(self, c, shortcut=True, e=0.5):
        super().__init__()
        c_hidden = int(c * e)
        self.cv1 = ConvBNAct(c, c_hidden, 1, 1)
        self.cv2 = ConvBNAct(c_hidden, c, 3, 1)
        self.use_shortcut = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.use_shortcut else y


class C2f(nn.Module):
    """
    YOLOv8/11 风格的 C2f：split + 多个 Bottleneck + concat + fuse
    """
    def __init__(self, c_in, c_out, n=3, e=0.5, shortcut=True):
        super().__init__()
        c_hidden = int(c_out * e)
        self.cv1 = ConvBNAct(c_in, c_hidden * 2, 1, 1)
        self.m = nn.ModuleList([Bottleneck(c_hidden, shortcut=shortcut, e=1.0) for _ in range(n)])
        self.cv2 = ConvBNAct(c_hidden * (2 + n), c_out, 1, 1)

    def forward(self, x):
        y = self.cv1(x)
        y1, y2 = torch.chunk(y, 2, dim=1)
        outs = [y1, y2]
        for block in self.m:
            y2 = block(y2)
            outs.append(y2)
        return self.cv2(torch.cat(outs, dim=1))

# -------------------------------------------------------
# YOLOv12 近似模块：A2C2f（Attention-Augmented C2f）
# -------------------------------------------------------

class DepthwiseSepConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = ConvBNAct(c, c, k=7, s=1, g=c)
        self.pw = ConvBNAct(c, c, k=1, s=1)
    def forward(self, x):
        return self.pw(self.dw(x))

class SimpleMHA2D(nn.Module):
    """
    简化多头注意力：在空间维度进行注意力（H*W 为序列长度），
    在通道维度上做线性降维以控制开销。
    """
    def __init__(self, c, num_heads=4, attn_ratio=0.5, drop=0.0):
        super().__init__()
        d = int(c * attn_ratio)
        d = max(32, (d // num_heads) * num_heads)  # 对齐 heads
        self.in_proj = nn.Conv2d(c, d, 1, 1, 0, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, dropout=drop, batch_first=True)
        self.out_proj = nn.Conv2d(d, c, 1, 1, 0, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        q = self.in_proj(x)                    # (B, d, H, W)
        q = q.flatten(2).transpose(1, 2)       # (B, L, d)  L = H*W
        q = self.norm(q)
        y, _ = self.attn(q, q, q, need_weights=False)
        y = y.transpose(1, 2).view(B, -1, H, W)  # (B, d, H, W)
        return self.out_proj(y)                  # (B, C, H, W)

class A2C2f(nn.Module):
    """
    Attention-Augmented C2f：
    - 位置感知器：Depthwise 7x7
    - 简化空间注意力（MHA over H*W）
    - 融合：C2f 输出 + Attention 输出 -> 1x1 融合
    """
    def __init__(self, c_in, c_out, n=3, e=0.5, num_heads=4, attn_ratio=0.5, drop=0.0):
        super().__init__()
        self.c2f = C2f(c_in, c_out, n=n, e=e, shortcut=True)
        self.pos = DepthwiseSepConv(c_out)
        self.attn = SimpleMHA2D(c_out, num_heads=num_heads, attn_ratio=attn_ratio, drop=drop)
        self.fuse = ConvBNAct(2 * c_out, c_out, 1, 1)

    def forward(self, x):
        y = self.c2f(x)
        pa = self.pos(y)
        ya = self.attn(pa)
        return self.fuse(torch.cat([y, ya], dim=1))

# -------------------------------------------------------
# YOLOv12 Backbone（替换原 MambaVision）
# 默认参数与原工程一致：
#   dim=128, depths=(3,3,10,5), num_heads=(2,4,8,16)
# 结构：Stem -> Stage1/2/3/4（返回多尺度）
# 步长：4/8/16/32；通道：[dim, 2*dim, 4*dim, 8*dim]
# -------------------------------------------------------

class YOLOv12Backbone(nn.Module):
    def __init__(self,
                 dim=128,
                 in_chans=3,
                 depths=(3, 3, 10, 5),
                 mlp_ratio=2.0,
                 num_heads=(2, 4, 8, 16),
                 drop_path_rate=0.2,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 layer_scale=None,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__()

        c1, c2, c3, c4 = dim, dim*2, dim*4, dim*8  # 通道配置

        # Stem：两次 stride=2 -> 输出 P2 (stride=4)
        self.stem = nn.Sequential(
            ConvBNAct(in_chans, dim//2, 3, 2),   # stride 2
            ConvBNAct(dim//2, dim, 3, 2),        # stride 4
        )

        # Stage1 (stride 4 -> 输出 stride 4)
        self.stage1 = nn.Sequential(
            C2f(c1, c1, n=depths[0], e=0.5, shortcut=True)
        )

        # Down -> Stage2 (stride 8)
        self.down1 = ConvBNAct(c1, c2, 3, 2)
        self.stage2 = nn.Sequential(
            A2C2f(c2, c2, n=depths[1], e=0.5, num_heads=num_heads[1], attn_ratio=0.5, drop=attn_drop_rate)
        )

        # Down -> Stage3 (stride 16)
        self.down2 = ConvBNAct(c2, c3, 3, 2)
        self.stage3 = nn.Sequential(
            A2C2f(c3, c3, n=depths[2], e=0.5, num_heads=num_heads[2], attn_ratio=0.5, drop=attn_drop_rate)
        )

        # Down -> Stage4 (stride 32)
        self.down3 = ConvBNAct(c3, c4, 3, 2)
        self.stage4 = nn.Sequential(
            A2C2f(c4, c4, n=depths[3], e=0.5, num_heads=num_heads[3], attn_ratio=0.5, drop=attn_drop_rate)
        )

        # 归一化层（用于检测/分割多尺度特征）
        self.out_norms = nn.ModuleList([
            nn.BatchNorm2d(c1),
            nn.BatchNorm2d(c2),
            nn.BatchNorm2d(c3),
            nn.BatchNorm2d(c4),
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, LayerNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B,3,H,W)
        p2 = self.stem(x)        # stride 4, c1
        p2 = self.stage1(p2)     # (B, c1, H/4, W/4)

        p3 = self.down1(p2)      # stride 8, c2
        p3 = self.stage2(p3)

        p4 = self.down2(p3)      # stride 16, c3
        p4 = self.stage3(p4)

        p5 = self.down3(p4)      # stride 32, c4
        p5 = self.stage4(p5)

        outs = [p2, p3, p4, p5]
        outs = [norm(o).contiguous() for norm, o in zip(self.out_norms, outs)]
        return outs  # [P2, P3, P4, P5]

    # 兼容旧风格加载
    def _load_state_dict(self, pretrained, strict: bool = False):
        _load_checkpoint(self, pretrained, strict=strict)

# -------------------------------------------------------
# MMDet/MMseg 适配骨干（默认参数与原工程一致）
# -------------------------------------------------------

@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class MM_yolov12(YOLOv12Backbone):
    """适配 MMDetection 和 MMSegmentation 的 YOLOv12 风格骨干网络。"""
    def __init__(self,
                 dim=128,                         # ← 与原工程一致
                 in_dim=None,                     # 兼容旧签名，不使用
                 depths=(3, 3, 10, 5),            # ← 与原工程一致
                 window_size=None,                # 兼容旧签名，不使用
                 mlp_ratio=2.0,
                 num_heads=(2, 4, 8, 16),         # ← 与原工程一致
                 out_indices=(0, 1, 2, 3),        # ← 与原工程一致
                 pretrained=None,
                 norm_layer="ln2d",               # 兼容旧签名，不使用
                 layer_scale=None,
                 use_checkpoint=False,
                 enable_pcs=True,                 # 兼容旧签名，不使用
                 enable_sac=True,                 # 兼容旧签名，不使用
                 enable_sl_bridge=True,           # 兼容旧签名，不使用
                 in_chans=3,
                 **kwargs):
        super().__init__(
            dim=dim,
            in_chans=in_chans,
            depths=depths,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
        self.dims = [int(dim * 2 ** i) for i in range(0, 4)]
        self.channel_first = True
        self.out_indices = out_indices

        # 为与原版保持一致，仍提供 outnorm 层名（这里是占位）
        for i in out_indices:
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, nn.Identity())

        # 顶层分类相关（原文件里删除了分类 head）这里不提供分类 head
        self.init_weights(pretrained)

    def load_pretrained(self, ckpt=None, key="state_dict"):
        if ckpt is None:
            return
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt.get(key, _ckpt), strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def init_weights(self, pretrained=None):
        """初始化骨干网络的权重。"""
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            # super().__init__ 已经调用过 self.apply(_init_weights)
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        # 返回与 out_indices 对应的多尺度特征
        outs_all = super().forward(x)  # [P2,P3,P4,P5]
        outs = []
        for i, feat in enumerate(outs_all):
            if i in self.out_indices:
                _ = getattr(self, f'outnorm{i}')
                outs.append(feat)
        if len(self.out_indices) == 0:
            return outs_all[-1]
        return outs

# -------------------------------------------------------
# 可选：注册到 timm（如需）
# -------------------------------------------------------

@register_model
def yolov12_backbone_tiny(pretrained=False, **kwargs):
    # 与原工程的风格保持：提供不同规模的便捷构造
    model = YOLOv12Backbone(dim=96, depths=(2, 2, 6, 2), num_heads=(2, 4, 8, 8), **kwargs)
    model.default_cfg = default_cfgs['yolov12_T']
    if pretrained and 'pretrained' in kwargs:
        model._load_state_dict(kwargs['pretrained'], strict=False)
    return model

@register_model
def yolov12_backbone_small(pretrained=False, **kwargs):
    model = YOLOv12Backbone(dim=128, depths=(3, 3, 10, 5), num_heads=(2, 4, 8, 16), **kwargs)
    model.default_cfg = default_cfgs['yolov12_S']
    if pretrained and 'pretrained' in kwargs:
        model._load_state_dict(kwargs['pretrained'], strict=False)
    return model

@register_model
def yolov12_backbone_base(pretrained=False, **kwargs):
    model = YOLOv12Backbone(dim=160, depths=(3, 6, 12, 6), num_heads=(2, 4, 8, 16), **kwargs)
    model.default_cfg = default_cfgs['yolov12_B']
    if pretrained and 'pretrained' in kwargs:
        model._load_state_dict(kwargs['pretrained'], strict=False)
    return model

@register_model
def yolov12_backbone_large(pretrained=False, **kwargs):
    model = YOLOv12Backbone(dim=192, depths=(4, 8, 16, 8), num_heads=(4, 8, 8, 16), **kwargs)
    model.default_cfg = default_cfgs['yolov12_L']
    if pretrained and 'pretrained' in kwargs:
        model._load_state_dict(kwargs['pretrained'], strict=False)
    return model
