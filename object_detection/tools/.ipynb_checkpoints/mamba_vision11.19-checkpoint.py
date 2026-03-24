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
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mmdet.registry import MODELS as MODELS_MMDET
from mmseg.registry import MODELS as MODELS_MMSEG
from timm.models._builder import resolve_pretrained_cfg
try:
    from timm.models._builder import _update_default_kwargs as update_args
except Exception:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import Mlp, PatchEmbed as TimmPatchEmbed
from torch.cuda.amp import autocast
from torch.utils import checkpoint

# -------------------------------------------------------
# 基础配置
# -------------------------------------------------------

def _cfg(url='', **kwargs):
    return {
        'url': url,
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
    'mamba_vision_T': _cfg(
        url='https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar',
        crop_pct=1.0,
        input_size=(3, 224, 224),
        crop_mode='center'),
    'mamba_vision_T2': _cfg(
        url='https://huggingface.co/nvidia/MambaVision-T2-1K/resolve/main/mambavision_tiny2_1k.pth.tar',
        crop_pct=0.98,
        input_size=(3, 224, 224),
        crop_mode='center'),
    'mamba_vision_S': _cfg(
        url='https://huggingface.co/nvidia/MambaVision-S-1K/resolve/main/mambavision_small_1k.pth.tar',
        crop_pct=0.93,
        input_size=(3, 224, 224),
        crop_mode='center'),
    'mamba_vision_B': _cfg(
        url='https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar',
        crop_pct=1.0,
        input_size=(3, 224, 224),
        crop_mode='center'),
    'mamba_vision_B_21k': _cfg(
        url='https://huggingface.co/nvidia/MambaVision-B-21K/resolve/main/mambavision_base_21k.pth.tar',
        crop_pct=1.0,
        input_size=(3, 224, 224),
        crop_mode='center'),
    'mamba_vision_L': _cfg(
        url='https://huggingface.co/nvidia/MambaVision-L-1K/resolve/main/mambavision_large_1k.pth.tar',
        crop_pct=1.0,
        input_size=(3, 224, 224),
        crop_mode='center'),
    'mamba_vision_L_21k': _cfg(
        url='https://huggingface.co/nvidia/MambaVision-L-21K/resolve/main/mambavision_large_21k.pth.tar',
        crop_pct=1.0,
        input_size=(3, 224, 224),
        crop_mode='center'),
    'mamba_vision_L2': _cfg(
        url='https://huggingface.co/nvidia/MambaVision-L2-1K/resolve/main/mambavision_large2_1k.pth.tar',
        crop_pct=1.0,
        input_size=(3, 224, 224),
        crop_mode='center'),
    'mamba_vision_L2_512_21k': _cfg(
        url='https://huggingface.co/nvidia/MambaVision-L2-512-21K/resolve/main/mambavision_L2_21k_240m_512.pth.tar',
        crop_pct=0.93,
        input_size=(3, 512, 512),
        crop_mode='squash'),
    'mamba_vision_L3_256_21k': _cfg(
        url='https://huggingface.co/nvidia/MambaVision-L3-256-21K/resolve/main/mambavision_L3_21k_740m_256.pth.tar',
        crop_pct=1.0,
        input_size=(3, 256, 256),
        crop_mode='center'),
    'mamba_vision_L3_512_21k': _cfg(
        url='https://huggingface.co/nvidia/MambaVision-L3-512-21K/resolve/main/mambavision_L3_21k_740m_512.pth.tar',
        crop_pct=0.93,
        input_size=(3, 512, 512),
        crop_mode='squash'),
}

# -------------------------------------------------------
# 工具函数（窗口分割/恢复、权重加载）
# -------------------------------------------------------

def window_partition(x, window_size):
    """
    x: (B, C, H, W)
    return: (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(
        B, C,
        H // window_size, window_size,
        W // window_size, window_size)
    windows = (
        x.permute(0, 2, 4, 3, 5, 1)
         .reshape(-1, window_size * window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    windows: (num_windows*B, window_size*window_size, C)
    return: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    C = windows.shape[-1]
    x = windows.reshape(
        B,
        H // window_size, W // window_size,
        window_size, window_size,
        C,
    )
    x = (
        x.permute(0, 5, 1, 3, 2, 4)
         .reshape(B, C, H, W)
    )
    return x


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
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True,
            all_missing_keys, unexpected_keys, err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append(
            'unexpected key in source '
            f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: '
            f'{", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
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
    """加载检查点文件并处理其中的 state_dict。"""
    checkpoint_file = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint_file, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint_file:
        state_dict = checkpoint_file['state_dict']
    elif 'model' in checkpoint_file:
        state_dict = checkpoint_file['model']
    else:
        state_dict = checkpoint_file
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {
            k.replace('encoder.', ''): v
            for k, v in state_dict.items()
            if k.startswith('encoder.')
        }
    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint_file

# -------------------------------------------------------
# 归一化与下采样、PatchEmbed
# -------------------------------------------------------

class LayerNorm2d(nn.LayerNorm):
    """2D 版本的 LayerNorm。"""
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(
            x, self.normalized_shape,
            self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Downsample(nn.Module):
    """下采样模块，使用步长为2的卷积实现。"""
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """将图像转换为 Patch Embeddings。"""
    def __init__(self, in_chans=3, in_dim=64, dim=96):
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x

# -------------------------------------------------------
# ConvBlock
# -------------------------------------------------------

class ConvBlock(nn.Module):
    """传统的卷积残差块。"""
    def __init__(self, dim, drop_path=0., layer_scale=None, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            dim, dim, kernel_size=kernel_size,
            stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv2d(
            dim, dim, kernel_size=kernel_size,
            stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale_flag = False
        if (layer_scale is not None
                and isinstance(layer_scale, (int, float))):
            safe_gamma = float(layer_scale)
            safe_gamma = max(min(safe_gamma, 10.0), 1e-4)
            self.gamma = nn.Parameter(
                safe_gamma * torch.ones(dim))
            self.layer_scale_flag = True
        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0. else nn.Identity()
        )

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale_flag:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = shortcut + self.drop_path(x)
        x = torch.nan_to_num(x)
        return x

# -----------------------------
# MambaVisionMixer（含稳定性修复）
# -----------------------------

class MambaVisionMixer(nn.Module):
    """Mamba 混合器模块，用于处理 token 序列。"""
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = (
            math.ceil(self.d_model / 16)
            if dt_rank == "auto" else dt_rank
        )
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner,
            bias=bias, **factory_kwargs)

        self.x_proj = nn.Linear(
            self.d_inner // 2,
            self.dt_rank + self.d_state * 2,
            bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(
            self.dt_rank,
            self.d_inner // 2,
            bias=True, **factory_kwargs
        )
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(
                self.dt_proj.weight,
                -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(
                self.d_inner // 2, **factory_kwargs
            ) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(
                1, self.d_state + 1,
                dtype=torch.float32,
                device=device,
            ),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(
            torch.ones(
                self.d_inner // 2,
                device=device
            )
        )
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model,
            bias=bias, **factory_kwargs
        )

        # conv1d depthwise
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

        # LayerNorm for x_dbl in fp32 for stability
        self.ln_x_dbl = nn.LayerNorm(
            self.dt_rank + self.d_state * 2,
            dtype=torch.float32
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: (B, L, D)
        """
        B, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)          # (B,L,d_inner)
        xz = rearrange(xz, "b l d -> b d l")      # (B,d_inner,L)
        x, z = xz.chunk(2, dim=1)                 # (B,d_inner/2,L),(B,d_inner/2,L)

        with autocast(enabled=False):
            x_float = x.float()
            z_float = z.float()

            bias_x_float = (
                self.conv1d_x.bias.float()
                if self.conv1d_x.bias is not None else None
            )
            bias_z_float = (
                self.conv1d_z.bias.float()
                if self.conv1d_z.bias is not None else None
            )

            x_conv = F.silu(
                F.conv1d(
                    input=x_float,
                    weight=self.conv1d_x.weight.float(),
                    bias=bias_x_float,
                    padding='same',
                    groups=self.d_inner // 2,
                )
            )
            z_conv = F.silu(
                F.conv1d(
                    input=z_float,
                    weight=self.conv1d_z.weight.float(),
                    bias=bias_z_float,
                    padding='same',
                    groups=self.d_inner // 2,
                )
            )

            A = -torch.exp(self.A_log.float())
            D = self.D.float()
            delta_bias = self.dt_proj.bias.float()

            x_dbl = self.x_proj(
                rearrange(x_conv, "b d l -> (b l) d")
            )
            x_dbl = self.ln_x_dbl(x_dbl)

            dt, B_par, C_par = torch.split(
                x_dbl,
                [self.dt_rank,
                 self.d_state,
                 self.d_state],
                dim=-1
            )
            dt = rearrange(
                self.dt_proj(dt),
                "(b l) d -> b d l",
                l=seqlen
            )
            B_par = rearrange(
                B_par,
                "(b l) dstate -> b dstate l",
                l=seqlen
            ).contiguous()
            C_par = rearrange(
                C_par,
                "(b l) dstate -> b dstate l",
                l=seqlen
            ).contiguous()

            y = selective_scan_fn(
                x_conv, dt, A, B_par, C_par, D, z=None,
                delta_bias=delta_bias,
                delta_softplus=True,
                return_last_state=None
            )

            # 防 NaN / Inf 护栏
            y = torch.nan_to_num(y)

            y = torch.cat([y, z_conv], dim=1)

        y = y.to(dtype=hidden_states.dtype)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

# -----------------------------
# Attention（增强版，支持外部 Query 偏置）
# -----------------------------

class AttentionSAC(nn.Module):
    """
    支持外部 query 偏置（state-aware bias）的局部窗口注意力；
    兼容 PyTorch SDPA。
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.,
        proj_drop=0.,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = (
            norm_layer(self.head_dim)
            if qk_norm else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim)
            if qk_norm else nn.Identity()
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x,
        q_bias_per_head: Optional[torch.Tensor] = None
    ):
        """
        x: (B, N, C)
        q_bias_per_head: (B, num_heads, head_dim) 或 (1, num_heads, head_dim)
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3,
                     self.num_heads,
                     self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if q_bias_per_head is not None:
            q = q + q_bias_per_head.unsqueeze(-2)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# -------------------------------------------------------
# SAC：状态感知交叉注意 + 写回门控
# -------------------------------------------------------

class SACBridge(nn.Module):
    """
    将 M-Stream（Mamba）状态用于生成 T-Stream 的查询偏置，
    并用 T-Stream 的聚合特征门控写回 M-Stream。
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_bias_gen = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_heads * self.head_dim)
        )
        self.writeback_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.writeback_proj = nn.Linear(dim, dim)

    def forward(
        self,
        m_tokens: torch.Tensor,   # (B, N, C)
        t_tokens: torch.Tensor    # (B, N, C)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m_state = m_tokens.mean(dim=1)  # (B, C)
        q_bias = self.q_bias_gen(m_state).view(
            -1, self.num_heads, self.head_dim
        )  # (B, H, Dh)

        t_state = t_tokens.mean(dim=1)  # (B, C)
        gate = self.writeback_gate(t_state)          # (B, C)
        write_feat = self.writeback_proj(
            t_state
        ).unsqueeze(1)                                # (B,1,C)
        m_tokens_enh = m_tokens + gate.unsqueeze(1) * write_feat  # (B,N,C)

        return q_bias, m_tokens_enh

# -------------------------------------------------------
# PCS-Scan：四向扫描的 Mamba（H/V/diag/anti-diag）
# -------------------------------------------------------

def _diag_indices(w: int):
    inds = []
    for s in range(2 * w - 1):
        row_start = max(0, s - (w - 1))
        row_end = min(w - 1, s)
        line = []
        for r in range(row_start, row_end + 1):
            c = s - r
            line.append((r, c))
        inds.extend(line)
    return inds

def _anti_diag_indices(w: int):
    inds = []
    for s in range(2 * w - 1):
        row_start = max(0, s - (w - 1))
        row_end = min(w - 1, s)
        line = []
        for r in range(row_start, row_end + 1):
            c = s - r
            line.append((r, w - 1 - c))
        inds.extend(line)
    return inds

class PCSScanMamba(nn.Module):
    """
    Polarized Cross-Scan SSM:
    - 将窗口内 token 重排为 4 种序列（水平/垂直/主对角/副对角）分别过 Mamba，
      以可学习 gate 融合四路输出。
    """
    def __init__(self, dim, d_state=8, d_conv=3, expand=1, window_size=7):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        self.pre_h = nn.Linear(dim, dim)
        self.pre_v = nn.Linear(dim, dim)
        self.pre_d = nn.Linear(dim, dim)
        self.pre_a = nn.Linear(dim, dim)

        self.mamba_core = MambaVisionMixer(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.post_h = nn.Linear(dim, dim)
        self.post_v = nn.Linear(dim, dim)
        self.post_d = nn.Linear(dim, dim)
        self.post_a = nn.Linear(dim, dim)

        self.gate_logits = nn.Parameter(torch.zeros(4))

    def _reorder(self, x: torch.Tensor, mode: str):
        """
        x: (B, N, C) where N = w*w
        returns (B, N, C) reordered
        """
        B, N, C = x.shape
        w = self.window_size
        assert N == w * w, (
            f"N ({N}) must equal window_size^2 ({w}^2)."
        )

        grid = x.view(B, w, w, C)  # (B,w,w,C)

        if mode == 'h':
            y = grid.reshape(B, N, C)
            return y
        elif mode == 'v':
            y = (
                grid.permute(0, 2, 1, 3)
                .contiguous()
                .view(B, N, C)
            )
            return y
        elif mode == 'd':
            idxs = _diag_indices(w)
        elif mode == 'a':
            idxs = _anti_diag_indices(w)
        else:
            raise ValueError

        r_idx = torch.tensor(
            [rc[0] for rc in idxs],
            device=x.device,
            dtype=torch.long
        )
        c_idx = torch.tensor(
            [rc[1] for rc in idxs],
            device=x.device,
            dtype=torch.long
        )
        y = grid[:, r_idx, c_idx, :]  # (B,N,C)
        return y

    def forward(self, tokens_window: torch.Tensor):
        """
        tokens_window: (B, N, C), N = window_size^2
        """
        B, N, C = tokens_window.shape

        x_h = self.pre_h(tokens_window)
        x_v = self.pre_v(tokens_window)
        x_d = self.pre_d(tokens_window)
        x_a = self.pre_a(tokens_window)

        rh = self._reorder(x_h, 'h')
        rv = self._reorder(x_v, 'v')
        rd = self._reorder(x_d, 'd')
        ra = self._reorder(x_a, 'a')

        r_all = torch.cat([rh, rv, rd, ra], dim=0)    # (4B,N,C)
        y_all = self.mamba_core(r_all)                # (4B,N,C)
        yh, yv, yd, ya = torch.chunk(y_all, 4, dim=0)

        yh = self.post_h(yh)
        yv = self.post_v(yv)
        yd = self.post_d(yd)
        ya = self.post_a(ya)

        g = torch.softmax(self.gate_logits, dim=0)    # (4,)
        y = g[0] * yh + g[1] * yv + g[2] * yd + g[3] * ya
        y = torch.nan_to_num(y)
        return y  # (B,N,C)

# -------------------------------------------------------
# CSDSBlock（稳定化版本：禁用 DropPath 乘法，并在 fp32 中主干计算）
# -------------------------------------------------------

class CSDSBlock(nn.Module):
    """
    双流并行：
      - M-Stream: PCSScanMamba（四向扫描）
      - T-Stream: 局部窗口注意力（支持 SAC 的 Query 偏置）

    我们做了两件额外的稳定性处理：
      1. 主干计算在 autocast(False) 下用 fp32 完成，再 cast 回去，降低 AMP 下的梯度爆炸。
      2. DropPath 禁用成恒等映射，避免 MulBackward0 在 backward 里产生 NaN。

    SAC：
      - 用 M-Stream 的全局状态生成 T-Stream Query 偏置；
      - 用 T-Stream 的聚合门控写回 M-Stream 状态。

    融合：
      - 拼接后线性融合，后接前馈。
    """
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        mlp_ratio=4.,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        layer_scale=None,
        d_state=8,
        d_conv=3,
        expand=1,
        norm_layer=nn.LayerNorm,
        enable_pcs: bool = True,
        enable_sac: bool = True,
        enable_sl_bridge: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.enable_pcs = enable_pcs
        self.enable_sac = enable_sac
        self.enable_sl_bridge = enable_sl_bridge

        # 归一化
        self.norm_m = norm_layer(dim)
        self.norm_t = norm_layer(dim)

        # 两条分支
        self.m_stream = (
            PCSScanMamba(
                dim, d_state=d_state,
                d_conv=d_conv, expand=expand,
                window_size=window_size
            )
            if self.enable_pcs else nn.Identity()
        )
        self.t_stream = AttentionSAC(
            dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=False,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer
        )

        # SAC 桥
        self.sac = SACBridge(dim, num_heads) if self.enable_sac else None

        # 融合
        self.fuse = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # MLP
        self.norm_ffn = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = Mlp(
            in_features=dim,
            hidden_features=hidden,
            act_layer=nn.GELU,
            drop=drop
        )

        # 🔒 DropPath 禁用，避免 MulBackward0 的随机乘法
        self.drop_path = nn.Identity()

        # layer scale
        def _safe_gamma_init(v, ch):
            safe_v = float(v)
            safe_v = max(min(safe_v, 10.0), 1e-4)
            return nn.Parameter(safe_v * torch.ones(ch))

        if (layer_scale is not None
                and isinstance(layer_scale, (int, float))):
            self.gamma_1 = _safe_gamma_init(layer_scale, dim)
            self.gamma_2 = _safe_gamma_init(layer_scale, dim)
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def _scale(self, x, gamma):
        # x: (B,N,C)
        if gamma is None:
            x_safe = torch.nan_to_num(x)
            return x_safe
        x_safe = torch.nan_to_num(x)
        out = x_safe * gamma.view(1, 1, -1)
        out = torch.nan_to_num(out)
        return out

    def _residual_add_identitydroppath(self, base, update, gamma):
        # base, update: (B,N,C)
        upd_scaled = self._scale(update, gamma)
        # DropPath 已经是恒等，这里不会再走随机mask乘法
        out = base + self.drop_path(upd_scaled)
        out = torch.nan_to_num(out)
        return out

    def forward(
        self,
        x_tokens: torch.Tensor,
        sl_bridge_state: Optional[torch.Tensor] = None
    ):
        """
        x_tokens: (B, N, C)
        sl_bridge_state: (B, C) or None
        """
        inp_dtype = x_tokens.dtype

        # 在 fp32 下做整块运算，最后再 cast 回原 dtype
        with autocast(enabled=False):
            x_tokens32 = x_tokens.float()

            # SL-Bridge：FiLM-like 调制
            if self.enable_sl_bridge and sl_bridge_state is not None:
                alpha = 0.1
                x_tokens32 = x_tokens32 + alpha * sl_bridge_state.float().unsqueeze(1)

            # 归一化
            xm = self.norm_m(x_tokens32)
            xt = self.norm_t(x_tokens32)

            # M-Stream
            if self.enable_pcs:
                y_m0 = self.m_stream(xm)  # (B,N,C)
            else:
                y_m0 = xm

            # SAC 路由 + 注意力
            if self.enable_sac:
                q_bias, y_m_enh = self.sac(y_m0, xt)
                y_t = self.t_stream(xt, q_bias_per_head=q_bias)
            else:
                y_m_enh = y_m0
                y_t = self.t_stream(xt, q_bias_per_head=None)

            # 融合
            y = torch.cat([y_m_enh, y_t], dim=-1)  # (B,N,2C)
            y = self.fuse(y)                       # (B,N,C)
            y = torch.nan_to_num(y)

            # 残差1（稳态）
            x1 = self._residual_add_identitydroppath(x_tokens32, y, self.gamma_1)

            # FFN
            y_ffn = self.ffn(self.norm_ffn(x1))
            y_ffn = torch.nan_to_num(y_ffn)

            # 残差2（稳态）
            out32 = self._residual_add_identitydroppath(x1, y_ffn, self.gamma_2)

            # 当前 block 的 M-Stream 状态，用于 SL-Bridge
            m_state = y_m_enh.mean(dim=1)  # (B,C)
            m_state = torch.nan_to_num(m_state)

        out = out32.to(dtype=inp_dtype)
        m_state = m_state.to(dtype=inp_dtype)
        return out, m_state

# -------------------------------------------------------
# CSDSLayer（调用上面稳定版 block，并把 DropPath 统一置0）
# -------------------------------------------------------

class CSDSLayer(nn.Module):
    """
    一个 Stage：
      - 输入 (B,C,H,W)
      - 窗口分块 -> (B*num_windows, win^2, C)
      - 堆叠 L 个 CSDSBlock（每个返回局部 M 状态）
      - 将所有窗口恢复到 (B,C,H,W)
      - 降采样到下一 stage，并返回：
          * x_down: (B, C_out, H/2, W/2) 作为下一 stage 输入
          * x_res:  (B, C, H, W)         作为当前 stage 输出特征
          * sl_state_stage: (B, C)       本 stage 聚合的 M 状态
    """
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.,
        qkv_bias=True,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        layer_scale=None,
        downsample=True,
        d_state=8,
        d_conv=3,
        expand=1,
        use_checkpoint=False,
        prev_dim: Optional[int] = None,
        enable_pcs: bool = True,
        enable_sac: bool = True,
        enable_sl_bridge: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.enable_sl_bridge = enable_sl_bridge
        self.downsample = Downsample(dim=dim) if downsample else None

        # 不再把外部的 drop_path rate 用到 block 内部，直接在 block 里禁用 DropPath
        self.blocks = nn.ModuleList([
            CSDSBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=0.0,  # 强制0，防止MulBackward0 NaN
                layer_scale=layer_scale,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                norm_layer=nn.LayerNorm,
                enable_pcs=enable_pcs,
                enable_sac=enable_sac,
                enable_sl_bridge=enable_sl_bridge,
            )
            for _ in range(depth)
        ])

        # 跨 stage 状态投影层
        if prev_dim is not None and self.enable_sl_bridge:
            self.sl_proj_in = nn.Linear(prev_dim, self.dim)
        else:
            self.sl_proj_in = None

        self.sl_proj_agg = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        sl_bridge_state: Optional[torch.Tensor] = None
    ):
        """
        x: (B,C,H,W)
        sl_bridge_state: (B,C_prev) 来自上一 Stage 的记忆状态
        """
        B, C, H, W = x.shape
        ws = self.window_size

        # padding 到窗口倍数
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))
            _, _, Hp, Wp = x.shape
        else:
            Hp, Wp = H, W

        tokens = window_partition(x, ws)  # (Bnw, ws*ws, C)

        # 跨 stage 的 SL-Bridge 状态
        if sl_bridge_state is not None and self.sl_proj_in is not None:
            target_dtype = self.sl_proj_in.weight.dtype
            sl_bridge_state_casted = sl_bridge_state.to(
                dtype=target_dtype
            )
            sl = self.sl_proj_in(sl_bridge_state_casted)  # (B,self.dim)

            # Bnw = B * (Hp/ws) * (Wp/ws)
            num_wins = tokens.shape[0] // B
            sl = sl.repeat_interleave(num_wins, dim=0)  # (Bnw,self.dim)
        else:
            sl = None

        # 堆叠多层 CSDSBlock
        m_states = []
        for blk in self.blocks:
            if self.training and self.use_checkpoint:
                tokens, m_state_blk = checkpoint.checkpoint(
                    blk,
                    tokens,
                    sl_bridge_state=sl,
                    use_reentrant=False
                )
            else:
                tokens, m_state_blk = blk(
                    tokens,
                    sl_bridge_state=sl
                )
            m_states.append(m_state_blk)

        # 还原窗口
        x_feat = window_reverse(tokens, ws, Hp, Wp)  # (B,C,Hp,Wp)
        if pad_r > 0 or pad_b > 0:
            x_feat = x_feat[:, :, :H, :W].contiguous()

        x_feat = torch.nan_to_num(x_feat)

        # 汇聚 stage 记忆状态
        if len(m_states) > 0:
            m_last = m_states[-1]              # (Bnw,C)
            num_wins = m_last.shape[0] // B
            m_last = m_last.view(B, num_wins, -1).mean(dim=1)  # (B,C)
            sl_state_stage = self.sl_proj_agg(m_last)          # (B,C)
        else:
            sl_state_stage = torch.zeros(
                B, C,
                device=x.device,
                dtype=x.dtype
            )

        sl_state_stage = torch.nan_to_num(sl_state_stage)

        # 下采样给下一 stage
        x_down = (
            self.downsample(x_feat)
            if self.downsample is not None else None
        )
        if x_down is not None:
            x_down = torch.nan_to_num(x_down)

        return (
            x_down if x_down is not None else x_feat,
            x_feat,
            sl_state_stage
        )

# -------------------------------------------------------
# 顶层骨干：CSDS-Backbone
# -------------------------------------------------------

class MambaVision(nn.Module):
    """
    CSDS-Backbone（替换原 MambaVision）：
      - Stage1/2/3/4：CSDSLayer
      - 跨 Stage 的 SL-Bridge：将上一 stage 输出的 sl_state 传入下一 stage
    """
    def __init__(
        self,
        dim=128,
        in_dim=64,
        depths=(3, 3, 10, 5),
        window_size=(8, 8, 14, 7),
        mlp_ratio=4.0,
        num_heads=(2, 4, 8, 16),
        drop_path_rate=0.2,
        in_chans=3,
        num_classes=1000,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        layer_scale=None,
        layer_scale_conv=None,
        use_checkpoint=False,
        enable_pcs: bool = True,
        enable_sac: bool = True,
        enable_sl_bridge: bool = True,
        **kwargs,
    ):
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            in_dim=in_dim,
            dim=dim
        )

        # DropPath 分配（仍然计算 dpr 但不会真的用进 block，因为 block 内硬置0）
        dpr = [
            x.item()
            for x in torch.linspace(
                0, drop_path_rate, sum(depths)
            )
        ]

        # Stages
        self.levels = nn.ModuleList()
        cur = 0
        prev_dim = None
        for i in range(len(depths)):
            current_dim = int(dim * 2 ** i)
            level = CSDSLayer(
                dim=current_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur:cur + depths[i]],  # 会被CSDSLayer忽略并传0.0给Block
                layer_scale=layer_scale,
                downsample=(i < len(depths) - 1),
                d_state=8,
                d_conv=3,
                expand=1,
                use_checkpoint=use_checkpoint,
                prev_dim=prev_dim,
                enable_pcs=enable_pcs,
                enable_sac=enable_sac,
                enable_sl_bridge=enable_sl_bridge,
            )
            cur += depths[i]
            prev_dim = current_dim
            self.levels.append(level)

        self.norm = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = (
            nn.Linear(num_features, num_classes)
            if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        x = self.patch_embed(x)   # (B,C,H,W)
        sl_state = None
        for idx, level in enumerate(self.levels):
            x, feat, sl_state = level(
                x,
                sl_bridge_state=sl_state
            )
        x = self.norm(feat)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = torch.nan_to_num(x)
        return x

    def _load_state_dict(self, pretrained, strict: bool = False):
        _load_checkpoint(self, pretrained, strict=strict)

# -------------------------------------------------------
# MMDet/MMseg 适配骨干
# -------------------------------------------------------

@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class MM_mamba_vision(MambaVision):
    """适配 MMDetection 和 MMSegmentation 的 MambaVision 骨干网络。"""
    def __init__(
        self,
        dim,
        in_dim,
        depths,
        window_size,
        mlp_ratio,
        num_heads,
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        norm_layer="ln2d",
        layer_scale=None,
        use_checkpoint=False,
        enable_pcs: bool = True,
        enable_sac: bool = True,
        enable_sl_bridge: bool = True,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            in_dim=in_dim,
            depths=depths,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            layer_scale=layer_scale,
            use_checkpoint=use_checkpoint,
            enable_pcs=enable_pcs,
            enable_sac=enable_sac,
            enable_sl_bridge=enable_sl_bridge,
            **kwargs,
        )
        self.dims = [int(dim * 2 ** i) for i in range(0, 4)]
        self.channel_first = True
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer_mod: nn.Module = _NORMLAYERS.get(
            norm_layer.lower(), None
        )

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer_mod(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        # 顶层分类头不需要
        del self.norm
        del self.head
        self.init_weights(pretrained)

    def load_pretrained(self, ckpt=None, key="state_dict"):
        if ckpt is None:
            return
        try:
            _ckpt = torch.load(
                open(ckpt, "rb"),
                map_location=torch.device("cpu")
            )
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(
                _ckpt[key], strict=False
            )
            print(incompatibleKeys)
        except Exception as e:
            print(
                f"Failed loading checkpoint form {ckpt}: {e}"
            )

    def init_weights(self, pretrained=None):
        """初始化骨干网络的权重。"""
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            # 父类构造函数里已经调用了 self.apply(self._init_weights)
            pass
        else:
            raise TypeError(
                'pretrained must be a str or None'
            )

    def forward(self, x):
        """
        前向传播，返回多尺度特征图列表。
        """
        x = self.patch_embed(x)  # (B,C,H,W)

        outs = []
        sl_state = None
        for i, level in enumerate(self.levels):
            x, feat, sl_state = level(
                x,
                sl_bridge_state=sl_state
            )  # feat: (B,C_i,H_i,W_i)
            if i in self.out_indices:
                norm_layer_i = getattr(self, f'outnorm{i}')
                out = norm_layer_i(feat)
                out = torch.nan_to_num(out)
                outs.append(out.contiguous())

        if len(self.out_indices) == 0:
            return x

        return outs
