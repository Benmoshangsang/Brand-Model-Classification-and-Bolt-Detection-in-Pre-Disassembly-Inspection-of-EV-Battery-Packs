#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch                                            # Import the main PyTorch library
import torch.nn as nn                                   # Import PyTorch neural network modules
from timm.models.registry import register_model         # Import model registry from timm
import math                                             # Import math library
from timm.models.layers import trunc_normal_, DropPath  # Import weight initialization and DropPath from timm
from timm.models._builder import resolve_pretrained_cfg # Import pretrained config resolver from timm
try:
    from timm.models._builder import _update_default_kwargs as update_args # Try to import function for updating default arguments
except:
    from timm.models._builder import _update_default_model_kwargs as update_args # Compatibility with older timm versions
from timm.models.vision_transformer import Mlp, PatchEmbed as TimmPatchEmbed # Import MLP and PatchEmbed from timm
from timm.models.layers import DropPath, trunc_normal_  # Re-import DropPath and trunc_normal_
from timm.models.registry import register_model         # Re-import model registry
import torch.nn.functional as F                         # Import PyTorch functional API
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn # Import the core scan operation from mamba_ssm
from einops import rearrange, repeat                     # Import tensor manipulation tools from einops
from pathlib import Path                                # Import path handling library
import os                                               # Import operating system interface
from functools import partial                           # Import partial function utility
from typing import Callable, Optional, Tuple            # Import type hints
from torch.utils import checkpoint                      # Import gradient checkpointing
from mmengine.model import BaseModule                   # Import base model module from mmengine
from mmdet.registry import MODELS as MODELS_MMDET       # Import model registry from mmdet
from mmseg.registry import MODELS as MODELS_MMSEG       # Import model registry from mmseg
import mmcv                                             # Import mmcv library
from mmengine.runner import load_checkpoint             # Import checkpoint loading function from mmengine
from torch.cuda.amp import autocast                     # Import autocast for mixed precision control

# -------------------------------------------------------
# Basic configuration (kept unchanged)
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
    'mamba_vision_T': _cfg(url='https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_T2': _cfg(url='https://huggingface.co/nvidia/MambaVision-T2-1K/resolve/main/mambavision_tiny2_1k.pth.tar',
                            crop_pct=0.98,
                            input_size=(3, 224, 224),
                            crop_mode='center'),
    'mamba_vision_S': _cfg(url='https://huggingface.co/nvidia/MambaVision-S-1K/resolve/main/mambavision_small_1k.pth.tar',
                           crop_pct=0.93,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_B': _cfg(url='https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_B_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-B-21K/resolve/main/mambavision_base_21k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_L': _cfg(url='https://huggingface.co/nvidia/MambaVision-L-1K/resolve/main/mambavision_large_1k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_L_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-L-21K/resolve/main/mambavision_large_21k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_L2': _cfg(url='https://huggingface.co/nvidia/MambaVision-L2-1K/resolve/main/mambavision_large2_1k.pth.tar',
                            crop_pct=1.0,
                            input_size=(3, 224, 224),
                            crop_mode='center'),
    'mamba_vision_L2_512_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-L2-512-21K/resolve/main/mambavision_L2_21k_240m_512.pth.tar',
                            crop_pct=0.93,
                            input_size=(3, 512, 512),
                            crop_mode='squash'),
    'mamba_vision_L3_256_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-L3-256-21K/resolve/main/mambavision_L3_21k_740m_256.pth.tar',
                            crop_pct=1.0,
                            input_size=(3, 256, 256),
                            crop_mode='center'),
    'mamba_vision_L3_512_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-L3-512-21K/resolve/main/mambavision_L3_21k_740m_512.pth.tar',
                            crop_pct=0.93,
                            input_size=(3, 512, 512),
                            crop_mode='squash'),                               
}

# -------------------------------------------------------
# Utility functions (window partition/reverse, weight loading)
# -------------------------------------------------------

def window_partition(x, window_size):
    """
    Partition the feature map into non-overlapping windows.
    Args:
        x: (B, C, H, W) input feature map
        window_size: window size
    Returns:
        local window features: (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape                                                                      # Get input dimensions
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)           # Reshape into window view
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)            # Rearrange dimensions and flatten
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Restore window features back to the original feature map.
    Args:
        windows: (num_windows*B, window_size*window_size, C) window features
        window_size: window size
        H, W: height and width of the original feature map
    Returns:
        x: (B, C, H, W) restored feature map
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))                           # Compute the original batch size
    C = windows.shape[-1]                                                                     # Get the number of channels
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, C)  # Reshape to window view
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)                                       # Rearrange and restore
    return x


def _load_state_dict(module, state_dict, strict=False, logger=None):
    """Custom weight loading function for handling partially matched state_dict."""
    unexpected_keys = []                                                                      # Unexpected keys
    all_missing_keys = []                                                                     # All missing keys
    err_msg = []                                                                              # Error messages

    metadata = getattr(state_dict, '_metadata', None)                                         # Get metadata
    state_dict = state_dict.copy()                                                            # Copy state_dict
    if metadata is not None:
        state_dict._metadata = metadata
    
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)                                                                              # Recursive loading
    load = None
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    
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
    """Load a checkpoint file and process the contained state_dict."""
    checkpoint = torch.load(filename, map_location=map_location)                              # Load file
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):                                      # Remove 'module.' prefix
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):                              # Remove 'encoder.' prefix
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    _load_state_dict(model, state_dict, strict, logger)                                       # Call custom loading function
    return checkpoint

# -------------------------------------------------------
# Normalization, downsampling, and PatchEmbed (kept unchanged)
# -------------------------------------------------------

class LayerNorm2d(nn.LayerNorm):
    """2D version of LayerNorm."""
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)                                                             # (B, C, H, W) -> (B, H, W, C)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps) # Apply LayerNorm
        x = x.permute(0, 3, 1, 2)                                                             # (B, H, W, C) -> (B, C, H, W)
        return x


class Downsample(nn.Module):
    """Downsampling module implemented with a stride-2 convolution."""
    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
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
    """Convert an image into patch embeddings."""
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
# Old ConvBlock/Mixer/Attention are kept and reused as components
# (Attention will be enhanced into the SAC version)
# -------------------------------------------------------

class ConvBlock(nn.Module):
    """Standard convolutional residual block."""
    def __init__(self, dim, drop_path=0., layer_scale=None, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x

# -----------------------------
# Base Mamba Mixer (token sequence) —— [Core bug fix]
# -----------------------------

class MambaVisionMixer(nn.Module):
    """Mamba mixer module for processing token sequences."""
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
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        # ✅ [Core bug fix] Replace conv_bias//2 with conv_bias
        # to correctly pass the boolean value
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias, # Fixed bug: directly pass the boolean value
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias, # Fixed bug: directly pass the boolean value
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        
        # ✅ [Numerical stability fix] Pre-create a float32 LayerNorm
        # to use inside autocast
        self.ln_x_dbl = nn.LayerNorm(self.dt_rank + self.d_state * 2, dtype=torch.float32)

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: (B, L, D)
        """
        _, seqlen, _ = hidden_states.shape                                     # Get sequence length
        xz = self.in_proj(hidden_states)                                       # Input linear projection
        xz = rearrange(xz, "b l d -> b d l")                                   # Rearrange for 1D convolution
        x, z = xz.chunk(2, dim=1)                                              # Split projection into x and z
        
        # ✅ [Core fix]: improve numerical stability
        with autocast(enabled=False):
            x_float = x.float()
            z_float = z.float()
            
            # ✅ [Core bug fix] Robustly handle the case where bias may be None
            bias_x_float = self.conv1d_x.bias.float() if self.conv1d_x.bias is not None else None
            bias_z_float = self.conv1d_z.bias.float() if self.conv1d_z.bias is not None else None

            # 1D convolution part
            x_conv = F.silu(F.conv1d(input=x_float, weight=self.conv1d_x.weight.float(), bias=bias_x_float, padding='same', groups=self.d_inner//2))
            z_conv = F.silu(F.conv1d(input=z_float, weight=self.conv1d_z.weight.float(), bias=bias_z_float, padding='same', groups=self.d_inner//2))
            
            # SSM (state space model) parameter computation
            A = -torch.exp(self.A_log.float())
            D = self.D.float()
            delta_bias = self.dt_proj.bias.float()

            # Dynamic parameter computation (dt, B, C)
            x_dbl = self.x_proj(rearrange(x_conv, "b d l -> (b l) d"))
            
            x_dbl = self.ln_x_dbl(x_dbl)
            
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            
            # Core selective scan function
            y = selective_scan_fn(
                x_conv, dt, A, B, C, D, z=None,
                delta_bias=delta_bias,
                delta_softplus=True,
                return_last_state=None
            )

            # ✅ [Speed and stability optimization] Add a safeguard here
            # to prevent NaN or Inf from selective_scan_fn,
            # avoiding gradient explosion
            y = torch.nan_to_num(y)

            # Concatenate z
            y = torch.cat([y, z_conv], dim=1)

        y = y.to(dtype=hidden_states.dtype)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        
        return out

# -----------------------------
# Attention (enhanced version, supports external Query bias) — part of SAC
# -----------------------------

class AttentionSAC(nn.Module):
    """
    Local window attention supporting external query bias (state-aware bias);
    compatible with PyTorch SDPA.
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
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, q_bias_per_head: Optional[torch.Tensor] = None):
        """
        x: (B, N, C)
        q_bias_per_head: (B, num_heads, head_dim) or (1, num_heads, head_dim)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if q_bias_per_head is not None:
            # Add the state-aware bias to q for every token
            q = q + q_bias_per_head.unsqueeze(-2)  # (B, heads, 1, head_dim), broadcast to N

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
# SAC: State-aware cross attention + write-back gating
# -------------------------------------------------------

class SACBridge(nn.Module):
    """
    Use the M-Stream (Mamba) state to generate the query bias for T-Stream,
    and use the aggregated feature from T-Stream to gate the write-back to M-Stream.
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
        m_tokens: torch.Tensor,   # (B, N, C) M-Stream output
        t_tokens: torch.Tensor    # (B, N, C) T-Stream temporary output or input
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate Query bias from the global state of M-Stream (mean pooling)
        m_state = m_tokens.mean(dim=1)  # (B, C)
        q_bias = self.q_bias_gen(m_state).view(-1, self.num_heads, self.head_dim)  # (B, H, Dh)

        # Write-back gating from T-Stream aggregation
        t_state = t_tokens.mean(dim=1)  # (B, C)
        gate = self.writeback_gate(t_state)                                      # (B, C)
        write_feat = self.writeback_proj(t_state).unsqueeze(1)                   # (B, 1, C)
        m_tokens_enh = m_tokens + gate.unsqueeze(1) * write_feat                 # (B, N, C)

        return q_bias, m_tokens_enh

# -------------------------------------------------------
# PCS-Scan: four-direction Mamba (H/V/diag/anti-diag)
# -------------------------------------------------------

def _diag_indices(w: int):
    # Generate (row, col) indices in main diagonal order
    # and then flatten them into token indices
    inds = []
    for s in range(2*w-1):
        row_start = max(0, s-(w-1))
        row_end = min(w-1, s)
        line = []
        for r in range(row_start, row_end+1):
            c = s - r
            line.append((r, c))
        inds.extend(line)
    return inds

def _anti_diag_indices(w: int):
    # Generate anti-diagonal order (top-right -> bottom-left)
    inds = []
    for s in range(2*w-1):
        row_start = max(0, s-(w-1))
        row_end = min(w-1, s)
        line = []
        for r in range(row_start, row_end+1):
            c = s - r
            line.append((r, w-1-c))
        inds.extend(line)
    return inds

class PCSScanMamba(nn.Module):
    """
    Polarized Cross-Scan SSM:
    - Reorder tokens within a window into 4 sequences
      (horizontal / vertical / main diagonal / anti-diagonal),
      pass each through Mamba, and fuse the four outputs
      with a learnable gate.
    - Does not introduce extra supervision or modify the data.
    """
    def __init__(self, dim, d_state=8, d_conv=3, expand=1, window_size=7):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        # Shared or independent weights for the four branches:
        # here a shared Mamba core is used,
        # with learnable linear layers before and after to model "polarized" differences
        self.pre_h = nn.Linear(dim, dim)
        self.pre_v = nn.Linear(dim, dim)
        self.pre_d = nn.Linear(dim, dim)
        self.pre_a = nn.Linear(dim, dim)

        self.mamba_core = MambaVisionMixer(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

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
        assert N == w*w, f"N ({N}) must equal window_size^2 ({w}^2)."

        grid = x.view(B, w, w, C)  # (B, w, w, C)

        if mode == 'h':
            # Row-major: flatten as is
            y = grid.reshape(B, N, C)
            return y
        elif mode == 'v':
            # Column-major: transpose rows and columns before flattening
            y = grid.permute(0, 2, 1, 3).contiguous().view(B, N, C)
            return y
        elif mode == 'd':
            idxs = _diag_indices(w)  # list of (r,c)
        elif mode == 'a':
            idxs = _anti_diag_indices(w)
        else:
            raise ValueError

        # Sample according to idxs
        r_idx = torch.tensor([rc[0] for rc in idxs], device=x.device, dtype=torch.long)
        c_idx = torch.tensor([rc[1] for rc in idxs], device=x.device, dtype=torch.long)
        y = grid[:, r_idx, c_idx, :]  # (B, N, C)
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

        # Reordering
        rh = self._reorder(x_h, 'h')
        rv = self._reorder(x_v, 'v')
        rd = self._reorder(x_d, 'd')
        ra = self._reorder(x_a, 'a')

        # Speed optimization: batch the four Mamba calls
        # Concatenate the four inputs along the batch dimension
        # and pass them through mamba_core in one shot
        r_all = torch.cat([rh, rv, rd, ra], dim=0)   # (4*B, N, C)
        y_all = self.mamba_core(r_all)               # (4*B, N, C), only one call!
        yh, yv, yd, ya = torch.chunk(y_all, 4, dim=0) # Split the outputs back into four branches

        # Restore to original order
        # For h branch it is already aligned;
        # for the others we simply align by corresponding position
        yh = self.post_h(yh)
        yv = self.post_v(yv)
        yd = self.post_d(yd)
        ya = self.post_a(ya)

        # Polarized gate
        g = torch.softmax(self.gate_logits, dim=0)  # (4,)
        y = g[0]*yh + g[1]*yv + g[2]*yd + g[3]*ya
        return y  # (B, N, C)

# -------------------------------------------------------
# CSDSBlock: parallel dual-stream + SAC + fusion + FFN
# -------------------------------------------------------

class CSDSBlock(nn.Module):
    """
    Dual-stream parallel design:
      - M-Stream: PCSScanMamba (four-direction scan)
      - T-Stream: local window attention (supports SAC query bias)

    SAC:
      - Use the global state of M-Stream to generate the query bias for T-Stream;
      - Use the aggregated T-Stream feature to gate write-back to the M-Stream state.

    Fusion:
      - Concatenate and linearly fuse, followed by feed-forward network.
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

        # Normalization
        self.norm_m = norm_layer(dim)
        self.norm_t = norm_layer(dim)

        # Two branches
        self.m_stream = PCSScanMamba(dim, d_state=d_state, d_conv=d_conv, expand=expand, window_size=window_size) \
                        if self.enable_pcs else nn.Identity()
        self.t_stream = AttentionSAC(dim, num_heads=num_heads, qkv_bias=True, qk_norm=False,
                                     attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer)
        # SAC bridge
        self.sac = SACBridge(dim, num_heads) if self.enable_sac else None

        # Fusion
        self.fuse = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # MLP
        self.norm_ffn = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = Mlp(in_features=dim, hidden_features=hidden, act_layer=nn.GELU, drop=drop)

        # Residual
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Optional layer scale
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if (layer_scale is not None and isinstance(layer_scale,(int,float))) else None
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if (layer_scale is not None and isinstance(layer_scale,(int,float))) else None

    def _scale(self, x, gamma):
        if gamma is None:
            return x
        return x * gamma.view(1, 1, -1)

    def forward(self, x_tokens: torch.Tensor, sl_bridge_state: Optional[torch.Tensor] = None):
        """
        x_tokens: (B, N, C), window tokens
        sl_bridge_state: (B, C) or None, memory bridge from the previous Stage
        """
        B, N, C = x_tokens.shape

        # SL-Bridge: use the state from the previous stage for light FiLM-like modulation
        if self.enable_sl_bridge and sl_bridge_state is not None:
            alpha = 0.1
            x_tokens = x_tokens + alpha * sl_bridge_state.unsqueeze(1)

        # Dual streams in parallel
        # First normalize the input separately for each stream
        xm = self.norm_m(x_tokens)
        xt = self.norm_t(x_tokens)

        # Run M-Stream first to get initial M tokens
        if self.enable_pcs:
            y_m0 = self.m_stream(xm)  # (B,N,C)
        else:
            y_m0 = xm  # Bypass when PCS is disabled

        # SAC routing and attention
        if self.enable_sac:
            q_bias, y_m_enh = self.sac(y_m0, xt)  # q_bias: (B,H,Dh); y_m_enh: (B,N,C)
            y_t = self.t_stream(xt, q_bias_per_head=q_bias)  # (B,N,C)
        else:
            y_m_enh = y_m0
            y_t = self.t_stream(xt, q_bias_per_head=None)

        # Fusion
        y = torch.cat([y_m_enh, y_t], dim=-1)  # (B,N,2C)
        y = self.fuse(y)                       # (B,N,C)

        # Residual 1
        x1 = x_tokens + self.drop_path(self._scale(y, self.gamma_1))

        # FFN
        y_ffn = self.ffn(self.norm_ffn(x1))
        out = x1 + self.drop_path(self._scale(y_ffn, self.gamma_2))  # (B,N,C)

        # Return the current block's M-Stream state (after pooling)
        # for SL-Bridge aggregation
        m_state = y_m_enh.mean(dim=1)  # (B,C)
        return out, m_state

# -------------------------------------------------------
# Stage — [Core fix] add numerical stability safeguards
# -------------------------------------------------------

class CSDSLayer(nn.Module):
    """
    A single Stage:
      - Input (B,C,H,W)
      - Window partition -> (B*num_windows, win^2, C)
      - Stack L CSDSBlocks (each returns a local M state)
      - Restore all windows back to (B,C,H,W)
      - Downsample to the next stage, and return:
          * x_down: (B, C_out, H/2, W/2) as input to the next stage
          * x_res:  (B, C, H, W) as the current stage output feature
          * sl_state_stage: (B, C) aggregated M state of this stage
            (for the SL-Bridge of the next stage)
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
        prev_dim: Optional[int] = None, # ✅ New: receive previous-stage dimension to pre-create projection layer
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

        if isinstance(drop_path, list):
            dp_list = drop_path
        else:
            dp_list = [drop_path for _ in range(depth)]

        self.blocks = nn.ModuleList([
            CSDSBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dp_list[i],
                layer_scale=layer_scale,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                norm_layer=nn.LayerNorm,
                enable_pcs=enable_pcs,
                enable_sac=enable_sac,
                enable_sl_bridge=enable_sl_bridge,
            ) for i in range(depth)
        ])

        # [DDP robustness fix] Pre-create cross-stage state projection layer
        if prev_dim is not None and self.enable_sl_bridge:
            self.sl_proj_in = nn.Linear(prev_dim, self.dim)
        else:
            self.sl_proj_in = None # No input state for the first stage or when SL-Bridge is disabled

        self.sl_proj_agg = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, sl_bridge_state: Optional[torch.Tensor] = None):
        """
        x: (B,C,H,W)
        sl_bridge_state: (B,C_prev), memory state from the previous Stage;
                         channels may differ from the current stage
        """
        B, C, H, W = x.shape
        ws = self.window_size

        # Transform into window tokens
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))
            _, _, Hp, Wp = x.shape
        else:
            Hp, Wp = H, W

        tokens = window_partition(x, ws)  # (Bnw, ws*ws, C)

        # If a cross-stage state exists, first linearly modulate it before passing it to the block
        if sl_bridge_state is not None and self.sl_proj_in is not None:
            target_dtype = self.sl_proj_in.weight.dtype
            sl_bridge_state_casted = sl_bridge_state.to(dtype=target_dtype)
            sl = self.sl_proj_in(sl_bridge_state_casted)

            # Expand sl to match every window batch: Bnw = B * (Hp/ws) * (Wp/ws)
            num_wins = tokens.shape[0] // B
            sl = sl.repeat_interleave(num_wins, dim=0)  # (Bnw, self.dim)
        else:
            sl = None

        # Pass through multiple CSDSBlocks
        m_states = []
        for blk in self.blocks:
            if self.training and self.use_checkpoint:
                tokens, m_state_blk = checkpoint.checkpoint(blk, tokens, sl_bridge_state=sl, use_reentrant=False)
            else:
                tokens, m_state_blk = blk(tokens, sl_bridge_state=sl)
            m_states.append(m_state_blk)

        # Restore into feature map
        x_feat = window_reverse(tokens, ws, Hp, Wp)  # (B,C,Hp,Wp)
        if pad_r > 0 or pad_b > 0:
            x_feat = x_feat[:, :, :H, :W].contiguous()
        
        # Numerical stability
        x_feat = torch.nan_to_num(x_feat)

        # Aggregate memory state for this stage
        # average over all windows or blocks
        if len(m_states) > 0:
            m_last = m_states[-1]  # (Bnw, C)
            num_wins = m_last.shape[0] // B
            m_last = m_last.view(B, num_wins, -1).mean(dim=1)  # (B,C)
            sl_state_stage = self.sl_proj_agg(m_last)          # (B,C)
        else:
            sl_state_stage = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        
        sl_state_stage = torch.nan_to_num(sl_state_stage)

        # Downsampling
        x_down = self.downsample(x_feat) if self.downsample is not None else None
        if x_down is not None:
            x_down = torch.nan_to_num(x_down)
        
        return x_down if x_down is not None else x_feat, x_feat, sl_state_stage

# -------------------------------------------------------
# Top-level backbone: CSDS-Backbone
# -------------------------------------------------------

class MambaVision(nn.Module):
    """
    CSDS-Backbone (replacing the original MambaVision):
      - Stage1/2/3/4: CSDSLayer
      - Cross-stage SL-Bridge: pass the sl_state from the previous stage
        into the next stage
    """

    def __init__(self,
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
                 **kwargs):
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)

        # DropPath allocation
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

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
                drop_path=dpr[cur:cur + depths[i]],
                layer_scale=layer_scale,
                downsample=(i < len(depths)-1),
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
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
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
            x, feat, sl_state = level(x, sl_bridge_state=sl_state)  # x: next stage input; feat: current feature
        # The channel dimension of the last feature map equals num_features
        x = self.norm(feat)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def _load_state_dict(self, pretrained, strict: bool = False):
        _load_checkpoint(self, pretrained, strict=strict)

# -------------------------------------------------------
# MMDet/MMseg compatible backbone
# -------------------------------------------------------

@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class MM_mamba_vision(MambaVision):
    """MambaVision backbone adapted for MMDetection and MMSegmentation."""
    def __init__(self, 
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
                 **kwargs):
        # ✅ Pass use_checkpoint and feature switches to the parent class
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
        self.dims = [int(dim * 2 ** i) for i in range(0,4)]
        self.channel_first = True
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        # Top classification head is not needed
        del self.norm
        del self.head
        self.init_weights(pretrained)

    def load_pretrained(self, ckpt=None, key="state_dict"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")
    
    def init_weights(self, pretrained=None):
        """Initialize the weights of the backbone."""
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            # The parent class already called apply(_init_weights) in MambaVision.__init__
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """
        Forward pass, returning a list of multi-scale feature maps.
        """
        x = self.patch_embed(x)  # (B,C,H,W)

        outs = []
        sl_state = None
        for i, level in enumerate(self.levels):
            x, feat, sl_state = level(x, sl_bridge_state=sl_state)  # feat: (B, C_i, H_i, W_i)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(feat)
                outs.append(out.contiguous())

        if len(self.out_indices) == 0:
            return x
        
        return outs
