#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
import os
from functools import partial
from typing import Optional, Tuple
from torch.utils import checkpoint

from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from timm.models._builder import resolve_pretrained_cfg
try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp, PatchEmbed as TimmPatchEmbed

from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from torch.cuda.amp import autocast

from mmengine.model import BaseModule
from mmdet.registry import MODELS as MODELS_MMDET
from mmseg.registry import MODELS as MODELS_MMSEG
import mmcv
from mmengine.runner import load_checkpoint

# -------------------------------------------------------
# Basic configuration
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
# Utility functions: window partition, restoration, and weight loading
# -------------------------------------------------------

def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    C = windows.shape[-1]
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
    return x

def _load_state_dict(module, state_dict, strict=False, logger=None):
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
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
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
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)

def _load_checkpoint(model, filename, map_location='cpu', strict=False, logger=None):
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
# Normalization, downsampling, and PatchEmbed
# -------------------------------------------------------

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class Downsample(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        dim_out = dim if keep_dim else 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )
    def forward(self, x):
        return self.reduction(x)

class PatchEmbed(nn.Module):
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
# Pure Mamba Mixer module (enhanced numerical stability)
# -------------------------------------------------------

class MambaVisionMixer(nn.Module):
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
        self.x_proj = nn.Linear(self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
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

        self.ln_x_dbl = nn.LayerNorm(self.dt_rank + self.d_state * 2, dtype=torch.float32)

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        return: (B, L, D)
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        with autocast(enabled=False):
            x_float = x.float()
            z_float = z.float()

            # 1) depthwise convolution branches + safeguard
            bias_x_float = self.conv1d_x.bias.float() if self.conv1d_x.bias is not None else None
            bias_z_float = self.conv1d_z.bias.float() if self.conv1d_z.bias is not None else None

            x_conv = F.conv1d(input=x_float, weight=self.conv1d_x.weight.float(), bias=bias_x_float,
                              padding='same', groups=self.d_inner // 2)
            z_conv = F.conv1d(input=z_float, weight=self.conv1d_z.weight.float(), bias=bias_z_float,
                              padding='same', groups=self.d_inner // 2)

            x_conv = F.silu(torch.nan_to_num(x_conv))
            z_conv = F.silu(torch.nan_to_num(z_conv))

            # 2) SSM parameters
            A = -torch.exp(self.A_log.float())
            D = self.D.float()
            delta_bias = self.dt_proj.bias.float()

            x_dbl = self.x_proj(rearrange(x_conv, "b d l -> (b l) d"))
            x_dbl = self.ln_x_dbl(x_dbl)

            dt_raw, Bp, Cp = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj(dt_raw)

            # Mild numerical clipping to avoid explosion before or after softplus
            dt = torch.clamp(dt, -6.0, 6.0)

            dt = rearrange(dt, "(b l) d -> b d l", l=seqlen)
            Bp = rearrange(Bp, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            Cp = rearrange(Cp, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

            # 3) selective scan + safeguard
            y = selective_scan_fn(
                x_conv, dt, A, Bp, Cp, D, z=None,
                delta_bias=delta_bias,
                delta_softplus=True,
                return_last_state=None
            )
            y = torch.nan_to_num(y)

            # 4) Concatenate the other branch, then output projection + safeguard
            y = torch.cat([y, z_conv], dim=1)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
            out = torch.nan_to_num(out)

        return out

# -------------------------------------------------------
# Single MambaBlock and hierarchical MambaLayer (with safeguards)
# -------------------------------------------------------

class MambaBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        d_state=16,
        d_conv=4,
        expand=2,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mixer = MambaVisionMixer(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if (layer_scale is not None and isinstance(layer_scale,(int,float))) else None

        self.norm2 = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = Mlp(in_features=dim, hidden_features=hidden, act_layer=nn.GELU, drop=drop)
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if (layer_scale is not None and isinstance(layer_scale,(int,float))) else None

    def _scale(self, x, gamma):
        if gamma is None:
            return x
        return x * gamma.view(1, 1, -1)

    def forward(self, x_tokens: torch.Tensor):
        # Mixer
        y = self.mixer(self.norm1(x_tokens))
        y = torch.nan_to_num(y)  # Safeguard
        x = x_tokens + self.drop_path(self._scale(y, self.gamma_1))

        # FFN
        y_ffn = self.ffn(self.norm2(x))
        y_ffn = torch.nan_to_num(y_ffn)  # Safeguard
        x = x + self.drop_path(self._scale(y_ffn, self.gamma_2))
        return x

class MambaLayer(nn.Module):
    """
    One stage
      Input: B C H W
      Window partition -> Bnw win^2 C
      Stack L MambaBlocks
      Window restoration
      Optional downsampling
    """
    def __init__(
        self,
        dim,
        depth,
        window_size,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        downsample=True,
        d_state=16,
        d_conv=4,
        expand=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.downsample = Downsample(dim=dim) if downsample else None

        if isinstance(drop_path, list):
            dp_list = drop_path
        else:
            dp_list = [drop_path for _ in range(depth)]

        self.blocks = nn.ModuleList([
            MambaBlock(
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=dp_list[i],
                layer_scale=layer_scale,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                norm_layer=nn.LayerNorm
            ) for i in range(depth)
        ])

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        ws = self.window_size

        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))
            _, _, Hp, Wp = x.shape
        else:
            Hp, Wp = H, W

        tokens = window_partition(x, ws)  # Bnw, ws*ws, C

        for blk in self.blocks:
            if self.training and self.use_checkpoint:
                tokens = checkpoint.checkpoint(blk, tokens, use_reentrant=False)
            else:
                tokens = blk(tokens)

        x_feat = window_reverse(tokens, ws, Hp, Wp)
        if pad_r > 0 or pad_b > 0:
            x_feat = x_feat[:, :, :H, :W].contiguous()

        x_feat = torch.nan_to_num(x_feat)

        x_down = self.downsample(x_feat) if self.downsample is not None else None
        if x_down is not None:
            x_down = torch.nan_to_num(x_down)

        return x_down if x_down is not None else x_feat, x_feat

# -------------------------------------------------------
# Top-level backbone: pure Mamba version
# -------------------------------------------------------

class MambaVision(nn.Module):
    """
    Pure Mamba backbone
      Stage1..4 use MambaLayer
    """
    def __init__(self,
                 dim=128,
                 in_dim=64,
                 depths=(3, 3, 10, 5),
                 window_size=(8, 8, 14, 7),
                 mlp_ratio=4.0,
                 num_heads=None,            # Kept for compatibility; unused
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,             # Compatibility placeholder; unused
                 qk_scale=None,             # Compatibility placeholder; unused
                 drop_rate=0.0,
                 attn_drop_rate=0.0,        # Compatibility placeholder; unused
                 layer_scale=None,
                 layer_scale_conv=None,     # Compatibility placeholder; unused
                 use_checkpoint=False,
                 **kwargs):
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.levels = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            current_dim = int(dim * 2 ** i)
            level = MambaLayer(
                dim=current_dim,
                depth=depths[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[cur:cur + depths[i]],
                layer_scale=layer_scale,
                downsample=(i < len(depths)-1),
                d_state=16,
                d_conv=4,
                expand=2,
                use_checkpoint=use_checkpoint,
            )
            cur += depths[i]
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
        x = self.patch_embed(x)   # B C H W
        feat = None
        for idx, level in enumerate(self.levels):
            x, feat = level(x)
        x = self.norm(feat)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.nan_to_num(x)  # Safeguard
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = torch.nan_to_num(x)  # Safeguard
        return x

    def _load_state_dict(self, pretrained, strict: bool = False):
        _load_checkpoint(self, pretrained, strict=strict)

# -------------------------------------------------------
# MMDet / MMSeg compatible backbone (registered name: mamba_custom)
# -------------------------------------------------------

@MODELS_MMSEG.register_module(name='mamba_custom')
@MODELS_MMDET.register_module(name='mamba_custom')
class MambaCustom(MambaVision):
    """
    Pure Mamba backbone adapted for MMDetection and MMSegmentation
    Registered name: mamba_custom (to avoid conflicts with existing implementations)
    """
    def __init__(self, 
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads=None,
                 out_indices=(0, 1, 2, 3),
                 pretrained=None,
                 norm_layer="ln2d",
                 layer_scale=None,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__(
            dim=dim,
            in_dim=in_dim,
            depths=depths,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            layer_scale=layer_scale,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
        self.dims = [int(dim * 2 ** i) for i in range(0, 4)]
        self.channel_first = True
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer_mod: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer_mod(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        # The classification head is not needed for detection/segmentation backbones
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
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.patch_embed(x)  # B C H W
        outs = []
        for i, level in enumerate(self.levels):
            x, feat = level(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(feat)
                out = torch.nan_to_num(out)  # Safeguard
                outs.append(out.contiguous())
        if len(self.out_indices) == 0:
            return x
        return outs
