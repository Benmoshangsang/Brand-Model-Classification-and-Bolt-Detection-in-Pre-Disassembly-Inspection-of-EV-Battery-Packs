#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch                                            # 导入 PyTorch 主库
import torch.nn as nn                                   # 导入 PyTorch 神经网络模块
from timm.models.registry import register_model         # 从 timm 库导入模型注册器
import math                                             # 导入数学库
from timm.models.layers import trunc_normal_, DropPath  # 从 timm 导入权重初始化和 DropPath
from timm.models._builder import resolve_pretrained_cfg # 从 timm 导入预训练配置解析
try:
    from timm.models._builder import _update_default_kwargs as update_args # 尝试导入更新默认参数的函数
except:
    from timm.models._builder import _update_default_model_kwargs as update_args # 兼容旧版 timm
from timm.models.vision_transformer import Mlp, PatchEmbed as TimmPatchEmbed # 从 timm 导入 MLP 和 PatchEmbed
from timm.models.layers import DropPath, trunc_normal_  # 再次导入 DropPath 和 trunc_normal_
from timm.models.registry import register_model         # 再次导入模型注册器
import torch.nn.functional as F                         # 导入 PyTorch 函数库
from pathlib import Path                                # 导入路径处理库
import os                                               # 导入操作系统接口
from functools import partial                           # 导入偏函数工具
from typing import Callable, Optional, Tuple            # 导入类型提示
from torch.utils import checkpoint                      # 导入梯度检查点功能
from mmengine.model import BaseModule                   # 从 mmengine 导入基础模型模块
from mmdet.registry import MODELS as MODELS_MMDET       # 从 mmdet 导入模型注册表
from mmseg.registry import MODELS as MODELS_MMSEG       # 从 mmseg 导入模型注册表
import mmcv                                             # 导入 mmcv 库
from mmengine.runner import load_checkpoint             # 从 mmengine 导入加载检查点的函数

# -------------------------------------------------------
# 基础配置（可以按需更新为 TransNeXt 的权重 URL）
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


# 这里仅保留占位 default_cfg，可根据实际 TransNeXt 预训练权重自行补充 URL
default_cfgs = {
    'transnext_tiny': _cfg(
        crop_pct=1.0,
        input_size=(3, 224, 224),
        crop_mode='center'),
    'transnext_small': _cfg(
        crop_pct=0.98,
        input_size=(3, 224, 224),
        crop_mode='center'),
    'transnext_base': _cfg(
        crop_pct=0.93,
        input_size=(3, 224, 224),
        crop_mode='center'),
}

# -------------------------------------------------------
# 工具函数（窗口分割/恢复、权重加载）
# -------------------------------------------------------

def window_partition(x, window_size):
    """
    将特征图分割成不重叠的窗口。
    Args:
        x: (B, C, H, W) 输入特征图
        window_size: 窗口大小
    Returns:
        local window features: (num_windows*B, window_size*window_size, C) 分割后的窗口特征
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将窗口特征恢复为原始特征图。
    Args:
        windows: (num_windows*B, window_size*window_size, C) 窗口特征
        window_size: 窗口大小
        H, W: 原始特征图的高度和宽度
    Returns:
        x: (B, C, H, W) 恢复后的特征图
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    C = windows.shape[-1]
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
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
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
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
    """加载检查点文件并处理其中的 state_dict。"""
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {
            k.replace('encoder.', ''): v
            for k, v in state_dict.items()
            if k.startswith('encoder.')
        }

    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint

# -------------------------------------------------------
# 归一化与下采样、PatchEmbed
# -------------------------------------------------------

class LayerNorm2d(nn.LayerNorm):
    """2D 版本的 LayerNorm。"""
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return x


class Downsample(nn.Module):
    """下采样模块，使用步长为2的卷积实现。"""
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
    """将图像转换为 Patch Embeddings（与原代码保持一致，用两次 stride=2 卷积实现 1/4 下采样）。"""
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
# TransNeXt 组件：ConvGLU + Aggregated Attention
# -------------------------------------------------------

class ConvGLU(nn.Module):
    """
    卷积版 GLU 通道混合器：
      - 1x1 卷积 -> 2 * hidden_dim
      - 分成 (u, v)，对 u 做深度卷积建模局部信息
      - 用 v 的非线性作为 gate
    """
    def __init__(self, dim, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u, v = self.fc1(x).chunk(2, dim=1)  # (B, hidden_dim, H, W) * 2
        u = self.dwconv(u)
        v = F.gelu(v)
        x = u * v
        x = self.fc2(x)
        return x


class AggregatedAttention(nn.Module):
    """
    TransNeXt 风格的 Aggregated Attention（简化版）：
      - 为每个阶段引入若干可学习的聚合 token；
      - 对 [agg_tokens, x] 做标准自注意力；
      - 只回写空间 token 对应的输出。
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_agg_tokens: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_agg_tokens = num_agg_tokens

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 聚合 token（全局可学习）
        self.agg_tokens = nn.Parameter(torch.zeros(1, num_agg_tokens, dim))
        trunc_normal_(self.agg_tokens, std=.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        return: (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W

        # (B, C, H, W) -> (B, N, C)
        x_tokens = x.flatten(2).transpose(1, 2)

        # 拼上聚合 token: (B, num_agg+N, C)
        agg = self.agg_tokens.expand(B, -1, -1)
        x_all = torch.cat([agg, x_tokens], dim=1)
        L = x_all.shape[1]

        qkv = self.qkv(x_all).reshape(
            B, L, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)  # (3, B, heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # PyTorch fused SDPA
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
        )  # (B, heads, L, head_dim)

        attn_out = attn_out.transpose(1, 2).reshape(B, L, C)
        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)

        # 丢弃聚合 token，只保留空间 token 部分
        attn_out = attn_out[:, self.num_agg_tokens:, :]  # (B, N, C)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)
        return attn_out

# -------------------------------------------------------
# TransNeXt Block 与 Stage
# -------------------------------------------------------

class TransNeXtBlock(nn.Module):
    """
    TransNeXt 基本块：
      - LayerNorm2d + Aggregated Attention
      - LayerNorm2d + ConvGLU
      - 支持 DropPath 和 LayerScale
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        layer_scale: Optional[float] = None,
    ):
        super().__init__()
        self.dim = dim

        self.norm1 = LayerNorm2d(dim)
        self.attn = AggregatedAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = LayerNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvGLU(dim, hidden_dim=mlp_hidden_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if layer_scale is not None and isinstance(layer_scale, (int, float)):
            self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))
            self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def _apply_gamma(self, x: torch.Tensor, gamma: Optional[torch.Tensor]) -> torch.Tensor:
        if gamma is None:
            return x
        return x * gamma.view(1, -1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention 分支
        shortcut = x
        y = self.attn(self.norm1(x))
        y = self._apply_gamma(y, self.gamma_1)
        x = shortcut + self.drop_path(y)

        # ConvGLU MLP 分支
        shortcut2 = x
        y2 = self.mlp(self.norm2(x))
        y2 = self._apply_gamma(y2, self.gamma_2)
        x = shortcut2 + self.drop_path(y2)
        return x


class TransNeXtStage(nn.Module):
    """
    一个 Stage：若干个 TransNeXtBlock 的堆叠（不含下采样）。
    下采样由骨干在 stage 之间单独控制。
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path=0.,
        layer_scale: Optional[float] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        if isinstance(drop_path, list):
            dp_list = drop_path
        else:
            dp_list = [drop_path for _ in range(depth)]

        self.blocks = nn.ModuleList([
            TransNeXtBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dp_list[i],
                layer_scale=layer_scale,
            ) for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.training and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

# -------------------------------------------------------
# 顶层骨干：TransNeXt-Backbone（替换原 MambaVision/CSDS-Backbone）
# -------------------------------------------------------

class MambaVision(nn.Module):
    """
    TransNeXt Backbone（保持外部接口不变）：
      - Stage1/2/3/4：TransNeXtStage（多头 Aggregated Attention + ConvGLU）
      - 使用 PatchEmbed 进行 1/4 下采样，然后每个 stage 之间再做一次 stride=2 卷积下采样
      - 分类模式下使用最后一个 stage 做全局池化 + 全连接
    """

    def __init__(self,
                 dim=128,
                 in_dim=64,
                 depths=(3, 3, 10, 5),
                 window_size=(8, 8, 14, 7),   # 仅为兼容保留，不再使用
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
                 enable_pcs: bool = True,      # 仅为兼容参数，占位不用
                 enable_sac: bool = True,      # 仅为兼容参数，占位不用
                 enable_sl_bridge: bool = True,# 仅为兼容参数，占位不用
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)

        # 每个 stage 的通道数：dim, 2*dim, 4*dim, 8*dim
        self.num_stages = len(depths)
        self.dims = [int(dim * 2 ** i) for i in range(self.num_stages)]
        num_features = self.dims[-1]

        # DropPath 分配
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 下采样层（stage 之间）
        self.downsample_layers = nn.ModuleList()
        # stage0: 不再额外下采样，由 patch_embed 完成
        self.downsample_layers.append(nn.Identity())
        for i in range(1, self.num_stages):
            in_c = self.dims[i - 1]
            self.downsample_layers.append(Downsample(dim=in_c))

        # Stages
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(self.num_stages):
            stage_dim = self.dims[i]
            stage = TransNeXtStage(
                dim=stage_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur:cur + depths[i]],
                layer_scale=layer_scale,
                use_checkpoint=use_checkpoint,
            )
            cur += depths[i]
            self.stages.append(stage)

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
        # PatchEmbed: (B, 3, H, W) -> (B, C0, H/4, W/4)
        x = self.patch_embed(x)
        # 多 stage 递进
        for i in range(self.num_stages):
            if i > 0:
                x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        # 最后一层输出的通道即 num_features
        x = self.norm(x)
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
# MMDet/MMseg 适配骨干（名称保持不变，仅内部改为 TransNeXt）
# -------------------------------------------------------

@MODELS_MMSEG.register_module(name='MM_TransNeXt')
@MODELS_MMDET.register_module(name='MM_TransNeXt')
class MM_TransNeXt(MambaVision):
    """适配 MMDetection 和 MMSegmentation 的 TransNeXt 骨干网络（兼容原 MM_mamba_vision 名称）。"""
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
        # 保持参数接口不变，将 use_checkpoint 与各开关传递给父类（内部已忽略 enable_*）
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
        # 这里的 self.dims 在父类中已经构建为 [dim,2*dim,4*dim,8*dim,...]
        self.channel_first = True
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer_cls: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer_cls(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        # 顶层分类头对检测/分割不需要
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
        """初始化骨干网络的权重。"""
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            # 父类在 __init__ 中已经调用了 apply(_init_weights)
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """
        前向传播，返回多尺度特征图列表：
          - out[0]: C2
          - out[1]: C3
          - out[2]: C4
          - out[3]: C5
        """
        # PatchEmbed
        x = self.patch_embed(x)  # (B, C0, H/4, W/4)

        outs = []
        # 逐 stage 前向
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsample_layers[i](x)
            x = stage(x)  # (B, Ci, Hi, Wi)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(x)
                outs.append(out.contiguous())

        if len(self.out_indices) == 0:
            return x
        
        return outs
