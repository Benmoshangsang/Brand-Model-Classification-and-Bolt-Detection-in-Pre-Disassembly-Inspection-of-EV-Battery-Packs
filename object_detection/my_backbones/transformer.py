import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# ✅ Import only from the registry to avoid circular imports
from mmdet.registry import MODELS
# ✅ MMDet 3.x recommends inheriting from BaseModule to support init_cfg / weight initialization
from mmengine.model import BaseModule

# timm DropPath (optional)
try:
    from timm.layers import DropPath  # timm>=0.9
except Exception:
    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight[:, None, None] + self.bias[:, None, None]

class PatchEmbed(nn.Module):
    """A simple two-layer convolutional downsampling module for patch embedding. Replace as needed."""
    def __init__(self, in_chans=3, in_dim=32, dim=80):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1),
        )

    def forward(self, x):
        return self.proj(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class AttentionSAC(nn.Module):
    """Placeholder attention module; replace with your own implementation if needed."""
    def __init__(self, dim, num_heads=4, qkv_bias=True, qk_norm=False):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        return self.pw(self.dw(x))

class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, window_size, depth):
        super().__init__()
        self.attn = AttentionSAC(dim, num_heads=num_heads, qkv_bias=True, qk_norm=False)
        self.ffn = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU)
        self.window_size = window_size
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)
        self.drop_path = DropPath(0.0)

    def forward(self, x):
        x = self.norm1(x + self.drop_path(self.attn(x)))
        x = self.norm2(x + self.drop_path(self.ffn(x)))
        return x

@MODELS.register_module()
class TransformerBackbone(BaseModule):
    """Custom Transformer backbone compatible with MMDet 3.3

    - Inherits from BaseModule (supports init_cfg / Pretrained)
    - Returns tuple(outs)
    - Exposes out_channels (for use with FPN and similar modules)
    """
    def __init__(self,
                 dim=128,
                 in_dim=64,
                 depths=(3, 3, 10, 5),
                 num_heads=(2, 4, 8, 16),
                 mlp_ratio=4.0,
                 window_size=(8, 8, 14, 7),
                 out_indices=(0, 1, 2, 3),
                 pretrained=None,          # Compatible with legacy config fields
                 init_cfg=None,            # Recommended entry for weight initialization
                 norm_layer='ln2d',
                 layer_scale=None,
                 use_checkpoint=False,
                 **kwargs):
        # Compatibility: if pretrained is provided but init_cfg is not,
        # convert it into a Pretrained init_cfg
        if pretrained is not None and init_cfg is None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        super().__init__(init_cfg=init_cfg)

        self.patch_embed = PatchEmbed(in_chans=3, in_dim=in_dim, dim=dim)
        self.out_indices = tuple(out_indices)
        self.depths = tuple(depths)

        # Layers for each stage
        # The channel dimension is kept as dim here;
        # if progressive channel expansion is needed, modify accordingly
        self.transformer_layers = nn.ModuleList([
            nn.Sequential(*[
                TransformerLayer(
                    dim=dim,
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratio,
                    window_size=window_size[i],
                    depth=depths[i]
                ) for _ in range(depths[i])
            ]) for i in range(len(depths))
        ])

        self.out_norms = nn.ModuleList([LayerNorm2d(dim) for _ in range(len(depths))])

        # Channel declaration for the Neck;
        # if stage channels differ, replace this with the corresponding list
        self.out_channels = [dim for _ in range(len(depths))]

    def forward(self, x):
        x = self.patch_embed(x)  # (B, dim, H/4, W/4)
        outs = []
        for i, stage in enumerate(self.transformer_layers):
            x = stage(x)
            if i in self.out_indices:
                outs.append(self.out_norms[i](x).contiguous())
        return tuple(outs)
