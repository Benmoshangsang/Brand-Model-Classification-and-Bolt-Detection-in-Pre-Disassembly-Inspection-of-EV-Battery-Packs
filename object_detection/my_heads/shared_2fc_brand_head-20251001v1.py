import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple
from mmengine.model import BaseModule
from mmdet.registry import MODELS


def _cosine_sim(a: Tensor, b: Tensor, eps: float = 1e-6) -> Tensor:
    """
    a: (B, D) ; b: (K, D)
    return: (B, K) cosine similarities
    """
    a_n = F.normalize(a, dim=-1, eps=eps)
    b_n = F.normalize(b, dim=-1, eps=eps)
    return a_n @ b_n.t()


class _StateConditioner(nn.Module):
    """Map state -> (gamma, beta) for FiLM on a vector (B, C)."""
    def __init__(self, state_dim: int, target_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * target_dim)
        )

    def forward(self, s: Tensor) -> Tuple[Tensor, Tensor]:
        gb = self.net(s)  # (B, 2C)
        gamma, beta = gb.chunk(2, dim=-1)  # (B, C), (B, C)
        return gamma, beta


class _FourDirSqueeze(nn.Module):
    """
    Four-direction squeeze over a feature map (B, C, H, W).
    Produces a fused global vector (B, C) and optional FiLM params.
    """
    def __init__(self, c: int, hidden: int = 256, make_film: bool = True):
        super().__init__()
        self.make_film = make_film
        self.gate_mlp = nn.Sequential(
            nn.Linear(4 * c, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 4)  # gates for 4 directions
        )
        if make_film:
            self.film_mlp = nn.Sequential(
                nn.Linear(4 * c, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, 2 * c)  # gamma, beta
            )
        # projection to fuse (optional, here direct weighted sum of 4 vectors)

    @staticmethod
    def _diag_pool(x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        L = min(H, W)
        idx = torch.arange(L, device=x.device)
        diag = x[:, :, idx, idx]          # (B, C, L)
        return diag.mean(dim=-1)          # (B, C)

    @staticmethod
    def _anti_diag_pool(x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        L = min(H, W)
        i = torch.arange(L, device=x.device)
        j = (W - 1) - i
        if H < W:
            i = i[:H]; j = j[:H]
        anti = x[:, :, i, j]              # (B, C, L)
        return anti.mean(dim=-1)          # (B, C)

    def forward(self, feat_map: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        feat_map: (B, C, H, W)
        returns:
          fused: (B, C)
          gamma: (B, C) or None
          beta:  (B, C) or None
        """
        B, C, H, W = feat_map.shape
        v_h = feat_map.mean(dim=3).mean(dim=2)  # (B, C)  horizontal
        v_v = feat_map.mean(dim=2).mean(dim=3)  # (B, C)  vertical
        v_d = self._diag_pool(feat_map)         # (B, C)  main diagonal
        v_a = self._anti_diag_pool(feat_map)    # (B, C)  anti-diagonal

        V = torch.stack([v_h, v_v, v_d, v_a], dim=1)   # (B, 4, C)
        V_cat = V.reshape(B, 4 * C)                    # (B, 4C)
        gate = F.softmax(self.gate_mlp(V_cat), dim=-1).unsqueeze(-1)  # (B, 4, 1)
        fused = (V * gate).sum(dim=1)                  # (B, C)

        if self.make_film:
            gb = self.film_mlp(V_cat)                  # (B, 2C)
            gamma, beta = gb.chunk(2, dim=-1)
            return fused, gamma, beta
        else:
            return fused, None, None


@MODELS.register_module()
class Shared2FCBrandHead(BaseModule):
    """
    SCM-Brand Head:
      - 2FC baseline for image-level brand classification
      - Optional State-conditioned FiLM on vector x
      - Optional Four-direction channel squeeze + gating on feature map
      - Optional Prototype-assisted logit fusion (cosine logits)

    用法：
      forward(x)  # 兼容原用法
      forward(x, state=..., feat_map=...)  # 开启新机制
    """
    def __init__(self,
                 in_channels: int = 512,         # 与你的骨干 GAP 输出对齐（原384也可）
                 fc_out_channels: int = 1024,
                 num_classes: int = 7,
                 # ---- 新增可选机制 ----
                 use_state_condition: bool = True,
                 state_dim: Optional[int] = None,  # 若为 None，默认与 in_channels 相同
                 state_hidden: int = 256,
                 use_four_dir_squeeze: bool = True,
                 fourdir_hidden: int = 256,
                 film_from_fourdir: bool = True,   # 是否从四向挤压生成一组 FiLM
                 use_prototype_branch: bool = True,
                 proto_dim: Optional[int] = None,  # 若为 None，使用 fc_out_channels
                 proto_tau: float = 10.0,          # 余弦logits温度（乘法缩放）
                 proto_lambda: float = 0.3,        # 原型logits的融合权重
                 # loss
                 loss_cls: dict = dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes

        # ---- 2FC 主干 ----
        self.fc1 = nn.Linear(in_channels, fc_out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc_out_channels, fc_out_channels)
        self.fc_cls = nn.Linear(fc_out_channels, num_classes)

        # ---- State-conditioned ----
        self.use_state_condition = use_state_condition
        self.state_dim = state_dim if state_dim is not None else in_channels
        if self.use_state_condition:
            self.state_cond = _StateConditioner(self.state_dim, in_channels, hidden=state_hidden)

        # ---- Four-direction squeeze ----
        self.use_four_dir_squeeze = use_four_dir_squeeze
        self.film_from_fourdir = film_from_fourdir
        if self.use_four_dir_squeeze:
            # make_film=True 时除了 fused 还会输出一组 (gamma,beta)
            self.fourdir = _FourDirSqueeze(c=in_channels, hidden=fourdir_hidden, make_film=film_from_fourdir)
            # 融合 fused->x 的线性映射（轻量）
            self.fused_proj = nn.Linear(in_channels, in_channels)

        # ---- Prototype-assisted logits ----
        self.use_prototype_branch = use_prototype_branch
        self.proto_dim = proto_dim if proto_dim is not None else fc_out_channels
        self.proto_tau = proto_tau
        self.proto_lambda = proto_lambda
        if self.use_prototype_branch:
            # 使用 2FC 后的特征作为对齐空间（维度 = fc_out_channels）
            self.prototypes = nn.Parameter(torch.randn(num_classes, self.proto_dim))
            nn.init.normal_(self.prototypes, std=0.02)

        self.loss_cls = MODELS.build(loss_cls)

    # ------------------------------ forward ------------------------------

    def _apply_film_on_vec(self, x: Tensor, gamma: Optional[Tensor], beta: Optional[Tensor]) -> Tensor:
        if gamma is None or beta is None:
            return x
        return x * (1.0 + gamma) + beta

    def forward(self,
                x: Tensor,
                return_logits: bool = True,
                state: Optional[Tensor] = None,
                feat_map: Optional[Tensor] = None) -> Tensor:
        """
        x: (B, C)  —— 通常来自 GAP
        state: (B, S)  —— 可选；若启用 use_state_condition，建议提供
        feat_map: (B, C, H, W)  —— 可选；若启用 use_four_dir_squeeze，建议提供
        """
        B = x.size(0)

        # 1) State-conditioned FiLM on vector x
        if self.use_state_condition and state is not None:
            gamma_s, beta_s = self.state_cond(state)  # (B, C), (B, C)
            x = self._apply_film_on_vec(x, gamma_s, beta_s)

        # 2) Four-direction squeeze on feature map (if provided)
        if self.use_four_dir_squeeze and feat_map is not None:
            fused, gamma_f, beta_f = self.fourdir(feat_map)  # (B, C), (B, C)?, (B, C)?
            # 2.1 将 fused 线性投影并残差加入 x
            x = x + self.fused_proj(fused)
            # 2.2 可选：再以四向挤压产生的一组 FiLM 调制 x
            if self.film_from_fourdir and gamma_f is not None and beta_f is not None:
                x = self._apply_film_on_vec(x, gamma_f, beta_f)

        # ---- 原始 2FC 主干 ----
        feat = self.relu(self.fc1(x))
        feat = self.relu(self.fc2(feat))
        cls_fc = self.fc_cls(feat)  # (B, num_classes)

        # 3) Prototype-assisted logits
        if self.use_prototype_branch:
            # 在 2FC 表征空间计算 cosine logits
            # prototypes: (K, D), feat: (B, D)
            logits_proto = self.proto_tau * _cosine_sim(feat, self.prototypes)  # (B, K)
            cls_score = cls_fc + self.proto_lambda * logits_proto
        else:
            cls_score = cls_fc

        return cls_score if return_logits else cls_score.softmax(dim=-1)

    # ------------------------------ losses/predict ------------------------------

    def loss(self,
             cls_score: Tensor,
             labels: Tensor,
             reduction_override: Optional[str] = None) -> Dict[str, Tensor]:
        """
        计算整图分类损失（仍为 CE），若融合了原型 logits，不需要额外损失即可工作。
        """
        if cls_score.numel() == 0:
            return {'loss_brand_cls': cls_score.new_tensor(0.)}

        if not isinstance(labels, Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=cls_score.device)
        elif labels.dtype != torch.long:
            labels = labels.to(dtype=torch.long)

        loss_cls = self.loss_cls(cls_score, labels, reduction_override=reduction_override)
        return {'loss_brand_cls': loss_cls}

    def predict(self, x: Tensor, state: Optional[Tensor] = None, feat_map: Optional[Tensor] = None) -> Tensor:
        """
        预测整图所属品牌（softmax 概率）。可选传入 state/feat_map 以启用新机制。
        """
        return self.forward(x, return_logits=False, state=state, feat_map=feat_map)
