import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple
from mmengine.model import BaseModule
from mmdet.registry import MODELS

# This print statement is used to verify if this specific file is being imported correctly
print("[LOAD CHECK] using NEW Shared2FCBrandHead (fp32 proto branch + safe cosine)", flush=True)


def _cosine_sim(a: Tensor, b: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Numerically safe cosine similarity.

    a: (B, D)
    b: (K, D)
    Returns: (B, K) cosine similarities
    """
    # Normalize with eps to prevent division by zero for zero vectors
    a_n = F.normalize(a, dim=-1, eps=eps)
    b_n = F.normalize(b, dim=-1, eps=eps)

    # Clean any unexpected fp16 underflow NaN / inf values
    a_n = torch.nan_to_num(a_n, nan=0.0, posinf=0.0, neginf=0.0)
    b_n = torch.nan_to_num(b_n, nan=0.0, posinf=0.0, neginf=0.0)

    # Cosine similarity is the dot product of normalized vectors
    sim = a_n @ b_n.t()  # (B, K)

    # Clamp minor numerical drifts outside the [-1, 1] range
    sim = sim.clamp(min=-1.0, max=1.0)

    return sim


class _StateConditioner(nn.Module):
    """Maps a state vector to (gamma, beta) for FiLM operations on a vector (B, C)."""
    def __init__(self, state_dim: int, target_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * target_dim)
        )

    def forward(self, s: Tensor) -> Tuple[Tensor, Tensor]:
        gb = self.net(s)  # (B, 2C)
        gamma, beta = gb.chunk(2, dim=-1)  # (B, C), (B, C)
        return gamma, beta


class _FourDirSqueeze(nn.Module):
    """
    Performs four-direction squeezing over a feature map (B, C, H, W).
    Produces a fused global vector (B, C) and optional FiLM parameters.
    """
    def __init__(self, c: int, hidden: int = 256, make_film: bool = True):
        super().__init__()
        self.make_film = make_film
        self.gate_mlp = nn.Sequential(
            nn.Linear(4 * c, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4)  # Softmax gates for the 4 directions
        )
        if make_film:
            self.film_mlp = nn.Sequential(
                nn.Linear(4 * c, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 2 * c)  # gamma, beta output
            )

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
        # Handle rectangular feature maps
        if H < W:
            i = i[:H]
            j = j[:H]
        anti = x[:, :, i, j]              # (B, C, L)
        return anti.mean(dim=-1)          # (B, C)

    def forward(self, feat_map: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Args:
            feat_map: (B, C, H, W)
        Returns:
            fused: (B, C) fused feature vector
            gamma: (B, C) or None for FiLM scaling
            beta:  (B, C) or None for FiLM shifting
        """
        B, C, H, W = feat_map.shape

        # Compute four directional summaries: Horizontal, Vertical, Diagonal, Anti-diagonal
        v_h = feat_map.mean(dim=3).mean(dim=2)  # (B, C)
        v_v = feat_map.mean(dim=2).mean(dim=3)  # (B, C)
        v_d = self._diag_pool(feat_map)         # (B, C)
        v_a = self._anti_diag_pool(feat_map)    # (B, C)

        V = torch.stack([v_h, v_v, v_d, v_a], dim=1)   # (B, 4, C)
        V_cat = V.reshape(B, 4 * C)                    # (B, 4C)

        gate_logits = self.gate_mlp(V_cat)              # (B, 4)
        gate = F.softmax(gate_logits, dim=-1).unsqueeze(-1)  # (B, 4, 1)

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
    SCM-Brand Head
      - 2FC baseline for image-level brand classification.
      - Optional State-conditioned FiLM on vector x.
      - Optional Four-direction channel squeeze + gating on feature maps.
      - Optional Prototype-assisted logit fusion (cosine similarity logits).
    """
    def __init__(self,
                 in_channels: int = 512,         # Matches GAP or pooled ROI dimension
                 fc_out_channels: int = 1024,
                 num_classes: int = 7,
                 # Optional mechanism toggles
                 use_state_condition: bool = True,
                 state_dim: Optional[int] = None,
                 state_hidden: int = 256,
                 use_four_dir_squeeze: bool = True,
                 fourdir_hidden: int = 256,
                 film_from_fourdir: bool = True,
                 # Prototype branch is OFF by default for numerical stability
                 use_prototype_branch: bool = False,
                 proto_dim: Optional[int] = None,
                 proto_tau: float = 10.0,        # Temperature multiplier
                 proto_lambda: float = 0.3,      # Fusion weight for the prototype branch
                 # Loss configuration
                 loss_cls: dict = dict(type='CrossEntropyLoss',
                                       use_sigmoid=False,
                                       loss_weight=1.0),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes

        # 2FC trunk architecture
        self.fc1 = nn.Linear(in_channels, fc_out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc_out_channels, fc_out_channels)
        self.fc_cls = nn.Linear(fc_out_channels, num_classes)

        # State-conditioned FiLM setup
        self.use_state_condition = use_state_condition
        self.state_dim = state_dim if state_dim is not None else in_channels
        if self.use_state_condition:
            self.state_cond = _StateConditioner(
                self.state_dim,
                in_channels,
                hidden=state_hidden
            )

        # Four-direction squeeze setup
        self.use_four_dir_squeeze = use_four_dir_squeeze
        self.film_from_fourdir = film_from_fourdir
        if self.use_four_dir_squeeze:
            self.fourdir = _FourDirSqueeze(
                c=in_channels,
                hidden=fourdir_hidden,
                make_film=film_from_fourdir
            )
            # Map fused global signal back into the x channel space
            self.fused_proj = nn.Linear(in_channels, in_channels)

        # Prototype-assisted logits setup
        self.use_prototype_branch = use_prototype_branch
        self.proto_dim = proto_dim if proto_dim is not None else fc_out_channels
        self.proto_tau = proto_tau
        self.proto_lambda = proto_lambda

        if self.use_prototype_branch:
            # K x D learnable prototypes in the same space as features after the 2FC trunk
            self.prototypes = nn.Parameter(torch.empty(num_classes, self.proto_dim))
            nn.init.normal_(self.prototypes, std=0.02)
        else:
            # Assign dummy attribute for compatibility during inspection
            self.prototypes = None

        # Build classification loss module
        self.loss_cls = MODELS.build(loss_cls)

    # helpers ---------------------------------------------------------------

    def _apply_film_on_vec(self, x: Tensor,
                           gamma: Optional[Tensor],
                           beta: Optional[Tensor]) -> Tensor:
        if gamma is None or beta is None:
            return x
        # FiLM formula: x * (1 + gamma) + beta
        return x * (1.0 + gamma) + beta

    # forward --------------------------------------------------------------

    def forward(self,
                x: Tensor,
                return_logits: bool = True,
                state: Optional[Tensor] = None,
                feat_map: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:        (B, C) feature vector from GAP or pooled ROI.
            state:    (B, S) optional state vector for conditioning.
            feat_map: (B, C, H, W) optional spatial feature map.
        """

        # 1) Optional state-conditioned FiLM on input x
        if self.use_state_condition and state is not None:
            gamma_s, beta_s = self.state_cond(state)  # (B, C), (B, C)
            x = self._apply_film_on_vec(x, gamma_s, beta_s)

        # 2) Optional four-direction squeeze branch
        if self.use_four_dir_squeeze and feat_map is not None:
            fused, gamma_f, beta_f = self.fourdir(feat_map)  # (B, C), (B, C)?, (B, C)?
            # Fuse global directional context back into x
            x = x + self.fused_proj(fused)
            # Optionally apply FiLM again using four-dir statistics
            if self.film_from_fourdir and gamma_f is not None and beta_f is not None:
                x = self._apply_film_on_vec(x, gamma_f, beta_f)

        # ---- 2FC Trunk ----
        feat = self.relu(self.fc1(x))
        feat = self.relu(self.fc2(feat))          # (B, D=fc_out_channels)
        cls_fc = self.fc_cls(feat)                # (B, num_classes)

        # Prototype-assisted logits branch
        if self.use_prototype_branch:
            # Perform prototype logic in stable fp32, detached from AMP (Automatic Mixed Precision)
            with torch.cuda.amp.autocast(enabled=False):
                feat32 = feat.float()
                proto32 = self.prototypes.float()

                # Robustness check for NaNs/Infs
                feat32 = torch.nan_to_num(
                    feat32, nan=0.0, posinf=0.0, neginf=0.0
                )
                proto32 = torch.nan_to_num(
                    proto32, nan=0.0, posinf=0.0, neginf=0.0
                )

                logits_proto32 = _cosine_sim(feat32, proto32)  # (B, K) result in fp32

                tau32 = torch.tensor(
                    self.proto_tau,
                    dtype=feat32.dtype,
                    device=feat32.device
                )
                tau32 = torch.nan_to_num(
                    tau32, nan=1.0, posinf=1.0, neginf=1.0
                )
                tau32 = tau32.clamp(min=0.01, max=100.0)

                logits_proto32 = tau32 * logits_proto32  # Temperature scaling

            logits_proto = logits_proto32.to(cls_fc.dtype)
            # Residual-like fusion of FC logits and prototype cosine logits
            cls_score = cls_fc + self.proto_lambda * logits_proto
        else:
            cls_score = cls_fc

        return cls_score if return_logits else cls_score.softmax(dim=-1)

    # loss / predict -------------------------------------------------------

    def loss(self,
             cls_score: Tensor,
             labels: Tensor,
             reduction_override: Optional[str] = None) -> Dict[str, Tensor]:
        """
        Compute Cross-Entropy loss for the brand class.
        """
        # Empty batch guard for distributed training corner cases
        if cls_score.numel() == 0:
            return {'loss_brand_cls': cls_score.new_tensor(0.)}

        # Ensure labels are Long tensors
        if not isinstance(labels, Tensor):
            labels = torch.tensor(labels, dtype=torch.long,
                                  device=cls_score.device)
        elif labels.dtype != torch.long:
            labels = labels.to(dtype=torch.long)

        # Compute classification loss
        loss_cls = self.loss_cls(
            cls_score,
            labels,
            reduction_override=reduction_override
        )

        # Apply nan_to_num to prevent a single exploding sample from crashing all ranks in DDP
        loss_cls = torch.nan_to_num(
            loss_cls,
            nan=0.0,
            posinf=1e4,
            neginf=1e4
        )

        return {'loss_brand_cls': loss_cls}

    def predict(self,
                x: Tensor,
                state: Optional[Tensor] = None,
                feat_map: Optional[Tensor] = None) -> Tensor:
        """
        Predict brand probabilities using softmax.
        """
        return self.forward(
            x,
            return_logits=False,
            state=state,
            feat_map=feat_map
        )
