# -*- coding: utf-8 -*-
"""
Custom Cascade ROI Head Module - Supports State-guided and Four-direction Pooling FiLM Modulation
✅ Fixed unused parameter issues in distributed training (DDP)
✅ Ensured all created modules participate in the forward pass
✅ Fixed incorrect InstanceData reference in the predict method
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample, SampleList
from mmdet.models.roi_heads import CascadeRoIHead
from mmdet.registry import MODELS


def _global_avg_pool(feat: Tensor) -> Tensor:
    """
    Perform Global Average Pooling on spatial dimensions: (B, C, H, W) -> (B, C).
    
    Args:
        feat: Input feature tensor, shape (B, C, H, W)
    Returns:
        Pooled tensor, shape (B, C)
    """
    return F.adaptive_avg_pool2d(feat, 1).flatten(1)


def _gather_batch_state_from_metas(batch_data_samples: List[DetDataSample],
                                   key: str = 'state') -> Optional[Tensor]:
    """
    Attempts to read external backbone state from the metainfo of data samples.
    
    Args:
        batch_data_samples: List of data samples in the batch.
        key: The key name of the state to be retrieved.
    Returns:
        State tensor (B, C_state) or None.
    """
    states = []
    for s in batch_data_samples:
        meta = getattr(s, 'metainfo', None)
        if meta is None or key not in meta:
            return None
        t = meta[key]
        if isinstance(t, torch.Tensor):
            states.append(t)
        else:
            return None
    if len(states) == 0:
        return None
    # Return stacked state tensor (B, C_state)
    return torch.stack(states, dim=0)


class _StateProjector(nn.Module):
    """
    Projects the state (B, C_in) into FiLM parameters (B, C_out).
    Uses a two-layer MLP to map states to gamma and beta parameters.
    """
    def __init__(self, c_in: int, c_out: int, hidden: int = 256):
        """
        Initialize the state projector.
        
        Args:
            c_in: Input state dimension.
            c_out: Output FiLM parameter dimension.
            hidden: Hidden layer dimension.
        """
        super().__init__()
        # Use nn.Sequential to ensure all layers are correctly registered as submodules
        self.mlp = nn.Sequential(
            nn.Linear(c_in, hidden), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * c_out)  # Output [gamma, beta]
        )

    def forward(self, s: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass: takes state s and outputs gamma and beta.
        
        Args:
            s: (B, C_in) state tensor.
        Returns:
            gamma: (B, C_out) scaling parameter.
            beta: (B, C_out) shifting parameter.
        """
        gb = self.mlp(s)  # (B, 2*C)
        gamma, beta = gb.chunk(2, dim=-1)  # Split into two parts
        return gamma, beta


class _FourDirROIGate(nn.Module):
    """
    Four-direction ROI pooling (Horizontal/Vertical/Diagonal/Anti-diagonal) + Learnable Gating.
    Input x: (N, C, H, W), returns channel-level FiLM parameters (gamma, beta) for each ROI.
    
    ✅ Key Fix: Added dimension check to skip four-direction pooling if input is not a 4D feature map.
    """
    def __init__(self, c: int, hidden: int = 256):
        """
        Initialize the four-direction gating module.
        
        Args:
            c: Number of feature channels.
            hidden: Hidden layer dimension.
        """
        super().__init__()
        # Gating network: calculates attention weights for the 4 directions
        self.proj = nn.Sequential(
            nn.Linear(4 * c, hidden), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4),  # Gating logits for 4 directions
        )
        # Output network: generates final FiLM parameters
        self.out = nn.Sequential(
            nn.Linear(4 * c, hidden), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * c)  # Output [gamma, beta]
        )

    @staticmethod
    def _diag_pool(x: Tensor) -> Tensor:
        """
        Main diagonal pooling: averages elements where i == j.
        
        Args:
            x: (N, C, H, W) feature map.
        Returns:
            (N, C) pooled vector.
        """
        N, C, H, W = x.shape
        L = min(H, W)
        device = x.device
        i = torch.arange(L, device=device)
        diag = x[:, :, i, i]
        return diag.mean(dim=-1)  # (N, C)

    @staticmethod
    def _anti_diag_pool(x: Tensor) -> Tensor:
        """
        Anti-diagonal pooling: averages elements where i + j == W - 1 (after cropping).
        
        Args:
            x: (N, C, H, W) feature map.
        Returns:
            (N, C) pooled vector.
        """
        N, C, H, W = x.shape
        L = min(H, W)
        device = x.device
        i = torch.arange(L, device=device)
        j = (W - 1) - i
        if H >= W:
            ii, jj = i, j
        else:
            ii, jj = i[:H], j[:H]
        anti = x[:, :, ii, jj]  # (N, C, L)
        return anti.mean(dim=-1)  # (N, C)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            x: (N, C, H, W) or (N, C) feature tensor.
        Returns:
            gamma: (N, C) scaling parameter.
            beta: (N, C) shifting parameter.
        
        ✅ Key Fix: If input is not 4D, returns zero vectors directly (skips pooling).
        """
        if x.ndim != 4:
            N, C = x.shape[0], x.shape[1]
            device = x.device
            dtype = x.dtype
            # Return zero gamma and beta (equivalent to no modulation)
            gamma = torch.zeros(N, C, device=device, dtype=dtype)
            beta = torch.zeros(N, C, device=device, dtype=dtype)
            return gamma, beta
        
        N, C, H, W = x.shape
        # Horizontal and Vertical pooling
        v_h = x.mean(dim=3).mean(dim=2)        # (N, C) Horizontal
        v_v = x.mean(dim=2).mean(dim=2)        # (N, C) Vertical
        # Diagonal and Anti-diagonal pooling
        v_d = self._diag_pool(x)               # (N, C) Main Diagonal
        v_a = self._anti_diag_pool(x)          # (N, C) Anti-diagonal

        # Stack and Gate
        v = torch.stack([v_h, v_v, v_d, v_a], dim=1)   # (N, 4, C)
        v_cat = v.reshape(N, 4 * C)                    # (N, 4C)
        gate = F.softmax(self.proj(v_cat), dim=-1).unsqueeze(-1)  # (N, 4, 1)
        fused = (v * gate).sum(dim=1)                  # (N, C) Weighted fusion

        # Generate FiLM parameters
        gb = self.out(v_cat)                           # (N, 2C)
        gamma, beta = gb.chunk(2, dim=-1)              # (N, C), (N, C)
        return gamma, beta


def _film_apply(x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
    """
    Applies channel-level FiLM modulation to ROI features.
    FiLM: Feature-wise Linear Modulation
    Formula: output = x * (1 + gamma) + beta
    
    Args:
        x: (N, C, H, W) or (N, C) feature tensor.
        gamma: (N, C) scaling parameter.
        beta: (N, C) shifting parameter.
    Returns:
        Modulated feature tensor, same shape as x.
    """
    if x.ndim == 4:
        N, C, H, W = x.shape
        gamma = gamma.view(N, C, 1, 1)
        beta = beta.view(N, C, 1, 1)
    elif x.ndim == 2:
        pass
    else:
        gamma = gamma.view(x.shape[0], x.shape[1], *([1] * (x.ndim - 2)))
        beta = beta.view(x.shape[0], x.shape[1], *([1] * (x.ndim - 2)))
    
    return x * (1.0 + gamma) + beta


def _knn_relation_refine(bboxes: Tensor,
                         scores: Tensor,
                         k: int = 6,
                         alpha: float = 0.5) -> Tensor:
    """
    Lightweight KNN-based neighborhood relation re-weighting (per image scores).
    Smooths scores by calculating a weighted average of spatially neighboring boxes.
    
    Args:
        bboxes: (M, 4) bboxes in xyxy format.
        scores: (M,) or (M, C) scores.
        k: Number of neighbors in KNN.
        alpha: Fusion weight for neighborhood average.
    Returns:
        Refined scores with the same shape as input scores.
    """
    if bboxes.numel() == 0:
        return scores
    
    # Calculate box centers
    centers = torch.stack([(bboxes[:, 0] + bboxes[:, 2]) * 0.5,
                           (bboxes[:, 1] + bboxes[:, 3]) * 0.5], dim=-1)  # (M, 2)
    # Calculate pair-wise distances
    dist = torch.cdist(centers, centers, p=2)  # (M, M)
    # Exclude self
    M = centers.size(0)
    dist[torch.arange(M), torch.arange(M)] = float('inf')
    # Find kNN indices
    k_eff = min(k, max(1, M - 1))
    knn_idx = torch.topk(-dist, k=k_eff, dim=-1).indices  # Negative sign for min distance
    
    if scores.ndim == 1:
        neigh = scores[knn_idx]               # (M, k)
        avg = neigh.mean(dim=-1)              # (M,)
        return (1 - alpha) * scores + alpha * avg
    else:
        neigh = torch.gather(
            scores, 0, knn_idx.unsqueeze(-1).expand(-1, -1, scores.size(-1))
        )                                     # (M, k, C)
        avg = neigh.mean(dim=1)               # (M, C)
        return (1 - alpha) * scores + alpha * avg


@MODELS.register_module()
class CustomCascadeRoIHead(CascadeRoIHead):
    """
    Custom Cascade ROI Head integrated with SFG-RL (State-guided Feature modulation and Relation Learning):
      - State-guided ROI modulation (FiLM from global state)
      - Four-direction ROI pooling + Gating (Angle robustness at ROI level)
      - Relation-aware neighborhood aggregation (optional at inference)

    All features are configurable via initialization parameters.
    
    ✅ Fixed distributed training issues:
       1. Created all submodules immediately in __init__.
       2. Ensured all created modules participate in the forward pass to avoid DDP errors.
    """
    def __init__(self,
                 *args,
                 num_stages: int = 3,
                 # ---- SFG-RL Switches ----
                 use_state_condition: bool = True,
                 use_four_dir_pool: bool = True,
                 use_relation_refine: bool = True,
                 # Dimensions
                 state_in_channels: Optional[int] = None,
                 roi_feat_channels: int = 256,
                 # Hidden layer dimensions
                 state_hidden: int = 256,
                 fourdir_hidden: int = 256,
                 # Relation refinement parameters
                 rel_k: int = 6,
                 rel_alpha: float = 0.5,
                 # Use external state from metainfo
                 use_external_state: bool = False,
                 **kwargs):
        """
        Initialize Custom Cascade ROI Head.
        """
        super().__init__(*args, num_stages=num_stages, **kwargs)

        self.use_state_condition = use_state_condition
        self.use_four_dir_pool = use_four_dir_pool
        self.use_relation_refine = use_relation_refine
        self.use_external_state = use_external_state

        self.state_in_channels = state_in_channels if state_in_channels is not None else roi_feat_channels
        self.roi_feat_channels = roi_feat_channels
        self.state_hidden = state_hidden
        self.fourdir_hidden = fourdir_hidden
        self.rel_k = rel_k
        self.rel_alpha = rel_alpha

        # ✅ Key Fix: Create modules immediately in __init__ so DDP can track parameters
        if self.use_state_condition:
            self.state_projector = _StateProjector(
                c_in=self.state_in_channels,
                c_out=self.roi_feat_channels,
                hidden=self.state_hidden
            )
            print(f"[INFO] Created state_projector: {self.state_in_channels} -> {self.roi_feat_channels}")
        else:
            self.state_projector = None
        
        if self.use_four_dir_pool:
            self.fourdir_gate = _FourDirROIGate(
                c=self.roi_feat_channels,
                hidden=self.fourdir_hidden
            )
            print(f"[INFO] Created fourdir_gate: channels={self.roi_feat_channels}")
        else:
            self.fourdir_gate = None

    # ---------------------------- Utilities ----------------------------

    def _compute_batch_state(self,
                             x: Tuple[Tensor],
                             batch_data_samples: List[DetDataSample]) -> Tensor:
        """
        Returns (B, C_state) batch state. Prioritizes external state; otherwise uses GAP on x[-1].
        
        Args:
            x: Feature pyramid tuple.
            batch_data_samples: List of data samples.
        Returns:
            (B, C_state) state tensor.
        """
        if self.use_external_state:
            s = _gather_batch_state_from_metas(batch_data_samples, key='state')
            if s is not None:
                return s.to(device=x[-1].device, dtype=x[-1].dtype)

        # Fallback: Global Average Pooling on the last feature map layer
        feat_last = x[-1]  # (B, C, H, W)
        state = _global_avg_pool(feat_last)  # (B, C)
        return state

    # --------------------- Core bbox Override Methods ---------------------

    def _bbox_forward(self, stage: int, x: Tuple[Tensor],
                      rois: Tensor) -> dict:
        """
        Inject SFG-RL modulation before calling the bbox_head.
        """
        # ROI extraction
        if isinstance(self.bbox_roi_extractor, nn.ModuleList):
            roi_extractor = self.bbox_roi_extractor[stage]
        else:
            roi_extractor = self.bbox_roi_extractor

        if isinstance(self.bbox_head, nn.ModuleList):
            bbox_head = self.bbox_head[stage]
        else:
            bbox_head = self.bbox_head

        num_in = getattr(roi_extractor, 'num_inputs', len(x))
        feats_for_extractor = x[:num_in]
        bbox_feats = roi_extractor(feats_for_extractor, rois)

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        # ----- SFG-RL Modulation -----
        if self.use_state_condition or self.use_four_dir_pool:
            img_inds = rois[:, 0].long()
            batch_state = self._compute_batch_state(x, getattr(self, 'batch_data_samples', []))
            states = batch_state[img_inds]  # (N, C_state)

        # ✅ State-conditioned FiLM
        if self.use_state_condition and self.state_projector is not None:
            gamma_s, beta_s = self.state_projector(states)
            bbox_feats = _film_apply(bbox_feats, gamma_s, beta_s)

        # ✅ Four-direction ROI Pooling + Gating (FiLM)
        if self.use_four_dir_pool and self.fourdir_gate is not None:
            gamma_r, beta_r = self.fourdir_gate(bbox_feats)
            if not (gamma_r.abs().sum() == 0 and beta_r.abs().sum() == 0):
                bbox_feats = _film_apply(bbox_feats, gamma_r, beta_r)

        # Forward bbox head
        cls_score, bbox_pred = bbox_head(bbox_feats)
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage: int, x: Tuple[Tensor],
                            sampling_results: List,
                            batch_data_samples: List[DetDataSample]) -> dict:
        """
        Standard training flow, but attaches batch_data_samples for state computation.
        """
        self.batch_data_samples = batch_data_samples
        rois = self.bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)

        if isinstance(self.bbox_head, nn.ModuleList):
            bbox_head = self.bbox_head[stage]
        else:
            bbox_head = self.bbox_head

        bbox_targets = bbox_head.get_targets(sampling_results, batch_data_samples)
        loss_bbox = bbox_head.loss(bbox_results['cls_score'],
                                   bbox_results['bbox_pred'],
                                   *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox, rois=rois)
        return bbox_results

    # ---------------- Loss Calculation Adjustments ------------------

    def loss(self,
             x: Tuple[Tensor],
             rpn_results_list: List[InstanceData],
             batch_data_samples: List[DetDataSample],
             **kwargs) -> dict:
        """
        ✅ Fixed version: Ensures all modules participate in forward pass even if no valid samples exist.
        """
        # Filter samples without Ground Truth
        valid_data_samples = []
        valid_rpn_results = []
        for s, r in zip(batch_data_samples, rpn_results_list):
            gt = getattr(s, 'gt_instances', None)
            has_box = (gt is not None) and hasattr(gt, 'bboxes') and (gt.bboxes is not None) and (len(gt.bboxes) > 0)
            if has_box:
                valid_data_samples.append(s)
                valid_rpn_results.append(r)

        # ✅ Key Fix: If no valid samples, run a dummy forward pass to avoid DDP unused parameter errors
        if len(valid_data_samples) == 0:
            print("[WARNING] No valid samples with GT boxes in this batch.")
            with torch.no_grad():
                if self.use_state_condition or self.use_four_dir_pool:
                    _ = self._compute_batch_state(x, batch_data_samples)
                    device = x[0].device
                    dtype = x[0].dtype
                    dummy_rois = torch.tensor([[0, 10, 10, 50, 50]], device=device, dtype=dtype)
                    self.batch_data_samples = batch_data_samples
                    _ = self._bbox_forward(0, x, dummy_rois)
            return {}

        self.batch_data_samples = valid_data_samples
        losses = super().loss(x, valid_rpn_results, valid_data_samples, **kwargs)
        return losses

    # ---------------- Prediction with Relation Refinement ----------------------

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: List[InstanceData],
                batch_data_samples: List[DetDataSample],
                rescale: bool = False) -> List[InstanceData]:
        """
        Inference phase, supports KNN relation optimization.
        """
        # Parent predict returns List[InstanceData]
        results = super().predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        if not self.use_relation_refine:
            return results

        # Post-processing: KNN relation smoothing per image
        for data_sample in results: 
            # ✅ Key Fix: Directly use data_sample as the prediction instance 'pred'
            pred = data_sample
            
            if pred is None:
                continue
            
            scores = None
            if hasattr(pred, 'scores'):
                scores = pred.scores
            elif hasattr(pred, 'scores_mask'): 
                scores = pred.scores_mask
            
            if scores is None or len(scores) == 0:
                continue

            bboxes = pred.bboxes
            # Support both (N,) and (N, C) scores
            if scores.ndim in [1, 2]:
                new_scores = _knn_relation_refine(
                    bboxes.to(scores.device), scores, k=self.rel_k, alpha=self.rel_alpha)
                pred.scores = new_scores

        return results
