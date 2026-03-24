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
    """GAP over spatial dims: (B,C,H,W) -> (B,C)."""
    return F.adaptive_avg_pool2d(feat, 1).flatten(1)


def _gather_batch_state_from_metas(batch_data_samples: List[DetDataSample],
                                   key: str = 'state') -> Optional[Tensor]:
    """Try to read external backbone state from data samples' metainfo."""
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
    # (B, C_state)
    return torch.stack(states, dim=0)


class _StateProjector(nn.Module):
    """Project (B, C_in) state to (B, C_roi) FiLM params."""
    def __init__(self, c_in: int, c_out: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(c_in, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * c_out)  # -> [gamma, beta]
        )

    def forward(self, s: Tensor) -> Tuple[Tensor, Tensor]:
        gb = self.mlp(s)  # (B, 2*C)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta


class _FourDirROIGate(nn.Module):
    """
    Four-direction ROI pooling (H/V/Diag/Anti) + learnable gating.
    Given x: (N, C, H, W), returns per-ROI channel-wise FiLM params (gamma, beta).
    """
    def __init__(self, c: int, hidden: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(4 * c, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 4),  # gate logits for 4 directions
        )
        self.out = nn.Sequential(
            nn.Linear(4 * c, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * c)  # -> [gamma, beta] per ROI
        )

    @staticmethod
    def _diag_pool(x: Tensor) -> Tensor:
        """
        Main-diagonal pooling: mean over elements where i==j.
        x: (N, C, H, W) -> (N, C)
        """
        N, C, H, W = x.shape
        L = min(H, W)
        device = x.device
        i = torch.arange(L, device=device)
        # (N, C, L)
        diag = x[:, :, i, i]
        return diag.mean(dim=-1)

    @staticmethod
    def _anti_diag_pool(x: Tensor) -> Tensor:
        """
        Anti-diagonal pooling: mean over elements where i+j==W-1 (after cropping).
        x: (N, C, H, W) -> (N, C)
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
        anti = x[:, :, ii, jj]
        return anti.mean(dim=-1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x: (N, C, H, W)  -> gamma,beta: (N, C), (N, C)
        """
        N, C, H, W = x.shape
        # Horizontal & Vertical pooled vectors (两次均值与全局均值等价，这里保持写法直观)
        v_h = x.mean(dim=3).mean(dim=2)        # (N, C)
        v_v = x.mean(dim=2).mean(dim=3)        # (N, C)
        # Diagonal & Anti-diagonal pooled vectors
        v_d = self._diag_pool(x)               # (N, C)
        v_a = self._anti_diag_pool(x)          # (N, C)

        # Stack & gate
        v = torch.stack([v_h, v_v, v_d, v_a], dim=1)   # (N, 4, C)
        v_cat = v.reshape(N, 4 * C)                    # (N, 4C)
        gate = F.softmax(self.proj(v_cat), dim=-1).unsqueeze(-1)  # (N, 4, 1)
        fused = (v * gate).sum(dim=1)                  # (N, C)  # 目前未直接用进 FiLM，可按需启用残差

        gb = self.out(v_cat)                           # (N, 2C)
        gamma, beta = gb.chunk(2, dim=-1)              # (N, C), (N, C)
        return gamma, beta


def _film_apply(x: Tensor, gamma: Tensor, beta: Tensor,
                eps: float = 1e-6) -> Tensor:
    """
    Apply channel-wise FiLM to ROI features.
    x: (N, C, H, W), gamma/beta: (N, C)
    """
    N, C, H, W = x.shape
    gamma = gamma.view(N, C, 1, 1)
    beta = beta.view(N, C, 1, 1)
    return x * (1.0 + gamma) + beta


def _knn_relation_refine(bboxes: Tensor,
                         scores: Tensor,
                         k: int = 6,
                         alpha: float = 0.5) -> Tensor:
    """
    Lightweight KNN neighborhood reweighting over scores (per image).
    bboxes: (M, 4) xyxy; scores: (M,) or (M, C)
    Return refined scores with same shape.
    """
    if bboxes.numel() == 0:
        return scores
    centers = torch.stack([(bboxes[:, 0] + bboxes[:, 2]) * 0.5,
                           (bboxes[:, 1] + bboxes[:, 3]) * 0.5], dim=-1)  # (M,2)
    # Pairwise distances
    dist = torch.cdist(centers, centers, p=2)  # (M, M)
    # Exclude self
    M = centers.size(0)
    dist[torch.arange(M), torch.arange(M)] = float('inf')
    # kNN indices
    k_eff = min(k, max(1, M - 1))
    knn_idx = torch.topk(-dist, k=k_eff, dim=-1).indices  # negative for smallest
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
    SFG-RL Head inside a Cascade ROI Head:
      - State-guided ROI conditioning (FiLM from global state)
      - Four-direction ROI pooling + gating (ROI-level angle robustness)
      - Relation-aware neighborhood aggregation (predict-time optional)

    All knobs are configurable via init kwargs.
    """
    def __init__(self,
                 *args,
                 num_stages: int = 3,
                 # ---- SFG-RL knobs ----
                 use_state_condition: bool = True,
                 use_four_dir_pool: bool = True,
                 use_relation_refine: bool = True,
                 # state dims (will be inferred if None)
                 state_in_channels: Optional[int] = None,
                 # projector hidden dims
                 state_hidden: int = 256,
                 fourdir_hidden: int = 256,
                 # relation refine params
                 rel_k: int = 6,
                 rel_alpha: float = 0.5,
                 # if True, read external state from data_samples.metainfo['state']
                 use_external_state: bool = False,
                 **kwargs):
        super().__init__(*args, num_stages=num_stages, **kwargs)

        # Flags
        self.use_state_condition = use_state_condition
        self.use_four_dir_pool = use_four_dir_pool
        self.use_relation_refine = use_relation_refine
        self.use_external_state = use_external_state

        # Buffers to be lazily built when first forward happens (since we don't
        # know ROI feat channels until roi_extractor is constructed)
        self._film_proj_built = False

        # State dims & relation params
        self.state_in_channels_cfg = state_in_channels
        self.state_hidden = state_hidden
        self.fourdir_hidden = fourdir_hidden
        self.rel_k = rel_k
        self.rel_alpha = rel_alpha

        # Placeholders for submodules (built lazily)
        self.state_projector: Optional[_StateProjector] = None
        self.fourdir_gate: Optional[_FourDirROIGate] = None

    # ---------------------------- utils ----------------------------

    def _build_film_modules_if_needed(self, x: Tuple[Tensor]):
        """Build FiLM/gate modules once we know channels."""
        if self._film_proj_built:
            return

        # 兼容单模块或 ModuleList
        if isinstance(self.bbox_roi_extractor, nn.ModuleList):
            roi_extractor0 = self.bbox_roi_extractor[0]
        else:
            roi_extractor0 = self.bbox_roi_extractor

        # Infer ROI feature channels from roi_extractor if available, else from last feat
        if hasattr(roi_extractor0, 'out_channels'):
            c_roi = roi_extractor0.out_channels
        else:
            c_roi = x[-1].shape[1]

        # Infer state_in from x[-1] if not provided
        if self.state_in_channels_cfg is None:
            c_state_in = x[-1].shape[1]
        else:
            c_state_in = self.state_in_channels_cfg

        if self.use_state_condition:
            self.state_projector = _StateProjector(c_in=c_state_in,
                                                   c_out=c_roi,
                                                   hidden=self.state_hidden)
        if self.use_four_dir_pool:
            self.fourdir_gate = _FourDirROIGate(c=c_roi, hidden=self.fourdir_hidden)

        self._film_proj_built = True

    def _compute_batch_state(self,
                             x: Tuple[Tensor],
                             batch_data_samples: List[DetDataSample]) -> Tensor:
        """
        Return (B, C_state). Prefer external state if provided; otherwise GAP(x[-1]).
        """
        if self.use_external_state:
            s = _gather_batch_state_from_metas(batch_data_samples, key='state')
            if s is not None:
                return s.to(x[-1].dtype).to(x[-1].device)

        # Fallback: GAP over the last feature map
        feat_last = x[-1]  # (B, C, H, W)
        return _global_avg_pool(feat_last)

    # --------------------- core overrides for bbox ---------------------

    def _bbox_forward(self, stage: int, x: Tuple[Tensor],
                      rois: Tensor) -> dict:
        """
        Inject SFG-RL conditioning before calling bbox_head.
        """
        # build submodules once
        self._build_film_modules_if_needed(x)

        # ROI feature extraction：从 ModuleList 中索引出当前 stage 的模块
        if isinstance(self.bbox_roi_extractor, nn.ModuleList):
            roi_extractor = self.bbox_roi_extractor[stage]
        else:
            roi_extractor = self.bbox_roi_extractor

        if isinstance(self.bbox_head, nn.ModuleList):
            bbox_head = self.bbox_head[stage]
        else:
            bbox_head = self.bbox_head

        # 按 num_inputs 裁剪输入特征数量，避免 extractor 只接收部分层
        num_in = getattr(roi_extractor, 'num_inputs', len(x))
        feats_for_extractor = x[:num_in]
        bbox_feats = roi_extractor(feats_for_extractor, rois)  # (N, C, H, W)

        # optional shared head
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        # ----- State-guided FiLM conditioning -----
        if self.use_state_condition or self.use_four_dir_pool:
            # per-image state
            img_inds = rois[:, 0].long()
            states = self._compute_batch_state(x, getattr(self, 'batch_data_samples', []))  # (B, C_state)
            states = states[img_inds]  # (N, C_state)

        if self.use_state_condition and self.state_projector is not None:
            gamma_s, beta_s = self.state_projector(states)  # (N, C_roi)
            bbox_feats = _film_apply(bbox_feats, gamma_s, beta_s)

        # ----- Four-direction ROI pooling + gating (FiLM) -----
        if self.use_four_dir_pool and self.fourdir_gate is not None:
            gamma_r, beta_r = self.fourdir_gate(bbox_feats)  # (N, C_roi)
            bbox_feats = _film_apply(bbox_feats, gamma_r, beta_r)

        # forward bbox head
        cls_score, bbox_pred = bbox_head(bbox_feats)
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage: int, x: Tuple[Tensor],
                            sampling_results: List,
                            batch_data_samples: List[DetDataSample]) -> dict:
        """
        Keep parent flow, but attach batch_data_samples for state computation.
        """
        # store for _bbox_forward
        self.batch_data_samples = batch_data_samples

        # 通过父类工具生成 RoIs（等价于 bbox2roi([res.bboxes ...])）
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

    # ---------------- original helpers / minor tweaks ------------------

    def _get_sampling_results(self,
                              stage: int,
                              batch_data_samples: SampleList,
                              batch_proposals: List[InstanceData]) -> List:
        sampling_results = []
        for i in range(len(batch_data_samples)):
            gt_instances = batch_data_samples[i].gt_instances
            gt_instances_ignore = getattr(batch_data_samples[i], 'ignored_instances', None)

            assign_result = self.bbox_assigner[stage].assign(
                batch_proposals[i], gt_instances, gt_instances_ignore)
            sampling_result = self.bbox_sampler[stage].sample(
                assign_result, batch_proposals[i], gt_instances)
            sampling_results.append(sampling_result)
        return sampling_results

    def loss(self,
             x: Tuple[Tensor],
             rpn_results_list: List[InstanceData],
             batch_data_samples: List[DetDataSample],
             **kwargs) -> dict:

        # 过滤无 GT 的样本，避免后续 target 构造报错
        valid_data_samples = []
        valid_rpn_results = []
        for s, r in zip(batch_data_samples, rpn_results_list):
            gt = getattr(s, 'gt_instances', None)
            has_box = (gt is not None) and hasattr(gt, 'bboxes') and (gt.bboxes is not None) and (len(gt.bboxes) > 0)
            if has_box:
                valid_data_samples.append(s)
                valid_rpn_results.append(r)

        if len(valid_data_samples) == 0:
            return {}

        # ensure availability for state in forward
        self.batch_data_samples = valid_data_samples

        losses = super().loss(x, valid_rpn_results, valid_data_samples, **kwargs)
        return losses

    # ---------------------- predict with relation refine ----------------------

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: List[InstanceData],
                batch_data_samples: List[DetDataSample],
                rescale: bool = False) -> List[DetDataSample]:

        results = super().predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        if not self.use_relation_refine:
            return results

        # Post-hoc KNN relation smoothing per image
        for data_sample in results:
            pred = data_sample.pred_instances
            if pred is None:
                continue
            # scores could be (N,) or (N, C). 支持两种
            scores = None
            if hasattr(pred, 'scores'):
                scores = pred.scores
            elif hasattr(pred, 'scores_mask'):
                scores = pred.scores_mask
            if scores is None:
                continue

            bboxes = pred.bboxes
            if scores.ndim == 1:
                new_scores = _knn_relation_refine(
                    bboxes.to(scores.device), scores, k=self.rel_k, alpha=self.rel_alpha)
                pred.scores = new_scores
            elif scores.ndim == 2:
                new_scores = _knn_relation_refine(
                    bboxes.to(scores.device), scores, k=self.rel_k, alpha=self.rel_alpha)
                pred.scores = new_scores
            # 其他字段（如 score_factors 等）保持不变

        return results
