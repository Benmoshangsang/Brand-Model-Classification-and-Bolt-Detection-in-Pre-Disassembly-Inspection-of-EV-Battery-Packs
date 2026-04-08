# my_metrics/custom_det_extra_evaluator.py
from __future__ import annotations
from typing import Dict, List, Sequence, Iterable
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS
from mmdet.structures import DetDataSample


def _to_numpy(x) -> np.ndarray:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _iou_matrix_xyxy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    N = 0 if boxes1 is None else boxes1.shape[0]
    M = 0 if boxes2 is None else boxes2.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)

    x11, y11, x12, y12 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
    x21, y21, x22, y22 = boxes2[:, 0:1], boxes2[:, 1:2], boxes2[:, 2:3], boxes2[:, 3:4]

    xa = np.maximum(x11, x21.T)
    ya = np.maximum(y11, y21.T)
    xb = np.minimum(x12, x22.T)
    yb = np.minimum(y12, y22.T)

    inter = np.clip(xb - xa, 0, None) * np.clip(yb - ya, 0, None)
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    union = area1 + area2.T - inter
    return (inter / np.maximum(union, 1e-12)).astype(np.float32)


@METRICS.register_module()
class CustomDetExtraEvaluator(BaseMetric):
    r"""Extra detection metrics over one or multiple IoU thresholds:
    - Matched-box classification accuracy (on TPs)
    - TP-mean IoU (average IoU over TPs)

    Assumptions:
      * Each DetDataSample contains:
        - pred_instances: fields `bboxes (xyxy)`, `labels`, `scores`
        - gt_instances:   fields `bboxes (xyxy)`, `labels`

    Args:
        iou_thrs (float | Iterable[float]): IoU threshold(s). e.g. 0.5 or (0.5, 0.75)
        report_cls_acc (bool): report matched classification accuracy.
        report_tp_mean_iou (bool): report TP-mean IoU.
        collect_device (str): mmengine collect device.
        debug (bool): print debug logs.
    """

    default_prefix: str = 'det'

    def __init__(self,
                 iou_thrs: float | Iterable[float] = 0.5,
                 report_cls_acc: bool = True,
                 report_tp_mean_iou: bool = True,
                 collect_device: str = 'cpu',
                 debug: bool = False) -> None:
        super().__init__(collect_device=collect_device)
        # normalize thresholds to sorted unique list
        if isinstance(iou_thrs, (list, tuple, set)):
            self.iou_thrs = sorted({float(t) for t in iou_thrs})
        else:
            self.iou_thrs = [float(iou_thrs)]
        self.report_cls_acc = report_cls_acc
        self.report_tp_mean_iou = report_tp_mean_iou
        self.debug = debug

    def _greedy_match_once(self, ious: np.ndarray, pred_labels: np.ndarray,
                           gt_labels: np.ndarray, thr: float) -> Dict[str, float | int]:
        """Greedy one-to-one matching for a single IoU threshold."""
        matched_g = set()
        tp = 0
        cls_correct = 0
        iou_sum = 0.0

        # predictions assumed sorted by score desc before computing iou matrix
        for i in range(ious.shape[0]):
            if ious.shape[1] == 0:
                break
            ious_i = ious[i].copy()
            if matched_g:
                idx = np.fromiter(matched_g, dtype=np.int64)
                ious_i[idx] = -1.0
            g_best = int(np.argmax(ious_i))
            best_iou = float(ious_i[g_best])
            if best_iou >= thr:
                matched_g.add(g_best)
                tp += 1
                iou_sum += best_iou
                if int(pred_labels[i]) == int(gt_labels[g_best]):
                    cls_correct += 1
        return dict(tp=tp, cls_correct=cls_correct, iou_sum=iou_sum)

    def process(self,
                data_batch: Sequence[dict],
                data_samples: Sequence[DetDataSample]) -> None:
        for sample in data_samples:
            if isinstance(sample, DetDataSample):
                pred = getattr(sample, 'pred_instances', None)
                gt = getattr(sample, 'gt_instances', None)
            else:
                pred = sample.get('pred_instances', None)
                gt = sample.get('gt_instances', None)

            if pred is None or gt is None:
                continue

            pb = _to_numpy(getattr(pred, 'bboxes', None))
            pl = _to_numpy(getattr(pred, 'labels', None))
            ps = _to_numpy(getattr(pred, 'scores', None))
            gb = _to_numpy(getattr(gt, 'bboxes', None))
            gl = _to_numpy(getattr(gt, 'labels', None))

            if pb is None or gb is None or pl is None or gl is None:
                continue

            # sort predictions by score desc for stable greedy matching
            if ps is not None:
                order = np.argsort(-ps)
                pb, pl, ps = pb[order], pl[order], ps[order]

            ious = _iou_matrix_xyxy(pb, gb)

            # compute stats for each IoU threshold independently
            per_thr_stats = []
            for thr in self.iou_thrs:
                s = self._greedy_match_once(ious, pl, gl, thr)
                per_thr_stats.append(s)

            # accumulate into results (store per-thr arrays)
            self.results.append({
                'tp': [s['tp'] for s in per_thr_stats],
                'cls_correct': [s['cls_correct'] for s in per_thr_stats],
                'iou_sum': [s['iou_sum'] for s in per_thr_stats],
            })

    def compute_metrics(self, results: List[Dict]) -> Dict[str, float]:
        if len(results) == 0:
            out: Dict[str, float] = {}
            for thr in self.iou_thrs:
                if self.report_cls_acc:
                    out[f'det/matched_cls_acc@{thr:.2f}'] = 0.0
                if self.report_tp_mean_iou:
                    out[f'det/tp_mean_iou@{thr:.2f}'] = 0.0
            return out

        # reduce across images: sum per-threshold arrays
        tp_tot = np.sum(np.array([r['tp'] for r in results], dtype=float), axis=0)
        cls_tot = np.sum(np.array([r['cls_correct'] for r in results], dtype=float), axis=0)
        iou_sum_tot = np.sum(np.array([r['iou_sum'] for r in results], dtype=float), axis=0)

        out: Dict[str, float] = {}
        for idx, thr in enumerate(self.iou_thrs):
            tp = tp_tot[idx]
            cls_ok = cls_tot[idx]
            iou_s = iou_sum_tot[idx]
            if self.report_cls_acc:
                acc = (cls_ok / tp) if tp > 0 else 0.0
                out[f'det/matched_cls_acc@{thr:.2f}'] = float(acc)
            if self.report_tp_mean_iou:
                tp_mean_iou = (iou_s / tp) if tp > 0 else 0.0
                out[f'det/tp_mean_iou@{thr:.2f}'] = float(tp_mean_iou)

       
        if len(self.iou_thrs) > 1:
            if self.report_cls_acc:
                vals = [out[f'det/matched_cls_acc@{thr:.2f}'] for thr in self.iou_thrs]
                out['det/matched_cls_acc@mean'] = float(np.mean(vals))
            if self.report_tp_mean_iou:
                vals = [out[f'det/tp_mean_iou@{thr:.2f}'] for thr in self.iou_thrs]
                out['det/tp_mean_iou@mean'] = float(np.mean(vals))

        if len(self.iou_thrs) > 1:
            
            lo, hi = self.iou_thrs[0], self.iou_thrs[-1]
            out['det/_iou_range'] = float(lo + (hi - lo))  

        return out
