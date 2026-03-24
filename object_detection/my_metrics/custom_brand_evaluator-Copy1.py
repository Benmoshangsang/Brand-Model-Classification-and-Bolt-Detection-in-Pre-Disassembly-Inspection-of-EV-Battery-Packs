from typing import List, Sequence, Tuple, Union
import torch
from torch import Tensor
from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample


@METRICS.register_module()
class CustomBrandEvaluator(BaseMetric):
    """
    Evaluator for brand classification accuracy (top-k).
    Returns metrics like 'brand_acc_top1', 'brand_acc_top5'.
    """

    def __init__(self,
                 topk: Union[int, List[int]] = (1,),
                 collect_device: str = 'cpu',
                 prefix: str = '',
                 debug: bool = False,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.prefix = prefix or ''  # ✅ 防止为 None，关键修复点！
        self.topk = topk if isinstance(topk, (list, tuple)) else [topk]
        self.debug = debug  # ✅ 控制日志输出

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[DetDataSample]) -> List[Tuple[Tensor, Tensor]]:
        results = []
        for sample in data_samples:
            if not isinstance(sample, DetDataSample):
                continue

            pred: InstanceData = getattr(sample, 'pred_instances', None)
            gt: InstanceData = getattr(sample, 'gt_instances', None)
            if pred is None or gt is None:
                continue
            if not hasattr(pred, 'brand_score') or pred.brand_score is None:
                continue
            if not hasattr(gt, 'brand_labels') or gt.brand_labels is None:
                continue
            if gt.brand_labels.numel() == 0 or pred.brand_score.numel() == 0:
                continue

            pred_score = pred.brand_score  # shape: [N, C]
            if pred_score.ndim != 2 or pred_score.size(1) <= 1:
                continue

            avg_score = pred_score.mean(dim=0, keepdim=True)  # [1, C]
            gt_label = gt.brand_labels[0].view(1)             # [1]
            results.append((avg_score.detach().cpu(), gt_label.detach().cpu()))

        if self.debug:
            if len(results) == 0:
                print("[WARNING] No valid brand classification samples collected in process().", flush=True)
            else:
                print(f"[INFO] Collected {len(results)} samples for brand accuracy evaluation.", flush=True)

        return results

    def compute_metrics(self, results: List[Tuple[Tensor, Tensor]]) -> dict:
        if len(results) == 0:
            if self.debug:
                print("[WARNING] No brand classification results to evaluate.", flush=True)
            return {f'{self.prefix}brand_acc_top{k}': 0.0 for k in self.topk}

        preds, targets = zip(*results)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        acc_topk = self.accuracy(preds, targets, topk=self.topk)
        metrics = {f'{self.prefix}brand_acc_top{k}': acc for k, acc in zip(self.topk, acc_topk)}

        if self.debug:
            print(f"[DEBUG] Returned brand metrics: {metrics}", flush=True)

        return metrics

    @staticmethod
    def accuracy(output: Tensor, target: Tensor, topk=(1,)) -> List[float]:
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # [maxk, B]
        pred = pred.t()  # [maxk, B]
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # [maxk, B]

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            acc = (correct_k / batch_size).item()
            res.append(acc)
        return res
