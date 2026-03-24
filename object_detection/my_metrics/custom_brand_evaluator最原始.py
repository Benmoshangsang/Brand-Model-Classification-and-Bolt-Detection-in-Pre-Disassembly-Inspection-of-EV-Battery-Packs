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
    Works at image level, using brand_score in pred_instances
    and brand_id in metainfo.
    """

    def __init__(self,
                 topk: Union[int, List[int]] = (1, 5),
                 collect_device: str = 'cpu',
                 debug: bool = False,
                 **kwargs):
        super().__init__(collect_device=collect_device)
        self.topk = topk if isinstance(topk, (list, tuple)) else [topk]
        self.debug = debug

    def process(self,
                data_batch: Sequence[dict],
                data_samples: Sequence[DetDataSample]) -> None:
        results = []

        for sample in data_samples:
            if not isinstance(sample, DetDataSample):
                continue
            if not hasattr(sample, 'pred_instances'):
                continue
            pred: InstanceData = sample.pred_instances

            if not hasattr(pred, 'brand_score') or pred.brand_score is None:
                continue
            brand_score = pred.brand_score
            if not isinstance(brand_score, Tensor):
                continue

            # ✅ 兼容 [C] 和 [1, C] 情形
            if brand_score.ndim == 2 and brand_score.size(0) == 1:
                brand_score = brand_score.squeeze(0)
            elif brand_score.ndim != 1:
                continue

            avg_score = brand_score.view(1, -1)

            # ✅ 从 metainfo 中获取 brand_id
            brand_id = sample.metainfo.get('brand_id', -1)
            if not isinstance(brand_id, int) or brand_id < 0:
                continue
            gt_label = torch.tensor([brand_id], dtype=torch.long)

            results.append((avg_score.detach().cpu(), gt_label.detach().cpu()))

        if self.debug:
            if results:
                print(f"[INFO] Collected {len(results)} valid brand samples.", flush=True)
            else:
                print(f"[WARNING] No valid samples for brand evaluation.", flush=True)

        # ✅ 修复关键问题：添加到 self.results
        self.results.extend(results)

    def compute_metrics(self, results: List[Tuple[Tensor, Tensor]]) -> dict:
        if len(results) == 0:
            print('[WARNING] No valid brand predictions for evaluation.', flush=True)
            return {
                f'brand_top{k}_acc': 0.0
                for k in self.topk
            }

        preds, targets = zip(*results)
        preds = torch.cat(preds, dim=0)     # [B, C]
        targets = torch.cat(targets, dim=0) # [B]

        acc_topk = self.accuracy(preds, targets, topk=self.topk)
        metrics = {
            f'brand_top{k}_acc': acc
            for k, acc in zip(self.topk, acc_topk)
        }

        if self.debug:
            print(f"[DEBUG] Computed metrics: {metrics}", flush=True)

        return metrics

    @staticmethod
    def accuracy(output: Tensor, target: Tensor, topk=(1,)) -> List[float]:
        """Compute top-K accuracy."""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # shape: [maxk, B]
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # shape: [maxk, B]

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append((correct_k / batch_size).item())
        return res
