from typing import List, Sequence, Tuple, Union, Dict
import torch
from torch import Tensor
from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
import numpy as np


@METRICS.register_module()
class CustomBrandEvaluator(BaseMetric):
    """
    Evaluator for brand classification with 5 metrics:
      - Top-1 Accuracy
      - Top-5 Accuracy
      - Macro Precision
      - Macro Recall
      - Macro F1

    Works at image level, using `brand_score` in pred_instances
    and `brand_id` in metainfo.

    Notes:
    - Accepts brand_score in shape [C] or [1, C].
    - Ignores samples without valid brand_score or brand_id.
    """

    def __init__(self,
                 topk: Union[int, List[int]] = (1, 5),
                 collect_device: str = 'cpu',
                 debug: bool = False,
                 **kwargs):
        super().__init__(collect_device=collect_device)
        self.topk = topk if isinstance(topk, (list, tuple)) else [topk]
        # 确保 top-1/top-5 都在
        if 1 not in self.topk:
            self.topk = [1] + list(self.topk)
        if 5 not in self.topk:
            self.topk = list(self.topk) + [5]
        # 排序去重
        self.topk = sorted(set(self.topk))
        self.debug = debug

    def process(self,
                data_batch: Sequence[dict],
                data_samples: Sequence[DetDataSample]) -> None:
        results: List[Tuple[Tensor, Tensor]] = []

        for sample in data_samples:
            if not isinstance(sample, DetDataSample):
                continue
            if not hasattr(sample, 'pred_instances'):
                continue
            pred: InstanceData = sample.pred_instances

            # 读取品牌分数
            if not hasattr(pred, 'brand_score') or pred.brand_score is None:
                continue
            brand_score = pred.brand_score
            if not isinstance(brand_score, Tensor):
                continue

            # 兼容 [C] 和 [1, C]
            if brand_score.ndim == 2 and brand_score.size(0) == 1:
                brand_score = brand_score.squeeze(0)
            elif brand_score.ndim != 1:
                continue

            # 保持 batch 维度为 1，后续拼接
            avg_score = brand_score.view(1, -1)

            # 取 brand_id（int）
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

        # 累积到 self.results（BaseMetric 会在分布式下聚合）
        self.results.extend(results)

    def compute_metrics(self, results: List[Tuple[Tensor, Tensor]]) -> Dict[str, float]:
        # 无有效样本的兜底
        if len(results) == 0:
            if self.debug:
                print('[WARNING] No valid brand predictions for evaluation.', flush=True)
            out = {f'brand_top{k}_acc': 0.0 for k in self.topk}
            out.update({
                'brand_macro_precision': 0.0,
                'brand_macro_recall': 0.0,
                'brand_macro_f1': 0.0
            })
            return out

        # 拼接成 [B, C] & [B]
        preds, targets = zip(*results)
        preds: Tensor = torch.cat(preds, dim=0)     # [B, C]
        targets: Tensor = torch.cat(targets, dim=0) # [B]

        # ===== 1) Top-k Accuracies =====
        acc_topk = self.accuracy(preds, targets, topk=self.topk)
        out = {f'brand_top{k}_acc': acc for k, acc in zip(self.topk, acc_topk)}

        # ===== 2) Macro Precision / Recall / F1 =====
        # 预测标签（top-1）
        pred_labels = torch.argmax(preds, dim=1)  # [B]
        y_true = targets.cpu().numpy().astype(np.int64)
        y_pred = pred_labels.cpu().numpy().astype(np.int64)

        # 类别数：优先用预测向量维度 C；若异常则回退到 max(label)+1
        C = int(preds.shape[1]) if preds.ndim == 2 and preds.shape[1] > 0 else int(max(y_true.max(), y_pred.max()) + 1)

        # 混淆矩阵 cm[t, p]
        cm = np.zeros((C, C), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < C and 0 <= p < C:
                cm[t, p] += 1

        TP = np.diag(cm).astype(np.float64)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP

        eps = 1e-12
        # 只在“验证集中出现过的类别”（支持 > 0）上做宏平均，避免全零类拉低结果
        support = cm.sum(axis=1)  # 每类的真值样本数
        mask = support > 0

        P_c = TP / np.maximum(TP + FP, eps)
        R_c = TP / np.maximum(TP + FN, eps)
        F1_c = 2 * P_c * R_c / np.maximum(P_c + R_c, eps)

        if mask.any():
            macro_p = float(P_c[mask].mean())
            macro_r = float(R_c[mask].mean())
            macro_f1 = float(F1_c[mask].mean())
        else:
            # 极端兜底：验证集里没有任何品牌标签
            macro_p = macro_r = macro_f1 = 0.0

        out.update({
            'brand_macro_precision': macro_p,
            'brand_macro_recall': macro_r,
            'brand_macro_f1': macro_f1
        })

        if self.debug:
            print(f"[DEBUG] metrics: {out}", flush=True)

        return out

    @staticmethod
    def accuracy(output: Tensor, target: Tensor, topk=(1,)) -> List[float]:
        """Compute top-K accuracy on probabilities/logits.

        Args:
            output: [B, C]
            target: [B]
            topk: Iterable[int]

        Returns:
            List[float]: accuracies for each k in topk
        """
        maxk = max(topk)
        batch_size = target.size(0)

        # top-k indices: [B, maxk] -> transpose -> [maxk, B]
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # [maxk, B]

        res = []
        for k in topk:
            # 统计前 k 行的命中数
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append((correct_k / batch_size).item())
        return res
