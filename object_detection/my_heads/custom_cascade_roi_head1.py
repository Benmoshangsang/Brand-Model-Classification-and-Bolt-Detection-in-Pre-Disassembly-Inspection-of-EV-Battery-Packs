import torch
from torch import Tensor
from typing import List, Tuple, Optional
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample, SampleList
from mmdet.models.roi_heads import CascadeRoIHead
from mmdet.registry import MODELS

@MODELS.register_module()
class CustomCascadeRoIHead(CascadeRoIHead):
    def __init__(self,
                 *args,
                 num_stages=3,
                 **kwargs):
        super().__init__(*args, num_stages=num_stages, **kwargs)

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

        valid_data_samples = [
            s for s in batch_data_samples
            if s.gt_instances and len(s.gt_instances.get('bboxes', [])) > 0
        ]
        valid_rpn_results = [
            r for s, r in zip(batch_data_samples, rpn_results_list)
            if s.gt_instances and len(s.gt_instances.get('bboxes', [])) > 0
        ]

        if len(valid_data_samples) == 0:
            return {}

        # 计算原有 ROI 检测 loss（Cascade）
        losses = super().loss(x, valid_rpn_results, valid_data_samples, **kwargs)

        # === 整图品牌分类任务（融合 OCR 特征） ===
        if hasattr(self, 'brand_head'):
            # 提取整图特征（通常来自主干最后一级）
            # 取特征金字塔的最后一层（e.g., P5），并进行 GAP
            feat = x[-1]  # [B, C, H, W]
            pooled_feat = torch.nn.functional.adaptive_avg_pool2d(feat, output_size=1).flatten(1)  # [B, C]

            # 提取品牌标签和 OCR 特征
            brand_labels = []
            ocr_feats = []

            for sample in batch_data_samples:
                brand_label = sample.metainfo.get('brand_label', torch.tensor(-1))
                brand_labels.append(brand_label)

                ocr_feat = sample.metainfo.get('ocr_feat', None)
                if ocr_feat is not None:
                    ocr_feats.append(ocr_feat.to(dtype=pooled_feat.dtype, device=pooled_feat.device))
                else:
                    ocr_feats.append(None)

            brand_labels = torch.stack(brand_labels, dim=0)

            # 构造 OCR 特征张量（None 替换为零向量）
            if any(f is not None for f in ocr_feats):
                ocr_dim = ocr_feats[0].shape[-1]
                ocr_tensor = torch.stack([
                    f if f is not None else pooled_feat.new_zeros(ocr_dim)
                    for f in ocr_feats
                ], dim=0)
            else:
                ocr_tensor = None

            # 前向 + 损失
            brand_logits = self.brand_head.forward(pooled_feat, ocr_feat=ocr_tensor)
            brand_loss_dict = self.brand_head.loss(brand_logits, brand_labels)

            losses.update(brand_loss_dict)

        return losses

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: List[InstanceData],
                batch_data_samples: List[DetDataSample],
                rescale: bool = False) -> List[DetDataSample]:

        results = super().predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        # === 整图品牌预测（融合 OCR 特征）===
        if hasattr(self, 'brand_head'):
            feat = x[-1]  # 主干最后一层
            pooled_feat = torch.nn.functional.adaptive_avg_pool2d(feat, output_size=1).flatten(1)

            ocr_feats = []
            for sample in batch_data_samples:
                ocr_feat = sample.metainfo.get('ocr_feat', None)
                if ocr_feat is not None:
                    ocr_feats.append(ocr_feat.to(dtype=pooled_feat.dtype, device=pooled_feat.device))
                else:
                    ocr_feats.append(None)

            if any(f is not None for f in ocr_feats):
                ocr_dim = ocr_feats[0].shape[-1]
                ocr_tensor = torch.stack([
                    f if f is not None else pooled_feat.new_zeros(ocr_dim)
                    for f in ocr_feats
                ], dim=0)
            else:
                ocr_tensor = None

            brand_probs = self.brand_head.predict(pooled_feat, ocr_feat=ocr_tensor)  # [B, num_classes]
            for i, sample in enumerate(results):
                sample.pred_instances.brand_score = brand_probs[i].detach()

        return results
