from typing import List
import torch
from torch import nn
from mmdet.models.detectors import CascadeRCNN
from mmdet.registry import MODELS
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample, SampleList


@MODELS.register_module()
class DualTaskDetector(CascadeRCNN):
    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 roi_head,
                 data_preprocessor,
                 brand_head=None,
                 loss_weight_brand=0.5,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):

        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg
        )

        self.loss_weight_brand = loss_weight_brand
        self.brand_head = MODELS.build(brand_head) if isinstance(brand_head, dict) else brand_head

        # 给 roi_head 注册品牌识别 head
        if hasattr(self.roi_head, 'brand_head') and self.brand_head is not None:
            self.roi_head.brand_head = self.brand_head

    def loss(self, inputs: torch.Tensor, data_samples: SampleList) -> dict:
        # 主任务：Cascade RCNN 检测损失
        losses = super().loss(inputs, data_samples)

        if self.brand_head is not None:
            feats = self.extract_feat(inputs)
            gap_feat = [f.mean(dim=[2, 3]) for f in feats]  # B×C
            pooled_feat = sum(gap_feat) / len(gap_feat)     # [B, C]

            # 提取 OCR 特征（None → 0向量）
            ocr_feats = []
            for sample in data_samples:
                ocr_feat = sample.metainfo.get('ocr_feat', None)
                if ocr_feat is not None:
                    ocr_feats.append(ocr_feat.to(dtype=pooled_feat.dtype, device=pooled_feat.device))
                else:
                    ocr_feats.append(torch.zeros_like(pooled_feat[0]))

            ocr_tensor = torch.stack(ocr_feats, dim=0)  # [B, D]

            # 品牌标签
            brand_labels = torch.cat([
                sample.metainfo['brand_label'] for sample in data_samples
            ], dim=0).to(pooled_feat.device)

            # 计算整图品牌损失
            cls_score = self.brand_head(pooled_feat, ocr_feat=ocr_tensor)
            loss_brand = self.brand_head.loss(cls_score, brand_labels)

            # 合并损失
            losses['loss_brand_cls'] = loss_brand['loss_brand_cls'] * self.loss_weight_brand

        return losses

    def predict(self,
                inputs: torch.Tensor,
                batch_data_samples: List[DetDataSample],
                **kwargs) -> List[DetDataSample]:

        results = super().predict(inputs, batch_data_samples=batch_data_samples, **kwargs)

        if self.brand_head is not None:
            feats = self.extract_feat(inputs)
            gap_feat = [f.mean(dim=[2, 3]) for f in feats]
            pooled_feat = sum(gap_feat) / len(gap_feat)

            ocr_feats = []
            for sample in batch_data_samples:
                ocr_feat = sample.metainfo.get('ocr_feat', None)
                if ocr_feat is not None:
                    ocr_feats.append(ocr_feat.to(dtype=pooled_feat.dtype, device=pooled_feat.device))
                else:
                    ocr_feats.append(torch.zeros_like(pooled_feat[0]))

            ocr_tensor = torch.stack(ocr_feats, dim=0)

            # 得到预测概率 [B, num_classes]
            brand_probs = self.brand_head.predict(pooled_feat, ocr_feat=ocr_tensor)

            for i, sample in enumerate(results):
                device = sample.pred_instances.bboxes.device if hasattr(sample.pred_instances, 'bboxes') else brand_probs[i].device
                score_tensor = brand_probs[i].detach().to(device)
                sample.pred_instances.set_field(score_tensor, name='brand_score', dtype=torch.Tensor)

        return results
