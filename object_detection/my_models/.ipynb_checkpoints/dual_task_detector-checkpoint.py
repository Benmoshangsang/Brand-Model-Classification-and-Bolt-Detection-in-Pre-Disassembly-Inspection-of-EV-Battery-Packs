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

    def loss(self, inputs: torch.Tensor, data_samples: SampleList) -> dict:
        # 主任务：目标检测损失
        losses = super().loss(inputs, data_samples)

        # 子任务：整图品牌分类损失
        if self.brand_head is not None:
            feats = self.extract_feat(inputs)
            gap_feat = [f.mean(dim=[2, 3]) for f in feats]  # B×C
            pooled_feat = sum(gap_feat) / len(gap_feat)

            cls_score = self.brand_head(pooled_feat)

            # ✅ 从 metainfo 中读取 brand_label（每个为 [1] Tensor），拼接
            brand_labels = torch.cat([
                sample.metainfo['brand_label'] for sample in data_samples
            ], dim=0).to(cls_score.device)

            loss_brand_cls = self.brand_head.loss(cls_score, brand_labels)
            # ✅ 修复 KeyError：正确读取键 'loss_brand_cls'
            losses['loss_brand_cls'] = loss_brand_cls['loss_brand_cls'] * self.loss_weight_brand

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

            brand_probs = self.brand_head.predict(pooled_feat)  # shape: [B, num_classes]

            for i, sample in enumerate(results):
                # ✅ 获取目标设备
                device = sample.pred_instances.bboxes.device if hasattr(sample.pred_instances, 'bboxes') else brand_probs[i].device
                score_tensor = brand_probs[i].detach().to(device)

                # ✅ 正确写入 brand_score 到 pred_instances（非 sample 本体）
                sample.pred_instances.set_field(score_tensor, name='brand_score', dtype=torch.Tensor)

        return results
