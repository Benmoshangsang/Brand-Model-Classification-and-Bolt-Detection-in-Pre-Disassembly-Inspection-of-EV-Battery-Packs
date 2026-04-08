from typing import List
import torch
from torch import nn
from mmdet.models.detectors import CascadeRCNN
from mmdet.registry import MODELS
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample, SampleList


@MODELS.register_module()
class DualTaskDetector(CascadeRCNN):
    """
    A dual-task detector based on Cascade R-CNN that performs both 
    object detection and image-level brand classification.
    """
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
        # Build brand_head if it's a config dict, otherwise use as is
        self.brand_head = MODELS.build(brand_head) if isinstance(brand_head, dict) else brand_head

    def loss(self, inputs: torch.Tensor, data_samples: SampleList) -> dict:
        """Calculate losses for both detection and brand classification."""
        # Main Task: Object Detection Loss
        losses = super().loss(inputs, data_samples)

        # Sub-task: Whole-image Brand Classification Loss
        if self.brand_head is not None:
            feats = self.extract_feat(inputs)
            # Global Average Pooling (GAP) across spatial dimensions: BxCxHxW -> BxC
            gap_feat = [f.mean(dim=[2, 3]) for f in feats]
            pooled_feat = sum(gap_feat) / len(gap_feat)

            cls_score = self.brand_head(pooled_feat)

            # ✅ Retrieve brand_label from metainfo (expected as [1] Tensor per sample) and concatenate
            brand_labels = torch.cat([
                sample.metainfo['brand_label'] for sample in data_samples
            ], dim=0).to(cls_score.device)

            loss_brand_cls = self.brand_head.loss(cls_score, brand_labels)
            
            # ✅ Fix KeyError: Correctly access the key 'loss_brand_cls' from the head's loss dict
            losses['loss_brand_cls'] = loss_brand_cls['loss_brand_cls'] * self.loss_weight_brand

        return losses

    def predict(self,
                inputs: torch.Tensor,
                batch_data_samples: List[DetDataSample],
                **kwargs) -> List[DetDataSample]:
        """Predict detection results and brand scores."""
        results = super().predict(inputs, batch_data_samples=batch_data_samples, **kwargs)

        if self.brand_head is not None:
            feats = self.extract_feat(inputs)
            gap_feat = [f.mean(dim=[2, 3]) for f in feats]
            pooled_feat = sum(gap_feat) / len(gap_feat)

            brand_probs = self.brand_head.predict(pooled_feat)  # Expected shape: [B, num_classes]

            for i, sample in enumerate(results):
                # ✅ Determine target device
                if hasattr(sample.pred_instances, 'bboxes') and sample.pred_instances.bboxes.numel() > 0:
                    device = sample.pred_instances.bboxes.device
                else:
                    device = brand_probs[i].device
                
                score_tensor = brand_probs[i].detach().to(device)

                # ✅ Correctly write brand_score to pred_instances (rather than the sample root)
                sample.pred_instances.set_field(score_tensor, name='brand_score', dtype=torch.Tensor)

        return results
