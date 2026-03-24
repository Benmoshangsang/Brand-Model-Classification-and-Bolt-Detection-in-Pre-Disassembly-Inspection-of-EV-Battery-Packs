import torch
from torch import Tensor
from typing import List, Tuple
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

        losses = super().loss(x, valid_rpn_results, valid_data_samples, **kwargs)
        return losses

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: List[InstanceData],
                batch_data_samples: List[DetDataSample],
                rescale: bool = False) -> List[DetDataSample]:

        # ✅ 直接返回父类结果，不污染结构
        return super().predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)
