print('[DEBUG] ✅ CustomCocoDataset successfully loaded.', flush=True)  # ←模块加载验证

from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset
from mmengine.structures import InstanceData
import torch


@DATASETS.register_module()
class CustomCocoDataset(CocoDataset):
    def parse_data_info(self, raw_data_info):
        """扩展：从 image 元数据中提取 brand_id"""
        data_info = super().parse_data_info(raw_data_info)

        brand_id = raw_data_info.get('brand_id', -1)
        if not isinstance(brand_id, int) or brand_id < 0:
            brand_id = -1

        data_info['brand_id'] = brand_id
        return data_info

    def parse_ann_info(self, img_info, ann_info):
        """扩展：为每个目标实例添加 brand_labels（图像级品牌标签）"""
        print(f"[DEBUG-ENTER] parse_ann_info: img_id={img_info.get('id')}", flush=True)

        results = super().parse_ann_info(img_info, ann_info)
        instances: InstanceData = results['instances']

        brand_id = img_info.get('brand_id', -1)
        if not hasattr(instances, 'bboxes') or not isinstance(instances.bboxes, torch.Tensor):
            num_instances = 0
        else:
            num_instances = len(instances.bboxes)

        # ✅ 广播 brand_id 到每个 bbox 实例
        if num_instances > 0 and brand_id >= 0:
            brand_labels = torch.full(
                (num_instances,),
                fill_value=int(brand_id),
                dtype=torch.long,
                device=instances.bboxes.device
            )
        else:
            brand_labels = torch.full(
                (num_instances,),
                fill_value=-1,
                dtype=torch.long,
                device=instances.bboxes.device if hasattr(instances, 'bboxes') else 'cpu'
            )

        instances.set_field('brand_labels', brand_labels)
        results['instances'] = instances

        print(f"[DEBUG] img_id={img_info.get('id')} | brand_id={brand_id} | num_boxes={num_instances} | brand_labels: {brand_labels.tolist()}", flush=True)

        return results
