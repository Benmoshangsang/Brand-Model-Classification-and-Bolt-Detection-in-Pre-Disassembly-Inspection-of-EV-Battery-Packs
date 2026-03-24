import os
import torch
from pycocotools.coco import COCO
from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS
from mmengine.structures import InstanceData


@DATASETS.register_module()
class CustomCocoDataset(CocoDataset):
    def __init__(self, ann_file, data_root=None, data_prefix=dict(img=''), *args, **kwargs):
        # 预处理 ann_file 路径
        if data_root is not None and not ann_file.startswith('/'):
            ann_file = os.path.join(data_root, ann_file)

        # 加载原始 COCO 数据结构
        self._raw_coco = COCO(ann_file)

        # 初始化父类（会自动调用 load_annotations）
        super().__init__(ann_file=ann_file, data_root=data_root,
                         data_prefix=data_prefix, *args, **kwargs)

    def load_annotations(self, ann_file):
        # 绑定 COCO API 对象
        self.coco = self._raw_coco
        return super().load_annotations(ann_file)

    def get_data_info(self, idx):
        # 获取基础信息
        data_info = super().get_data_info(idx)

        # 获取 img_id
        img_id = data_info.get('img_id', None)
        if img_id is None:
            filename = os.path.basename(data_info['img_path'])
            img_id = next((k for k, v in self._raw_coco.imgs.items()
                           if v.get('file_name', '') == filename), None)

        if img_id is None:
            raise ValueError(f"[CustomCocoDataset] Cannot find img_id for image: {data_info['img_path']}")

        # 获取 brand_id
        img_info = self._raw_coco.imgs[img_id]
        brand_id = img_info.get('brand_id', -1)

        # 写入 metainfo
        data_info['img_id'] = img_id
        data_info['brand_id'] = brand_id
        return data_info

    def prepare_data(self, idx):
        # 加载图像 + annotation
        data = super().prepare_data(idx)

        # 获取 brand_id 并转换为 Tensor
        brand_id = data['data_samples'].metainfo.get('brand_id', -1)
        brand_label_tensor = torch.tensor([brand_id], dtype=torch.long)

        # ✅ 将 brand_label 写入 metainfo（图像级别），而不是写入 gt_instances（目标级别）
        data['data_samples'].set_metainfo(dict(brand_label=brand_label_tensor))

        return data
