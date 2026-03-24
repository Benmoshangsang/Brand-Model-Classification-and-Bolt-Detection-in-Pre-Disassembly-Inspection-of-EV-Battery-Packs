import os
import torch
import json
from pycocotools.coco import COCO
from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS
from mmengine.structures import InstanceData
from mmengine.fileio import load
import numpy as np


@DATASETS.register_module()
class CustomCocoDataset(CocoDataset):
    def __init__(self,
                 ann_file,
                 data_root=None,
                 data_prefix=dict(img=''),
                 ocr_feat_path=None,
                 *args,
                 **kwargs):
        # ==== COCO ann 文件路径处理 ====
        if data_root is not None and not ann_file.startswith('/'):
            ann_file = os.path.join(data_root, ann_file)

        self._raw_coco = COCO(ann_file)
        self.ocr_feat_path = ocr_feat_path

        # ==== 加载 OCR 特征映射 ====
        self.ocr_feat_map = {}
        if ocr_feat_path and os.path.exists(ocr_feat_path):
            print(f'[CustomCocoDataset] ✅ 加载 OCR 特征: {ocr_feat_path}')
            with open(ocr_feat_path, 'r') as f:
                raw = json.load(f)
                # 假设结构: {"filename.jpg": [0.1, 0.2, ..., 0.8]}
                self.ocr_feat_map = {
                    k: torch.tensor(v, dtype=torch.float32)
                    for k, v in raw.items()
                }

        super().__init__(ann_file=ann_file,
                         data_root=data_root,
                         data_prefix=data_prefix,
                         *args, **kwargs)

    def load_annotations(self, ann_file):
        self.coco = self._raw_coco
        return super().load_annotations(ann_file)

    def get_data_info(self, idx):
        # 调用父类获取数据
        data_info = super().get_data_info(idx)

        # 获取 img_id
        img_id = data_info.get('img_id', None)
        if img_id is None:
            filename = os.path.basename(data_info['img_path'])
            img_id = next((k for k, v in self._raw_coco.imgs.items()
                           if v.get('file_name', '') == filename), None)
        if img_id is None:
            raise ValueError(f"[CustomCocoDataset] Cannot find img_id for image: {data_info['img_path']}")

        # 获取 brand_id（整图标签）
        img_info = self._raw_coco.imgs[img_id]
        brand_id = img_info.get('brand_id', -1)
        data_info['img_id'] = img_id
        data_info['brand_id'] = brand_id
        return data_info

    def prepare_data(self, idx):
        # 调用父类读取图像和 annotation
        data = super().prepare_data(idx)

        # 设置 brand_label（整图分类标签）
        brand_id = data['data_samples'].metainfo.get('brand_id', -1)
        brand_label_tensor = torch.tensor([brand_id], dtype=torch.long)
        data['data_samples'].set_metainfo(dict(brand_label=brand_label_tensor))

        # ==== 添加 OCR 特征向量 ====
        filename = data['data_samples'].metainfo.get('ori_filename', '')
        ocr_vec = self.ocr_feat_map.get(filename, None)
        if ocr_vec is None:
            ocr_vec = torch.zeros(128, dtype=torch.float32)  # 默认维度为 128
        data['ocr_vector'] = ocr_vec

        return data
