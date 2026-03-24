from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine.registry import MODELS
import torch

@MODELS.register_module()
class CustomDetDataPreprocessor(DetDataPreprocessor):
    def forward(self, data, training=False):
        """
        自定义 DataPreprocessor，兼容 MMEngine 的 tuple 输出要求。
        并添加 OCR 向量支持，供后续模型融合使用。

        Args:
            data (dict): 包含 'inputs' 和 'data_samples' 字段，
                         'inputs' 是图像张量，
                         'data_samples' 是 DetDataSample 对象的列表。
            training (bool): 是否训练模式。

        Returns:
            tuple: (inputs, data_samples)
                - inputs: Tensor[B, C, H, W]，已标准化图像张量。
                - data_samples: List[DetDataSample]，每张图像的标注与 metainfo。
            or dict: {
                'inputs': Tensor,
                'data_samples': List[DetDataSample],
                'ocr_vector': Tensor[B, D]
            }（包含额外 OCR 特征）
        """
        out_dict = super().forward(data, training)
        inputs = out_dict['inputs']                # 图像张量 [B, C, H, W]
        data_samples = out_dict['data_samples']    # List[DetDataSample]

        # === OCR 特征提取 ===
        if isinstance(data, dict) and 'ocr_vector' in data:
            # 单图推理（非batch）
            ocr_vec = data['ocr_vector']
            if isinstance(ocr_vec, torch.Tensor):
                ocr_vector = ocr_vec.unsqueeze(0)  # [1, D]
            else:
                ocr_vector = torch.stack(ocr_vec)  # [B, D]
        elif isinstance(data, list) or isinstance(data, tuple):
            # 批量模式
            ocr_list = []
            for item in data:
                vec = item.get('ocr_vector', None)
                if vec is None:
                    vec = torch.zeros(128)
                ocr_list.append(vec)
            ocr_vector = torch.stack(ocr_list)  # [B, 128]
        else:
            ocr_vector = torch.zeros((inputs.shape[0], 128))

        return {
            'inputs': inputs,
            'data_samples': data_samples,
            'ocr_vector': ocr_vector.to(inputs.device)
        }
