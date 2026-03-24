from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine.registry import MODELS

@MODELS.register_module()
class CustomDetDataPreprocessor(DetDataPreprocessor):
    def forward(self, data, training=False):
        """
        自定义 DataPreprocessor，兼容 MMEngine 的 tuple 输出要求。

        Args:
            data (dict): 包含 'inputs' 和 'data_samples' 字段，
                         'inputs' 是图像张量或元数据，
                         'data_samples' 是 DetDataSample 对象的列表。
            training (bool): 是否训练模式。

        Returns:
            tuple: (inputs, data_samples)
                - inputs: Tensor[B, C, H, W]，已标准化图像张量。
                - data_samples: List[DetDataSample]，每张图像的标注与 metainfo，
                                包含 brand_id、gt_instances（bbox 与 label）。
        """
        out_dict = super().forward(data, training)
        inputs = out_dict['inputs']            # 图像张量
        data_samples = out_dict['data_samples']  # 标注数据（含 brand_id）
        return inputs, data_samples
