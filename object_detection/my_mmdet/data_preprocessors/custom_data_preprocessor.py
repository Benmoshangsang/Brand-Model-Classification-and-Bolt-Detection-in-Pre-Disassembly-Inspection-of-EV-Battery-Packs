from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine.registry import MODELS

@MODELS.register_module()
class CustomDetDataPreprocessor(DetDataPreprocessor):
    def forward(self, data, training=False):
        """
        Custom DataPreprocessor compatible with MMEngine's tuple output requirement.

        Args:
            data (dict): Contains the 'inputs' and 'data_samples' fields,
                         where 'inputs' is the image tensor or metadata,
                         and 'data_samples' is a list of DetDataSample objects.
            training (bool): Whether the model is in training mode.

        Returns:
            tuple: (inputs, data_samples)
                - inputs: Tensor[B, C, H, W], the normalized image tensor.
                - data_samples: List[DetDataSample], annotations and metainfo for each image,
                                including brand_id and gt_instances (bbox and label).
        """
        out_dict = super().forward(data, training)
        inputs = out_dict['inputs']               # Image tensor
        data_samples = out_dict['data_samples']   # Annotation data (including brand_id)
        return inputs, data_samples
