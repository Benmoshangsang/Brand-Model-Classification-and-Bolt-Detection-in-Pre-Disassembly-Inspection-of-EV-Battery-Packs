from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine.registry import MODELS

@MODELS.register_module()
class CustomDetDataPreprocessor(DetDataPreprocessor):
    """Custom DataPreprocessor compatible with MMEngine's tuple output requirements."""

    def forward(self, data, training=False):
        """
        Custom forward pass for data preprocessing.

        Args:
            data (dict): Dictionary containing 'inputs' and 'data_samples'.
                         'inputs' is the image tensor or metadata.
                         'data_samples' is a list of DetDataSample objects.
            training (bool): Whether the model is in training mode.

        Returns:
            tuple: (inputs, data_samples)
                - inputs: Tensor[B, C, H, W], normalized image tensors.
                - data_samples: List[DetDataSample], containing annotations and metainfo 
                                for each image (including brand_id, gt_instances, etc.).
        """
        # Process the data using the base class DetDataPreprocessor
        out_dict = super().forward(data, training)
        
        # Extract inputs and data_samples from the output dictionary
        inputs = out_dict['inputs']              # Image tensors
        data_samples = out_dict['data_samples']  # Annotation data (includes brand_id)
        
        # Return as a tuple to comply with specific engine requirements
        return inputs, data_samples
