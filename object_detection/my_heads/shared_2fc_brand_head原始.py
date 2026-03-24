import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict
from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class Shared2FCBrandHead(BaseModule):
    def __init__(self,
                 in_channels: int = 384,  # 根据 Mamba 最后一层输出设置，通常来自 flatten 后维度
                 fc_out_channels: int = 1024,
                 num_classes: int = 7,  # 品牌型号数
                 loss_cls: dict = dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)

        self.fc1 = nn.Linear(in_channels, fc_out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc_out_channels, fc_out_channels)
        self.fc_cls = nn.Linear(fc_out_channels, num_classes)

        self.loss_cls = MODELS.build(loss_cls)

    def forward(self, x: Tensor, return_logits: bool = True) -> Tensor:
        """
        前向传播，返回 logits 或 softmax 概率。
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        cls_score = self.fc_cls(x)
        return cls_score if return_logits else cls_score.softmax(dim=-1)

    def loss(self,
             cls_score: Tensor,
             labels: Tensor,
             reduction_override: Optional[str] = None) -> Dict[str, Tensor]:
        """
        计算整图分类损失。
        """
        if cls_score.numel() == 0:
            return {'loss_brand_cls': cls_score.new_tensor(0.)}

        # 强制转换标签类型
        if not isinstance(labels, Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=cls_score.device)
        elif labels.dtype != torch.long:
            labels = labels.to(dtype=torch.long)

        loss_cls = self.loss_cls(cls_score, labels, reduction_override=reduction_override)
        return {'loss_brand_cls': loss_cls}

    def predict(self, x: Tensor) -> Tensor:
        """
        预测整图所属品牌（softmax 概率）。
        """
        return self.forward(x, return_logits=False)
