# ========================
# Dual-Task Detector Config (with ViT+OCR+FocalLoss for SCI Q1)
# ========================
_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# ===== 数据集配置 =====
dataset_type = 'CustomCocoDataset'
data_root = '/root/autodl-tmp/coco_dataset1/'
backend_args = None

classes = [f'bolt{i+1}' for i in range(22)] + ['nameplate']

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [dict(type='RandomChoiceResize',
                  scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                          (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                          (736, 1333), (768, 1333), (800, 1333)],
                  keep_ratio=True)],
            [dict(type='RandomChoiceResize',
                  scales=[(400, 1333), (500, 1333), (600, 1333)],
                  keep_ratio=True),
             dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
             dict(type='RandomChoiceResize',
                  scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                          (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                          (736, 1333), (768, 1333), (800, 1333)],
                  keep_ratio=True)]
        ]),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'brand_id'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'brand_id'))
]

train_dataloader = dict(...)
val_dataloader = dict(...)
test_dataloader = val_dataloader

val_evaluator = [...]
test_evaluator = val_evaluator

# ===== 数据预处理器 =====
data_preprocessor = dict(
    type='CustomDetDataPreprocessor',
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    bgr_to_rgb=False,
    pad_size_divisor=32
)

# ===== 模型结构（Mamba+ViT+OCR+Focal） =====
model = dict(
    type='DualTaskDetector',
    backbone=dict(
        type='MM_mamba_vision',
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        depths=(1, 3, 8, 4),
        num_heads=(2, 4, 8, 16),
        window_size=(8, 8, 112, 56),
        dim=80,
        in_dim=32,
        mlp_ratio=4,
        drop_path_rate=0.2,
        norm_layer='ln2d',
        layer_scale=None,
        use_vit_patch_embed=True,
        use_token_mixer=True,
        use_ocr_feature=True
    ),
    neck=dict(type='FPN', in_channels=[80, 160, 320, 640], out_channels=256, num_outs=4),
    rpn_head=dict(...),
    roi_head=dict(...),
    brand_head=dict(
        type='Shared2FCBrandHead',
        in_channels=256,
        fc_out_channels=1024,
        num_classes=70,
        use_ocr_fusion=True,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, reduction='mean')
    ),
    loss_weight_brand=0.5,
    train_cfg=dict(max_epochs=12)
)

# ===== 优化器 =====
optim_wrapper = dict(...)

# ===== 日志 & 可视化 =====
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        max_keep_ckpts=3,
        save_best=['coco/bbox_mAP', 'brand_acc_top1', 'brand_acc_top5'],
        rule='greater'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    timer=dict(type='IterTimerHook')
)

log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=True
)

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ]
)

# ===== 自定义模块导入（包含 ocr_utils） =====
custom_imports = dict(
    imports=[
        'my_models.dual_task_detector',
        'my_heads.custom_cascade_roi_head',
        'my_heads.shared_2fc_brand_head',
        'my_metrics.custom_brand_evaluator',
        'my_mmdet.datasets.custom_coco_dataset',
        'my_mmdet.data_preprocessors.custom_data_preprocessor',
        'my_utils.ocr_utils'  # ✅ 加入 OCR 特征提取模块
    ],
    allow_failed_imports=False
)

randomness = dict(seed=42)
launcher = 'pytorch'
env_cfg = dict(
    dist_cfg=dict(backend='nccl'),
    cudnn_benchmark=True
)
