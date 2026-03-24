# ========================
# Dual-Task Detector Config (stable: coco save_best on, brand eval restored)
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

# 修正为 14 类（与 bbox heads 对齐）
classes = [f'bolt{i+1}' for i in range(14)]

# 暂时移除 RandomCrop 以验证是否能跑通
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

# ===== 动态 batch size：每卡 6 张 =====
from mmengine.dist import get_world_size
_gpu_count = get_world_size()
_batch_size_per_gpu = 6
_num_workers_per_gpu = 4

train_dataloader = dict(
    batch_size=_batch_size_per_gpu,
    num_workers=_num_workers_per_gpu,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    collate_fn=dict(type='pseudo_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train'),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline,
        backend_args=backend_args))

# —— 验证/测试：更稳的设置 —— #
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,   # 稳定起见关闭复用
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='pseudo_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val'),
        metainfo=dict(classes=classes),
        test_mode=True,
        filter_cfg=None,         # 验证/测试阶段不做过滤，避免被筛成 0
        pipeline=test_pipeline,
        backend_args=backend_args)
)

test_dataloader = val_dataloader

# —— 评估器：恢复 COCO + Brand，save_best 仅看 COCO —— #
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/val.json',
        metric=['bbox'],
        format_only=False,
        collect_device='cpu',
        backend_args=backend_args
    ),
    dict(
        type='CustomBrandEvaluator',
        topk=(1, 5),
        by_epoch=True
    )
]

test_evaluator = val_evaluator

# ===== 预处理器 =====
data_preprocessor = dict(
    type='CustomDetDataPreprocessor',
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    bgr_to_rgb=False,
    pad_size_divisor=32
)

# ===== 模型 =====
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
        layer_scale=None),
    neck=dict(
        type='FPN',
        in_channels=[80, 160, 320, 640],
        out_channels=256,
        num_outs=4),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    roi_head=dict(
        type='CustomCascadeRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[dict(
            type='ConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=14,  # 与 classes 对齐
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0))] * 3,
        test_cfg=dict(
            rcnn=dict(
                score_thr=0.05,   # 已恢复为常规阈值
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))),
    # —— brand 头参与训练；确保 num_classes=你的真实品牌类数（这里示例为 7） —— #
    brand_head=dict(
        type='Shared2FCBrandHead',
        in_channels=256,
        fc_out_channels=1024,
        num_classes=7,  # <—— 如与你数据不一致，请改为真实数
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            reduction='mean')),
    loss_weight_brand=0.5,
    train_cfg=dict(max_epochs=12)
)

max_epochs = 12
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[8, 11], gamma=0.1)
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0004,
        betas=(0.9, 0.999),
        weight_decay=0.05))

work_dir = './work_dirs/cascade_mask_rcnn_mamba_dualtask'

# —— save_best 恢复，显式指定 coco/bbox_mAP —— #
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',  # 显式指定，避免 auto 误判
        rule='greater'
    ),
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

env_cfg = dict(
    dist_cfg=dict(backend='nccl'),
    cudnn_benchmark=True
)
launcher = 'pytorch'

custom_imports = dict(
    imports=[
        'my_models.dual_task_detector',
        'my_heads.custom_cascade_roi_head',
        'my_heads.shared_2fc_brand_head',
        'my_metrics.custom_brand_evaluator',
        'my_mmdet.datasets.custom_coco_dataset',
        'my_mmdet.data_preprocessors.custom_data_preprocessor'
    ],
    allow_failed_imports=False
)

randomness = dict(seed=42)
