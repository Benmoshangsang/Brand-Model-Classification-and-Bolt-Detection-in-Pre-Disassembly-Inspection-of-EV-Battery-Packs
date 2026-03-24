# ========================
# Dual-Task Detector Config (stable, 14 classes, runtime bbox filtering)
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

# 14 类螺栓
classes = [f'bolt{i+1}' for i in range(14)]

# —— 数据管道：仅在运行时过滤，不修改原始标注文件 —— #
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=False
    ),
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
    # 运行时过滤：丢弃过小框；若图片无有效框则剔除
    dict(
        type='FilterAnnotations',
        min_gt_bbox_wh=(4, 4),   # ← 从 (2,2) 提高到 (4,4)
        keep_empty=False
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'brand_id')
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'brand_id')
    )
]

# ===== 动态 batch size：每卡 2 张 =====
from mmengine.dist import get_world_size
_gpu_count = get_world_size()
_batch_size_per_gpu = 2
_num_workers_per_gpu = 2

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

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
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
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/val.json',
        metric=['bbox'],
        format_only=False,
        backend_args=backend_args),
    dict(
        type='CustomBrandEvaluator',
        topk=(1, 5),
        by_epoch=True
    )
]
test_evaluator = val_evaluator

data_preprocessor = dict(
    type='CustomDetDataPreprocessor',
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    bgr_to_rgb=False,
    pad_size_divisor=32
)

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
            target_stds=[1.0, 1.0, 1.0, 1.0]  # ← 更稳
        ),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    roi_head=dict(
        type='CustomCascadeRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(  # stage 1
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=14,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=False,   # ← 先关
                norm_cfg=None,            # ← 去掉 BN
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            ),
            dict(  # stage 2
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=14,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=False,
                norm_cfg=None,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            ),
            dict(  # stage 3
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=14,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=False,
                norm_cfg=None,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            )
        ],
        stage_loss_weights=[1.0, 0.5, 0.25],
        test_cfg=dict(
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))),
    brand_head=dict(
        type='Shared2FCBrandHead',
        in_channels=256,
        fc_out_channels=1024,
        num_classes=7,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            reduction='mean')),
    loss_weight_brand=0.2,   # ← 先降权重，检测先学稳
    train_cfg=dict(max_epochs=12)
)

max_epochs = 12
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1500),  # ← warmup 延长
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[8, 11], gamma=0.1)
]

# ==== 纯 FP32 优化器 + 更稳的梯度裁剪 ====
optim_wrapper = dict(
    type='OptimWrapper',                     # ← 关闭 AMP
    clip_grad=dict(max_norm=0.5, norm_type=2),
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,                           # ← 更稳的起步 LR
        betas=(0.9, 0.999),
        weight_decay=0.05)
)

work_dir = './work_dirs/cascade_mask_rcnn_mamba_dualtask_fp32_stable'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        max_keep_ckpts=3,
        save_best='auto',
        rule='greater'
    ),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    timer=dict(type='IterTimerHook')
)

# === 只要出现 NaN / Inf 就立刻停止训练 ===
custom_hooks = [
    dict(type='CheckInvalidLossHook', interval=1)  # 每 iter 检测，异常即抛错中止
]

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
