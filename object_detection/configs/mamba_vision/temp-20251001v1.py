# ========================
# Dual-Task Detector Config (with brand top1/top5 + macro P/R/F1, det extra metrics)
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

# ===== 动态 batch size：每卡 4 张 =====
from mmengine.dist import get_world_size
_gpu_count = get_world_size()
_batch_size_per_gpu = 3
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

val_dataloader = dict(
    batch_size=3,
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

# ====== 评估器：两组各5项 ======
# 检测侧（COCO mAP@[.50:.95]、AP50、AR@K）+ 额外两项（匹配框分类准确率、TP-mean IoU）
# 品牌分类侧（Top-1/Top-5 + Macro P/R/F1 —— 已在 CustomBrandEvaluator 内实现）
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
    ),
    dict(
        type='CustomDetExtraEvaluator',
        iou_thrs=(0.50,),          # 如需 0.50/0.75 同时报：改为 (0.50, 0.75)
        report_cls_acc=True,       # 输出 det/matched_cls_acc@0.50
        report_tp_mean_iou=True    # 输出 det/tp_mean_iou@0.50
    ),
]

test_evaluator = val_evaluator

data_preprocessor = dict(
    type='CustomDetDataPreprocessor',
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    bgr_to_rgb=False,
    pad_size_divisor=32
)

# ========================
# 模型：CSDS-Backbone + FPN + SFG-RL Head（检测）+ SCM-Brand Head（整图品牌）
# ========================
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
        # === SFG-RL Head: 三项可开关机制 ===
        use_state_condition=True,      # state→FiLM（ROI 级）
        use_four_dir_pool=True,        # 四向 ROI 池化 + 门控
        use_relation_refine=True,      # 预测阶段KNN关系重标定
        # 可选：外部传 state（若 backbone/数据管道提供）
        use_external_state=False,
        # 隐层/关系参数（可按需调优/消融）
        state_in_channels=None,        # None 则自动从 x[-1] 通道数推断
        state_hidden=256,
        fourdir_hidden=256,
        rel_k=6,
        rel_alpha=0.5,
        # === 原级联 ROIHead 配置 ===
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
            num_classes=14,
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
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))),
    # === SCM-Brand Head（整图品牌分类） ===
    brand_head=dict(
        type='Shared2FCBrandHead',
        in_channels=640,                 # 与 backbone 最后一级通道对齐（Stage4: 640）
        fc_out_channels=1024,
        num_classes=7,
        # —— 三项可开关机制 ——
        use_state_condition=True,        # state→FiLM（向量级）
        state_dim=640,                   # 若未显式提供 state，则可设为与 in_channels 一致
        state_hidden=256,
        use_four_dir_squeeze=True,       # 四向通道挤压（需 feat_map；若未传则跳过）
        fourdir_hidden=256,
        film_from_fourdir=True,          # 由四向挤压再产生一组 FiLM
        use_prototype_branch=True,       # 原型辅助 logits
        proto_dim=None,                  # None→使用 fc_out_channels 维度
        proto_tau=10.0,
        proto_lambda=0.3,
        # loss
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            reduction='mean')),
    # 多任务权重
    loss_weight_brand=0.5,
    train_cfg=dict(max_epochs=12)
)

# ========================
# 训练与优化
# ========================
max_epochs = 12
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[8, 11], gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05))

work_dir = './work_dirs/cascade_mask_rcnn_mamba_dualtask'

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
        'my_heads.custom_cascade_roi_head',   # ← 已改造成 SFG-RL Head
        'my_heads.shared_2fc_brand_head',     # ← 已改造成 SCM-Brand Head
        'my_metrics.custom_brand_evaluator',
        'my_metrics.custom_det_extra_evaluator',
        'my_mmdet.datasets.custom_coco_dataset',
        'my_mmdet.data_preprocessors.custom_data_preprocessor'
    ],
    allow_failed_imports=False
)

randomness = dict(seed=42)
