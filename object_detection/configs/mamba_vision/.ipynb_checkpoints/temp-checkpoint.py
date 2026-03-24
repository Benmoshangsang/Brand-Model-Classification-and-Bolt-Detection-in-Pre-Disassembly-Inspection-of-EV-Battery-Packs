# ========================
# Dual-Task Detector Config (with brand top1/top5 + macro P/R/F1, det extra metrics)
# ✅ 已修复：删除了 AdamW 中的 momentum 参数
# ✅ 已修复：添加了 roi_feat_channels 参数以支持立即创建 FiLM 模块
# ✅ 已修复：移除了 env_cfg.dist_cfg 中错误的 find_unused_parameters 参数
# ✅ 已优化：统一了 model.train_cfg.rcnn 中的 sampler 数量
# ✅【核心修复】修正了 backbone 配置，移除了无效的继承参数
# ✅【核心修复 & 加速】修正了不合理的超大 window_size
# ✅【核心加速】禁用了梯度检查点，用显存换取速度
# ✅【速度优化】将 SyncBN 替换为 GroupNorm，减少多卡通信开销
# ✅【速度与指标修复】增大默认物理批次大小，并修复自定义评估器在验证时不工作的问题
# ========================
_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# ===== 数据集配置 =====
dataset_type = 'CustomCocoDataset'
data_root = '/root/autodl-tmp/coco_dataset2/'
backend_args = None

classes = [f'bolt{i+1}' for i in range(14)]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [dict(type='RandomChoiceResize',
                  scales=[
                      (480, 900), (512, 900), (544, 900),
                      (576, 900), (608, 900), (640, 900)
                  ],
                  keep_ratio=True)]
        ]),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'brand_id'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(900, 608), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'brand_id'))
]

# ===== 动态 batch size =====
from mmengine.dist import get_world_size
_gpu_count = get_world_size()
_batch_size_per_gpu = 4
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
    batch_size=4,
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

# ====== 评估器 ======
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
        iou_thrs=(0.50,),
        report_cls_acc=True,
        report_tp_mean_iou=True
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
# 模型配置
# ========================
model = dict(
    type='DualTaskDetector',
    backbone=dict(
        _delete_=True,
        type='MM_mamba_vision',
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        depths=(1, 3, 8, 4),
        num_heads=(2, 4, 8, 16),
        window_size=(8, 8, 8, 8),
        dim=80,
        in_dim=32,
        mlp_ratio=4,
        drop_path_rate=0.2,
        norm_layer='ln2d',
        layer_scale=None,
        use_checkpoint=False,

        # ===== 新增：CSDS-Backbone 消融开关（默认开启以保持行为不变） =====
        enable_pcs=True,          # PCS-Scan（四向扫描 Mamba）
        enable_sac=True,          # SAC：状态感知交叉注意 + 写回门控
        enable_sl_bridge=True,    # SL-Bridge：跨 Stage 记忆调制
    ),
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
        use_state_condition=True,
        use_four_dir_pool=True,
        use_relation_refine=True,
        use_external_state=False,
        state_in_channels=256,
        roi_feat_channels=256,
        state_hidden=256,
        fourdir_hidden=256,
        rel_k=6,
        rel_alpha=0.5,
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
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
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0))] * 3,
        train_cfg=dict(
            rcnn=[
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=False,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=256,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    pos_weight=-1,
                    debug=False),
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.6,
                        neg_iou_thr=0.6,
                        min_pos_iou=0.6,
                        match_low_quality=False,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=256,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    pos_weight=-1,
                    debug=False),
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.7,
                        min_pos_iou=0.7,
                        match_low_quality=False,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=256,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    pos_weight=-1,
                    debug=False)
            ]),
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
        use_state_condition=True,
        state_dim=256,
        state_hidden=256,
        use_four_dir_squeeze=True,
        fourdir_hidden=256,
        film_from_fourdir=True,
        use_prototype_branch=True,
        proto_dim=None,
        proto_tau=10.0,
        proto_lambda=0.3,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            reduction='mean')),
    loss_weight_brand=0.5,
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100),
        output_brand_score=True)
)

# ========================
# 训练与优化
# ========================
max_epochs = 20
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[8, 11], gamma=0.1)
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05
    )
)

work_dir = './work_dirs/cascade_mask_rcnn_mamba_vision_tiny_3x_coco128'

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='auto',
        rule='greater'
    ),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    timer=dict(type='IterTimerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    visualization=dict(type='DetVisualizationHook')
)

log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=True
)

visualizer = dict(
    type='DetLocalVisualizer',
    name='visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ]
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)
launcher = 'pytorch'

custom_imports = dict(
    imports=[
        'my_models.dual_task_detector',
        'my_heads.custom_cascade_roi_head',
        'my_heads.shared_2fc_brand_head',
        'my_metrics.custom_brand_evaluator',
        'my_metrics.custom_det_extra_evaluator',
        'my_mmdet.datasets.custom_coco_dataset',
        'my_mmdet.data_preprocessors.custom_data_preprocessor'
    ],
    allow_failed_imports=False
)

randomness = dict(seed=42)
resume = False
load_from = None
log_level = 'INFO'
auto_scale_lr = dict(base_batch_size=16, enable=False)
