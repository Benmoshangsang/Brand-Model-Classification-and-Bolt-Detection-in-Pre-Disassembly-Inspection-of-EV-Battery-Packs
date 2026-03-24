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
dataset_type = 'CustomCocoDataset'                                  # 数据集类型
data_root = '/root/autodl-tmp/coco_dataset2/'                       # 数据集根目录
backend_args = None                                                 # 后端参数，通常用于云存储

classes = [f'bolt{i+1}' for i in range(14)]                          # 定义类别名称列表

# ✅ 可选优化：略微降低输入图像分辨率，以进一步加速训练
train_pipeline = [                                                  # 训练数据预处理流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),      # 从文件加载图像
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),  # 加载标注，只关心边界框
    dict(type='RandomFlip', prob=0.5),                              # 随机水平翻转
    dict(
        type='RandomChoice',                                        # 随机选择一种变换
        transforms=[
            [dict(type='RandomChoiceResize',                        # 随机选择一种尺寸进行缩放
                  scales=[
                      (480, 900), (512, 900), (544, 900),
                      (576, 900), (608, 900), (640, 900)
                  ],
                  keep_ratio=True)]                                 # 保持图像宽高比
        ]),
    dict(type='PackDetInputs',                                      # 将数据打包成 MMDetection 需要的格式
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'brand_id')) # 需要保留的元信息
]

test_pipeline = [                                                   # 测试数据预处理流水线
    dict(type='LoadImageFromFile', backend_args=backend_args),      # 从文件加载图像
    dict(type='Resize', scale=(900, 608), keep_ratio=True),         # 固定尺寸缩放
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),  # 加载标注
    dict(type='PackDetInputs',                                      # 打包数据
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'brand_id')) # 需要保留的元信息
]

# ===== 动态 batch size =====
from mmengine.dist import get_world_size                            # 导入获取分布式世界大小的函数
_gpu_count = get_world_size()                                       # 获取当前使用的 GPU 数量
# ✅【速度优化】将默认物理批次大小从3提升到4。在RTX 4090上应有足够显存。
# 更大的物理批次能更充分利用GPU，显著减少训练时间。
# 如果发生OOM，可以调回3，或通过命令行 --bs 参数覆盖。
_batch_size_per_gpu = 4                                             # 设置每个 GPU 的 batch size
_num_workers_per_gpu = 4                                            # 设置每个 GPU 的数据加载进程数

train_dataloader = dict(                                            # 训练数据加载器配置
    batch_size=_batch_size_per_gpu,                                 # 每 GPU 的 batch size
    num_workers=_num_workers_per_gpu,                               # 每 GPU 的 worker 数量
    persistent_workers=True,                                        # 保持 worker 进程，加速数据加载
    sampler=dict(type='DefaultSampler', shuffle=True),              # 默认采样器，打乱数据
    batch_sampler=dict(type='AspectRatioBatchSampler'),             # 根据图像宽高比采样，节省显存
    collate_fn=dict(type='pseudo_collate'),                         # 自定义的数据整理函数
    dataset=dict(                                                   # 数据集配置
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',                          # 训练集标注文件
        data_prefix=dict(img='images/train'),                       # 训练集图片路径前缀
        metainfo=dict(classes=classes),                             # 类别元信息
        pipeline=train_pipeline,                                    # 使用训练数据流水线
        backend_args=backend_args))

val_dataloader = dict(                                              # 验证数据加载器配置
    batch_size=4,                                                   # 验证时 batch size (可适当增大)
    num_workers=2,                                                  # 验证时 worker 数量
    persistent_workers=True,                                        # 保持 worker 进程
    drop_last=False,                                                # 不丢弃最后一个不完整的 batch
    sampler=dict(type='DefaultSampler', shuffle=False),             # 默认采样器，不打乱
    collate_fn=dict(type='pseudo_collate'),                         # 自定义数据整理函数
    dataset=dict(                                                   # 验证数据集配置
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',                            # 验证集标注文件
        data_prefix=dict(img='images/val'),                         # 验证集图片路径前缀
        metainfo=dict(classes=classes),
        test_mode=True,                                             # 设置为测试模式
        pipeline=test_pipeline,                                     # 使用测试数据流水线
        backend_args=backend_args))

test_dataloader = val_dataloader                                    # 测试数据加载器配置与验证集相同

# ====== 评估器 ======
val_evaluator = [                                                   # 验证评估器列表
    dict(
        type='CocoMetric',                                          # COCO 指标评估器
        ann_file=data_root + 'annotations/val.json',                # 验证集标注文件
        metric=['bbox'],                                            # 评估边界框
        format_only=False,
        backend_args=backend_args),
    dict(
        type='CustomBrandEvaluator',                                # 自定义品牌评估器
        topk=(1, 5),                                                # 计算 top-1 和 top-5 准确率
        by_epoch=True                                               # 按 epoch 评估
    ),
    dict(
        type='CustomDetExtraEvaluator',                             # 自定义检测附加指标评估器
        iou_thrs=(0.50,),                                           # IoU 阈值
        report_cls_acc=True,                                        # 报告分类准确率
        report_tp_mean_iou=True                                     # 报告 True Positive 的平均 IoU
    ),
]

test_evaluator = val_evaluator                                      # 测试评估器与验证评估器相同

data_preprocessor = dict(                                           # 数据预处理器配置
    type='CustomDetDataPreprocessor',                               # 自定义数据预处理器
    mean=[103.53, 116.28, 123.675],                                 # 图像归一化均值 (ImageNet BGR)
    std=[57.375, 57.12, 58.395],                                    # 图像归一化标准差 (ImageNet BGR)
    bgr_to_rgb=False,                                               # 输入是 BGR，不需要转换
    pad_size_divisor=32                                             # 填充图像尺寸为 32 的倍数
)

# ========================
# 模型配置
# ========================
model = dict(
    type='DualTaskDetector',                                        # 模型类型：双任务检测器
    backbone=dict(
        _delete_=True,                                              # ✅ 【关键修复】删除所有从 base config 继承的无关配置项 (如 resnet50 的配置)
        type='MM_mamba_vision',                                     # Backbone 类型
        out_indices=(0, 1, 2, 3),                                   # 输出特征图的索引
        pretrained=None,                                            # 不使用预训练权重
        depths=(1, 3, 8, 4),                                        # 每个 stage 的层数
        num_heads=(2, 4, 8, 16),                                    # 每个 stage 的头数
        window_size=(8, 8, 8, 8),                                   # ✅ 【关键修复 & 加速】修正为合理的窗口大小
        dim=80,                                                     # 基础维度
        in_dim=32,                                                  # 输入维度
        mlp_ratio=4,                                                # MLP 层的扩展比例
        drop_path_rate=0.2,                                         # DropPath 随机失活率
        norm_layer='ln2d',                                          # 归一化层类型
        layer_scale=None,
        use_checkpoint=False),                                      # ✅ 【核心加速】禁用梯度检查点，用显存换取速度
    neck=dict(
        type='FPN',                                                 # Neck 类型：特征金字塔网络
        in_channels=[80, 160, 320, 640],                            # 输入通道数，与 backbone 输出对应
        out_channels=256,                                           # 输出通道数
        num_outs=4),                                                # 输出特征图数量
    rpn_head=dict(
        type='RPNHead',                                             # RPN 头
        in_channels=256,                                            # 输入通道数
        feat_channels=256,                                          # 特征通道数
        anchor_generator=dict(                                      # Anchor 生成器
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32]),
        bbox_coder=dict(                                            # Bbox 编码器
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), # 分类损失
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),          # 回归损失
    roi_head=dict(
        type='CustomCascadeRoIHead',                                # RoI 头类型：自定义级联 RoI 头
        use_state_condition=False,                                   # 启用状态条件调制
        use_four_dir_pool=False,                                     # 启用四方向池化
        use_relation_refine=True,                                   # 启用关系精炼
        use_external_state=False,                                   # 不使用外部状态
        state_in_channels=256,                                      # 明确指定状态输入维度
        roi_feat_channels=256,                                      # 明确指定 ROI 特征维度
        state_hidden=256,                                           # 状态投影器隐藏层维度
        fourdir_hidden=256,                                         # 四方向池化隐藏层维度
        rel_k=6,                                                    # 关系精炼的 K 值
        rel_alpha=0.5,                                              # 关系精炼的融合权重
        num_stages=3,                                               # 级联数量
        stage_loss_weights=[1, 0.5, 0.25],                          # 每个 stage 的损失权重
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',                              # RoI 特征提取器
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0), # RoIAlign 层
            out_channels=256,                                       # 输出通道数，应与 roi_feat_channels 一致
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[dict(                                            # Bbox 头配置 (3个 stage)
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
            # ✅ 【速度优化】将 SyncBN 替换为 GN (Group Normalization)，以减少多GPU通信开销
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
                        num=256,                                    # 从512降低到256
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
                        num=256,                                    # 从512降低到256
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
                        num=256,                                    # 从512降低到256
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
        in_channels=256,                                            # 与 FPN 输出对齐
        fc_out_channels=1024,
        num_classes=7,
        use_state_condition=False,
        state_dim=256,                                              # 与 FPN 输出对齐
        state_hidden=256,
        use_four_dir_squeeze=False,
        fourdir_hidden=256,
        film_from_fourdir=True,
        use_prototype_branch=False,
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
            nms_pre=1000,                                           # 从2000降低到1000
            max_per_img=1000,                                       # 从2000降低到1000
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
        # ✅【指标修复】添加此标志，让 DualTaskDetector 在 predict/val 阶段输出 brand_score,
        # 这样 CustomBrandEvaluator 和 CustomDetExtraEvaluator 才能接收到数据并正常工作。
        output_brand_score=True)
)

# ========================
# 训练与优化
# ========================
max_epochs = 20                                                      # 【修改点】最大训练轮数从12改为3，用于快速测试
train_cfg = dict(
    type='EpochBasedTrainLoop',                                     # 基于 Epoch 的训练循环
    max_epochs=max_epochs,
    val_interval=1                                                  # 每 1 个 epoch 验证一次
)
val_cfg = dict(type='ValLoop')                                      # 验证循环配置
test_cfg = dict(type='TestLoop')                                    # 测试循环配置

param_scheduler = [                                                 # 学习率调度器
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500), # 线性热身
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[8, 11], gamma=0.1) # 多步长衰减
]

optim_wrapper = dict(
    type='AmpOptimWrapper',                                         # 启用混合精度训练 (AMP)
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}),  # 对归一化层不使用权重衰减
    clip_grad=dict(max_norm=1.0, norm_type=2),                      # 梯度裁剪
    optimizer=dict(
        _delete_=True,                                              # 删除继承的 optimizer 配置
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05
    )
)

work_dir = './work_dirs/cascade_mask_rcnn_mamba_vision_tiny_3x_coco128' # 工作目录

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',                               # 分布式数据并行封装器
    find_unused_parameters=True                                     # 允许存在未使用的参数
)

default_hooks = dict(                                               # 默认钩子配置
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,                                                 # 验证完每个epoch都保存，以便快速拿到模型
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

log_processor = dict(                                               # 日志处理器配置
    type='LogProcessor',
    window_size=50,
    by_epoch=True
)

visualizer = dict(                                                  # 可视化器配置
    type='DetLocalVisualizer',
    name='visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ]
)

env_cfg = dict(
    cudnn_benchmark=True,                                           # 启用 cudnn benchmark 加速
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),      # 多进程配置
    dist_cfg=dict(backend='nccl')                                   # 分布式后端
)
launcher = 'pytorch'                                                # 启动器类型

custom_imports = dict(                                              # 自定义导入模块
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

randomness = dict(seed=42)                                          # 随机种子
resume = False                                                      # 是否从断点恢复
load_from = None                                                    # 从预训练模型加载
log_level = 'INFO'                                                  # 日志级别
auto_scale_lr = dict(base_batch_size=16, enable=False)              # 自动缩放学习率（禁用）