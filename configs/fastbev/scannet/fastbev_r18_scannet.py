# -*- coding: utf-8 -*-
"""
FastBEV for ScanNet Indoor 3D Detection

适配说明:
- 室内场景: 10m x 10m x 3m (vs nuScenes 100m x 100m x 6m)
- 18 类检测 (vs nuScenes 10 类)
- 无时序帧 (单场景多视角)
- 无 BEV 分割
- 无速度预测 (box 7维 vs nuScenes 9维)
"""

# ======================== 模型配置 ========================
model = dict(
    type='FastBEV',
    style='v1',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        style='pytorch'
    ),
    neck=dict(
        type='FPN',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        in_channels=[64, 128, 256, 512],
        out_channels=64,
        num_outs=4),
    neck_fuse=dict(in_channels=[256], out_channels=[64]),
    neck_3d=dict(
        type='M2BevNeck',
        in_channels=64 * 4,  # 64 channels * n_times(1)
        out_channels=192,
        num_layers=2,
        stride=1,  # BEV 不下采样，保持 160x160 分辨率
        is_transpose=False,
        fuse=None,  # 无时序帧，不需要 fuse 降维
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    seg_head=None,  # ScanNet 无 BEV 分割
    bbox_head=dict(
        type='FreeAnchor3DHead',
        is_transpose=True,
        num_classes=18,  # ScanNet 18 类
        in_channels=192,
        feat_channels=192,
        num_convs=0,
        use_direction_classifier=True,
        pre_anchor_topk=25,
        bbox_thr=0.5,
        gamma=2.0,
        alpha=0.5,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            # 室内场景: Z 固定在 0.8m (GT z 均值)
            ranges=[[-4, -4, 0.8, 4, 4, 0.8]],
            sizes=[
                # 基于 GT 分布设计的 anchor 尺寸 (w, l, h)
                [0.35, 0.50, 0.40],  # 垃圾桶 (w=0.37, l=0.50, h=0.42)
                [0.65, 0.45, 0.70],  # 桌子 (w=0.65, l=0.45, h=0.69)
                [0.83, 0.74, 0.76],  # 椅子 (w=0.83, l=0.74, h=0.76)
                [0.69, 0.76, 0.57],  # 马桶 (w=0.69, l=0.76, h=0.57)
                [0.50, 0.94, 0.67],  # 水槽 (w=0.50, l=0.94, h=0.67)
                [0.88, 2.00, 0.23],  # 门 (w=0.88, l=2.00, h=0.23) 扁平
                [0.66, 1.00, 0.86],  # 窗户 (w=0.66, l=1.00, h=0.86)
                [0.64, 1.86, 0.97],  # 柜子 (w=0.64, l=1.86, h=0.97)
                [0.83, 0.82, 1.47],  # 沙发 (w=0.83, l=0.82, h=1.47)
                [0.60, 0.29, 1.85],  # 台面 (w=0.60, l=0.29, h=1.85) 细高
                [1.22, 1.76, 1.14],  # 窗帘 (w=1.22, l=1.76, h=1.14)
                [0.93, 1.81, 0.94],  # 冰箱 (w=0.93, l=1.81, h=0.94)
                [1.57, 0.83, 1.99],  # 床 (w=1.57, l=0.83, h=1.99) 高大
                [1.38, 1.05, 0.71],  # 书桌 (w=1.38, l=1.05, h=0.71)
            ],
            custom_values=[],
            rotations=[0, 1.57],  # 0° 和 90°
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.8),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8)),
    multi_scale_id=[0],
    # 室内体素: 8m x 8m x 3m, 分辨率 0.05m
    n_voxels=[[160, 160, 4]],
    voxel_size=[[0.05, 0.05, 0.75]],
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.5,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        score_thr=0.001,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=500,
        use_rotate_nms=True,
        nms_thr=0.2,
        nms_type_list=['rotate'] * 18,
        nms_thr_list=[0.2] * 18,
    )
)

# ======================== 数据配置 ========================
# 室内场景范围: 根据实际 GT 分布调整
# X: [-3.43, 3.64], Y: [-3.59, 3.58], Z: [-0.07, 2.05]
point_cloud_range = [-4, -4, -0.5, 4, 4, 2.5]

class_names = (
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'desk', 'curtain',
    'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
    'garbagebin'
)

dataset_type = 'ScanNetMultiViewDataset'
data_root = 'dataset/scannet/mm3Ddetection/'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# 图像归一化 (ImageNet 标准)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# ScanNet 图像尺寸: 1296x968
data_config = {
    'src_size': (968, 1296),
    'input_size': (256, 704),
    # train-aug
    'resize': (-0.06, 0.11),
    'crop': (-0.05, 0.05),
    'rot': (-5.4, 5.4),
    'flip': True,
    # test-aug
    'test_input_size': (256, 704),
    'test_resize': 0.0,
    'test_rotate': 0.0,
    'test_flip': False,
    # top, right, bottom, left
    'pad': (0, 0, 0, 0),
    'pad_divisor': 32,
    'pad_color': (0, 0, 0),
}

file_client_args = dict(backend='disk')

# ======================== 数据管道 ========================
# ScanNet 专用管道：不使用 RandomAugImageMultiViewImage（rts2proj 不兼容）
# 直接使用原始外参矩阵，不进行图像增强
train_pipeline = [
    dict(type='MultiViewPipeline',
         sequential=False,  # ScanNet 非时序
         n_images=4,         # 每次采样 4 个视角
         n_times=1,          # 无时序
         transforms=[
             dict(type='LoadImageFromFile',
                  file_client_args=file_client_args)]),
    dict(type='LoadAnnotations3D',
         with_bbox=True,
         with_label=True,
         with_bbox_3d=True,
         with_label_3d=True,
         with_bev_seg=False),  # ScanNet 无 BEV 分割
    dict(type='LoadPointsFromFile',
         dummy=True,
         coord_type='LIDAR',
         load_dim=5,
         use_dim=5),
    dict(type='RandomFlip3D',
         flip_2d=False,
         sync_2d=False,
         flip_ratio_bev_horizontal=0.5,
         flip_ratio_bev_vertical=0.5,
         update_img2lidar=True),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.3925, 0.3925],
         scale_ratio_range=[0.95, 1.05],
         translation_std=[0.05, 0.05, 0.05],
         update_img2lidar=True),
    # 使用简单 resize 替代 RandomAugImageMultiViewImage
    dict(type='SimpleResizeImage', target_size=(256, 704)),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes', 'gt_labels',
                                 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(type='MultiViewPipeline',
         sequential=False,
         n_images=4,
         n_times=1,
         transforms=[
             dict(type='LoadImageFromFile',
                  file_client_args=file_client_args)]),
    dict(type='LoadPointsFromFile',
         dummy=True,
         coord_type='LIDAR',
         load_dim=5,
         use_dim=5),
    # 使用简单 resize 替代 RandomAugImageMultiViewImage
    dict(type='SimpleResizeImage', target_size=(256, 704)),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D',
         class_names=class_names,
         with_label=False),
    dict(type='Collect3D', keys=['img']),
]

# ======================== 数据加载 ========================
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d='Depth',  # ScanNet 使用 Depth 坐标系
        ann_file=data_root + 'scannet_fastbev_infos_train.pkl',
        num_frames=4,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Depth',
        ann_file=data_root + 'scannet_fastbev_infos_val.pkl',
        num_frames=4,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Depth',
        ann_file=data_root + 'scannet_fastbev_infos_val.pkl',
        num_frames=4,
    ),
)

# ======================== 训练配置 ========================
optimizer = dict(
    type='AdamW2',
    lr=0.005,  # 极端过拟合测试
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=35., norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=3,
    warmup_ratio=0.1,
    step=[200, 250])

total_epochs = 300  # 极端过拟合测试
checkpoint_config = dict(interval=20)
log_config = dict(
    interval=1,  # 每个 iteration 都记录（数据量少）
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
evaluation = dict(interval=5)
dist_params = dict(backend='nccl')
find_unused_parameters = True
log_level = 'INFO'

load_from = None  # 不加载 nuScenes 预训练权重（类别不同）
resume_from = None
workflow = [('train', 1)]

# fp16 settings - disabled for debugging
# fp16 = dict(loss_scale='dynamic')
