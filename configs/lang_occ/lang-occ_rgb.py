# Copyright (c) Phigent Robotics. All rights reserved.

# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

############ CUSTOM ###############
bev_h_ = 200
bev_w_ = 200
bev_z_ = 16
voxel_resolution = 0.4
pc_range = [-40., -40., -1.0, 40., 40., 5.4]
voxel_size = [0.1, 0.1, 0.2]
eval_threshold_range=[.1]
vocabulary_version = [1]
sh_deg = 3

# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
gt_root = 'data/gts'
embeddings_root = 'data/embeddings'
rays_root = 'data/rays'
embedding_model = 'MaskCLIP'
file_client_args = dict(backend='disk')

# Other settings
batch_size = 4
total_rays = 32768

###################################

numC_Trans = 32
voxel_feat_dim = 256
hidden_dim = 256

multi_adj_frame_id_cfg = (1, 1+1, 1)
T = 12
render_frame_ids = list(range(-T,T+1,1))

model = dict(
    type='LangOcc',
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    out_dim=voxel_feat_dim,
    eval_threshold_range=eval_threshold_range,
    class_embeddings_path=f'{embeddings_root}/{embedding_model}',
    vocabulary_version=vocabulary_version,
    use_templates=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=hidden_dim,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVStereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=hidden_dim,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=False,
        loss_depth_weight=0.05,
        depthnet_cfg=dict(use_dcn=False,
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_layer=[1, 2, 4],
        with_cp=False,
        num_channels=[numC_Trans,numC_Trans*2,numC_Trans*4],
        stride=[1,2,2],
        backbone_output_ids=[0,1,2]),
    img_bev_encoder_neck=dict(type='LSSFPN3D',
                              in_channels=numC_Trans*7,
                              out_channels=numC_Trans),
    pre_process=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans,
        with_cp=False,
        num_layer=[1,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    renderer=dict(
        type='LangRenderer',
        render_modules=[
            dict(type='SH_RGBRenderModule', degree=sh_deg)
        ],
            pc_range=pc_range,
            samples_per_ray=100,
            prop_samples_per_ray=50,
            grid_cfg=[bev_h_, bev_w_, bev_z_, voxel_resolution, 1],
            use_proposal=True,
            render_frame_ids=render_frame_ids,
    ),
    density_decoder=dict(
        type='PointDecoder',
        in_channels=voxel_feat_dim,
        embed_dims=hidden_dim,
        num_hidden_layers=3,
        num_classes=1,
        final_act_cfg=dict(type='Sigmoid'),
    ),
    language_decoder=None,
    rgb_decoder=dict(
        type='PointDecoder',
        in_channels=voxel_feat_dim,
        num_hidden_layers=3,
        embed_dims=hidden_dim,
        num_classes=3 * sh_deg**2 if sh_deg>0 else 3, # RGB
        final_act_cfg=dict(type='Sigmoid'), # scale to 0-1 range
    )
)

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='GenerateRaysMaskCLIP', embeddings_root=embeddings_root, model=embedding_model, rays_root=rays_root, 
         num_rays=total_rays // batch_size, render_frame_ids=render_frame_ids, use_rgb=True),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'lang_ray_dataset'])
]

val_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
   dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

test_pipeline = val_pipeline

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
    render_frame_ids=render_frame_ids,
    eval_threshold_range=eval_threshold_range,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file='data/bevdetv2-nuscenes_infos_val.pkl',
    gt_root=gt_root)

val_data_config = dict(
    pipeline=val_pipeline,
    ann_file='data/bevdetv2-nuscenes_infos_val.pkl',
    gt_root=gt_root)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file='data/bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        gt_root=gt_root,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=val_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='CustomCosineAnealing',
    start_at=7,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    min_lr_ratio=1e-2
)
runner = dict(type='EpochBasedRunner', max_epochs=18)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval = 6
    ),
]
evaluation = dict(interval = 6, pipeline=test_pipeline)
checkpoint_config=dict(interval=6)
load_from="ckpts/bevdet-r50-4d-stereo-cbgs.pth"
# fp16 = dict(loss_scale='dynamic')
