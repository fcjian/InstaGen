_base_ = [
    '../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py'
]

base_data_root = '../outputs/coco_ovd_images/coco_base_ft6_1250/'
base_train_ann = 'anns_thr0.8_pos_images.json'
novel_data_root = '../outputs/coco_ovd_images/coco_novel_ft6_3000/'
novel_train_ann = 'anns_thr0.8_all_images_top1250.json'
test_data_root = '../outputs/coco_ovd_images/coco_val_ft6_200/'
test_ann = 'anns_thr0.8_pos_images.json'

cls_name_file = 'instagen_resources/coco_classes.txt'
cls_emb_file = 'instagen_resources/coco_classes.pt'
base_ind_file = 'instagen_resources/coco_base_inds.txt'
novel_ind_file = 'instagen_resources/coco_novel_inds.txt'

model = dict(
    type='InstaGen',
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[128, 128, 128], # mean=[123.675, 116.28, 103.53],
        std=[128, 128, 128], # std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='StableDiffusion',
        last_sample=False,
        return_img=True,
        config="../configs/stable-diffusion/v1-inference.yaml",
        sd_ckpt="../checkpoints/stable-diffusion-v-1-4-original/sd-v1-4.ckpt"
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[5120, 8320, 14080, 15360],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        use_text_enhancer=True,  # fcj add
        use_fusion_layer=True,  # fcj add
        dropout=0.1, # fcj add
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            use_text_cross_attention=True, # fcj add
            dropout=0.1, # fcj add
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='GroundingHead',
        num_classes=2,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR

# data loader
base_dataset=dict(
    type='CocoDataset',
    data_root=base_data_root,
    ann_file=base_train_ann,
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=[
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomFlip', prob=0.0),
        dict(type='Resize', scale=(512, 512), keep_ratio=True),
        dict(
            type='LoadFeatures',
            cls_name_file=cls_name_file,
            cls_emb_file=cls_emb_file,
            base_ind_file=base_ind_file,
            novel_ind_file=novel_ind_file
        ),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                        'scale_factor', 'flip', 'flip_direction',
                        'text_embeddings', 'diffusion_feats', 'prompts',
                        'label_mappings', 'base_cls_inds', 'novel_cls_inds',
                        'novel_image_flag'))
    ],
    backend_args=None
)

novel_dataset=dict(
    type='CocoDataset',
    data_root=novel_data_root,
    ann_file=novel_train_ann,
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=[
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomFlip', prob=0.0),
        dict(type='Resize', scale=(512, 512), keep_ratio=True),
        dict(
            type='LoadFeatures',
            cls_name_file=cls_name_file,
            cls_emb_file=cls_emb_file,
            base_ind_file=base_ind_file,
            novel_ind_file=novel_ind_file
        ),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                        'scale_factor', 'flip', 'flip_direction',
                        'text_embeddings', 'diffusion_feats', 'prompts',
                        'label_mappings', 'base_cls_inds', 'novel_cls_inds',
                        'novel_image_flag'))
    ],
    backend_args=None
)

batch_size=4
base_train_dataloader = dict(
    batch_size=batch_size,  # fcj default: 4
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=base_dataset,
)

novel_train_dataloader = dict(
    batch_size=batch_size,  # fcj default: 4
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=novel_dataset,
)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='LoadFeatures',
        cls_name_file=cls_name_file,
        cls_emb_file=cls_emb_file,
        base_ind_file=base_ind_file,
        novel_ind_file=novel_ind_file
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text_embeddings', 'diffusion_feats', 
                   'prompts', 'label_mappings')
    )
]

val_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        data_root=test_data_root,
        ann_file=test_ann,
        data_prefix=dict(img='images/'),
        pipeline=test_pipeline
    )
)
test_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        data_root=test_data_root,
        ann_file=test_ann,
        data_prefix=dict(img='images/'),
        pipeline=test_pipeline
    )
)

val_evaluator = dict(
    ann_file=test_data_root + test_ann,
    classwise=True,
    base_ind_file=base_ind_file,
    novel_ind_file=novel_ind_file,
    # metric=[],  # 'bbox',
)
test_evaluator = dict(
    ann_file=test_data_root + test_ann,
    classwise=True,
    base_ind_file=base_ind_file,
    novel_ind_file=novel_ind_file,
    # metric=[],  # 'bbox',
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
max_epochs = 12
train_cfg = dict(type='EpochDatasetTrainLoop', max_epochs=max_epochs, val_interval=1, \
                 train_dataloaders=[base_train_dataloader, novel_train_dataloader], \
                 self_train=True, score_thresh=0.4, start_epoch=6)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(base_batch_size=32)

work_dir = './work_dirs/instagen-4scale_fd_8xb2-12e_coco_demo'

find_unused_parameters = True
