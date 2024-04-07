_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

base_data_root = '../outputs/coco_ovd_images/coco_base_ft6_1250/'
base_train_ann = 'anns_thr0.8_pos_images.json'
novel_data_root = '../outputs/coco_ovd_images/coco_novel_ft6_3000/'
novel_train_ann = 'anns_base_and_novel_thr0.4.json'

coco_data_root='data/coco/'
coco_train_ann='annotations/instances_train2017_seen.json'
test_data_root = 'data/coco/'
test_ann = 'annotations/instances_val2017.json'

base_ind_file = 'instagen_resources/coco_base_inds.txt'
novel_ind_file = 'instagen_resources/coco_novel_inds.txt'

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe'
        )
    ),
    train_cfg = dict(
        rpn_proposal = dict(
            ignore_loss=True,
            ignored_data='data/coco/'
        ),
        rcnn = dict(
            ignore_novel=True,
            ignored_data='data/coco/',
        )
    ),
    roi_head=dict(
        bbox_head=dict(
            reg_class_agnostic=False,
            loss_cls=dict(
                type='MaskedCrossEntropyLoss', 
                use_sigmoid=True, 
                loss_weight=1.0,
                num_classes=80,
                base_ind_file=base_ind_file 
            ),
        )
    )
)

base_dataset=dict(
    type='CocoDataset',
    data_root=base_data_root,
    ann_file=base_train_ann,
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=[
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='RandomChoiceResize',
            scales=[(1333, 224), (1333, 256), (1333, 288), (1333, 320),
                    (1333, 352), (1333, 384), (1333, 416), (1333, 448),
                    (1333, 480), (1333, 512), (1333, 544), (1333, 576),
                    (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                    (1333, 736), (1333, 768), (1333, 800)],
            keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PackDetInputs')
    ],
    backend_args=None
)

novel_dataset=dict(
    type='CocoDataset',
    data_root=novel_data_root,
    ann_file=novel_train_ann,
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=[
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='RandomChoiceResize',
            scales=[(1333, 224), (1333, 256), (1333, 288), (1333, 320),
                    (1333, 352), (1333, 384), (1333, 416), (1333, 448),
                    (1333, 480), (1333, 512), (1333, 544), (1333, 576),
                    (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                    (1333, 736), (1333, 768), (1333, 800)],
            keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PackDetInputs')
    ],
    backend_args=None
)

base_train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=base_dataset,
)

novel_train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=novel_dataset,
)

coco_train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root=coco_data_root,
        ann_file=coco_train_ann,
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ],
        backend_args=None))

test_pipeline=[
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]

val_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        data_root=test_data_root,
        ann_file=test_ann,
        data_prefix=dict(img='val2017/'),
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        data_root=test_data_root,
        ann_file=test_ann,
        data_prefix=dict(img='val2017/'),
        pipeline=test_pipeline
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=test_data_root + test_ann,
    metric='bbox',
    format_only=False,
    backend_args=None,
    classwise=True,
    base_ind_file=base_ind_file,
    novel_ind_file=novel_ind_file
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=test_data_root + test_ann,
    metric='bbox',
    format_only=False,
    backend_args=None,
    classwise=True,
    base_ind_file=base_ind_file,
    novel_ind_file=novel_ind_file
)

train_cfg = dict(type='EpochDatasetTrainLoop', max_epochs=12, val_interval=1, \
                 train_dataloaders=[base_train_dataloader, novel_train_dataloader, coco_train_dataloader], \
                 self_train=False, score_thresh=0., start_epoch=-1)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(base_batch_size=32)

work_dir = './work_dirs/faster-rcnn_r50-caffe_c4-1x_coco-ovd_instagen-dataset'

find_unused_parameters = True
