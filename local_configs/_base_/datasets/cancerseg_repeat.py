# dataset settings
dataset_type = 'CancersegDataset'
# data_root = '/home/ubuntu/ssd/data/20211206/seg/HNSC/pyramid_slide/20X/'
data_root = '/data/sdc/medicine_svs/multi_scale_datas'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=(2048, 2048), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios = [1.0],
        # img_scale=(8192, 8192),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=500,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train_datas/img_dir',#'leftImg8bit/train',
            ann_dir='train_datas/ann_dir',#'gtFine/train',
            pipeline=train_pipeline,
            ignore_index=255)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train_datas/img_dir',#'leftImg8bit/val',
        ann_dir='train_datas/ann_dir',#'gtFine/val',
        pipeline=test_pipeline,
            ignore_index=255),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train_datas/img_dir',#'leftImg8bit/val',
        ann_dir='train_datas/ann_dir',#'gtFine/val',
        pipeline=test_pipeline,
        ignore_index=255))
