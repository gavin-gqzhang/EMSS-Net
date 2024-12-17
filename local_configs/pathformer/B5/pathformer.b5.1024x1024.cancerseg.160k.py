_base_ = [
    '../../_base_/models/segformer.py',
    # '../../_base_/datasets/cityscapes_1024x1024_repeat.py',
    '../../_base_/datasets/cancerseg_repeat.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
# norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    # pretrained='../work_dirs/mit_b5.pth',
    pretrained='/data/sdc/checkpoints/medicine_res/ori_scale_segformer_bgfg/iter_160000.pth',
    backbone=dict(
        type='mit_b5',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels= [64, 128, 320], #  [64, 128, 320, 512],
        in_index=[0,1,2],  # [0, 1, 2, 3],
        feature_strides=[4,8,16], #[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=5, # 
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        # loss_decode=dict(type='Merge_loss',n_classes=3,loss_name=['lovasz','dice','ce'], use_sigmoid=False, loss_weight=1.0,reduction='none'),
        ignore_index=255),
        # loss_decode=dict(type='LovaszLoss', use_sigmoid=True, loss_weight=1.0,reduction='none')),
    # model training and testing settings
    # If using shallow features ==> aux_decode = True
    # Note: Setting backbone param aux_depth  default: using last layer
    train_cfg=dict(freeze_name=('linear_c3','linear_c2','linear_c1','linear_fuse','dou_cls_pred'),use_mse_loss=True,aux_decode=False),
    # test_cfg=dict(mode='whole'))
    test_cfg=dict(mode='slide', crop_size=(1024,1024), stride=(768,768)))

# data
data = dict(samples_per_gpu=2, pin_memory=True)
runner = dict(type='IterBasedRunner', max_iters=320000)  # batch size ADE : 16 Cityscapes : 8 iter : 160000 
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=320000, metric='mIoU',save_thres=70.0,save_metric='F1',rtn_name=['nor_hyp_cls','dys_car_cls'])
# save_metric : mIoU,mAcc,aAcc,Precision,Specificity`,Sensitivity,F1  default : mIoU

resume_from=None

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


