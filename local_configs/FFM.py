_base_ = [
    # '../_base_/models/segformer.py',
    # '../_base_/datasets/cityscapes_1024x1024_repeat.py',
    '_base_/datasets/cancerseg_repeat.py',
    '_base_/default_runtime.py',
    '_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
# norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='../work_dirs/mit_b5.pth',
    backbone=dict(
        type='mit_b5',
        aux_depth=[1,3,20,1],
        style='pytorch'),
    decode_head=dict(
        type='UnetHead',
        in_channels=[ 64, 128, 320],
        in_index=[0, 1, 2],
        channels=128,
        dropout_ratio=0.1,
        num_classes=5, # background,nor,hyp,dys,car
        multiple=[4,  2, 2],
        dilation={'dilation': [6,12], 'kernel_size': [3,3],
                  'use_pooling': True, 'up_mode': 'mul',
                  'fusion_conv': False},
        norm_cfg=norm_cfg,
        align_corners=False,
        # decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    # If using shallow features ==> aux_decode = True
    # Note: Setting backbone param aux_depth  default: using last layer
    train_cfg=dict(freeze_name=(),use_mse_loss=True,aux_decode=True),
    # train_cfg=dict(),
    # test_cfg=dict(mode='whole'))
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

# data
data = dict(samples_per_gpu=4, pin_memory=True)
runner = dict(type='IterBasedRunner', max_iters=160000)  # batch size ADE : 16 Cityscapes : 8 iter : 160000 
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=160000, metric='mIoU',save_thres=70.0,save_metric='F1',rtn_name=['nor_hyp_cls','dys_car_cls'])
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
