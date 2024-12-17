_base_ = [
    # '../_base_/models/segformer.py',
    '../_base_/datasets/cityscapes_1024x1024_repeat.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
# norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='../res/pretrain/mit_b1.pth',
    backbone=dict(
        type='mit_b1',
        style='pytorch'),
    decode_head=dict(
        type='UnetHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        multiple=[4, 2, 2, 2],
        dilation={'dilation': [6], 'kernel_size': [3],
                  'use_pooling': True, 'up_mode': 'mul',
                  'fusion_conv': False},
        insert_attn=False,
        insert_aspp=False,
        insert_fusion='norm',
        fusion_cat=False,  # default:false  Integration use accumulation
        aux_val=False,
        distill=False,
        norm_cfg=norm_cfg,
        align_corners=False,
        # decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    # If additional characteristic maps are used to calculate losses ==> extra_aux=True  Note: Setting backbone param aux_depth，default：using last layer
    # Additional feature map calculation loss weight settings ==>extra_loss_weight={loss weight}
    train_cfg=dict(extra_aux=True, extra_loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.1)),
    # train_cfg=dict(),
    # test_cfg=dict(mode='whole'))
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

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

work_dir = '../res/1024/Norm/sum'
MASTER_ADDR = 'localhost'
MASTER_PORT = '12347'
gpu_ids = [0,1,2,3]
# distributed data parallel setting  word_size= --nproc_per_node  rank= --node_rank
# dist_params = dict(backend='nccl',world_size=2,rank=1)

# distributed  option : none pytorch slurm mpi
launcher='pytorch'
load_from = None
# load model weight ,optimizer ,iter ....
resume_from = None

data = dict(samples_per_gpu=2, workers_per_gpu=4, pin_memory=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=5000, metric='mIoU')
