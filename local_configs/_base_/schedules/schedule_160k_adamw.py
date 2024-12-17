# optimizer
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=500000)
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=5000, metric='mIoU')
