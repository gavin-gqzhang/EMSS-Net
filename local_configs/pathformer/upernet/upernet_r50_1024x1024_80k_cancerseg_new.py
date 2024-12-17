# _base_ = [
#     '../_base_/models/pspnet_r50-d8.py',
#     # '../_base_/datasets/cityscapes_769x769.py',
#     '../_base_/datasets/cancerseg_repeat.py',
#     '../_base_/default_runtime.py',
#     '../_base_/schedules/schedule_80k.py'
# ]
_base_ = [
    '../../_base_/models/upernet_r50.py',
    # '../../_base_/datasets/cityscapes_1024x1024_repeat.py',
    '../../_base_/datasets/cancerseg_repeat.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]
model = dict(
    decode_head=dict(align_corners=True),
    auxiliary_head=dict(align_corners=True),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(683, 683)))
