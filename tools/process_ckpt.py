
import torch
"""
ckpt=torch.load('/media/ubuntu/Seagate Basic1/work_dirs/optim_data/multi_cls/eval_model/cls_3/mAcc_85.52.pth',map_location='cpu')

print(ckpt.keys())
print(ckpt['state_dict'].keys())

backbone,decode={},{}
for key in list(ckpt['state_dict'].keys()):
    if 'linear_pred' in key or 'mul_cls_linear_fuse' in key:
        print(key)
        del ckpt['state_dict'][key]

print(ckpt['state_dict'].keys())

new_ckpt={
    'state_dict':ckpt['state_dict']
}
torch.save(new_ckpt,'/media/ubuntu/Seagate Basic1/work_dirs/04-06/pretrain_model.pth')
"""
ckpt=torch.load('/media/ubuntu/Seagate Basic1/work_dirs/04-06/eval_model/cls_5/F1_91.08.pth',map_location='cpu')
print(ckpt.keys())

print(ckpt['state_dict'].keys())