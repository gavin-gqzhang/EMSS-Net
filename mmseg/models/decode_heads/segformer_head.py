# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import os
import random

import cv2
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
from mmseg.models.backbones.mix_transformer import Attention
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr
import matplotlib.pyplot as plt

from IPython import embed


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        # c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        c1_in_channels, c2_in_channels, c3_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        # self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            # in_channels=embedding_dim * 4,
            in_channels=embedding_dim * 3,
            out_channels=embedding_dim,
            kernel_size=1,
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.dou_cls_pred=nn.Conv2d(embedding_dim,2,kernel_size=1)  # background NOR+HYP+DYS+CAR

        """
        self.mul_cls_linear_fuse = ConvModule(
            in_channels=embedding_dim * 3,
            out_channels=embedding_dim,
            kernel_size=1,
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        """
        if self.num_classes!=2:
            self.filter_fg=nn.Sequential(
                nn.Conv2d(2*embedding_dim,embedding_dim,kernel_size=1),
                nn.Sigmoid(),
                nn.Dropout(0.1),
                nn.Conv2d(embedding_dim,embedding_dim,kernel_size=1),
                nn.ReLU()
            )
            self.linear_pred = nn.Sequential(
                nn.Conv2d(embedding_dim,embedding_dim,kernel_size=1),
                nn.Dropout(0.1),
                nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
                )
            # assert self.num_classes==5, 'Please confirm the number of categories!' # background,nor,hyp,dys,car

            # ************* restrict nor-hyp classifier *************
            # self.restrict_nor_hyp=nn.Conv2d(embedding_dim,3,kernel_size=1)  # nor/hyp classifier
            self.fused_c3_c2=FFM(embedding_dim,embedding_dim)
            self.fused_c2_mix=Mix_FNN(embedding_dim)    
            
            self.fused_c2_c1=FFM(embedding_dim,embedding_dim)
            self.fused_c1_mix=Mix_FNN(embedding_dim)
            
            self.filter_fused_c1=nn.Sequential(
                    nn.Conv2d(2*embedding_dim,embedding_dim,kernel_size=1),
                    nn.Sigmoid(),
                    nn.Dropout(0.1),
                    nn.Conv2d(embedding_dim,embedding_dim,kernel_size=1),
                )
        
    def freeze_module(self):
        for name,parm in self.named_parameters():
            if 'linear_c3' in name or 'linear_c2' in name or 'linear_c1' in name or 'linear_fuse' in name or 'dou_cls_pred' in name:
                parm.requires_grad=False
    
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def forward(self, inputs,img_metas):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        # c1, c2, c3, c4 = x
        c1, c2, c3 = x
        
        # vis_features(c1,f'{img_metas[0]["save_path"]}/c1/')
        # vis_features(c2,f'{img_metas[0]["save_path"]}/c2/')
        # vis_features(c3,f'{img_metas[0]["save_path"]}/c3/')

        # vis_features(c1,f'{img_metas[0]["save_path"]}/c1/',channel=3)
        # vis_features(c2,f'{img_metas[0]["save_path"]}/c2/',channel=3)
        # vis_features(c3,f'{img_metas[0]["save_path"]}/c3/',channel=3)
        ############## MLP decoder on C1-C4 ###########
        # n, _, h, w = c4.shape
        n, _, h, w = c3.shape

        # _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        # _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        proj_c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(proj_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        proj_c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(proj_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c3, _c2, _c1], dim=1))
        
        # vis_features(_c,f'{img_metas[0]["save_path"]}/shallow_fusion/')
        # vis_features(_c,f'{img_metas[0]["save_path"]}/shallow_fusion/',channel=3)

        dou_x=self.dou_cls_pred(self.dropout(_c)) # 2  classes   => 2*h*w
            
        if self.num_classes!=2:
            _c,dou_x=_c.detach(),dou_x.detach()
            weighted_fg=_c+_c*dou_x[:,1:,...]-_c*dou_x[:,:1,...]
            fused_fg=weighted_fg+self.filter_fg(torch.cat([weighted_fg,_c],dim=1))*_c 
            
            fused_c2=self.fused_c3_c2(resize(proj_c3,size=proj_c2.size()[2:],mode='bilinear',align_corners=False),proj_c2)
            fused_c2=self.fused_c2_mix(fused_c2)
            
            fused_c1=self.fused_c2_c1(resize(fused_c2,size=_c1.size()[2:],mode='bilinear',align_corners=False),_c1)
            fused_c1=self.fused_c1_mix(fused_c1)
            
            # vis_features(fused_c1,f'{img_metas[0]["save_path"]}/mix_fusion/')
            # vis_features(fused_c1,f'{img_metas[0]["save_path"]}/mix_fusion/',channel=3)
        
            # ****** predict *******
            # nor_hyp=self.restrict_nor_hyp(fused_fg) # predicte nor/hyp classes
            fused_fg=self.filter_fused_c1(torch.cat([fused_c1,fused_fg],dim=1))
            multi_cls=self.linear_pred(fused_fg)

            if not self.training:
                # multi_cls[:,0,...]=multi_cls[:,0,...]*dou_x[:,0,...]
                # multi_cls[:,1:,...]=multi_cls[:,1:,...]*(dou_x[:,1,...].unsqueeze(1).expand(-1,self.num_classes-1,-1,-1))
                multi_cls=multi_cls*dou_x.argmax(dim=1).unsqueeze(1).expand(-1,self.num_classes,-1,-1)
            return (dou_x,multi_cls)
        else:
            return (dou_x)
        
class FFM(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super(FFM, self).__init__()

        out_dim = out_dim or in_dim

        self.fun_conv = nn.Sequential(
            nn.Conv2d(2 * in_dim, out_dim, kernel_size=1, stride=1, groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )

        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim, out_dim // 4, kernel_size=1, stride=1, groups=out_dim // 4),
            nn.GELU(),
            nn.Conv2d(out_dim // 4, out_dim, kernel_size=1, stride=1, groups=out_dim // 4),
            nn.Sigmoid()
        )

    def forward(self, up_x, original_x):
        feat = torch.cat([up_x, original_x], dim=1)
        feat = self.fun_conv(feat)

        attn = self.pooling(feat)
        feat_attn = torch.mul(feat, attn)

        out = feat_attn + feat

        return out

class Mix_FNN(nn.Module):
    def __init__(self, in_dim, dil_steps=[6,12], dil_kernels=[3,3], drop=0.):
        super(Mix_FNN, self).__init__()

        self.reshape = False

        depth = 1 + len(dil_steps)

        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, in_dim)

        self.dwconv = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_dim, in_dim, kernel_size=dil_kernel, dilation=dil_step, padding=dil_step if dil_kernel !=1 else 0, bias=False,
                            groups=in_dim),
                    nn.BatchNorm2d(in_dim),
                    nn.GELU()
                ) for dil_step,dil_kernel in zip(dil_steps,dil_kernels)
            ]),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_dim, in_dim, 1, bias=False, groups=in_dim),
                nn.GELU()
                ),
            nn.Sequential(
                nn.Conv2d(depth * in_dim, in_dim, 1, bias=False, groups=in_dim),
                nn.BatchNorm2d(in_dim),
                nn.GELU(),
                nn.Dropout(0.5),
                Rearrange('b c h w -> b (h w) c'),
            )
        ])

        self.dropout = nn.Dropout(drop)

    def dwconv_forward(self,x):
        dil_convs,pooling,proj_fuse=self.dwconv
        dil_x = []
        for aspp in dil_convs:
            aspp_x = aspp(x)
            dil_x.append(aspp_x)

        pool_x = pooling(x)
        pool_x = torch.mul(x, pool_x)
        dil_x.append(pool_x)

        x = torch.cat(dil_x, dim=1)

        x = proj_fuse(x)

        return x
    
    def forward(self, x):
        _, _, h, w = x.shape
        re_x = rearrange(x, 'b c h w -> b (h w) c')
        re_x = self.norm(re_x)
        re_x = self.fc1(re_x)
        re_x = rearrange(re_x, 'b (h w) c -> b c h w', h=h, w=w)
        re_x = self.dwconv_forward(re_x)
        re_x = self.dropout(re_x)
        re_x = self.fc2(re_x)
        re_x = self.dropout(re_x)
        re_x = rearrange(re_x, 'b (h w) c -> b c h w', h=h, w=w)

        out = x + re_x
        return out


def vis_features(reps,save_path,channel=1):
    for rep_idx,rep in enumerate(reps):
        sample_channels=random.sample(range(rep.shape[0]//channel),min(100,rep.shape[0]//channel))
        for chann_idx in sample_channels:

            save_reps=rep[chann_idx*channel:(chann_idx+1)*channel,...].permute(1,2,0).contiguous().detach().cpu().data.numpy()

            img_save_path=f"{save_path}/sample_{rep_idx}/{channel}/{chann_idx}.png"
            os.makedirs(os.path.dirname(img_save_path),exist_ok=True)

            norm_reps=((save_reps-save_reps.min())/(save_reps.max()-save_reps.min())*255).astype(np.uint8)
            if channel==3:
                color_reps=cv2.applyColorMap(norm_reps,cv2.COLORMAP_JET)

                cv2.imwrite(img_save_path,color_reps)
            else:
                cv2.imwrite(img_save_path,norm_reps)
        