import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import termcolor
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from einops.layers.torch import Rearrange
from torchvision import transforms

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *


# 融合模块，从下往上全部融合，高分辨率的降分辨率后进行融合
class Fusion_Block(nn.Module):
    def __init__(self, in_dims, out_dim):
        super(Fusion_Block, self).__init__()
        self.fusion_conv = nn.ModuleList([])
        for index in range(len(in_dims)):
            dim = in_dims[index]
            if dim == out_dim:
                conv = nn.Identity()
            elif dim < out_dim:
                ratio = len(in_dims) - index - 1
                kernel_size = 2 ** ratio + 1
                conv = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=2 ** ratio, padding=kernel_size // 2,
                              groups=dim),
                    nn.BatchNorm2d(dim),
                    nn.Conv2d(dim, out_dim, kernel_size=1, stride=1),
                    nn.BatchNorm2d(out_dim)
                )
            else:
                raise ValueError('Fusion Block get error in channel list')
            self.fusion_conv.append(conv)
        self.act = nn.GELU()

    def forward(self, features,in_feature=None, visual=False):
        if in_feature==None:
            out = None
            for (feature, block) in zip(features, self.fusion_conv):
                out = out + block(feature) if out is not None else block(feature)
            if visual:
                visual_feature(out, 'after Fusion Block(use all feature) Module processing')
        else:
            out=[in_feature]
            for (feature, block) in zip(features, self.fusion_conv):
                out.append(block(feature))
            out=torch.cat(out,dim=1)

        return self.act(out)


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

    def forward(self, up_x, original_x, visual=False):
        feat = torch.cat([up_x, original_x], dim=1)
        feat = self.fun_conv(feat)
        if visual:
            visual_feature(feat, 'after FFM(use two feature) fusion Module processing')

        attn = self.pooling(feat)
        feat_attn = torch.mul(feat, attn)
        if visual:
            visual_feature(feat, 'after FFM(use two feature) pooling Module processing')

        out = feat_attn + feat
        if visual:
            visual_feature(feat, 'after FFM(use two feature) Module processing')

        return out


class Conv_Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv_Block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, visual=False):
        x = self.conv(x)
        if visual:
            visual_feature(x, 'after Conv Block fusion Module processing')

        return x


class Mix_FNN(nn.Module):
    def __init__(self, in_dim, out_dim, dilations=None, drop=0.):
        super(Mix_FNN, self).__init__()

        self.reshape = False
        if in_dim != out_dim:
            self.deconv = nn.Sequential(
                # nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, groups=out_dim),
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=out_dim if in_dim>out_dim else in_dim),
                nn.GELU()
            )
            self.reshape = True

        depth = 1 if dilations == None else len(dilations['dilation']) + 1 if dilations['use_pooling'] else len(
            dilations['dilation'])

        self.norm = nn.LayerNorm(out_dim)
        self.fc1 = nn.Linear(out_dim, out_dim)
        self.fc2 = nn.Linear(out_dim if dilations != None and dilations['fusion_conv'] else depth * out_dim, out_dim)

        self.dwconv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=6, dilation=6, bias=True,
                      groups=out_dim) if dilations == None else dil_module(out_dim, dilations=dilations['dilation'],
                                                                           kernel_size=dilations['kernel_size'],
                                                                           use_pooling=dilations['use_pooling'],
                                                                           fusion_conv=dilations['fusion_conv'],
                                                                           mode=dilations['up_mode']),
            nn.GELU() if dilations == None or not dilations['fusion_conv'] else nn.Identity(),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.dropout = nn.Dropout(drop)

    def forward(self, x, visual=False):
        if self.reshape:
            x = self.deconv(x)
            if visual:
                visual_feature(x, 'after Mix FNN drop channel Module processing')
        _, _, h, w = x.shape
        re_x = rearrange(x, 'b c h w -> b (h w) c')
        re_x = self.norm(re_x)
        re_x = self.fc1(re_x)
        re_x = rearrange(re_x, 'b (h w) c -> b c h w', h=h, w=w)
        if visual:
            visual_feature(re_x, 'after Mix FNN fc1 Module processing')
        re_x = self.dwconv(re_x)
        if visual:
            visual_feature(rearrange(re_x, 'b (h w) c -> b c h w', h=h, w=w), 'after Mix FNN DWConv Module processing')
        re_x = self.dropout(re_x)
        re_x = self.fc2(re_x)
        re_x = self.dropout(re_x)
        re_x = rearrange(re_x, 'b (h w) c -> b c h w', h=h, w=w)
        if visual:
            visual_feature(re_x, 'after Mix FNN fc2 Module processing')

        out = x + re_x
        if visual:
            visual_feature(out, 'after Mix FNN Module processing')
        return out


class dil_module(nn.Module):
    def __init__(self, input_channel, out_channel=None, dilations=[6, 12, 18], kernel_size=[3, 3, 3],
                 use_pooling=False, fusion_conv=False, mode='bilinear'):
        super(dil_module, self).__init__()

        out_channel = out_channel or input_channel

        assert len(dilations) == len(kernel_size), termcolor.colored(
            'The length of the dilation rate list is not equal to the length of the kernel list', 'red')

        depth = len(dilations)

        self.use_pooling, self.fusion_conv = use_pooling, fusion_conv
        self.mode = mode
        self.dil = nn.ModuleList([])
        for index in range(len(dilations)):
            self.dil.append(
                # kernel_size:kernel+(dilation-1)*(kernel-1)
                ASPP_Module(input_channel, out_channel, kernel_size=kernel_size[index], dilation=dilations[index],
                            padding=dilations[index] if kernel_size[index] != 1 else 0)
            )

        if use_pooling:
            self.pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                         nn.Conv2d(input_channel, out_channel, 1, bias=False, groups=out_channel),
                                         # nn.BatchNorm2d(out_channel),
                                         
                                         nn.GELU() if mode != 'mul' else nn.Sigmoid()
                                         )
            depth = depth + 1

        if fusion_conv:
            self.projection = nn.Sequential(
                nn.Conv2d(depth * out_channel, out_channel, 1, bias=False, groups=out_channel),
                nn.BatchNorm2d(out_channel),
                nn.GELU(),
                nn.Dropout(0.5),
            )

        self.act = nn.GELU()

    def forward(self, x, visual=False):
        dil_x = []
        for aspp in self.dil:
            aspp_x = aspp(x)
            dil_x.append(aspp_x)

            if visual:
                visual_feature(aspp_x, 'after Mix FNN DWConv-dilation_conv{} Module processing'.format(len(dil_x)))

        if self.use_pooling:
            pool_x = self.pooling(x)
            if self.mode == 'mul':
                pool_x = torch.mul(x, pool_x)
            else:
                pool_x = F.interpolate(pool_x, aspp_x.size()[-2:], mode=self.mode, align_corners=False)
            dil_x.append(pool_x)
            if visual:
                visual_feature(pool_x, 'after Mix FNN DWConv-pooling Module processing')

        x = torch.cat(dil_x, dim=1)
        x = self.act(x)

        if self.fusion_conv:
            x = self.projection(x)
            if visual:
                visual_feature(x, 'after Mix FNN DWConv-fusion_conv Module processing')

        return x


class ASPP(nn.Module):
    def __init__(self, input_channel, out_channel=None, dilations=[6, 12, 18], kernel_size=[3, 3, 3]):
        super(ASPP, self).__init__()

        out_channel = out_channel or input_channel

        assert len(dilations) == len(kernel_size), termcolor.colored(
            'The length of the dilation rate list is not equal to the length of the kernel list', 'red')

        # kernel_size:kernel+(dilation-1)*(kernel-1)
        self.aspp1 = ASPP_Module(input_channel, out_channel, kernel_size=kernel_size[0],
                                 dilation=dilations[0])  # (54,54)
        self.aspp2 = ASPP_Module(input_channel, out_channel, kernel_size=kernel_size[1], dilation=dilations[1],
                                 padding=dilations[1])  # kernel=3+(6-1)(3-1)=3+5*2=13
        self.aspp3 = ASPP_Module(input_channel, out_channel, kernel_size=kernel_size[2], dilation=dilations[2],
                                 padding=dilations[2])  # kernel=3+(12-1)(3-1)=3+11*2=25
        self.aspp4 = ASPP_Module(input_channel, out_channel, kernel_size=kernel_size[3], dilation=dilations[3],
                                 padding=dilations[3])  # kernel=3+(18-1)(3-1)=3+17*2=37

        self.pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(input_channel, out_channel, 1, bias=False, groups=out_channel),
                                     nn.BatchNorm2d(out_channel),
                                     nn.ReLU()
                                     )

        self.projection = nn.Sequential(nn.Conv2d(5 * out_channel, out_channel, 1, bias=False, groups=out_channel),
                                        nn.BatchNorm2d(out_channel),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        )

    def forward(self, x, visual=False):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x_pool = self.pooling(x)
        x_pool = F.interpolate(x_pool, size=x4.size()[-2:], mode='bilinear', align_corners=False)

        x = torch.cat((x1, x2, x3, x4, x_pool), dim=1)

        x = self.projection(x)

        return x


class ASPP_Module(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation, padding=0):
        super(ASPP_Module, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=False,
                      groups=out_channel),
            nn.BatchNorm2d(out_channel),
            nn.GELU()
        )

    def forward(self, x, visual=False):
        return self.conv(x)


class up_sampling(nn.Module):
    def __init__(self, in_dim, out_dim, multiple=2):
        super(up_sampling, self).__init__()

        self.up_sampling = nn.Sequential(
            # nn.ConvTranspose2d(in_dim, out_dim, kernel_size=multiple, stride=multiple),
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=multiple, mode='bilinear', align_corners=True)
        )

    def forward(self, x, visual=False):
        x = self.up_sampling(x)
        return x


class PAM(nn.Module):
    def __init__(self, in_dim):
        super(PAM, self).__init__()

        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, visual=False):
        _, _, h, w = x.shape
        proj_query = self.query(x).flatten(2).permute(0, 2, 1).contiguous()
        proj_key = self.key(x).flatten(2)
        q_k = torch.bmm(proj_query, proj_key)
        atten = self.softmax(q_k)
        proj_value = self.value(x).flatten(2)

        res = torch.bmm(proj_value, atten.permute(0, 2, 1).contiguous())
        res = rearrange(res, 'b c (h w) -> b c h w', h=h, w=w)
        out = res * self.gamma + x
        if visual:
            visual_feature(out, 'after Attention-PAM Module processing')
        return out


class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, visual=False):
        _, _, h, w = x.shape
        proj_query = x.flatten(2)
        proj_key = x.flatten(2).permute(0, 2, 1).contiguous()
        q_k = torch.bmm(proj_query, proj_key)
        q_k = torch.max(q_k, -1, keepdim=True)[0].expand_as(q_k) - q_k
        attn = self.softmax(q_k)
        proj_value = x.flatten(2)

        res = torch.bmm(attn, proj_value)
        res = rearrange(res, 'b c (h w) -> b c h w', h=h, w=w)
        out = res * self.gamma + x
        if visual:
            visual_feature(out, 'after Attention-CAM Module processing')
        return out


class Atten_Head(nn.Module):
    def __init__(self, in_dim, out_dim=None, dropout=0.1):
        super(Atten_Head, self).__init__()
        inter_dim = in_dim // 4
        out_dim = out_dim or in_dim

        self.position_conv = nn.Sequential(
            nn.Conv2d(in_dim, inter_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU()
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_dim, inter_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU()
        )

        self.pam = PAM(inter_dim)
        self.cam = CAM()

        self.after_pos_conv = nn.Sequential(
            nn.Conv2d(inter_dim, inter_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU()
        )
        self.after_chann_conv = nn.Sequential(
            nn.Conv2d(inter_dim, inter_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU()
        )

        self.fusion_conv = nn.Sequential(
            nn.Dropout2d(dropout, False),
            nn.Conv2d(inter_dim, out_dim, kernel_size=1)
        )

    def forward(self, x, visual=False):
        pos_x = self.position_conv(x)
        pos_x = self.pam(pos_x)
        pos_x = self.after_pos_conv(pos_x)
        if visual:
            visual_feature(pos_x, 'after Attention-PAM Conv processing')

        chann_x = self.channel_conv(x)
        chann_x = self.cam(chann_x)
        chann_x = self.after_chann_conv(chann_x)
        if visual:
            visual_feature(chann_x, 'after Attention-CAM Conv processing')

        fusion_x = pos_x + chann_x
        out = self.fusion_conv(fusion_x)
        if visual:
            visual_feature(out, 'after Attention-PAM-CAM Fusion Conv processing')
        return out


@HEADS.register_module()
class UnetHead(BaseDecodeHead):
    def __init__(self, multiple=[4,  2, 2], dilation=None,**kwargs):
        super(UnetHead, self).__init__(input_transform='multiple_select',**kwargs)

        self.module = nn.ModuleList([])
        self.depth = len(self.in_channels)

        assert self.depth == len(multiple)

        if dilation != None:
            assert 'dilation' in dilation and 'kernel_size' in dilation and 'use_pooling' in dilation, termcolor.colored(
                'ERROR: Missing dilation key...')

        for i in range(self.depth - 1):
            self.module.append(nn.ModuleList([
                up_sampling(self.in_channels[self.depth - 1 - i], self.in_channels[self.depth - 2 - i],
                            multiple[self.depth - 1 - i]),
                FFM(self.in_channels[self.depth - 2 - i], self.in_channels[self.depth - 2 - i]),
                Mix_FNN(self.in_channels[self.depth - 2 - i], self.in_channels[self.depth - 2 - i], dilations=dilation)
            ]))
        
        self.pre = nn.Conv2d(self.in_channels[0], self.num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, features, visual=False):
        """
        :param features: list[feature1,....]
        :return: tensor
        """

        assert self.depth == len(features)
        distils_fea = []

        feature = features[self.depth - 1]
        idx = 0
        for up_, conn_, fusion_ in self.module:
            # up_,fusion_=self.modules[idx]
            up_feature = up_(feature, visual=visual)
        
            next_feature = features[self.depth - 2 - idx]
            fusion_feature = conn_(up_feature, next_feature, visual=visual)
            
            feature = fusion_(fusion_feature, visual=visual)
            idx = idx + 1

        if visual:
            visual_feature(feature, 'before predict feature')
        pre = self.pre(feature)

        return pre

'''
class Segformer_Unet(nn.Module):
    def __init__(self, img_size, num_classes, backbone='mit_b0', insert_attn=False, dilations=None, insert_aspp=False,
                 insert_fusion=None, aux_val=False, distill=False):
        super(Segformer_Unet, self).__init__()

        assert backbone in ['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5']
        self.backbone_type = backbone
        self.aux_val = aux_val
        self.dil = distill

        if backbone == 'mit_b0':
            aux_depth = [0, 0, 0, 0] if aux_val else None
            self.backbone = mit_b0(crop_size=img_size, num_classes=num_classes, aux_depth=aux_depth)
            self.decoder = Decoder(num_classes=num_classes, embed_dims=self.backbone.embed_dims, dilation=dilations,
                                   insert_attn=insert_attn, insert_aspp=insert_aspp, insert_fusion=insert_fusion,
                                   aux_val=aux_val, distill=distill)

        if backbone == 'mit_b1':
            aux_depth = [0, 0, 0, 0] if aux_val else None
            self.backbone = mit_b1(crop_size=img_size, num_classes=num_classes, aux_depth=aux_depth)
            self.decoder = Decoder(num_classes=num_classes, embed_dims=self.backbone.embed_dims, dilation=dilations,
                                   insert_attn=insert_attn, insert_aspp=insert_aspp, insert_fusion=insert_fusion,
                                   aux_val=aux_val, distill=distill)

        if backbone == 'mit_b2':  # 3, 4, 6, 3
            aux_depth = [1, 2, 3, 1] if aux_val else None
            self.backbone = mit_b2(crop_size=img_size, num_classes=num_classes, aux_depth=aux_depth)
            self.decoder = Decoder(num_classes=num_classes, embed_dims=self.backbone.embed_dims, dilation=dilations,
                                   insert_attn=insert_attn, insert_aspp=insert_aspp, insert_fusion=insert_fusion,
                                   aux_val=aux_val, distill=distill)

        if backbone == 'mit_b3':  # 3, 4, 18, 3
            aux_depth = [1, 2, 9, 1] if aux_val else None
            self.backbone = mit_b3(crop_size=img_size, num_classes=num_classes, aux_depth=aux_depth)
            self.decoder = Decoder(num_classes=num_classes, embed_dims=self.backbone.embed_dims[:-1],
                                   dilation=dilations,
                                   multiple=[4, 2, 2], insert_attn=insert_attn, insert_aspp=insert_aspp,
                                   insert_fusion=insert_fusion, aux_val=aux_val, distill=distill)

        if backbone == 'mit_b4':  # 3, 8, 27, 3
            aux_depth = [1, 4, 13, 1] if aux_val else None
            self.backbone = mit_b4(crop_size=img_size, num_classes=num_classes, aux_depth=aux_depth)
            self.decoder = Decoder(num_classes=num_classes, embed_dims=self.backbone.embed_dims[:-1],
                                   dilation=dilations,
                                   multiple=[4, 2, 2], insert_attn=insert_attn, insert_aspp=insert_aspp,
                                   insert_fusion=insert_fusion, aux_val=aux_val, distill=distill)

        if backbone == 'mit_b5':  # 3, 6, 40, 3
            aux_depth = [1, 3, 20, 1] if aux_val else None
            self.backbone = mit_b5(crop_size=img_size, num_classes=num_classes, aux_depth=aux_depth)
            self.decoder = Decoder(num_classes=num_classes, embed_dims=self.backbone.embed_dims[:-1],
                                   dilation=dilations,
                                   multiple=[4, 2, 2], insert_attn=insert_attn, insert_aspp=insert_aspp,
                                   insert_fusion=insert_fusion, aux_val=aux_val, distill=distill)

    def forward(self, x, visual=False):
        if self.backbone_type in ['mit_b3', 'mit_b4', 'mit_b5']:
            if self.aux_val:
                feature_list, aux_list = self.backbone(x, part_feature=True)
            else:
                feature_list = self.backbone(x, part_feature=True)
        else:
            if self.aux_val:
                feature_list, aux_list = self.backbone(x)
            else:
                feature_list = self.backbone(x)

        if visual:
            for i in range(len(feature_list)):
                visual_feature(feature_list[i], 'feature depth_{}'.format(i + 1))

        if self.dil:
            out, dil_fea = self.decoder(feature_list, visual=visual)
            if self.aux_val:
                aux_out, _ = self.decoder(aux_list, visual=visual)
                return out, dil_fea, aux_out
            return out, dil_fea
        else:
            out = self.decoder(feature_list, visual=visual)
            if self.aux_val:
                aux_out = self.decoder(aux_list, visual=visual)
                return out, aux_out
            return out
'''

def visual_feature(x, name, visual_num=16, save_path='../feature_maps'):
    b, c, h, w = x.shape
    if visual_num > c:
        print(
            termcolor.colored('ERROR : The number of visual features is greater than the number of channels...', 'red'))
        visual_num = c
    index = np.random.randint(0, c, visual_num)
    index = list(np.unique(index))
    for i in range(len(index)):
        feature = x[0, index[i], :, :]
        plt.subplot(4, 4, i + 1)
        plt.imshow(feature.cpu().detach().numpy(), 'gray')
        plt.axis('off')

    save_path = "{}/{}".format(save_path, 'dil_unet_original')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if '{}.png'.format(name) in os.listdir(save_path):
        num = 1
        while '{}_{}.png'.format(name, num) in os.listdir(save_path):
            num = num + 1
        name = '{}_{}'.format(name, num)

    plt.title(name)
    plt.savefig('{}/{}.png'.format(save_path, name))
    # plt.show()
    plt.close()


def empty_feature(name, save_path='../feature_maps'):
    path = '{}/{}'.format(save_path, name)
    if os.path.exists(path):
        for file in os.listdir(path):
            os.remove('{}/{}'.format(path, file))
        print(termcolor.colored('del all feature maps {}'.format(len(os.listdir(path))), 'green'))


if __name__ == '__main__':
    device = torch.device('cuda:0')
    crop_size = 512
    num_classes = 19
    backbone = 'mit_b1'
    decoder = 'dil_unet_original'
    # load_data='miou_0.7025455664134682.pth.tar'
    load_data = 'miou_0.7021762563249968.pth.tar'
    # dilations = None
    # up_model : 'bilinear','nearest','mul'==>torch.mul(x,pool)
    dilations = {'dilation': [6], 'kernel_size': [3],
                 'use_pooling': True, 'up_mode': 'mul',
                 'fusion_conv': False}  # aspp中dilation为[6,12,18]  # 从后向前使用膨胀率  低分辨率-->高分辨率
    insert_aspp = True
    insert_fusion = 'FFM'  # option: 'norm' ==> fusion block ,'FFM' ==> bisenet的FFM ,None
    insert_attn = False
    finetune = True

    input = Image.open('H:/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png').convert('RGB')
    train_transform = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input = train_transform(input).unsqueeze(0)

    model = Segformer_Unet(img_size=crop_size, num_classes=num_classes, backbone=backbone, insert_attn=insert_attn,
                           dilations=dilations, insert_aspp=insert_aspp, insert_fusion=insert_fusion)

    load_data = torch.load('../res/mit_b1/{}/{}'.format(decoder, load_data), map_location=device)
    pretrain_model = load_data['model']
    if finetune:
        state = model.state_dict()
        for key in pretrain_model.keys():
            if key not in state.keys():
                continue
            if pretrain_model[key].shape != state[key].shape:
                print(termcolor.colored(f"Warning: Removing key {key} from pretrained checkpoint", 'yellow'))
                pretrain_model[key] = state[key]
    model.load_state_dict(pretrain_model, strict=False)

    empty_feature(decoder)
    model, input = model.to(device), input.to(device)

    model(input, True)
