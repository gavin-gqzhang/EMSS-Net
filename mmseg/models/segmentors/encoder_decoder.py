import glob
import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import numpy as np
from mmcv.runner import load_checkpoint

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from mmseg.utils import get_root_logger

@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoder, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        logger = get_root_logger()
        self.logger=logger

        if self.train_cfg.get('extra_aux',False):
            self.extra_loss_decode=builder.build_loss(self.train_cfg.get('extra_loss_decode',dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.1)))

        self.init_weights(pretrained=pretrained)

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad=False
        logger.info('Freeze backbone success')

        if len(self.train_cfg.get('freeze_name', ())) != 0:
            freeze_name=[]
            for layer_name,param in self.decode_head.named_parameters():
                if layer_name.split('.')[0] in self.train_cfg.get('freeze_name', ()):
                    param.requires_grad = False
                    freeze_name.append(layer_name)
                else:
                    param.requires_grad = True

            logger.info(f'Freeze decoder success, freeze layer: {freeze_name}')

        self.max_acc=0.0
        assert self.with_decode_head


    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained)
        if isinstance(pretrained, str):
            if 'mit' in pretrained:
                self.backbone.init_weights(pretrained=pretrained)
            else:
                logger = get_root_logger()
                # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
                load_ckpts=torch.load(pretrained,map_location='cpu')
                if 'backbone' in load_ckpts.keys():
                    load_backbone_res=self.backbone.load_state_dict(load_ckpts['backbone'],strict=False)
                    load_decoder_params=load_ckpts['decode_head']
                    remove_decode_infos=[key for key in load_decoder_params.keys() if 'conv_seg' in key]
                    for remove_decode_key in remove_decode_infos:
                        del load_decoder_params[remove_decode_key]
                    load_decoder_res=self.decode_head.load_state_dict(load_decoder_params,strict=False)
                    logger.info(f'load backbone pretrain checkpoint result: {load_backbone_res}, load decoder pretrain checkpoint result: {load_decoder_res}')
                else:
                    load_checkpoint(self,pretrained,map_location='cpu',strict=False,logger=logger)
                
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        try:
            x,aux_x = self.backbone(img)
            if self.with_neck:
                x = self.neck(x)
            return x,aux_x
        except:
            x=self.backbone(img)
            if self.with_neck:
                x = self.neck(x)
            return x


    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        try:
            x, aux = self.extract_feat(img)
        except:
            x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        if not isinstance(out, (list, tuple)):
            out=[out]

        new_out=[]
        for ou in out:
            ou = resize(
                input=ou,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            new_out.append(ou)
        return new_out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg,prefix_name=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                        gt_semantic_seg,
                                                        self.train_cfg)
        if isinstance(loss_decode,(list)):
            if len(loss_decode)==2:
                losses.update(add_prefix(loss_decode[0], 'dou_decode' if prefix_name is None else f"{prefix_name}_dou_decode"))
                losses.update(add_prefix(loss_decode[1], 'decode' if prefix_name is None else f"{prefix_name}_decode"))
            else:
                losses.update(add_prefix(loss_decode[-1], 'decode' if prefix_name is None else f"{prefix_name}_decode"))
        else:
            losses.update(add_prefix(loss_decode, 'decode' if prefix_name is None else f"{prefix_name}_decode"))

        if isinstance(loss_decode,(list)):
            acc_seg=loss_decode[-1]['acc_seg']
        else:
            acc_seg=loss_decode['acc_seg']
        
        if acc_seg>self.max_acc and acc_seg>92 and prefix_name is None:
            if os.path.isdir(f"{self.train_cfg.work_dir}/acc_ckpts"):
                glob_acc_files=glob.glob(f"{self.train_cfg.work_dir}/acc_ckpts/acc_*.pth")
                sorted_acc=[float(os.path.basename(file_name).split('.pth')[0].split("_")[-1]) for file_name in glob_acc_files]

                for acc_ in sorted(sorted_acc)[:-3]:
                    del_files=f"{self.train_cfg.work_dir}/acc_ckpts/acc_{acc_}.pth"
                    if os.path.exists(del_files):
                        os.remove(del_files)
            else:
                os.makedirs(f"{self.train_cfg.work_dir}/acc_ckpts",exist_ok=True)
            ckpts=dict(backbone=self.backbone.state_dict(),decode_head=self.decode_head.state_dict(),align_corners=self.align_corners,num_classes=self.num_classes)
            torch.save(ckpts,f"{self.train_cfg.work_dir}/acc_ckpts/acc_{acc_seg.item()}.pth")
            self.logger.info(f"save max accuracy checkpoint: {self.train_cfg.work_dir}/acc_ckpts/acc_{acc_seg.item()}.pth")
            self.max_acc=acc_seg.item()
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        """
        feature_blobs=[]
        def hook_feature(module,input,output):
            feature_blobs.append(output.data.cpu().numpy())

        self.decode_head._modules.get('linear_pred')[0].register_forward_hook(hook_feature)
        weight_softmax,weight_bias=None,None
        for name,param in self.decode_head.named_parameters():
            if 'linear_pred.2.weight' in name:
                weight_softmax=np.squeeze(param.cpu().data.numpy())
            if 'linear_pred.2.bias' in name:
                weight_bias=param.cpu().data.numpy()
        if weight_softmax is None and weight_bias is None:
            raise ValueError(f'without weight and bias param, decode_head modules: {self.decode_head._modules}')

        def return_cam(feature_conv,weight_softmax,weight_bias,class_idx):
            size_upsample=(256,256)
            bs,nc,h,w=feature_conv.shape
            output_cam=[]
            for idx in class_idx:
                if weight_bias is not None:
                    cam=weight_softmax[idx].dot(feature_conv.reshape(nc,h*w))+weight_bias
                else:
                    cam = weight_softmax[idx].dot(feature_conv.reshape(nc, h * w))
                cam=cam.reshape(h,w)
                cam=cam-np.min(cam)
                cam_img=cam/np.max(cam)
                cam_img=np.uint8(255*cam_img)
                output_cam.append(cv2.resize(cam_img,size_upsample))
            return output_cam

        seg_logits=self.decode_head.forward_test(x,img_metas,self.test_cfg)
        # import pdb
        # pdb.set_trace()
        try:
            cams = return_cam(feature_blobs[0], weight_softmax, weight_bias, [1, 2, 3, 4])
            for i in range(4):
                heat_map = cv2.applyColorMap(cams[i], cv2.COLORMAP_JET)

                result = heat_map * 0.3 + (x[0] * 0.5).squeeze(0).permute(1, 2, 0).cpu().data.numpy()[..., :3]
                cv2.imwrite(f'{img_metas[0]["save_path"]}/cls_{i + 1}_cam.png', result)
        except Exception as e:
            print(f'error info: {e}')
        
        """
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        # print(f'exception {seg_logits.size()}')
        # if isinstance(seg_logits, (list, tuple)):
        #     seg_logits = seg_logits[-1]

        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        try:
            x,aux = self.extract_feat(img)
        except:
            x = self.extract_feat(img)
            aux=None

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)
        
        # ********************* auxiliary decoder based on shallow features *********************
        if self.train_cfg.get('aux_decode',False) and aux:
            loss_decode = self._decode_head_forward_train(aux, img_metas,
                                                      gt_semantic_seg,prefix_name='aux')
            losses.update(loss_decode)
        # ***************************************************************************************
        
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        # num_classes = 2
        pre_lists,count_lists=[],[]
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        # preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        # count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logits = self.encode_decode(crop_img, img_meta)
                if not isinstance(crop_seg_logits,(list,type)):
                    crop_seg_logits=[crop_seg_logits]
                for idx,seg_logist in enumerate(crop_seg_logits):
                    if idx==len(pre_lists):
                        preds = img.new_zeros((batch_size, seg_logist.shape[-3], h_img, w_img))
                        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
                        pre_lists.append(preds)
                        count_lists.append(count_mat)
                    # preds += F.pad(crop_seg_logit,
                    #                (int(x1), int(preds.shape[3] - x2), int(y1),
                    #                 int(preds.shape[2] - y2)))
                    pre_lists[idx] += F.pad(seg_logist,
                                       (int(x1), int(pre_lists[idx].shape[3] - x2), int(y1),
                                        int(preds.shape[2] - y2)))
                    count_mat=count_lists[idx]
                    count_mat[:,:,y1:y2,x1:x2]+=1
                    count_lists[idx]=count_mat
        # assert (count_mat == 0).sum() == 0
        for idx in range(len(count_lists)):
            assert (count_lists[idx] == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            for idx,count_ in enumerate(count_lists):
                count_ = torch.from_numpy(
                    count_.cpu().detach().numpy()).to(device=img.device)
                count_lists[idx]=count_
        # preds = preds / count_mat
        # if rescale:
        #     preds = resize(
        #         preds,
        #         size=img_meta[0]['ori_shape'][:2],
        #         mode='bilinear',
        #         align_corners=self.align_corners,
        #         warning=False)
        for idx in range(len(pre_lists)):
            pre_lists[idx] = pre_lists[idx] / count_lists[idx]
            if rescale:
                pre_lists[idx] = resize(
                    pre_lists[idx],
                    size=img_meta[0]['ori_shape'][:2],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
        return pre_lists

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        # output = F.softmax(seg_logit, dim=1)
        if isinstance(seg_logit,(list,tuple)):
            output=[]
            for idx,logit in enumerate(seg_logit):
                pre=F.softmax(logit, dim=1)
                flip = img_meta[0]['flip']
                if flip:
                    flip_direction = img_meta[0]['flip_direction']
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        pre = pre.flip(dims=(3, ))
                    elif flip_direction == 'vertical':
                        pre = pre.flip(dims=(2,))
                output.append(pre)
            return output

        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        if isinstance(seg_logit,(list,tuple)):
            seg_pred=[]
            for idx in range(len(seg_logit)):
                pre=seg_logit[idx].argmax(dim=1)
                if torch.onnx.is_in_onnx_export():
                    # our inference backend only support 4D output
                    pre = pre.unsqueeze(0)
                    seg_pred.append(pre)
                    continue
                pre = pre.cpu().numpy()
                seg_pred.append(pre)
            # seg_pred = list(seg_pred)
            return np.array(seg_pred)

        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        # seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
