from PIL import Image as im
im.MAX_IMAGE_PIXELS=None
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=pow(2,63).__str__()
import random
import openslide
from openslide import  OpenSlideError
from libtiff import TIFF
import cv2
import numpy as np
import scipy.misc
from glob import glob

def open_slide(filename):
    try:
        slide = openslide.open_slide(filename)
    except OpenSlideError:
        slide = None
    except FileNotFoundError:
        slide = None
    return slide

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.uint8)
    else:
        return scipy.misc.imread(path).astype(np.uint8)
# 40X level=0, 20X level=1, 10X level=2
def svsread(path,level):
    slide = open_slide(path)
    image = slide.read_region((0,0),level,slide.level_dimensions[level])
    image = np.array(image.convert("RGB"))
    slide.close()
    return image

def image_crop(image,gt,size,scale,name,type='train'):
    if scale =='40X':
        level = 0
        slide = open_slide(image)
        w,h = slide.level_dimensions[level]
        print(w,h)
        count = 0
        image_path = '/home/ubuntu/ssd/data/20211206/seg/HNSC/pyramid_slide/40X/images/{}/image/'.format(type)
        gt_path = '/home/ubuntu/ssd/data/20211206/seg/HNSC/pyramid_slide/40X/labels/{}/image/'.format(type)
        gt = scipy.misc.imresize(gt, [h, w])
        for i_h in range(int(h / size)):
            for i_w in range(int(w / size)):
                x = size * i_w
                y = size * i_h
                patch_image = slide.read_region((x, y), level, [size,size])
                patch_image = np.array(patch_image.convert("RGB"))
                gt_image = gt[y:y + size, x:x + size]
                num = np.shape(np.where(patch_image[:, :, 0] < 220))
                if type == 'val':
                    num1 = np.shape(np.where(gt_image > 100))
                else:
                    num1 = [2, 102400]
                if num[1] / size / size > 0.05 and num1[1] > 1000:
                    scipy.misc.imsave(image_path + name.split('.svs')[0] + '_{}_{:06d}.png'.format(scale, count),
                                      patch_image)
                    scipy.misc.imsave(gt_path + name.split('.svs')[0] + '_{}_{:06d}.png'.format(scale, count), gt_image)
                    count = count + 1
            #     if count > 10:
            #         break
            # if count >  10:
            #     break
        slide.close()
    else:
        h,w,c = image.shape
        count = 0
        image_path = '/home/ubuntu/ssd/data/20211206/seg/HNSC/pyramid_slide/20X/images/{}/image/'.format(type)
        gt_path = '/home/ubuntu/ssd/data/20211206/seg/HNSC/pyramid_slide/20X/labels/{}/image/'.format(type)
        for i_h in range(int(h / size)):
            for i_w in range(int(w / size)):
                y = size * i_h
                x = size * i_w
                patch_image = image[y:y + size, x:x + size, :]
                gt_image = gt[y:y + size, x:x + size]
                num = np.shape(np.where(patch_image[:, :, 0] < 220))
                if type == 'val':
                    num1 = np.shape(np.where(gt_image > 100))
                else:
                    num1 = [2,102400]
                if num[1] / size / size > 0.05 and num1[1] > 1000:
                    scipy.misc.imsave(image_path+name.split('.svs')[0]+'_{}_{:06d}.png'.format(scale,count),patch_image)
                    scipy.misc.imsave(gt_path + name.split('.svs')[0] + '_{}_{:06d}.png'.format(scale, count),gt_image)
                    count = count + 1

def main():
    type = 'train'
    size = 2048
    train_svs_path = '/home/ubuntu/ssd/data/20211206/seg/HNSC/'
    scales = [8192]
    count = 1
    with open(train_svs_path+'{}_new.txt'.format(type), 'r') as f:
        for line_all in f.readlines():
            line = line_all.split(' ')[0].strip()
            print('{}:{}'.format(count,line))
            count = count + 1
            # print(train_svs_path + line.replace('.svs', '.png').replace('svs_first','gt_first').replace('svs_second','gt_second'))
            gt = imread(train_svs_path + line.replace('.svs', '.png').replace('svs_first','gt_first').replace('svs_second','gt_second'))
            # print('40X')
            # image_crop(train_svs_path + 'svs/' + line, gt, size=size, scale='40X', name=line, type=type)
            # print('10X')
            # slide_image = svsread(train_svs_path + 'svs/' + line, level=2)
            # h, w, c = slide_image.shape
            # gt = scipy.misc.imresize(gt, [h, w])
            # image_crop(slide_image, gt, size=size, scale='10X', name=line, type=type)
            print('20X')
            slide_image = svsread(train_svs_path + line,level=1)
            h,w,c = slide_image.shape
            gt = scipy.misc.imresize(gt,[h,w])
            image_crop(slide_image, gt, size=size, scale='20X', name=line.replace('svs_first','').replace('svs_second',''), type=type)
            for scale in scales:
                if type == 'train':
                    print(scale)
                    if scale < 1:
                        py_slide_image = scipy.misc.imresize(slide_image,scale)
                    else:
                        py_slide_image = scipy.misc.imresize(slide_image, [scale,scale])
                    h, w, c = py_slide_image.shape
                    gt = scipy.misc.imresize(gt, [h, w])
                    image_crop(py_slide_image, gt, size=size, scale=scale, name=line.replace('svs_first','').replace('svs_second',''), type=type)

if __name__ == '__main__':
    # image_to_xml()
    main()