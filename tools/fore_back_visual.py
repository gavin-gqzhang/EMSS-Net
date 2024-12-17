from argparse import ArgumentParser

import numpy as np
import scipy.misc
import os

import xlrd

from tools.load_data import xml_to_region

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2, 63).__str__()
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from glob import glob
from mmseg.core import eval_metrics
import cv2
from terminaltables import AsciiTable
from mmcv.utils import print_log

import openslide
from openslide import OpenSlideError

PALETTE = [[255,255,255],[128, 138, 135], [135,206,235], [255,128,0], [135,38,87], [176,23,31]]

def main_test_slide():
    parser = ArgumentParser()
    parser.add_argument('--imgpath', default='/media/ubuntu/Seagate Basic/data_6/img_dir/test', help='Image path')
    parser.add_argument('--config', default='../local_configs/pathformer/B5/pathformer.b5.1024x1024.cancerseg.160k.py',
                        help='Config file')
    parser.add_argument('--checkpoint',
                        default='/media/ubuntu/Seagate Basic/work_dirs/data_v3/classes_6/SegFormer_mul_cls_head/latest.pth',
                        help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cancerseg',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    """
    read_xls = xlrd.open_workbook('/media/ubuntu/Seagate Basic1/data-v3/test-train-validation-list.xlsx')
    xls_context = read_xls.sheets()[0]

    test_lists = xls_context.col_values(colx=0, start_rowx=1)
    test_lists = list(filter(None, test_lists))
    assert len(test_lists) == 30, 'Data extraction failed , some data is lost'

    files = glob('/media/ubuntu/Seagate Basic1/data-v3/*.svs') + glob('/media/ubuntu/Seagate Basic1/data-v3/*.tif')

    test_files = []
    for file in files:
        file_name = file.split('/')[-1].split('.')[0] + '.xml'
        if file_name in test_lists:
            test_files.append(file)

    if not os.path.exists(f'{args.imgpath}/gt/'):
        os.makedirs(f'{args.imgpath}/gt/')
    if not  os.path.exists(f'{args.imgpath}/img'):
        os.makedirs(f'{args.imgpath}/img')
    if not  os.path.exists(f'{args.imgpath}/pre_gt'):
        os.makedirs(f'{args.imgpath}/pre_gt')


    process_test_data(test_files,args.imgpath)

    """

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    # print(model)

    data_all = glob(args.imgpath + '/*.png')
    for data in data_all:
        print(f'read and process img : {data} ')
        image_path = data
        result = inference_segmentor(model, image_path)
        result = result[0]
        print(result.shape, np.unique(np.array(result)))  # (2048,2048)
        # print(result)
        img = np.array(cv2.imread(data))
        # rgb_res=np.zeros((result[0],result[1],3))
        rgb_res = np.tile(np.array([255], dtype=np.uint8), (result.shape[0], result.shape[1], 3))
        for x in range(result.shape[0]):
            for y in range(result.shape[1]):
                rgb_res[x, y, :] = PALETTE[result[x, y]]
                if result[x, y] != 0:
                    img[x, y, :] = PALETTE[result[x, y]]
        os.makedirs(args.imgpath + '/fore_back_pre', exist_ok=True)

        cv2.imwrite(f'{args.imgpath}/fore_back_pre/{data.split("/")[-1].split(".png")[0]}_fore_in_ori.png', img)
        # cv2.imwrite(image_path.replace('/img/','/pre_rgb/'),rgb_res)

        # final_results = np.array(result * 255).astype('uint8')
        final_results = np.array(result).astype('uint8')
        cv2.imwrite(f'{args.imgpath}/fore_back_pre/{data.split("/")[-1].split(".png")[0]}_mask.png', final_results)

if __name__ == '__main__':
    main_test_slide()