import copy
import shutil
from argparse import ArgumentParser

import numpy as np
import scipy.misc
import os
import matplotlib.pyplot as plt
import xlrd
from tqdm import tqdm

import sys
sys.path.insert(0,'/home/dell/zgq/medicine_code')
from tools.load_data import xml_to_region
from tools.evaluate_seg import get_xml_metrics
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

PALETTE = [[255, 255, 255], [0,255,0], [0,0,255], [255,255,0], [255, 0, 0]]
# white,yellow,blue,purple, red

label_mapping = {
    'Q': 0,
    'NOR': 1,
    'HYP': 2,
    'DYS': 3,
    'CAR': 4,
}


def main_test_slide(dirs='cls_5_pre'):
    parser = ArgumentParser()
    # parser.add_argument('--imgpath', default='/media/ubuntu/Seagate Basic/test_datas/img_dir/', help='Image path')
    parser.add_argument('--imgpath', default='/media/ubuntu/Seagate Basic1/new_optim_data/test_datas/img_dir', help='Image path')
    parser.add_argument('--savepath', default='/media/ubuntu/Seagate Basic/work_dirs/240506/05-16/latest.pth', help='Image path')
    parser.add_argument('--config', default='/media/ubuntu/Seagate Basic/workspaces/local_configs/pathformer/B5/pathformer.b5.1024x1024.cancerseg.160k.py',
                        help='Config file')
    parser.add_argument('--checkpoint',
                        default='/media/ubuntu/Seagate Basic/work_dirs/05-16/latest.pth',
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

    if args.savepath==None:
        args.savepath=args.imgpath

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    # print(model)

    data_all = glob(args.imgpath + '/*.png')
    print(f'all img len:{len(data_all)}')
    for data in tqdm(data_all):
        print(f'read and process img : {data} ')
        image_path = data
        result = inference_segmentor(model, image_path) # shape:(segmentor_rtn,batch_size,h,w)
        """
        nor_hyp_result = result[-2][0]
        dys_car_result = result[-1][0]

        dys_car_result[dys_car_result != 0] = dys_car_result[dys_car_result != 0] + 2

        sum_res = nor_hyp_result + dys_car_result
        res = copy.deepcopy(sum_res)
        sum_res[sum_res <= 4] = 0

        res[sum_res != 0] = nor_hyp_result[sum_res != 0]
        """
        res=result[-1][0]
        result = copy.deepcopy(res)

        # print(result)
        img = np.array(cv2.imread(data))
        ori_shape = img.shape[:2]
        if ori_shape != result.shape:
            # print(ori_shape,result.shape,result.shape[0])
            img = cv2.resize(img, result.shape)
        # img=np.array(img)
        # rgb_res=np.zeros((result[0],result[1],3))
        # rgb_res = np.tile(np.array([255], dtype=np.uint8), (result.shape[0], result.shape[1], 3))
        for x in range(result.shape[0]):
            for y in range(result.shape[1]):
                # rgb_res[x, y, :] = PALETTE[result[x, y]]
                if result[x, y] != 0:
                    # img[x, y, :] = PALETTE[result[x, y]]
                    img[x, y, :] = PALETTE[1]
        os.makedirs(args.savepath + f'/{dirs}', exist_ok=True)
        if result.shape != ori_shape:
            result = cv2.resize(result.astype("float"), ori_shape)
            img = cv2.resize(img.astype("float"), ori_shape)
        cv2.imwrite(f'{args.savepath}/{dirs}/{data.split("/")[-1].split(".png")[0]}_mask_in_ori.png', img)
        # cv2.imwrite(image_path.replace('/img/','/pre_rgb/'),rgb_res)

        # final_results = np.array(result * 255).astype('uint8')
        final_results = np.array(result).astype('uint8')
        cv2.imwrite(f'{args.savepath}/{dirs}/{data.split("/")[-1].split(".png")[0]}_mask.png', final_results)


def mask_to_rgb():
    # mask_path='/media/ubuntu/Seagate Basic/test_datas/ann_dir/'
    mask_path = '.'
    mask_lists = glob(mask_path + '/*_mask.png')
    for file in mask_lists:
        mask = np.array(cv2.imread(file))
        rgb_res = np.tile(np.array([255], dtype=np.uint8), (mask.shape[0], mask.shape[1], 3))
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                rgb_res[x, y, :] = PALETTE[mask[x, y][0]]
        cv2.imwrite(f'{file.split(".png")[0]}_rgb.png', rgb_res)


def mask_in_ori_img():
    mask_path = '/media/ubuntu/Seagate Basic/data_6/ann_dir/test'
    mask_lists = glob(mask_path + '/*_rgb.png')
    for file in mask_lists:
        mask = np.array(cv2.imread(file))
        ori_img = np.array(cv2.imread(f'{file.replace("ann_dir", "img_dir").split("_rgb.png")[0]}.png'))
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x, y, :][0] == 255 and mask[x, y, :][1] == 255 and mask[x, y, :][2] == 255:
                    mask[x, y, :] = ori_img[x, y, :]
        cv2.imwrite(file, mask)


def img_to_patch(resize_scale=4096):
    read_xls = xlrd.open_workbook('/media/ubuntu/Seagate Basic1/data-v3/test-train-validation-list.xlsx')
    xls_context = read_xls.sheets()[0]

    train_lists = []
    test_lists = []
    validation_lists = []
    for col_num in [0, 2, 4]:
        col_datas = xls_context.col_values(colx=col_num, start_rowx=0)
        if col_datas[0] == 'train':
            for data in col_datas[1:]:
                if data != '':
                    train_lists.append(str(data).replace('0F', 'OF'))
        if col_datas[0] == 'test':
            for data in col_datas[1:]:
                if data != '':
                    test_lists.append(str(data).replace('0F', 'OF'))
        if col_datas[0] == 'validation':
            for data in col_datas[1:]:
                if data != '':
                    validation_lists.append(str(data).replace('0F', 'OF'))

    files = glob('/media/ubuntu/Seagate Basic1/data-v3/*.svs') + glob('/media/ubuntu/Seagate Basic1/data-v3/*.tif')

    train_files, test_files, val_files = [], [], []
    for file in files:
        file_name = file.split('/')[-1].split('.')[0] + '.xml'
        if file_name in train_lists:
            train_files.append(file)
        if file_name in test_lists:
            test_files.append(file)
        if file_name in validation_lists:
            val_files.append(file)

    cls_1, cls_2, cls_3, cls_4 = 0, 0, 0, 0
    train_cls_1, train_cls_2, train_cls_3, train_cls_4 = 0, 0, 0, 0
    for file in train_files:
        xml_file = '/media/ubuntu/Seagate Basic1/data-v3/' + file.split('/')[-1].split('.')[0] + '.xml'
        svs_file = file
        mask = open_slide_to_mask(svs_file, xml_file, 'train', resize_scale)
        cls_1 = cls_1 + np.sum(mask == 1)
        cls_2 = cls_2 + np.sum(mask == 2)
        cls_3 = cls_3 + np.sum(mask == 3)
        cls_4 = cls_4 + np.sum(mask == 4)
    train_cls_1, train_cls_2, train_cls_3, train_cls_4 = cls_1, cls_2, cls_3, cls_4

    cls_1, cls_2, cls_3, cls_4 = 0, 0, 0, 0
    test_cls_1, test_cls_2, test_cls_3, test_cls_4 = 0, 0, 0, 0
    for file in test_files:
        xml_file = '/media/ubuntu/Seagate Basic1/data-v3/' + file.split('/')[-1].split('.')[0] + '.xml'
        svs_file = file
        mask = open_slide_to_mask(svs_file, xml_file, 'test', resize_scale)
        cls_1 = cls_1 + np.sum(mask == 1)
        cls_2 = cls_2 + np.sum(mask == 2)
        cls_3 = cls_3 + np.sum(mask == 3)
        cls_4 = cls_4 + np.sum(mask == 4)
    test_cls_1, test_cls_2, test_cls_3, test_cls_4 = cls_1, cls_2, cls_3, cls_4

    cls_1, cls_2, cls_3, cls_4 = 0, 0, 0, 0
    val_cls_1, val_cls_2, val_cls_3, val_cls_4 = 0, 0, 0, 0
    for file in val_files:
        xml_file = '/media/ubuntu/Seagate Basic1/data-v3/' + file.split('/')[-1].split('.')[0] + '.xml'
        svs_file = file
        mask = open_slide_to_mask(svs_file, xml_file, 'val', resize_scale)
        cls_1 = cls_1 + np.sum(mask == 1)
        cls_2 = cls_2 + np.sum(mask == 2)
        cls_3 = cls_3 + np.sum(mask == 3)
        cls_4 = cls_4 + np.sum(mask == 4)
    val_cls_1, val_cls_2, val_cls_3, val_cls_4 = cls_1, cls_2, cls_3, cls_4

    print(
        f'Number of pixels  NOR:{train_cls_1 + test_cls_1 + val_cls_1} HYP:{train_cls_2 + test_cls_2 + val_cls_2} DYS:{train_cls_3 + val_cls_3 + test_cls_3} CAR:{train_cls_4 + test_cls_4 + val_cls_4}')
    print(f'Train number of pixels  NOR:{train_cls_1} HYP:{train_cls_2} DYS:{train_cls_3} CAR:{train_cls_4}')
    print(f'Test number of pixels  NOR:{test_cls_1} HYP:{test_cls_2} DYS:{test_cls_3} CAR:{test_cls_4}')
    print(f'Val number of pixels  NOR:{val_cls_1} HYP:{val_cls_2} DYS:{val_cls_3} CAR:{val_cls_4}')


def open_slide_to_mask(slide_file, xml_file, split, resized=0):
    slide = openslide.open_slide(slide_file)
    img_size = slide.level_dimensions[0]

    slide_region = slide.read_region((0, 0), 0, img_size).convert('RGB')
    slide_region = np.array(slide_region)  # size1,size2,channel  ==> size2,size1,channel

    img_size = (img_size[1], img_size[0])

    slide_points = xml_to_region(xml_file)

    mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
    # mask=np.tile(np.array([255],dtype=np.uint8),(img_size[0],img_size[1]))

    ori_img = copy.deepcopy(slide_region)

    for point_dict in slide_points:
        cls = list(point_dict.keys())[0]

        points = np.asarray([point_dict[cls]], dtype=np.int32)

        cv2.fillPoly(img=mask, pts=points, color=(label_mapping[cls], label_mapping[cls], label_mapping[cls]))

        # cv2.fillPoly(img=mask, pts=points, color=label_mapping[cls])
        # cv2.fillPoly(img=ori_img, pts=points, color=label_color[cls])

    # cv2.imwrite(f'../data/label/mask.png',cv2.resize(mask,(2048,2048)))
    # cv2.imwrite(f'../data/label/original.png',cv2.resize(slide_region,(2048,2048)))
    '''
    save_path='/media/ubuntu/Seagate Basic/data_6/test_dir'
    os.makedirs(save_path+'/masks/',exist_ok=True)
    os.makedirs(save_path+'/imgs/',exist_ok=True)

    cv2.imwrite(f'{save_path}/masks/{slide_file.split("/")[-1].split(".")[0]}.png',cv2.resize(mask,(2048,2048)))
    cv2.imwrite(f'{save_path}/imgs/{slide_file.split("/")[-1].split(".")[0]}.png',cv2.resize(slide_region,(2048,2048)))
    cv2.imwrite(f'{save_path}/imgs/{slide_file.split("/")[-1].split(".")[0]}_mask.png',cv2.resize(ori_img,(2048,2048)))

    '''
    img_dir = '/media/ubuntu/Seagate Basic/data_{}_1024/img_dir/{}/{}'.format(resized, split,
                                                                              slide_file.split('/')[-1].split('.')[0])
    ann_dir = '/media/ubuntu/Seagate Basic/data_{}_1024/ann_dir/{}/{}'.format(resized, split,
                                                                              slide_file.split('/')[-1].split('.')[0])

    if not os.path.exists(f'/media/ubuntu/Seagate Basic/data_{resized}_1024/img_dir/{split}'):
        os.makedirs(f'/media/ubuntu/Seagate Basic/data_{resized}_1024/img_dir/{split}')
    if not os.path.exists(f'/media/ubuntu/Seagate Basic/data_{resized}_1024/ann_dir/{split}'):
        os.makedirs(f'/media/ubuntu/Seagate Basic/data_{resized}_1024/ann_dir/{split}')

    # patch_split(slide_region,mask,img_size,img_dir,ann_dir)
    if resized != 0:
        ratio = resized / min(img_size)
        img_size = (int(img_size[0] * ratio), int(img_size[1] * ratio))
        mask = cv2.resize(mask, img_size)
        slide_region = cv2.resize(slide_region, img_size)

    num_key, drop_win = crop_split(slide_region, mask, img_size, img_dir, ann_dir, crop_size=(1024, 1024))

    print(
        f'process {slide_file} success , resize ratio :{resized} , resize :{img_size}, classes : {len(slide_points)} , threshold: 0.05 save patch : {num_key} drop patch : {drop_win}')

    return mask


def crop_split(slide_region, mask_region, img_size, img_dir, ann_dir, crop_size=(2048, 2048)):
    start_point = (0, 0)
    # region_size=(int(img_size[0]/num_path),int(img_size[1]/num_path))
    split_x = int(img_size[0] / crop_size[0])
    split_y = int(img_size[1] / crop_size[1])
    redun_x, redun_y = False, False
    if img_size[0] % crop_size[0] != 0:
        split_x = split_x + 1
        redun_x = True
    if img_size[1] % crop_size[1] != 0:
        split_y = split_y + 1
        redun_y = True

    num_key = 0
    drop_win = 0
    for dimension_0 in range(split_x):
        for dimension_1 in range(split_y):
            img_patch = slide_region[start_point[0]:start_point[0] + crop_size[0],
                        start_point[1]:start_point[1] + crop_size[1], :]
            mask_patch = mask_region[start_point[0]:start_point[0] + crop_size[0],
                         start_point[1]:start_point[1] + crop_size[1]]

            # cv2.imwrite(f'../data/{dimension_0+dimension_1}.png',img_patch)
            # img_patch.save(f'../data/{num_key}.png')
            if np.sum(mask_patch != 0) / (crop_size[0] * crop_size[1]) > 0.05:
                cv2.imwrite(f'{img_dir}-{num_key}.png', img_patch)
                cv2.imwrite(f'{ann_dir}-{num_key}.png', mask_patch)
                num_key = num_key + 1
            else:
                drop_win = drop_win + 1
            # print(f'save {num_key}th img patch and mask patch success.... start_point:{start_point} region_size:{region_size}')

            if dimension_1 == split_y - 2 and redun_y:
                start_point = (start_point[0], img_size[1] - crop_size[1])
            else:
                start_point = (start_point[0], start_point[1] + crop_size[1])

        if dimension_0 == split_x - 2 and redun_x:
            start_point = (img_size[0] - crop_size[0], 0)
        else:
            start_point = (start_point[0] + crop_size[0], 0)

    return num_key, drop_win


def svsread(path, level):
    slide = openslide.open_slide(path)
    image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    image = np.array(image.convert("RGB"))
    slide.close()
    return image


def image_to_xml(dirs='best_pre'):
    ori_imgpath='/media/ubuntu/Seagate Basic/optim_data/test_datas/img_dir'
    imgpath = f'/media/ubuntu/Seagate Basic1/work_dirs/04-06/{dirs}'
    read_xls = xlrd.open_workbook('/media/ubuntu/Seagate Basic/v2-test-train-validation-list.xlsx')
    xls_context = read_xls.sheets()[0]

    test_lists = []
    col_datas = xls_context.col_values(colx=0, start_rowx=1)

    for data in col_datas[1:]:
        if data != '':
            test_lists.append(str(data).replace('0F', 'OF'))  # xml lists

    for f_img_name in test_lists:
        f_img_name = f_img_name.split('.')[0]  # overall data name
        svs_path = glob(f'/media/ubuntu/Seagate Basic/data-v3/{f_img_name}.svs') + glob(
            f'/media/ubuntu/Seagate Basic/data-v3/{f_img_name}.tif')
        data_paths = glob(imgpath + f'/{f_img_name}-*_mask.png')  # split data path lists
        svs = openslide.open_slide(svs_path[0])
        w, h = svs.level_dimensions[0]
        overall_mask = np.tile(np.array([255]), (h, w))
        # mask_in_ori=np.tile(np.array([255]),(h,w,3))
        row, col = 0, 0

        # ori_imgs=np.zeros((h,w,3))
        print(f'read svs paths : {svs_path[0]}  file name : {f_img_name} patch img num : {len(data_paths)}')

        for data_index in range(len(data_paths)):
            ori_img=np.array(cv2.imread(f"{ori_imgpath}/{f_img_name}-{data_index}.png"))
            patch_img = np.array(cv2.imread(imgpath + f'/{f_img_name}-{data_index}_mask.png'))  # read split data
            # patch_ori_img=np.array(cv2.imread(imgpath + f'{f_img_name}-{data_index}_mask_in_ori.png'))
            if patch_img.shape!=ori_img.shape:
                patch_img=cv2.resize(patch_img,(ori_img.shape[1],ori_img.shape[0]))
                # patch_ori_img=cv2.resize(patch_ori_img,(ori_img.shape[1],ori_img.shape[0]))
                print('resize patch img , new shape : {} ori img shape : {}'.format(patch_img.shape,ori_img.shape[:-1]))
            assert ori_img.shape==patch_img.shape,f'ori img patch shape : {ori_img.shape}  mask patch img shape : {patch_img.shape}'

            patch_size = patch_img.shape
            # print('start point :{}  crop size :{} img_size :{}'.format(f'({row},{col})',patch_size,img_size))
            if row + patch_size[0] > h:
                patch_img = cv2.resize(patch_img, (h - row,patch_size[1]))
                patch_img = np.array(patch_img)

                # patch_ori_img=cv2.resize(patch_ori_img,(h-row,patch_size[1]))
                # patch_ori_img=np.array(patch_ori_img)

                ori_img=cv2.resize(ori_img,(h-row,patch_size[1]))
                ori_img=np.array(ori_img)
                print(
                    f'img size: {(h, w)} patch size: {patch_size} new patch size: {patch_img.shape} start_row: {row} patch size row over img size row')

            if col + patch_size[1] > w:
                patch_img = cv2.resize(patch_img, (patch_size[0],w - col))
                patch_img = np.array(patch_img)

                # patch_ori_img=cv2.resize(patch_ori_img,(patch_size[0],w - col))
                # patch_ori_img=np.array(patch_ori_img)

                ori_img = cv2.resize(ori_img, ( patch_size[0],w - col))
                ori_img = np.array(ori_img)
                print(
                    f'img size: {(h, w)} patch size: {patch_size} new patch size: {patch_img.shape} start_col: {col} patch size col over img size col')

            patch_size = patch_img.shape

            overall_mask[row:row + patch_size[0], col:col + patch_size[1]] = patch_img[..., 0]
            # mask_in_ori[row:row+patch_size[0],col:col+patch_size[1]]=patch_ori_img[...]

            # ori_imgs[row:row + patch_size[0], col:col + patch_size[1]]=ori_img[...]
            if col + patch_size[1] == w:
                row = row + patch_size[0]
                col = 0
            else:
                col = col + patch_size[1]


        # cv2.imwrite(f'{f_img_name}_ori_mask.png', cv2.resize(np.array(mask_in_ori, dtype='float'), (4096, 4096)))
        # cv2.imwrite(f'{f_img_name}_mask.png', cv2.resize(np.array(overall_mask, dtype='float'), (4096, 4096)))
        # cv2.imwrite(f'{f_img_name}.png',
        #             cv2.resize(np.array(svs.read_region((0, 0), 0, (w, h)).convert('RGB')), (4096, 4096)))
        print(f'process {f_img_name} success  different label id : {np.unique(overall_mask)}')
        # '''
        # final_results = cv2.resize(overall_mask.astype('float'), (svs.level_dimensions[level][1],svs.level_dimensions[level][0]))
        results = overall_mask

        # print(final_results.shape)
        # cv2.imwrite(data.replace('/results/', '/fine_xml/').replace('.svs', '.png'), results)
        kernel = np.ones((7, 7), dtype=np.uint8)
        binary = cv2.morphologyEx((results).astype('uint8'), cv2.MORPH_OPEN, kernel, iterations=5)
        # binary = results
        # cv2.imwrite(data.replace('/results/', '/fine_xml/').replace('.svs', '.png'), (binary).astype('uint8'))
        # contours_to_xml((binary[:, :] // 128).astype('uint8'),
        #                 f'{imgpath}/xml/{f_img_name}.xml', level)
        # contours_to_xml((binary[:, :]).astype('uint8'),
        #                 f'{imgpath}/xml/{f_img_name}.xml', level)
        os.makedirs(f'{imgpath}/no_cover_xml/',exist_ok=True)
        get_contour_to_xml((binary[:, :]).astype('uint8'),
                           f'{imgpath}/no_cover_xml/{f_img_name}.xml', )
        # os.makedirs(f'{imgpath}/sec_xml/',exist_ok=True)
        # for_back_xml((binary[:, :]).astype('uint8'),
        #                    f'{imgpath}/sec_xml/{f_img_name}.xml', )
        # break
        # '''


def plot_cut(binary, axis):
    for i in range(len(axis)):
        x0 = axis[i][0]
        y0 = axis[i][1]
        x1 = axis[i][2]
        y1 = axis[i][3]
        seg = 20
        if x1 > x0:
            k = (y1 - y0) * 1.0 / (x1 - x0) * 1.0
            for x in range(x0 - seg, x1 + seg):
                y = int(k * (x - x0) + y0)
                binary[y - seg:y + seg, x - seg:x + seg] = 0
        elif x1 < x0:
            k = (y1 - y0) * 1.0 / (x1 - x0) * 1.0
            for x in range(x1 - seg, x0 + seg):
                y = int(k * (x - x0) + y0)
                binary[y - seg:y + seg, x - seg:x + seg] = 0
    return binary


def decrease_axis(con):
    ax = con[0][0][0]
    ay = con[0][0][1]
    thresh = 100
    con1 = []
    for j in range(len(con)):
        if j == 0:
            con1.append([ax, ay])
        elif j < len(con) - 1:
            if np.abs(ax - con[j][0][0]) > thresh or np.abs(ay - con[j][0][1]) > thresh:
                ax = con[j][0][0]
                ay = con[j][0][1]
                con1.append([ax, ay])
        else:
            con1.append([con[j][0][0], con[j][0][1]])
    con1.append([con[0][0][0] - 1, con[0][0][1] - 1])
    return con1


def nn_dis(con1, con2):
    dis0 = 100000000000000
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0
    con1 = decrease_axis(con1)
    con2 = decrease_axis(con2)
    for i in range(len(con1)):
        for j in range(len(con2)):
            dis = np.sqrt(np.power(con1[i][0] - con2[j][0], 2) + np.power(con1[i][1] - con2[j][1], 2))
            if dis < dis0:
                dis0 = dis
                x0 = con1[i][0]
                y0 = con1[i][1]
                x1 = con2[j][0]
                y1 = con2[j][1]
    return x0, y0, x1, y1


def find_nn_axis(contours, hierarchy, binary):
    thresh = 10000
    axis = []
    for i in range(len(contours)):
        con1 = contours[i]
        if cv2.contourArea(contours[i]) > thresh:
            if hierarchy[0][i][3] >= 0:
                con2 = contours[hierarchy[0][i][3]]
                x0, y0, x1, y1 = nn_dis(con1, con2)
                # print(x0,y0,x1,y1)
                axis.append([x0, y0, x1, y1])
    if len(axis) > 0:
        binary = plot_cut(binary, axis)
    return binary


def contours_to_xml(binary, xml_name, level, scale=4):
    # if level == 1:
    #     h, w = binary.shape
    #     binary = cv2.resize(np.array(binary * 255).astype('uint8'), (w * scale, h * scale))
    # if level == 2:
    #     h, w = binary.shape
    #     binary = cv2.resize(np.array(binary * 255).astype('uint8'), (w * scale * scale, h * scale * scale))
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    binary = find_nn_axis(contours, hierarchy, binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    file = open(xml_name, 'w')
    file.write('<?xml version=\"1.0\"?>\n')
    file.write('<ASAP_Annotations>\n')
    file.write('\t<Annotations>\n')
    thresh = 10000
    ann_count = 0
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > thresh:
            file.write(
                '\t\t<Annotation Name=\"Annotation {:d}\" Type=\"Spline\" PartOfGroup=\"None\" Color=\"#F4FA58\">\n'.format(
                    ann_count))
            file.write('\t\t\t<Coordinates>\n')
            ann_count += 1
            ax = contours[i][0][0][0]
            ay = contours[i][0][0][1]
            count = 0
            thresh1 = 100
            for j in range(len(contours[i])):
                if j == 0:
                    file.write(
                        '\t\t\t\t<Coordinate Order=\"{:d}\" X=\"{:.2f}\" Y=\"{:.2f}\" />\n'.format(count, ax, ay))
                    count = count + 1
                elif j < len(contours[i]) - 1:
                    if np.abs(ax - contours[i][j][0][0]) > thresh1 or np.abs(ay - contours[i][j][0][1]) > thresh1:
                        ax = contours[i][j][0][0]
                        ay = contours[i][j][0][1]
                        file.write(
                            '\t\t\t\t<Coordinate Order=\"{:d}\" X=\"{:.2f}\" Y=\"{:.2f}\" />\n'.format(count, ax, ay))
                        count = count + 1
                else:
                    file.write(
                        '\t\t\t\t<Coordinate Order=\"{:d}\" X=\"{:.2f}\" Y=\"{:.2f}\" />\n'.format(count,
                                                                                                   contours[i][j][0][0],
                                                                                                   contours[i][j][0][
                                                                                                       1]))
                    count = count + 1
                # print(contours[i][j][0])
            file.write('\t\t\t\t<Coordinate Order=\"{:d}\" X=\"{:.2f}\" Y=\"{:.2f}\" />\n'.format(count,
                                                                                                  contours[i][0][0][
                                                                                                      0] - 1,
                                                                                                  contours[i][0][0][
                                                                                                      1] - 1))
            file.write('\t\t</Coordinates>\n')
            file.write('\t\t</Annotation>\n')
    file.write('\t</Annotations>\n')
    file.write('\t<AnnotationGroups />\n')
    file.write('</ASAP_Annotations>')
    file.close()


def for_back_xml(binary,path):
    binary[binary!=0]=1
    file = open(path, 'w')
    file.write('<?xml version=\"1.0\"?>\n')
    file.write('<ASAP_Annotations>\n')
    file.write('\t<Annotations>\n')
    thresh = 15000
    ann_count = 0
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    binary = find_nn_axis(contours, hierarchy, binary)
    # print(binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thread_1 = 25
    # thresh=thresh/100
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > thresh:
            count_list = []
            ax = contours[i][0][0][0]
            ay = contours[i][0][0][1]
            for j in range(len(contours[i])):
                if j == 0:
                    # ax, ay = (ax * mask.shape[0]) / new_shape[0], (ay * mask.shape[1]) / new_shape[1]
                    count_list.append([ax, ay])
                elif j < len(contours[i]) - 1:
                    if np.abs(ax - contours[i][j][0][0]) > thread_1 or np.abs(ay - contours[i][j][0][1]) > thread_1:
                        ax = contours[i][j][0][0]
                        ay = contours[i][j][0][1]
                        # ax, ay = (ax * mask.shape[0]) / new_shape[0], (ay * mask.shape[1]) / new_shape[1]
                        count_list.append([ax, ay])
                else:
                    ax = contours[i][j][0][0]
                    ay = contours[i][j][0][1]
                    # ax, ay = (ax * mask.shape[0]) / new_shape[0], (ay * mask.shape[1]) / new_shape[1]
                    count_list.append([ax, ay])
                # print(contours[i][j][0])
            ax = contours[i][0][0][0] - 1
            ay = contours[i][0][0][1] - 1
            # ax, ay = (ax * mask.shape[0]) / new_shape[0], (ay * mask.shape[1]) / new_shape[1]
            count_list.append([ax, ay])

            count = 0
            for index in range(len(count_list)):
                [ax, ay] = count_list[index]
                if index == 0:
                    file.write(
                        '\t\t<Annotation Name=\"Annotation {:d}\" Type=\"Spline\" PartOfGroup=\"{}\" Color=\"{}\">\n'.format(
                            ann_count, 'foreground', "#64FE2E"))
                    file.write('\t\t\t<Coordinates>\n')
                    ann_count = ann_count + 1

                file.write(
                    '\t\t\t\t<Coordinate Order=\"{:d}\" X=\"{:.2f}\" Y=\"{:.2f}\" />\n'.format(count, ax, ay))

                count = count + 1
                if index == len(count_list) - 1:
                    file.write('\t\t</Coordinates>\n')
                    file.write('\t\t</Annotation>\n')
    file.write('\t</Annotations>\n')
    file.write('\t<AnnotationGroups>\n')
    file.write('\t\t<Group Name="foreground" PartOfGroup="None" Color="#64FE2E">\n')
    file.write('\t\t\t<Attributes/>\n')
    file.write('\t\t</Group>\n')
    file.write('\t</AnnotationGroups>\n')
    file.write('</ASAP_Annotations>')
    file.close()


CLASSES = ('Q', 'NOR', 'HYP', 'DYS', 'CAR')

def get_contour_to_xml(mask, save_path,pre_type,thresh=15000):

    # cv2.drawContours(image=svs_copy,contours=contours,contourIdx=-1,color=(255,0,0))
    # cv2.imwrite(f"{save_path.split('.xml')[0]}_mask_in_svs.png",svs_copy)

    file = open(save_path, 'w')
    file.write('<?xml version=\"1.0\"?>\n')
    file.write('<ASAP_Annotations>\n')
    file.write('\t<Annotations>\n')
    thresh = thresh
    # thresh = 0
    ann_count = 0
    if pre_type=='double_pre':
        nor_mask=copy.copy(mask)
        nor_mask[nor_mask!=1]=0
        ann_count=get_counter(nor_mask,file,thresh,ann_count,'NOR','#64FE2E')
    else:
        nor_mask=copy.copy(mask)
        nor_mask[nor_mask!=1]=0
        hyp_mask=copy.copy(mask)
        hyp_mask[hyp_mask!=2]=0
        dys_mask=copy.copy(mask)
        dys_mask[dys_mask!=3]=0
        car_mask=copy.copy(mask)
        car_mask[car_mask!=4]=0

        ann_count=get_counter(nor_mask,file,thresh,ann_count,'NOR','#64FE2E')
        ann_count=get_counter(hyp_mask,file,thresh,ann_count,'HYP','#0000ff')
        ann_count=get_counter(dys_mask,file,thresh,ann_count,'DYS','#ffff00')
        ann_count=get_counter(car_mask,file,thresh,ann_count,'CAR','#ff0000')


    file.write('\t</Annotations>\n')
    file.write('\t<AnnotationGroups>\n')
    file.write('\t\t<Group Name="NOR" PartOfGroup="None" Color="#64FE2E">\n')
    file.write('\t\t\t<Attributes/>\n')
    file.write('\t\t</Group>\n')
    file.write('\t\t<Group Name="HYP" PartOfGroup="None" Color="#0000ff">\n')
    file.write('\t\t\t<Attributes/>\n')
    file.write('\t\t</Group>\n')
    file.write('\t\t<Group Name="DYS" PartOfGroup="None" Color="#ffff00">\n')
    file.write('\t\t\t<Attributes/>\n')
    file.write('\t\t</Group>\n')
    file.write('\t\t<Group Name="CAR" PartOfGroup="None" Color="#ff0000">\n')
    file.write('\t\t\t<Attributes/>\n')
    file.write('\t\t</Group>\n')
    file.write('\t</AnnotationGroups>\n')
    file.write('</ASAP_Annotations>')
    file.close()


def get_counter(mask,xml_file,thresh,ann_count,name,color):
    # new_shape=(int(mask.shape[0]/10),int(mask.shape[1]/10))
    # resized_mask=cv2.resize(mask,new_shape)
    resized_mask=mask
    contours, hierarchy = cv2.findContours(resized_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    binary = find_nn_axis(contours, hierarchy, resized_mask)
    # print(binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thread_1=30
    # thresh=thresh/100
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > thresh:
            count_list = []
            ax = contours[i][0][0][0]
            ay = contours[i][0][0][1]
            for j in range(len(contours[i])):
                if j==0:
                    # ax, ay = (ax * mask.shape[0]) / new_shape[0], (ay * mask.shape[1]) / new_shape[1]
                    count_list.append([ax, ay])
                elif j<len(contours[i])-1:
                    if np.abs(ax-contours[i][j][0][0])>thread_1 or np.abs(ay-contours[i][j][0][1])>thread_1:
                        ax=contours[i][j][0][0]
                        ay=contours[i][j][0][1]
                        # ax, ay = (ax * mask.shape[0]) / new_shape[0], (ay * mask.shape[1]) / new_shape[1]
                        count_list.append([ax, ay])
                else:
                    ax = contours[i][j][0][0]
                    ay = contours[i][j][0][1]
                    # ax, ay = (ax * mask.shape[0]) / new_shape[0], (ay * mask.shape[1]) / new_shape[1]
                    count_list.append([ax, ay])
                # print(contours[i][j][0])
            ax = contours[i][0][0][0] - 1
            ay = contours[i][0][0][1] - 1
            # ax, ay = (ax * mask.shape[0]) / new_shape[0], (ay * mask.shape[1]) / new_shape[1]
            count_list.append([ax,ay])

            count = 0
            for index in range(len(count_list)):
                [ax, ay] = count_list[index]
                if index == 0:
                    xml_file.write(
                        '\t\t<Annotation Name=\"Annotation {:d}\" Type=\"Spline\" PartOfGroup=\"{}\" Color=\"{}\">\n'.format(
                            ann_count,name,color))
                    xml_file.write('\t\t\t<Coordinates>\n')
                    ann_count = ann_count + 1

                xml_file.write(
                    '\t\t\t\t<Coordinate Order=\"{:d}\" X=\"{:.2f}\" Y=\"{:.2f}\" />\n'.format(count, ax, ay))

                count = count + 1
                if index == len(count_list) - 1:
                    xml_file.write('\t\t</Coordinates>\n')
                    xml_file.write('\t\t</Annotation>\n')

    return ann_count


def debug_xml(config,save_path,checkpoint,pre_type,res_idx=[-1],thresh=25000):
    # checkpoint='/media/ubuntu/Seagate Basic/work_dirs/04-06/pretrain_model.pth'
    if "acc" in checkpoint and not os.path.exists(f'{os.path.dirname(checkpoint)}/check_{os.path.basename(checkpoint)}'):
        import torch,pdb
        load_ckpts=torch.load(checkpoint,map_location='cpu')
        state_dict=dict()
        for key in load_ckpts.keys():
            if not isinstance(load_ckpts[key],dict):
                state_dict[key]=load_ckpts[key]
                continue
            for sub_key in load_ckpts[key].keys():
                state_dict[f'{key}.{sub_key}']=load_ckpts[key][sub_key]
        torch.save(state_dict,f'{os.path.dirname(checkpoint)}/check_{os.path.basename(checkpoint)}')
        checkpoint=f'{os.path.dirname(checkpoint)}/check_{os.path.basename(checkpoint)}'
    elif os.path.exists(f'{os.path.dirname(checkpoint)}/check_{os.path.basename(checkpoint)}'):
        checkpoint=f'{os.path.dirname(checkpoint)}/check_{os.path.basename(checkpoint)}'
    
    model = init_segmentor(config, checkpoint, device='cuda:0')

    read_xls = xlrd.open_workbook('/data/sdc/medicine_svs/v3-test-train-validation-list.xlsx')
    xls_context = read_xls.sheets()[0]

    xml_list = []
    col_datas = xls_context.col_values(colx=0, start_rowx=1)

    for data in col_datas[1:]:
        if data != '':
            xml_list.append(str(data).replace('0F', 'OF'))  # xml lists

    un_analy_xml=[]
    res_pixel_dict=dict()

    # xml_list=['OF-73-E.xml']
    import random
    # for file in tqdm(random.sample(xml_list,10)):
    for file in xml_list[:len(xml_list)//2]:
        svs_file=(glob('/data/sdc/medicine_svs/data-v3/'+file.split('.')[0]+'.svs')+glob('/data/sdc/medicine_svs/data-v3/'+file.split('.')[0]+'.tif'))[0]
        print(f'read file : {svs_file}')

        # read original silde file ----------------------
        slide = openslide.open_slide(svs_file)
        img_size = slide.level_dimensions[0]
        # try:
        slide_region = slide.read_region((0, 0), 0, img_size).convert('RGB')

        slide_region = np.array(slide_region)  # size1,size2,channel  ==> size2,size1,chann

        img_size = (img_size[1], img_size[0])

        mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        mask_lists = crop_pre(model, slide_region, mask, img_size,res_idx=res_idx)

        """
        for idx,this_mask in enumerate(mask_lists):
            this_mask=this_mask.astype('int32')
            this_slide_region = copy.deepcopy(slide_region)
            for row in range(this_mask.shape[0]):
                for col in range(this_mask.shape[1]):
                    this_slide_region[row, col] = PALETTE[this_mask[row, col]]

            cv2.imwrite(f'{save_path}/no_cover_xml/{file.split(".")[0]}_{idx}.png',
                        cv2.resize(this_slide_region, (4096, 4096)))

            print(f'save {idx} image ......')
        """
        
        if pre_type=="double_pre":
            mask=mask_lists[0]
        else:
            fg_mask,multi_mask=mask_lists
            mask=fg_mask*multi_mask
        
        img_ = cv2.resize(copy.deepcopy(slide_region), (1024, 1024))
        pre_ = cv2.resize(copy.deepcopy(mask), (1024, 1024))

        img_[pre_==1]=(0,255,0)   # GREEN
        img_[pre_==2]=(255,255,0)   # YELLOW
        img_[pre_==3]=(0,0,255)   # BLUE
        img_[pre_==4]=(255, 0, 0)   # RED
        
        plt.imshow(img_)
        os.makedirs(f'{save_path}/{pre_type}/{thresh}',exist_ok=True)
        plt.savefig(f'{save_path}/{pre_type}/{thresh}/{file.split(".")[0]}.png')
        

        print(f'Pre Mask shape : {mask.shape}, this mask classes number: {np.unique(mask)}, start generate xml file ......')

        assert slide_region.shape[:2]==mask.shape[:2], print(f"slide region shape: {slide_region.shape} mask shape: {mask.shape}")

        kernel = np.ones((7, 7), dtype=np.uint8)
        binary = cv2.morphologyEx((mask).astype('uint8'), cv2.MORPH_OPEN, kernel, iterations=5)
        os.makedirs(f'{save_path}/{pre_type}/{thresh}/',exist_ok=True)
        get_contour_to_xml((binary[:, :]).astype('uint8'),
                           f'{save_path}/{pre_type}/{thresh}/{file}',pre_type,thresh=thresh)

        print(f'generate xml file success, save file : {save_path}/{pre_type}/{thresh}/{file}')

    print(f'Un analysis file : {un_analy_xml}')

    for i in res_pixel_dict.keys():
        print(f'cls {i} pixel number : {res_pixel_dict[i]}')

    return save_path

def crop_pre(model,slide_region,mask, img_size,res_idx=[-1], crop_size=(1024,1024)):

    # region_size=(int(img_size[0]/num_path),int(img_size[1]/num_path))
    split_x = int(img_size[0] / crop_size[0])
    split_y = int(img_size[1] / crop_size[1])
    if img_size[0] % crop_size[0] != 0:
        split_x = split_x + 1
    if img_size[1] % crop_size[1] != 0:
        split_y = split_y + 1

    rtn_mask_list=[]
    for res_i in res_idx:
        rtn_mask=copy.deepcopy(mask)
        start_point = (0, 0)
        for dimension_0 in range(split_x):
            for dimension_1 in range(split_y):
                if dimension_1 == split_y - 1 and dimension_0 != split_x - 1:
                    img_patch = slide_region[start_point[0]:start_point[0] + crop_size[0],
                                start_point[1]:, :]
                    result = inference_segmentor(model, img_patch)  # shape:(segmentor_rtn,batch_size,h,w)
                    result=result[res_i][0]
                    if result.shape != img_patch.shape[:-1]:
                        # print(
                        #     f'result shape : {result.shape}, img_patch shape : {img_patch.shape[:-1]} resized shape : {cv2.resize(result.astype("float"),(img_patch.shape[1],img_patch.shape[0])).shape}')
                        result = cv2.resize(result.astype('float'), (img_patch.shape[1], img_patch.shape[0]))
                    rtn_mask[start_point[0]:start_point[0] + crop_size[0],
                    start_point[1]:] = result

                elif dimension_1 == split_y - 1 and dimension_0 == split_x - 1:
                    img_patch = slide_region[start_point[0]:, start_point[1]:, :]
                    result = inference_segmentor(model, img_patch)  # shape:(segmentor_rtn,batch_size,h,w)
                    result=result[res_i][0]
                    if result.shape != img_patch.shape[:-1]:
                        # print(
                        #     f'result shape : {result.shape}, img_patch shape : {img_patch.shape} resized shape : {cv2.resize(result.astype("float"),(img_patch.shape[1],img_patch.shape[0])).shape}')
                        result = cv2.resize(result.astype('float'), (img_patch.shape[1], img_patch.shape[0]))
                    rtn_mask[start_point[0]:, start_point[1]:] = result

                elif dimension_0 == split_x - 1 and dimension_1 != split_y - 1:
                    img_patch = slide_region[start_point[0]:, start_point[1]:start_point[1] + crop_size[1], :]
                    result = inference_segmentor(model, img_patch)  # shape:(segmentor_rtn,batch_size,h,w)
                    result=result[res_i][0]
                    if result.shape != img_patch.shape[:-1]:
                        # print(
                        #     f'result shape : {result.shape}, img_patch shape : {img_patch.shape} resized shape : {cv2.resize(result.astype("float"),(img_patch.shape[1],img_patch.shape[0])).shape}')
                        result = cv2.resize(result.astype('float'), (img_patch.shape[1], img_patch.shape[0]))
                    rtn_mask[start_point[0]:, start_point[1]:start_point[1] + crop_size[1]] = result

                else:
                    img_patch = slide_region[start_point[0]:start_point[0] + crop_size[0],
                                start_point[1]:start_point[1] + crop_size[1], :]

                    result = inference_segmentor(model, img_patch)  # shape:(segmentor_rtn,batch_size,h,w)
                    result=result[res_i][0]
                    if result.shape != img_patch.shape[:-1]:
                        # print(
                        #     f'result shape : {result.shape}, img_patch shape : {img_patch.shape} resized shape : {cv2.resize(result.astype("float"),(img_patch.shape[1],img_patch.shape[0])).shape}')
                        result = cv2.resize(result.astype('float'), (img_patch.shape[1], img_patch.shape[0]))
                    rtn_mask[start_point[0]:start_point[0] + crop_size[0],
                    start_point[1]:start_point[1] + crop_size[1]] = result


                start_point = (start_point[0], start_point[1] + crop_size[1])

            start_point = (start_point[0] + crop_size[0], 0)

        rtn_mask_list.append(rtn_mask)
        del rtn_mask
    return rtn_mask_list


def gen_xml_with_bg(config,save_path,checkpoint,pre_type,res_idx=[-1]):
        # checkpoint='/media/ubuntu/Seagate Basic/work_dirs/04-06/pretrain_model.pth'
    if "acc" in checkpoint and not os.path.exists(f'{os.path.dirname(checkpoint)}/check_{os.path.basename(checkpoint)}'):
        import torch,pdb
        load_ckpts=torch.load(checkpoint,map_location='cpu')
        state_dict=dict()
        for key in load_ckpts.keys():
            if not isinstance(load_ckpts[key],dict):
                state_dict[key]=load_ckpts[key]
                continue
            for sub_key in load_ckpts[key].keys():
                state_dict[f'{key}.{sub_key}']=load_ckpts[key][sub_key]
        torch.save(state_dict,f'{os.path.dirname(checkpoint)}/check_{os.path.basename(checkpoint)}')
        checkpoint=f'{os.path.dirname(checkpoint)}/check_{os.path.basename(checkpoint)}'
    elif os.path.exists(f'{os.path.dirname(checkpoint)}/check_{os.path.basename(checkpoint)}'):
        checkpoint=f'{os.path.dirname(checkpoint)}/check_{os.path.basename(checkpoint)}'
    
    model = init_segmentor(config, checkpoint, device='cuda:0')

    read_xls = xlrd.open_workbook('/data/sdc/medicine_svs/v3-test-train-validation-list.xlsx')
    xls_context = read_xls.sheets()[0]

    xml_list = []
    col_datas = xls_context.col_values(colx=0, start_rowx=1)

    for data in col_datas[1:]:
        if data != '':
            xml_list.append(str(data).replace('0F', 'OF'))  # xml lists

    un_analy_xml=[]
    res_pixel_dict=dict()

    bg_xml_path='/data/sdc/checkpoints/medicine_res/ori_scale_segformer_bgfg/iter_160000/double_pre/40000'
    load_bg_xmls=glob(f'{bg_xml_path}/*.xml')

    import random
    # for file in tqdm(random.sample(xml_list,10)):
    for file in xml_list:
        
        bg_xml=f'{bg_xml_path}/{os.path.basename(file)}'
        if bg_xml not in load_bg_xmls:
             continue
         
        svs_file=(glob('/data/sdc/medicine_svs/data-v3/'+file.split('.')[0]+'.svs')+glob('/data/sdc/medicine_svs/data-v3/'+file.split('.')[0]+'.tif'))[0]
        print(f'read file : {svs_file}')

        # read original silde file ----------------------
        slide = openslide.open_slide(svs_file)
        img_size = slide.level_dimensions[0]
        # try:
        slide_region = slide.read_region((0, 0), 0, img_size).convert('RGB')

        slide_region = np.array(slide_region)  # size1,size2,channel  ==> size2,size1,chann

        img_size = (img_size[1], img_size[0])
        
        bg_slide_points = xml_to_region(bg_xml)

        bg_mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)

        for point_dict in bg_slide_points:
            cls=list(point_dict.keys())[0]

            points=np.asarray([point_dict[cls]],dtype=np.int32)

            cv2.fillPoly(img=bg_mask, pts=points, color=(label_mapping[cls],label_mapping[cls],label_mapping[cls]))

        mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        mask_lists = crop_pre(model, slide_region, mask, img_size,res_idx=res_idx)

        """
        for idx,this_mask in enumerate(mask_lists):
            this_mask=this_mask.astype('int32')
            this_slide_region = copy.deepcopy(slide_region)
            for row in range(this_mask.shape[0]):
                for col in range(this_mask.shape[1]):
                    this_slide_region[row, col] = PALETTE[this_mask[row, col]]

            cv2.imwrite(f'{save_path}/no_cover_xml/{file.split(".")[0]}_{idx}.png',
                        cv2.resize(this_slide_region, (4096, 4096)))

            print(f'save {idx} image ......')
        """
        
        # print(bg_mask.shape,np.unique(bg_mask))
        fg_mask,multi_mask=mask_lists
        mask=bg_mask*multi_mask

        print(f'Pre Mask shape : {mask.shape}, this mask classes number: {np.unique(mask)}, start generate xml file ......')

        assert slide_region.shape[:2]==mask.shape[:2], print(f"slide region shape: {slide_region.shape} mask shape: {mask.shape}")

        img_=copy.deepcopy(slide_region)
        bg_img=copy.deepcopy(slide_region)
        pre_=copy.deepcopy(mask)

        img_ = cv2.resize(img_, (1024, 1024))
        pre_ = cv2.resize(pre_, (1024, 1024))
        
        bg_img = cv2.resize(bg_img, (1024, 1024))
        fg_mask=cv2.resize(fg_mask,(1024,1024))

        img_[pre_==1]=(0,255,0)   # GREEN
        img_[pre_==2]=(255,255,0)   # YELLOW
        img_[pre_==3]=(0,0,255)   # BLUE
        img_[pre_==4]=(255, 0, 0)   # RED
        
        plt.imshow(img_)
        os.makedirs(f'{save_path}/{pre_type}',exist_ok=True)
        plt.savefig(f'{save_path}/{pre_type}/{file.split(".")[0]}.png')
        
        bg_img[fg_mask==1]=(0,255,0)
        plt.clf()
        plt.imshow(bg_img)
        plt.savefig(f'{save_path}/{pre_type}/{file.split(".")[0]}_bg.png')

        kernel = np.ones((7, 7), dtype=np.uint8)
        binary = cv2.morphologyEx((mask).astype('uint8'), cv2.MORPH_OPEN, kernel, iterations=5)
        os.makedirs(f'{save_path}/{pre_type}/',exist_ok=True)
        get_contour_to_xml((binary[:, :]).astype('uint8'),
                           f'{save_path}/{pre_type}/{file}',pre_type)

        print(f'generate xml file success, save file : {save_path}/{pre_type}/{file}')

    print(f'Un analysis file : {un_analy_xml}')

    for i in res_pixel_dict.keys():
        print(f'cls {i} pixel number : {res_pixel_dict[i]}')

    return save_path

if __name__ == '__main__':
    # main_test_slide(dirs='best_pre')
    # image_to_xml(dirs='new_test_cls_5_pre')
    # print('------------------------ start generate xml ------------------------')

    # """
    # using pretrain model to generate cancer segmentation images
    base_path='/data/sdc/checkpoints/medicine_res/mix_scale_segformer_multi_cls_with_attn'
    # pretrain_ckpts=glob(f'{base_path}/iter_*.pth')
    pretrain_ckpts=[f'{base_path}/iter_240000.pth']
    for pth in pretrain_ckpts:
        iters=os.path.basename(pth).split(".pth")[0]
        save_path=f'{os.path.dirname(pth)}/15000/{iters}/'
        pre_type='multi_pre'  # double_pre
        res_idx=[0,1]
        config='/home/dell/zgq/medicine_code/local_configs/pathformer/B5/pathformer.b5.1024x1024.cancerseg.160k.py'
        save_file=gen_xml_with_bg(config,save_path=save_path,checkpoint=pth,pre_type=pre_type,res_idx=res_idx)
    # """
    
    """
    all_images=glob('/data/sdc/medicine_svs/crop_data/train_datas/img_dir/*.png')
    target_path='/data/sdc/medicine_svs/multi_scale_datas/train_datas'
    for img_path in tqdm(all_images):
        base_name=os.path.basename(img_path)
        rotation=int(base_name.split('-')[-2])
        assert rotation in [0,30,60,90,120,150,180,210,240,270,300,330,360]
        
        if rotation == 0:
            label_path=f'/data/sdc/medicine_svs/crop_data/train_datas/ann_dir/{base_name}'
            os.makedirs(f'{target_path}/img_dir/',exist_ok=True)
            os.makedirs(f'{target_path}/ann_dir/',exist_ok=True)
            
            shutil.copy(img_path,f'{target_path}/img_dir/{base_name}')
            shutil.copy(label_path,f'{target_path}/ann_dir/{base_name}')
    """
    
    # for iters in ['iter_145000','iter_170000','iter_235000']:
    #     save_file=debug_xml(dirs=iters)

    # text_file=f'{save_file}/{pth.split(".")[0]}.txt'
    # get_xml_metrics(text_file,num_classes=3,xml_path=f'{save_file}/no_cover_xml')
    # print('------------- latest_pre  -------------------')
    # main_test_slide(dirs='latest_pre')
    # image_to_xml(dirs='latest_pre')
    # mask_to_rgb()
    # mask_in_ori_img()

    '''
    path='/media/ubuntu/Seagate Basic1/test_datas/ann_dir'
    img_list=['OF-8-E-64.png','OF-8-E-65.png','OF-8-E-66.png','OF-8-E-67.png','OF-8-E-83.png']
    for i in img_list:
        data_path=f'{path}/{i}'
        img=cv2.imread(data_path)
        img=np.array(img)
        labels=np.tile(np.array([255]),img.shape)
        labels[img!=0]=0
        cv2.imwrite(f'./{i}',labels)

    imgpath = '/media/ubuntu/Seagate Basic/test_datas/img_dir/segformer_pre/'
    read_xls = xlrd.open_workbook('/media/ubuntu/Seagate Basic1/data-v3/test-train-validation-list.xlsx')
    xls_context = read_xls.sheets()[0]

    test_lists = []
    col_datas = xls_context.col_values(colx=0, start_rowx=1)

    for data in col_datas[1:]:
        if data != '':
            test_lists.append(str(data).replace('0F', 'OF'))  # xml lists

    level = 1
    for f_img_name in test_lists:
        f_img_name = f_img_name.split('.')[0]  # overall data name
        svs_path = glob(f'/media/ubuntu/Seagate Basic1/data-v3/{f_img_name}.svs') + glob(
            f'/media/ubuntu/Seagate Basic1/data-v3/{f_img_name}.tif')
        data_paths = glob(imgpath + f'{f_img_name}-*_mask_in_ori.png')  # split data path lists
        svs = openslide.open_slide(svs_path[0])
        w, h = svs.level_dimensions[0]
        overall_mask = np.tile(np.array([255]), (h, w,3))
        row, col = 0, 0

        print(svs_path, data_paths)
        for data_index in range(len(data_paths)):
            patch_img = np.array(cv2.imread(imgpath + f'{f_img_name}-{data_index}_mask_in_ori.png'))  # read split data
            print('read img : '+imgpath + f'{f_img_name}-{data_index}_mask_in_ori.png')
            patch_size = patch_img.shape
            # print('start point :{}  crop size :{} img_size :{}'.format(f'({row},{col})',patch_size,img_size))
            if row + patch_size[0] > h:
                patch_img = cv2.resize(patch_img, (patch_size[1], h - row))
                patch_img = np.array(patch_img)
                print(
                    f'img size: {(h, w)} patch size: {patch_size} new patch size: {patch_img.shape} start_row: {row} patch size row over img size row')

            if col + patch_size[1] > w:
                patch_img = cv2.resize(patch_img, (w - col, patch_size[0]))
                patch_img = np.array(patch_img)
                print(
                    f'img size: {(h, w)} patch size: {patch_size} new patch size: {patch_img.shape} start_col: {col} patch size col over img size col')

            patch_size = patch_img.shape

            overall_mask[row:row + patch_size[0], col:col + patch_size[1],:] = patch_img[...]
            if col + patch_size[1] == w:
                row = row + patch_size[0]
                col = 0
            else:
                col = col + patch_size[1]

        cv2.imwrite(f'{f_img_name}_mask.png', cv2.resize(np.array(overall_mask, dtype='float'), (4096, 4096)))
        cv2.imwrite(f'{f_img_name}.png',
                    cv2.resize(np.array(svs.read_region((0, 0), 0, (w, h)).convert('RGB')), (4096, 4096)))
        print(f'process {f_img_name} success')

        # final_results = cv2.resize(overall_mask.astype('float'), (svs.level_dimensions[level][1],svs.level_dimensions[level][0]))
        results = overall_mask

        # print(final_results.shape)
        # cv2.imwrite(data.replace('/results/', '/fine_xml/').replace('.svs', '.png'), results)
        kernel = np.ones((7, 7), dtype=np.uint8)
        binary = cv2.morphologyEx((results).astype('uint8'), cv2.MORPH_OPEN, kernel, iterations=5)
        # binary = results
        # cv2.imwrite(data.replace('/results/', '/fine_xml/').replace('.svs', '.png'), (binary).astype('uint8'))
        contours_to_xml((binary[:, :] // 128).astype('uint8'),
                        f'{imgpath}/xml/{f_img_name}.xml', level)
        break
    '''

    '''
    img_dir='/media/ubuntu/Seagate Basic/data_*_1024/img_dir/*/*.png'
    ann_dir=img_dir.replace('img_dir','ann_dir')
    img_lists=glob(img_dir)+glob(ann_dir)
    print(len(img_lists))

    new_dir='/media/ubuntu/Seagate Basic/all_data'
    for i in img_lists:
        old_path=i
        split=old_path.split('/')[-2]
        img_type=old_path.split('/')[-3]
        os.makedirs(f'{new_dir}/{img_type}/{split}',exist_ok=True)
        ori_name='-'.join(old_path.split('/')[-1].split('-')[:3])
        new_path=f'{new_dir}/{img_type}/{split}/{ori_name}'
        save_num=len(glob(f'{new_path}-*.png'))
        shutil.copy(old_path,f'{new_path}-{save_num}.png')



    resize_scale=[0]
    for resize in resize_scale:
        print(f'\n------ resize : {resize} ------')
        img_to_patch(resize)
    
    slide_region = slide.read_region((0, 0), 0, img_size).convert('RGB')
    slide_region = np.array(slide_region)  # size1,size2,channel  ==> size2,size1,channel
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3)

    ax[0].imshow(slide_region)
    ax[0].set_title(img_size)
    ax[0].axis('off')

    resized_region=copy.deepcopy(slide_region)
    ax[1].imshow(cv2.resize(resized_region,(8192,8192)))
    ax[1].set_title((8192,8192))
    ax[1].axis('off')

    resized_region = copy.deepcopy(slide_region)
    ax[2].imshow(cv2.resize(resized_region, (2048, 2048)))
    ax[2].set_title((2048,2048))
    ax[2].axis('off')

    plt.show()

    '''

    """
    datas = glob('/media/ubuntu/Seagate Basic/TCGA/pre/*_mask.png')
    svs = openslide.open_slide(
        '/media/ubuntu/Seagate Basic/TCGA-CV-6962-01Z-00-DX1.7BF8A1EF-06D7-4DC1-98F3-0A845744A90B.svs')
    w, h = svs.level_dimensions[0]
    overall_mask = np.tile(np.array([255.0]), (h, w))
    # overall_mask_in_ori = np.tile(np.array([255]), (h, w,3))
    row, col = 0, 0

    for data_index in range(len(datas)):
        patch_img = np.array(
            cv2.imread(f'/media/ubuntu/Seagate Basic/TCGA/pre/{data_index}_mask.png'))  # read split data
        mask_in_ori = np.array(cv2.imread(f'/media/ubuntu/Seagate Basic/TCGA/pre/{data_index}_mask_in_ori.png'))

        patch_size = patch_img.shape
        # print('start point :{}  crop size :{} img_size :{}'.format(f'({row},{col})',patch_size,img_size))
        if row + patch_size[0] > h:
            patch_img = cv2.resize(patch_img, (patch_size[1], h - row))
            patch_img = np.array(patch_img, dtype='float')

            mask_in_ori = cv2.resize(mask_in_ori, (patch_size[1], h - row))
            mask_in_ori = np.array(mask_in_ori, dtype='float')

            print(
                f'img size: {(h, w)} patch size: {patch_size} new patch size: {patch_img.shape} start_row: {row} patch size row over img size row')

        if col + patch_size[1] > w:
            patch_img = cv2.resize(patch_img, (w - col, patch_size[0]))
            patch_img = np.array(patch_img, dtype='float')

            mask_in_ori = cv2.resize(mask_in_ori, (w - col, patch_size[0]))
            mask_in_ori = np.array(mask_in_ori, dtype='float')

            print(
                f'img size: {(h, w)} patch size: {patch_size} new patch size: {patch_img.shape} start_col: {col} patch size col over img size col')

        patch_size = patch_img.shape
        # print(patch_size,overall_mask[row:row + patch_size[0], col:col + patch_size[1]].shape,row,col,overall_mask.shape)
        if overall_mask[row:row + patch_size[0], col:col + patch_size[1]].shape != patch_size[:2]:
            overall_mask_shape = overall_mask[row:row + patch_size[0], col:col + patch_size[1]].shape
            print(overall_mask_shape, patch_img.shape)
            patch_img = cv2.resize(patch_img, (overall_mask_shape[1], overall_mask_shape[0]))
            mask_in_ori = cv2.resize(mask_in_ori, (overall_mask_shape[1], overall_mask_shape[0]))

        overall_mask[row:row + patch_size[0], col:col + patch_size[1]] = patch_img[..., 0]
        # overall_mask_in_ori[row:row + patch_size[0], col:col + patch_size[1],:]=mask_in_ori[...]

        if col + patch_size[1] == w:
            row = row + patch_size[0]
            col = 0
        else:
            col = col + patch_size[1]

    # cv2.imwrite(f'{f_img_name}_ori_mask.png', cv2.resize(np.array(mask_in_ori, dtype='float'), (4096, 4096)))
    # cv2.imwrite(f'{f_img_name}_mask.png', cv2.resize(np.array(overall_mask, dtype='float'), (4096, 4096)))
    # cv2.imwrite(f'{f_img_name}.png',
    #             cv2.resize(np.array(svs.read_region((0, 0), 0, (w, h)).convert('RGB')), (4096, 4096)))
    # cv2.imwrite(f'/media/ubuntu/Seagate Basic/TCGA.png', cv2.resize(overall_mask_in_ori, (4096, 4096)))

    print(f'process success')
    # '''
    # final_results = cv2.resize(overall_mask.astype('float'), (svs.level_dimensions[level][1],svs.level_dimensions[level][0]))
    results = overall_mask
    # print(np.unique(cv2.resize(overall_mask, (4096, 4096))))

    # print(final_results.shape)
    # cv2.imwrite(data.replace('/results/', '/fine_xml/').replace('.svs', '.png'), results)
    kernel = np.ones((7, 7), dtype=np.uint8)
    binary = cv2.morphologyEx((results).astype('uint8'), cv2.MORPH_OPEN, kernel, iterations=5)
    # binary = results
    # cv2.imwrite(data.replace('/results/', '/fine_xml/').replace('.svs', '.png'), (binary).astype('uint8'))
    # contours_to_xml((binary[:, :] // 128).astype('uint8'),
    #                 f'{imgpath}/xml/{f_img_name}.xml', level)

"""