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
#[white,gray,yellow,light_blue,purple, dark_blue]

def open_slide(filename):
    try:
        slide = openslide.open_slide(filename)
    except OpenSlideError:
        slide = None
    except FileNotFoundError:
        slide = None
    return slide


# 40X level=0, 20X level=1, 10X level=2
def svsread(path, level):
    slide = open_slide(path)
    image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    image = np.array(image.convert("RGB"))
    slide.close()
    return image


class_names = ['background', 'cancer']
num_classes = len(class_names)


def evaluate_cancer_test(results, gt_seg_maps):
    ignore_index = 255
    mask = (results != ignore_index)
    pred_label = gt_seg_maps[mask]
    label = results[mask]
    total_area_intersect = np.zeros((num_classes,), dtype=np.float)
    total_area_union = np.zeros((num_classes,), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes,), dtype=np.float)
    total_area_label = np.zeros((num_classes,), dtype=np.float)

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect
    total_area_intersect += area_intersect
    total_area_label += area_label
    total_area_union += area_union
    total_area_pred_label += area_pred_label
    return total_area_intersect, total_area_label, total_area_union, total_area_pred_label


def final_result(total_area_intersect, total_area_label, total_area_union, total_area_pred_label, metric=['mIoU']):
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    ret_metrics = [all_acc, acc]
    if metric[0] == 'mIoU':
        iou = total_area_intersect / total_area_union
        ret_metrics.append(iou)
    elif metric[0] == 'mDice':
        dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
        ret_metrics.append(dice)
    eval_results = {}
    ret_metrics_round = [
        np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
    ]
    class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
    for i in range(num_classes):
        class_table_data.append([class_names[i]] +
                                [m[i] for m in ret_metrics_round[2:]] +
                                [ret_metrics_round[1][i]])
    summary_table_data = [['Scope'] +
                          ['m' + head
                           for head in class_table_data[0][1:]] + ['aAcc']]
    ret_metrics_mean = [
        np.round(np.nanmean(ret_metric) * 100, 2)
        for ret_metric in ret_metrics
    ]
    summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                              [ret_metrics_mean[1]] +
                              [ret_metrics_mean[0]])
    table = AsciiTable(class_table_data)
    print(table.table)
    table = AsciiTable(summary_table_data)
    print(table.table)
    for i in range(1, len(summary_table_data[0])):
        eval_results[summary_table_data[0]
        [i]] = summary_table_data[1][i] / 100.0
    return eval_results


def main():
    parser = ArgumentParser()
    parser.add_argument('imgpath', default='/media/ubuntu/Seagate Basic1/data/', help='Image path')
    parser.add_argument('config', default='../local_configs/pathformer/B5/pathformer.b5.1024x1024.cancerseg.160k.py',
                        help='Config file')
    parser.add_argument('checkpoint', default='/media/ubuntu/Seagate Basic1/work_dirs/data_v3/latest.pth',
                        help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cancerseg',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image

    # data_all = glob(args.imgpath + '*.png')
    # test_files = '/home/shichao/ssd/data/hnsc_dataset/2048/images/test/accuracy_test.txt'
    # test_files = '/home/shichao/ssd/data/hnsc_dataset/ori/images/train/train.txt'

    '''
    read_xls = xlrd.open_workbook('/media/ubuntu/Seagate Basic/data-v3/test-train-validation-list.xlsx')
    xls_context = read_xls.sheets()[0]

    test_lists = xls_context.col_values(colx=0, start_rowx=1)
    assert len(test_lists) == 30, 'Data extraction failed , some data is lost'

    files = glob('/media/ubuntu/Seagate Basic/data-v3/*.svs') + glob('/media/ubuntu/Seagate Basic/data-v3/*.tif')

    test_files = []
    for file in files:
        file_name = file.split('/')[-1].split('.')[0] + 'xml'
        if file_name in test_lists:
            test_files.append(file)

    for img_path in test_files:
        if '.svs' in img_path:
            size = 2048
            level = 1
            step = 1024
            slide = open_slide(img_path)
            # print(slide.level_dimensions[level])
            w, h = slide.level_dimensions[level]
            slide.close()
            result = np.zeros((h, w))
            for i_h in range(int(h / step)):
                for i_w in range(int(w / step)):
                    x = step * i_w
                    y = step * i_h
                    binary_result = inference_segmentor(model, img_path, x, y, size, level)
                    # result[y:y + size, x:x + size] = binary_result[0]
                    if y + size <= h and x + size <= w:
                        result[y:y + size, x:x + size] = binary_result[0]
                    elif y + size <= h and x + size > w:
                        result[y:y + size, x:w] = binary_result[0][:, :w - x]
                    elif y + size > h and x + size <= w:
                        result[y:h, x:x + size] = binary_result[0][:h - y, :]
                    else:
                        result[y:h, x:w] = binary_result[0][:h - y, :w - x]
        else:
            result = inference_segmentor(model, img_path, 0, 0, 0, level)
            result = result[0]
        result = result.astype('uint8')
        gt = cv2.imread(args.imgpath + data.replace('.svs', '.png').replace('svs_first/', '/gt_first/'))
        gt = gt // 128
        final_results = np.concatenate(((scipy.misc.imresize(gt[:, :, 0] * 255, [4096, 4096])),
                                        (scipy.misc.imresize(result * 255, [4096, 4096]))), axis=1)
        # cv2.imwrite(args.imgpath.replace('/image/', '/results/')+data.replace('.svs','.png'),final_results)
        cv2.imwrite(args.imgpath + data.replace('.svs', '.png').replace('svs_first/', '/results_da_small_new/'),
                    final_results)
        cv2.imwrite(args.imgpath + data.replace('.svs', '.png').replace('svs_first/', '/results_da_large_new/'),
                    result * 255)
    '''
    test_files = '/home/ubuntu/ssd/data/20211206/seg/HNSC/test_new.txt'
    save_failed_name = '/home/ubuntu/ssd/data/20211206/seg/HNSC/failed_test.txt'

    total_area_intersect = np.zeros((num_classes,), dtype=np.float)
    total_area_union = np.zeros((num_classes,), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes,), dtype=np.float)
    total_area_label = np.zeros((num_classes,), dtype=np.float)

    # for data in data_all:
    with open(test_files, 'r') as f, open(save_failed_name,'w') as f1:
        for line_all in f.readlines():
            data = line_all.split(' ')[0]
            num = int(line_all.split(' ')[1])
            wh = line_all.split(' ')[2]
            # data = line_all.split('\n')[0]
            # num = 1
            # wh = 'x'
            # image_path = args.imgpath+data.replace('.svs','.png')
            image_path = args.imgpath + data
            # img = cv2.imread(image_path)
            # gt = cv2.imread(args.imgpath.replace('/images/', '/labels/')+data.replace('.svs','.png'))

            # h,w,_ = gt.shape
            # if h*w > 20000*15000:
            #     f1.write(data+'\n')
            #     print(data)
            #     continue
            print(data)
            # img = cv2.imread(image_path)
            # gt = cv2.imread(args.imgpath.replace('/images/', '/labels/')+data.replace('.svs','.png'))
            # h,w,_ = gt.shape
            # if h*w > 24000*21000:
            #     f1.write(data+'\n')
            #     print(data)
            #     continue
            if '.svs' in image_path:
                size = 2048
                level = 1
                step = 1024
                slide = open_slide(image_path)
                # print(slide.level_dimensions[level])
                w, h = slide.level_dimensions[level]
                slide.close()
                result = np.zeros((h, w))
                for i_h in range(int(h / step)):
                    for i_w in range(int(w / step)):
                        x = step * i_w
                        y = step * i_h
                        binary_result = inference_segmentor(model, image_path, x, y, size, level)
                        # result[y:y + size, x:x + size] = binary_result[0]
                        if y + size <= h and x + size <= w:
                            result[y:y + size, x:x + size] = binary_result[0]
                        elif y + size <= h and x + size > w:
                            result[y:y + size, x:w] = binary_result[0][:, :w - x]
                        elif y + size > h and x + size <= w:
                            result[y:h, x:x + size] = binary_result[0][:h - y, :]
                        else:
                            result[y:h, x:w] = binary_result[0][:h - y, :w - x]
            else:
                result = inference_segmentor(model, image_path, 0, 0, 0, level)
                result = result[0]
            result = result.astype('uint8')
            gt = cv2.imread(args.imgpath + data.replace('.svs', '.png').replace('svs_first/', '/gt_first/'))
            gt = gt // 128
            final_results = np.concatenate(((scipy.misc.imresize(gt[:, :, 0] * 255, [4096, 4096])),
                                            (scipy.misc.imresize(result * 255, [4096, 4096]))), axis=1)
            # cv2.imwrite(args.imgpath.replace('/image/', '/results/')+data.replace('.svs','.png'),final_results)
            cv2.imwrite(args.imgpath + data.replace('.svs', '.png').replace('svs_first/', '/results_da_small_new/'), final_results)
            cv2.imwrite(args.imgpath + data.replace('.svs', '.png').replace('svs_first/', '/results_da_large_new/'), result*255)
        #     gt_seg_map = gt[:,:,0]
        #     h,w = result.shape
        #     gt_seg_map = scipy.misc.imresize(gt_seg_map,[h,w])
        #     print(np.shape(result), np.shape(gt_seg_map))
        #     h, w = gt_seg_map.shape
        #     if 'x' in wh:
        #         print('x')
        #         gt_seg_map=gt_seg_map
        #         result_map = result
        #     elif '10' in wh:
        #         if num ==2:
        #             if int(wh) == 101:
        #                 print('2-101')
        #                 gt_seg_map = gt_seg_map[:,:int(w/2)]
        #                 result_map = result[:, :int(w / 2)]
        #             else:
        #                 print('2-102')
        #                 gt_seg_map = gt_seg_map[:, int(w / 2):]
        #                 result_map = result[:, int(w / 2):]
        #         if num ==3:
        #             if int(wh) == 101:
        #                 print('3-101')
        #                 gt_seg_map = gt_seg_map[:, :int(w / 3)]
        #                 result_map = result[:, :int(w / 3)]
        #             elif int(wh) == 102:
        #                 print('3-102')
        #                 gt_seg_map = gt_seg_map[:, int(w / 3):int(w/3*2)]
        #                 result_map = result[:, int(w / 3):int(w / 3 * 2)]
        #             else:
        #                 print('3-103')
        #                 gt_seg_map = gt_seg_map[:, int(w / 3 * 2):]
        #                 result_map = result[:, int(w / 3 * 2):]
        #     elif '11' in wh:
        #         if int(wh) == 111:
        #             print('111')
        #             gt_seg_map = gt_seg_map[:int(h/2), :]
        #             result_map = result[:int(h / 2), :]
        #         else:
        #             print('112')
        #             gt_seg_map = gt_seg_map[int(h / 2):, :]
        #             result_map = result[int(h / 2):, :]
        #     area_intersect, area_label, area_union, area_pred_label = evaluate_cancer_test(result_map,gt_seg_map)
        #     total_area_intersect += area_intersect
        #     total_area_label += area_label
        #     total_area_union += area_union
        #     total_area_pred_label += area_pred_label
        #     final_results = np.concatenate(((scipy.misc.imresize(gt[:,:,0]*255,[4096,4096])),(scipy.misc.imresize(result*255,[4096,4096]))),axis=1)
        #     # cv2.imwrite(args.imgpath.replace('/image/', '/results/')+data.replace('.svs','.png'),final_results)
        #     cv2.imwrite(args.imgpath.replace('/svs/', '/results/') + data.replace('.svs', '.png'), final_results)
        #     contours_to_xml(result,args.imgpath.replace('/svs/', '/results_xml/') + data.replace('.svs', '.xml'))
        # mIoU = final_result(total_area_intersect, total_area_label, total_area_union, total_area_pred_label, metric=['mIoU'])
        # print(mIoU)
        # mDice = final_result(total_area_intersect, total_area_label, total_area_union, total_area_pred_label, metric=['mDice'])
        # print(mDice)
    f.close()
    f1.close()



def image_crop_to_test(image, size, scale, name):
    h, w, c = image.shape
    count = 0
    image_path = '/home/ubuntu/ssd/data/20211206/seg/HNSC/slide_test/'
    for i_h in range(int(h / size)):
        for i_w in range(int(w / size)):
            y = size * i_h
            x = size * i_w
            patch_image = image[y:y + size, x:x + size, :]
            scipy.misc.imsave(image_path + name.split('.svs')[0] + '_{}_{:06d}.png'.format(scale, count), patch_image)
            count = count + 1


def main_test():
    type = 'failed_test'
    size = 2048
    train_svs_path = '/home/ubuntu/ssd/data/20211206/seg/HNSC/'
    count = 1
    with open(train_svs_path + '{}.txt'.format(type), 'r') as f:
        for line_all in f.readlines():
            line = line_all.split(' ')[0].strip()
            print('{}:{}'.format(count, line))
            count = count + 1
            print('20X')
            slide_image = svsread(train_svs_path + 'svs/' + line, level=1)
            image_crop_to_test(slide_image, size=size, scale='20X', name=line)


def process_test_data(test_files,save_path):
    label_mapping = {
        'Q': 0,
        'NOR': 1,
        'HYP': 2,
        'DYS': 3,
        'CAR': 4,
    }
    for file in test_files:
        slide=openslide.open_slide(file)
        img_size=slide.level_dimensions[0]
        slide_region = slide.read_region((0, 0), 0, img_size).convert('RGB')
        slide_region = np.array(slide_region)  # size1,size2,channel  ==> size2,size1,channel

        img_size = (img_size[1], img_size[0])
        slide_points = xml_to_region(file.split('.')[0]+'.xml')

        mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)

        for point_dict in slide_points:
            cls = list(point_dict.keys())[0]

            points = np.asarray([point_dict[cls]], dtype=np.int32)

            cv2.fillPoly(img=mask, pts=points, color=(label_mapping[cls], label_mapping[cls], label_mapping[cls]))



        cv2.imwrite(f'{save_path}/gt/{file.split("/")[-1].split(".")[0]}.png',cv2.resize(mask,(2048,2048)))
        cv2.imwrite(f'{save_path}/img/{file.split("/")[-1].split(".")[0]}.png',cv2.resize(slide_region,(2048,2048)))


def main_test_slide():
    parser = ArgumentParser()
    parser.add_argument('--imgpath', default='/media/ubuntu/Seagate Basic/merge_data/img_dir', help='Image path')
    parser.add_argument('--config', default='../local_configs/pathformer/B5/pathformer.b5.1024x1024.cancerseg.160k.py',
                        help='Config file')
    parser.add_argument('--checkpoint', default='/media/ubuntu/Seagate Basic/work_dirs/optim_data/fore_back_head/latest.pth',
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
        print(result.shape,np.unique(np.array(result))) # (2048,2048)
        # print(result)
        os.makedirs(args.imgpath + '/pre', exist_ok=True)
        """
        img=np.array(cv2.imread(data))
        # rgb_res=np.zeros((result[0],result[1],3))
        # rgb_res=np.tile(np.array([255],dtype=np.uint8),(result.shape[0],result.shape[1],3))
        for x in range(result.shape[0]):
            for y in range(result.shape[1]):
                # rgb_res[x,y,:]=PALETTE[result[x,y]]
                if result[x,y]!=0:
                    img[x, y, :] = PALETTE[result[x, y]+1]

        cv2.imwrite(f'{args.imgpath}/pre/{data.split("/")[-1].split(".png")[0]}_mask_in_ori.png',img)
        # cv2.imwrite(image_path.replace('/img/','/pre_rgb/'),rgb_res)
        """
        # final_results = np.array(result * 255).astype('uint8')
        final_results = np.array(result).astype('uint8')
        cv2.imwrite(f'{args.imgpath}/pre/{data.split("/")[-1].split(".png")[0]}_mask.png', final_results)


def mask_in_ori_img():
    mask_path = '//media/ubuntu/Seagate Basic/data/ann_dir/test'
    # 5 classes  [Q,NOR,HYP,DYS,CAR]=[white,gray,yellow,light_blue,purple]
    mask_lists = glob(mask_path + '/*.png')
    for file in mask_lists:
        mask = np.array(cv2.imread(file))
        ori_img = np.array(cv2.imread(file.replace("ann_dir", "img_dir")))
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x,y][0]!=0:
                    ori_img[x,y,:]=PALETTE[mask[x,y][0]]
        cv2.imwrite(f'{file.split(".png")[0]}_rgb.png', ori_img)


def results_to_all():
    type = 'failed_test'
    size = 2048
    scale = '20X'
    train_svs_path = '/home/ubuntu/ssd/data/20211206/seg/HNSC/'
    count1 = 1
    with open(train_svs_path + '{}.txt'.format(type), 'r') as f:
        for line_all in f.readlines():
            line = line_all.split(' ')[0].strip()
            print('{}:{}'.format(count1, line))
            count1 = count1 + 1
            print('20X')
            image = svsread(train_svs_path + 'svs/' + line, level=1)
            h, w, c = image.shape
            count = 0
            binary_result = np.zeros((h, w))
            image_path = '/home/ubuntu/ssd/data/20211206/seg/HNSC/slide_results/'
            for i_h in range(int(h / size)):
                for i_w in range(int(w / size)):
                    y = size * i_h
                    x = size * i_w
                    patch_image = cv2.imread(image_path + line.split('.svs')[0] + '_{}_{:06d}.png'.format(scale, count))
                    binary_result[y:y + size, x:x + size] = patch_image[:, :, 0]
                    count = count + 1
            gt = cv2.imread(train_svs_path + 'gt/' + line.replace('.svs', '.png'))
            gt = gt // 128
            binary_result = binary_result // 128
            final_results = np.concatenate(((scipy.misc.imresize(gt[:, :, 0] * 255, [4096, 4096])),
                                            (scipy.misc.imresize(binary_result * 255, [4096, 4096]))), axis=1)
            cv2.imwrite(train_svs_path + 'results/' + line.replace('.svs', '.png'), final_results)
            contours_to_xml(binary_result, train_svs_path + '/results_xml/' + line.replace('.svs', '.xml'))


def results_to_xml():
    train_svs_path = '/home/ubuntu/ssd/data/20211206/seg/HNSC/'
    file = 'test'
    with open(train_svs_path + '{}.txt'.format(file), 'r') as f:
        for line_all in f.readlines():
            line = line_all.split(' ')[0].strip()
            image = svsread(train_svs_path + 'svs/' + line, level=1)
            h, w, c = image.shape
            image = cv2.imread(train_svs_path + 'results/' + line.replace('.svs', '.png'))
            h1, w1, _ = image.shape
            results = image[:, int(w1 / 2):, 0]
            results = cv2.resize(results, [w, h])
            contours_to_xml(results // 128, train_svs_path + 'results_xml/' + line.replace('.svs', '.xml'))


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
        # else:
        #     if y1 > y0:
        #         binary[y0-seg:y1 + seg,:] = 0
        #     else:
        #         binary[y1 - seg:y0 + seg, :] = 0
    return binary


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
    if level == 1:
        h, w = binary.shape
        binary = cv2.resize(np.array(binary * 255).astype('uint8'), (w * scale, h * scale))
    if level == 2:
        h, w = binary.shape
        binary = cv2.resize(np.array(binary * 255).astype('uint8'), (w * scale * scale, h * scale * scale))
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    binary = find_nn_axis(contours, hierarchy, binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    file = open(xml_name, 'w')
    file.write('<?xml version=\"1.0\"?>\n')
    file.write('<ASAP_Annotations>\n')
    file.write('\t<Annotations>\n')
    thresh = 10000
    count = 0
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > thresh:
            file.write(
                '\t\t<Annotation Name=\"Annotation {:d}\" Type=\"Spline\" PartOfGroup=\"None\" Color=\"#F4FA58\">\n'.format(
                    count))
            file.write('\t\t\t<Coordinates>\n')
            count += 1
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


def image_to_xml():
    file_name = 'results_20X_lscc'
    # imgpath = '/home/ubuntu/ssd/data/20211206/seg/HNSC/{}/'.format(file_name)
    imgpath = '/home/ubuntu/Downloads/{}/'.format(file_name)
    data_all = glob(imgpath + '*.png')
    level = 1
    for data in data_all:
        print(data)
        image_path = data
        results = cv2.imread(image_path)
        # h,w,_ = results.shape
        svs = open_slide(image_path.replace('/{}/'.format(file_name), '/LSCC/').replace('.png', '.svs'))
        w, h = svs.level_dimensions[level]

        final_results = cv2.resize(results, (w, h))
        results = final_results

        # print(final_results.shape)
        # cv2.imwrite(data.replace('/results/', '/fine_xml/').replace('.svs', '.png'), results)
        kernel = np.ones((7, 7), dtype=np.uint8)
        binary = cv2.morphologyEx((results).astype('uint8'), cv2.MORPH_OPEN, kernel, iterations=5)
        # binary = results
        # cv2.imwrite(data.replace('/results/', '/fine_xml/').replace('.svs', '.png'), (binary).astype('uint8'))
        contours_to_xml((binary[:, :, 0] // 128).astype('uint8'),
                        image_path.replace('/{}/'.format(file_name), '/LSCC/').replace('.png', '.xml'), level)


def main_300():
    parser = ArgumentParser()
    parser.add_argument('--imgpath', default='/media/ubuntu/Seagate Basic1/data/', help='Image path')
    parser.add_argument('--config', default='../local_configs/pathformer/B5/pathformer.b5.1024x1024.cancerseg.160k.py',
                        help='Config file')
    parser.add_argument('--checkpoint', default='/media/ubuntu/Seagate Basic1/work_dirs/data_v3/latest.pth',
                        help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cancerseg',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image

    read_xls = xlrd.open_workbook('/media/ubuntu/Seagate Basic/data-v3/test-train-validation-list.xlsx')
    xls_context = read_xls.sheets()[0]

    test_lists = xls_context.col_values(colx=0, start_rowx=1)
    test_lists=list(filter(None,test_lists))
    assert len(test_lists) == 30, 'Data extraction failed , some data is lost'

    files = glob('/media/ubuntu/Seagate Basic/data-v3/*.svs') + glob('/media/ubuntu/Seagate Basic/data-v3/*.tif')

    test_files = []
    for file in files:
        file_name = file.split('/')[-1].split('.')[0] + '.xml'
        if file_name in test_lists:
            test_files.append(file)

    count = 0
    # data_all = glob(args.imgpath + '*.tif')
    # data_all = glob(args.imgpath + '*.svs')
    data_all=test_files
    save_path=args.imgpath+'test/xml/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # print(data_all)
    for data in data_all:
        count = count + 1
        image_path = data
        # if count <=214:
        #     continue
        print('{}:{}'.format(count, data))
        if '.svs' in image_path:
            level = 1
        else:
            level = 2
        if '.tif' in image_path or '.svs' in image_path:
            size = 2048
            step = 1920
            slide = open_slide(image_path)
            w, h = slide.level_dimensions[level]
            print(h, w)
            slide.close()
            result = np.zeros((h, w))
            for i_h in range(int(h / step)):
                for i_w in range(int(w / step)):
                    x = step * i_w
                    y = step * i_h
                    binary_result = inference_segmentor(model, image_path, x, y, size, level)
                    if y + size <= h and x + size <= w:
                        result[y:y + size, x:x + size] = binary_result[0]
                    elif y + size <= h and x + size > w:
                        result[y:y + size, x:w] = binary_result[0][:, :w - x]
                    elif y + size > h and x + size <= w:
                        result[y:h, x:x + size] = binary_result[0][:h - y, :]
                    else:
                        result[y:h, x:w] = binary_result[0][:h - y, :w - x]
        else:
            result = inference_segmentor(model, image_path, 0, 0, 0, 0)
            result = result[0]
        result = result.astype('uint8')
        # final_results = scipy.misc.imresize(result * 255, [4096, 4096])
        # seg_filename_small = data.replace('zhuan/', 'results_small_seg/')
        # if not os.path.exists(seg_filename_small.split('seg/')[0]+'seg/'):
        #     os.mkdir(seg_filename_small.split('seg/')[0]+'seg/')
        # cv2.imwrite(seg_filename_small.replace('.svs', '.png'), final_results)
        # seg_filename_large = data.replace('zhuan/', 'results_large_seg/')
        # if not os.path.exists(seg_filename_large.split('seg/')[0]+'seg/'):
        #     os.mkdir(seg_filename_large.split('seg/')[0]+'seg/')
        # cv2.imwrite(seg_filename_large.replace('.svs', '.png'), (result*255).astype('uint8'))

        kernel = np.ones((7, 7), dtype=np.uint8)
        binary = cv2.morphologyEx((result * 255).astype('uint8'), cv2.MORPH_OPEN, kernel, iterations=5)
        if '.svs' in data:
            contours_to_xml((binary // 128).astype('uint8'), save_path+data.split('/')[-1].replace('.svs', '.xml'), level)
        if '.tif' in data:
            contours_to_xml((binary // 128).astype('uint8'), save_path+data.split('/')[-1].replace('.tif', '.xml'), level, scale=2)


if __name__ == '__main__':
    # main()
    # main_300()
    # main_test()
    main_test_slide()
    # mask_in_ori_img()
    # results_to_all()
    # results_to_xml()