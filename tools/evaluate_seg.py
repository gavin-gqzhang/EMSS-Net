import os
from glob import glob

import xml.etree.ElementTree as ET

from tqdm import tqdm
import xlrd

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=pow(2,63).__str__()
import cv2
import numpy as np
from terminaltables import AsciiTable
import scipy
import PIL
from PIL import Image
import scipy.misc
import openslide
from openslide import  OpenSlideError

class_names = ['Q','NOR','HYP','DYS','CAR']
# num_classes = len(class_names)
# class_names = ['B','Q']
num_classes = len(class_names)


def final_result(class_names,total_area_intersect,total_area_label,total_area_union, total_area_pred_label,metric=['mIoU'],rf=None):
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
    for i in range(len(class_names)):
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
    if rf!=None:
        rf.write(table.table)
    print(table.table)
    table = AsciiTable(summary_table_data)
    if rf!=None:
        rf.write(table.table)
    print(table.table)
    for i in range(1, len(summary_table_data[0])):
        eval_results[summary_table_data[0]
        [i]] = summary_table_data[1][i] / 100.0
    return eval_results

def open_slide(filename):
    try:
        slide = openslide.open_slide(filename)
    except OpenSlideError:
        slide = None
    except FileNotFoundError:
        slide = None
    return slide

# 40X level=0, 20X level=1, 10X level=2
def svsread(path,level):
    slide = open_slide(path)
    image = slide.read_region((0,0),level,slide.level_dimensions[level])
    image = np.array(image.convert("RGB"))
    slide.close()
    return image

# dataset_name = 'test_20211216'
# test_files = '/home/ubuntu/ssd/data/20211206/seg/HNSC/test.txt'
def get_metrics(txt_file,pre_file,num_classes):
    result_file = txt_file

    pre_files = glob(f'{pre_file}/*_mask.png')

    rf = open(result_file, 'w')
    rf.write('{},{},{},{},{},{},{}\n'.format('name', 'accuracy', 'precision', 'sensitive', 'specificity', 'F1', 'IOU'))
    mean_acc = 0
    mean_sen = 0
    mean_spe = 0
    mean_iou = 0
    mean_pre = 0
    mean_f1 = 0
    count = 0
    total_area_intersect = np.zeros((num_classes,), dtype=np.float)
    total_area_union = np.zeros((num_classes,), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes,), dtype=np.float)
    total_area_label = np.zeros((num_classes,), dtype=np.float)
    res = dict()
    for file in pre_files:
        # slide = open_slide('/media/ubuntu/Seagate Basic1/data-v3' + file.split('/')[-1].replace('png','svs'))
        # w1, h1 = slide.level_dimensions[1]
        gt = cv2.imread("/media/ubuntu/Seagate Basic/optim_data/test_datas/ann_dir/{}.png".format(
            file.split('/')[-1].split('_mask')[0]))
        pre_gt = cv2.imread(file)
        if num_classes==2:
            gt[gt!=0]=1

        gt, pre_gt = np.array(gt), np.array(pre_gt)
        if gt.shape != pre_gt.shape:
            # print(gt.shape, pre_gt.shape)
            pre_gt = cv2.resize(pre_gt, (gt.shape[1], gt.shape[0]))
            pre_gt = np.array(pre_gt)

        if '-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1]) not in res.keys():
            res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])] = dict()

        pre_gt = pre_gt[:, :, 0]
        gt = gt[:, :, 0]

        TP = 0
        FP = 0
        TN = 0
        FN = 0
        IOU = 0
        thresh = 0
        count = count + 1

        # pred_label = pre_gt // 128
        # label = gt // 128
        pred_label = pre_gt
        label = gt
        ignore_index = 255
        mask = (pred_label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]
        intersect = pred_label[pred_label == label]  # pre_true
        area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
        area_pred_label, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
        area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))

        area_union = area_pred_label + area_label - area_intersect  # all region
        total_area_intersect += area_intersect
        total_area_label += area_label
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        TP = area_intersect[1]
        if 'TP' not in res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])].keys():
            res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])]['TP'] = int(TP)
        else:
            res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])]['TP'] = \
            res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])]['TP'] + int(TP)

        TN = area_intersect[0]
        if 'TN' not in res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])].keys():
            res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])]['TN'] = int(TN)
        else:
            res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])]['TN'] = \
                res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])]['TN'] + int(TN)

        FP = area_pred_label[1] - area_intersect[1]
        if 'FP' not in res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])].keys():
            res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])]['FP'] = int(FP)
        else:
            res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])]['FP'] = \
                res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])]['FP'] + int(FP)

        FN = area_pred_label[0] - area_intersect[0]
        if 'FN' not in res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])].keys():
            res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])]['FN'] = int(FN)
        else:
            res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])]['FN'] = \
                res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])]['FN'] + int(FN)

        IOU = (TP / (area_union[1]) + TN / area_union[0]) / 2.
        if np.isnan(IOU):
            IOU = 0
        if 'IOU' not in res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])].keys():
            res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])]['IOU'] = IOU
        else:
            res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])]['IOU'] = \
                res['-'.join(file.split('/')[-1].split('_mask')[0].split('-')[:-1])]['IOU'] + IOU

    print(res.keys())
    for name in res.keys():
        TP = res[name]['TP']
        TN = res[name]['TN']
        FP = res[name]['FP']
        FN = res[name]['FN']
        IOU = res[name]['IOU']

        accuracy = (TP + TN) / (TP + FP + FN + TN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        if TP == 0 and FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)
        if precision == 0 and sensitivity == 0:
            F1 = 0
        else:
            F1 = 2. * precision * sensitivity / (precision + sensitivity)
        mean_acc = mean_acc + accuracy
        mean_sen = mean_sen + sensitivity
        mean_spe = mean_spe + specificity
        mean_iou = mean_iou + IOU
        mean_pre = mean_pre + precision
        mean_f1 = mean_f1 + F1
        print('accuracy:{},precision:{},sensitivity:{},specificity:{},f1:{},IOU:{}'.format(accuracy, precision,
                                                                                           sensitivity,
                                                                                           specificity, F1, IOU))
        rf.write(
            '{}.svs/tif,{},{},{},{},{},{}\n'.format(name, accuracy, precision, sensitivity, specificity, F1,
                                                    IOU))

        '''
        with open(test_files,'r') as f:
            for line_all in f.readlines():
                line = line_all.split(' ')[0]
                num = int(line_all.split(' ')[1])
                wh = line_all.split(' ')[2]
                print(line,num,wh)
                # print('/home/shichao/ssd/data/hnsc_dataset/test_xml_label/{}.png'.format(line.split('.svs')[0]))
                # mask = scipy.misc.imread('/home/shichao/ssd/data/hnsc_dataset/test_xml_label/{}.tiff'.format(line.split('.svs')[0]))
                # gt =  cv2.imread('/home/shichao/ssd/data/hnsc_dataset/gt/{}.png'.format(line.split('.svs')[0]))
                # print(gt)

                slide = open_slide('/home/ubuntu/ssd/data/20211206/seg/HNSC/svs/'+line.split('.svs')[0]+'.svs')
                w1,h1 = slide.level_dimensions[1]
                # data = cv2.imread('/home/shichao/ssd/data/hnsc_dataset/results_40X_small/'+line.split('.svs')[0]+'.png')
                # h, w,_ = data.shape
                # gt = data[:,:int(w/2),0]
                # mask = data[:,int(w/2):,0]

                mask = cv2.imread('/home/ubuntu/ssd/data/20211206/seg/HNSC/results_uper/' + line.split('.svs')[0] + '.png')
                gt = cv2.imread('/home/ubuntu/ssd/data/20211206/seg/HNSC/gt/' + line.split('.svs')[0] + '.png')

                mask = mask[:,:,0]
                gt = gt[:,:,0]

                gt = scipy.misc.imresize(gt,[h1,w1])
                mask = scipy.misc.imresize(mask,[h1,w1])

                h, w = mask.shape
                # gt = cv2.resize(gt, (w, h))
                print(h,w,np.shape(gt), np.shape(mask))
                if 'x' in wh:
                    print('x')
                    a_mask = mask
                    b_mask = gt
                elif '10' in wh:
                    if num ==2:
                        if int(wh) == 101:
                            print('2-101')
                            a_mask = mask[:,:int(w/2)]
                            b_mask = gt[:,:int(w/2)]
                        else:
                            print('2-102')
                            a_mask = mask[:, int(w / 2):]
                            b_mask = gt[:, int(w / 2):]
                    if num ==3:
                        if int(wh) == 101:
                            print('3-101')
                            a_mask = mask[:, :int(w / 3)]
                            b_mask = gt[:, :int(w / 3)]
                        elif int(wh) == 102:
                            print('3-102')
                            a_mask = mask[:, int(w / 3):int(w/3*2)]
                            b_mask = gt[:, int(w / 3):int(w/3*2)]
                        else:
                            print('3-103')
                            a_mask = mask[:, int(w / 3 * 2):]
                            b_mask = gt[:, int(w / 3 * 2):]
                elif '11' in wh:
                    if int(wh) == 111:
                        print('111')
                        a_mask = mask[:int(h/2), :]
                        b_mask = gt[:int(h/2), :]
                    else:
                        print('112')
                        a_mask = mask[int(h / 2):, :]
                        b_mask = gt[int(h / 2):, :]
                h, w = a_mask.shape
                TP = 0
                FP = 0
                TN = 0
                FN = 0
                IOU = 0
                thresh = 0
                count = count + 1
                # for i in range(h):
                #     for j in range(w):
                #         if (a_mask[i, j] > thresh or b_mask[i, j] > thresh):
                #             IOU += 1
                #         if (a_mask[i,j]>thresh and b_mask[i,j] > thresh):
                #             TP += 1
                #         if (a_mask[i, j] <= thresh and b_mask[i, j] > thresh):
                #             FP += 1
                #         if (a_mask[i,j]>thresh and b_mask[i,j] <= thresh):
                #             FN += 1
                #         if (a_mask[i,j]<=thresh and b_mask[i,j] <= thresh):
                #             TN += 1
                pred_label = a_mask // 128
                label = b_mask // 128
                ignore_index = 255
                mask = (pred_label!= ignore_index)
                pred_label = pred_label[mask]
                label = label[mask]
                intersect = pred_label[pred_label == label]
                area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
                area_pred_label, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
                area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
                area_union = area_pred_label + area_label - area_intersect
                total_area_intersect += area_intersect
                total_area_label += area_label
                total_area_union += area_union
                total_area_pred_label += area_pred_label
                TP = area_intersect[1]
                TN = area_intersect[0]
                FP = area_pred_label[1] - area_intersect[1]
                FN = area_pred_label[0] - area_intersect[0]
                IOU = area_union[1]
                accuracy = (TP + TN) / (TP + FP + FN + TN)
                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)
                IOU = (TP / area_union[1] + TN / area_union[0]) / 2.
                precision = TP / (TP+FP)
                F1 = 2.*precision*sensitivity/(precision+sensitivity)
                mean_acc = mean_acc + accuracy
                mean_sen = mean_sen + sensitivity
                mean_spe = mean_spe + specificity
                mean_iou = mean_iou + IOU
                mean_pre = mean_pre + precision
                mean_f1 = mean_f1 + F1
                print('accuracy:{},precision:{},sensitivity:{},specificity:{},f1:{},IOU:{}'.format(accuracy,precision,sensitivity,specificity,F1,IOU))
                rf.write('{}.svs,{},{},{},{},{},{}\n'.format(line.split('.svs')[0],accuracy,precision,sensitivity,specificity,F1,IOU))
        f.close()
        '''
    mean_acc = mean_acc / count
    mean_sen = mean_sen / count
    mean_spe = mean_spe / count
    mean_iou = mean_iou / count
    mean_pre = mean_pre / count
    mean_f1 = mean_f1 / count
    print('mean accuracy:{},mean precision:{}, mean sensitivity:{},mean specificity:{},mean F1:{}, mean IOU:{}'.format(
        mean_acc, mean_pre, mean_sen, mean_spe, mean_f1, mean_iou))
    rf.write('{},{},{},{},{},{},{}\n'.format('average', mean_acc, mean_pre, mean_sen, mean_spe, mean_f1, mean_iou))
    print(
        str(final_result(num_classes,total_area_intersect, total_area_label, total_area_union, total_area_pred_label, metric=['mIoU'])))
    rf.write(
        str(final_result(num_classes,total_area_intersect, total_area_label, total_area_union, total_area_pred_label, metric=['mIoU'])))
    print(
        str(final_result(num_classes,total_area_intersect, total_area_label, total_area_union, total_area_pred_label, metric=['mDice'])))
    rf.write(
        str(final_result(num_classes,total_area_intersect, total_area_label, total_area_union, total_area_pred_label, metric=['mDice'])))
    rf.close()


def xml_to_region(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    region_list=[]
    for ann in root.findall('Annotations/Annotation'):
        points = []
        for point in ann.findall('Coordinates/Coordinate'):
            x = float(point.get('X'))
            y = float(point.get('Y'))
            points.append([x, y])
        region_list.append({ann.get('PartOfGroup'):points})

    return region_list

def get_xml_metrics(txt_file,xml_path,num_classes):
    if num_classes==2:
        label_mapping={
            "background":0,
            "NOR":1
        }
    elif num_classes==5 or num_classes==4:
        label_mapping = {
            'Background':0,
            'NOR': 1,
            'HYP': 2,
            'DYS': 3,
            'CAR': 4,
        }
    result_file = txt_file
    base_path='/data/sdc/medicine_svs'

    read_xls = xlrd.open_workbook(f'{base_path}/v3-test-train-validation-list.xlsx')
    xls_context = read_xls.sheets()[0]

    xml_files = []
    col_datas = xls_context.col_values(colx=0, start_rowx=1)

    for data in col_datas[1:]:
        if data != '':
            xml_files.append(str(data).replace('0F', 'OF'))  # xml lists

    rf = open(result_file, 'w')
    rf.write('{},{},{},{},{},{},{}\n'.format('name', 'accuracy', 'precision', 'sensitive', 'specificity', 'F1', 'IOU'))
    mean_acc = 0
    mean_sen = 0
    mean_spe = 0
    mean_iou = 0
    mean_pre = 0
    mean_f1 = 0
    count = 0
    total_area_intersect = np.zeros((num_classes,), dtype=np.float)
    total_area_union = np.zeros((num_classes,), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes,), dtype=np.float)
    total_area_label = np.zeros((num_classes,), dtype=np.float)
    res=dict()

    for file in tqdm(xml_files):
        predict_xml=f'{xml_path}/{file}'
        slide_file=glob(f'{base_path}/data-v3/{file.split("/")[-1].split(".")[0]}.svs')+glob(f'{base_path}/data-v3/{file.split("/")[-1].split(".")[0]}.tif')
        slide = open_slide(slide_file[0])
        img_size = slide.level_dimensions[0]

        slide_region = slide.read_region((0, 0), 0, img_size).convert('RGB')
        slide_region = np.array(slide_region)

        img_size = (img_size[1], img_size[0])

        pre_slide_points = xml_to_region(predict_xml)
        predict_mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        for point_dict in pre_slide_points:
            cls = list(point_dict.keys())[0]
            points = np.asarray([point_dict[cls]], dtype=np.int32)

            if num_classes==2 and  cls in ['NOR','HYP','DYS','CAR']:
                cls="NOR"
                
            cv2.fillPoly(img=predict_mask, pts=points, color=(label_mapping[cls]))


        gt_slide_points=xml_to_region(f'{base_path}/data-v4/{file.split("/")[-1]}')
        gt_mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        for point_dict in gt_slide_points:
            cls = list(point_dict.keys())[0]
            points = np.asarray([point_dict[cls]], dtype=np.int32)

            if num_classes==2 and  cls in ['NOR','HYP','DYS','CAR']:
                cls="NOR"

            cv2.fillPoly(img=gt_mask, pts=points, color=(label_mapping[cls]))

        
        if f'{file.split("/")[-1].split(".")[0]}' not in res.keys():
            res[f'{file.split("/")[-1].split(".")[0]}'] = dict()

        TP = 0
        FP = 0
        TN = 0
        FN = 0
        IOU = 0
        thresh = 0
        count = count + 1

        if num_classes==4:
            gt_mask[gt_mask==0]=255

            gt_mask=gt_mask-1
            predict_mask=predict_mask-1
            
        ignore_index = 255
        filter_idx = (gt_mask != ignore_index)
        predict_mask = predict_mask[filter_idx]
        gt_mask = gt_mask[filter_idx]
        intersect = predict_mask[predict_mask == gt_mask]  # pre_true
        area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
        area_pred_label, _ = np.histogram(predict_mask, bins=np.arange(num_classes + 1))
        area_label, _ = np.histogram(gt_mask, bins=np.arange(num_classes + 1))

        area_union = area_pred_label + area_label - area_intersect  # all region
        total_area_intersect += area_intersect
        total_area_label += area_label
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        
        res[f'{file.split("/")[-1].split(".")[0]}'] = {'TP': [0] * num_classes,
                                        'TN': [0] * num_classes,
                                        'FP': [0] * num_classes,
                                        'FN': [0] * num_classes,
                                        'IOU': [0] * num_classes,
                                        'acc': [0] * num_classes,
                                        'sens': [0] * num_classes,
                                        'spec': [0] * num_classes,
                                        'precision': [0] * num_classes,
                                        'F1': [0] * num_classes}
        
        for class_idx in range(num_classes):
            TP = int(area_intersect[class_idx])  
            FP = int(area_pred_label[class_idx] - TP)  
            FN = int(area_label[class_idx] - TP)  
            TN = int(np.sum(area_label) - TP - FP - FN) 

            res[f'{file.split("/")[-1].split(".")[0]}']['TP'][class_idx] += int(TP)
            res[f'{file.split("/")[-1].split(".")[0]}']['FP'][class_idx] += int(FP)
            res[f'{file.split("/")[-1].split(".")[0]}']['FN'][class_idx] += int(FN)
            res[f'{file.split("/")[-1].split(".")[0]}']['TN'][class_idx] += int(TN)

            precision=TP / (TP + FP) if (TP + FP) > 0 else 0
            sensitivity=TP / (TP + FN) if (TP + FN) > 0 else 0
            
            res[f'{file.split("/")[-1].split(".")[0]}']['acc'][class_idx] += (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
            res[f'{file.split("/")[-1].split(".")[0]}']['sens'][class_idx] += sensitivity
            res[f'{file.split("/")[-1].split(".")[0]}']['spec'][class_idx] += TN / (TN + FP) if (TN + FP) > 0 else 0
            res[f'{file.split("/")[-1].split(".")[0]}']['precision'][class_idx] +=precision 
            res[f'{file.split("/")[-1].split(".")[0]}']['F1'][class_idx] += (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            # 计算 IOU，避免除以零
            union = area_union[class_idx]
            if union > 0:
                IOU = TP / union
            else:
                IOU = 0
            res[f'{file.split("/")[-1].split(".")[0]}']['IOU'][class_idx] += IOU


    for name in res.keys():
        TP = np.array(res[name]['TP'])
        TN = np.array(res[name]['TN'])
        FP = np.array(res[name]['FP'])
        FN = np.array(res[name]['FN'])
        IOU = np.array(res[name]['IOU'])
        
        acc = np.array(res[name]['acc'])
        sens = np.array(res[name]['sens'])
        spec = np.array(res[name]['spec'])
        precision = np.array(res[name]['precision'])
        F1 = np.array(res[name]['F1'])
        
        mean_acc = mean_acc + np.mean(acc)
        mean_sen = mean_sen + np.mean(sens)
        mean_spe = mean_spe + np.mean(spec)
        mean_pre = mean_pre + np.mean(precision)
        mean_f1 = mean_f1 + np.mean(F1)
        mean_iou = mean_iou + np.mean(IOU)

        print('name:{},accuracy:{},precision:{},sensitivity:{},specificity:{},f1:{},IOU:{}'.format(name,np.mean(acc), np.mean(precision),
                                                                                           np.mean(sens), np.mean(spec), np.mean(F1), np.mean(IOU)))
        rf.write(
            '{}.svs/tif,{},{},{},{},{},{}\n'.format(name, np.mean(acc), np.mean(precision), np.mean(sens), np.mean(spec), np.mean(F1),
                                                    np.mean(IOU)))

        '''
        with open(test_files,'r') as f:
            for line_all in f.readlines():
                line = line_all.split(' ')[0]
                num = int(line_all.split(' ')[1])
                wh = line_all.split(' ')[2]
                print(line,num,wh)
                # print('/home/shichao/ssd/data/hnsc_dataset/test_xml_label/{}.png'.format(line.split('.svs')[0]))
                # mask = scipy.misc.imread('/home/shichao/ssd/data/hnsc_dataset/test_xml_label/{}.tiff'.format(line.split('.svs')[0]))
                # gt =  cv2.imread('/home/shichao/ssd/data/hnsc_dataset/gt/{}.png'.format(line.split('.svs')[0]))
                # print(gt)

                slide = open_slide('/home/ubuntu/ssd/data/20211206/seg/HNSC/svs/'+line.split('.svs')[0]+'.svs')
                w1,h1 = slide.level_dimensions[1]
                # data = cv2.imread('/home/shichao/ssd/data/hnsc_dataset/results_40X_small/'+line.split('.svs')[0]+'.png')
                # h, w,_ = data.shape
                # gt = data[:,:int(w/2),0]
                # mask = data[:,int(w/2):,0]

                mask = cv2.imread('/home/ubuntu/ssd/data/20211206/seg/HNSC/results_uper/' + line.split('.svs')[0] + '.png')
                gt = cv2.imread('/home/ubuntu/ssd/data/20211206/seg/HNSC/gt/' + line.split('.svs')[0] + '.png')

                mask = mask[:,:,0]
                gt = gt[:,:,0]

                gt = scipy.misc.imresize(gt,[h1,w1])
                mask = scipy.misc.imresize(mask,[h1,w1])

                h, w = mask.shape
                # gt = cv2.resize(gt, (w, h))
                print(h,w,np.shape(gt), np.shape(mask))
                if 'x' in wh:
                    print('x')
                    a_mask = mask
                    b_mask = gt
                elif '10' in wh:
                    if num ==2:
                        if int(wh) == 101:
                            print('2-101')
                            a_mask = mask[:,:int(w/2)]
                            b_mask = gt[:,:int(w/2)]
                        else:
                            print('2-102')
                            a_mask = mask[:, int(w / 2):]
                            b_mask = gt[:, int(w / 2):]
                    if num ==3:
                        if int(wh) == 101:
                            print('3-101')
                            a_mask = mask[:, :int(w / 3)]
                            b_mask = gt[:, :int(w / 3)]
                        elif int(wh) == 102:
                            print('3-102')
                            a_mask = mask[:, int(w / 3):int(w/3*2)]
                            b_mask = gt[:, int(w / 3):int(w/3*2)]
                        else:
                            print('3-103')
                            a_mask = mask[:, int(w / 3 * 2):]
                            b_mask = gt[:, int(w / 3 * 2):]
                elif '11' in wh:
                    if int(wh) == 111:
                        print('111')
                        a_mask = mask[:int(h/2), :]
                        b_mask = gt[:int(h/2), :]
                    else:
                        print('112')
                        a_mask = mask[int(h / 2):, :]
                        b_mask = gt[int(h / 2):, :]
                h, w = a_mask.shape
                TP = 0
                FP = 0
                TN = 0
                FN = 0
                IOU = 0
                thresh = 0
                count = count + 1
                # for i in range(h):
                #     for j in range(w):
                #         if (a_mask[i, j] > thresh or b_mask[i, j] > thresh):
                #             IOU += 1
                #         if (a_mask[i,j]>thresh and b_mask[i,j] > thresh):
                #             TP += 1
                #         if (a_mask[i, j] <= thresh and b_mask[i, j] > thresh):
                #             FP += 1
                #         if (a_mask[i,j]>thresh and b_mask[i,j] <= thresh):
                #             FN += 1
                #         if (a_mask[i,j]<=thresh and b_mask[i,j] <= thresh):
                #             TN += 1
                pred_label = a_mask // 128
                label = b_mask // 128
                ignore_index = 255
                mask = (pred_label!= ignore_index)
                pred_label = pred_label[mask]
                label = label[mask]
                intersect = pred_label[pred_label == label]
                area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
                area_pred_label, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
                area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
                area_union = area_pred_label + area_label - area_intersect
                total_area_intersect += area_intersect
                total_area_label += area_label
                total_area_union += area_union
                total_area_pred_label += area_pred_label
                TP = area_intersect[1]
                TN = area_intersect[0]
                FP = area_pred_label[1] - area_intersect[1]
                FN = area_pred_label[0] - area_intersect[0]
                IOU = area_union[1]
                accuracy = (TP + TN) / (TP + FP + FN + TN)
                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)
                IOU = (TP / area_union[1] + TN / area_union[0]) / 2.
                precision = TP / (TP+FP)
                F1 = 2.*precision*sensitivity/(precision+sensitivity)
                mean_acc = mean_acc + accuracy
                mean_sen = mean_sen + sensitivity
                mean_spe = mean_spe + specificity
                mean_iou = mean_iou + IOU
                mean_pre = mean_pre + precision
                mean_f1 = mean_f1 + F1
                print('accuracy:{},precision:{},sensitivity:{},specificity:{},f1:{},IOU:{}'.format(accuracy,precision,sensitivity,specificity,F1,IOU))
                rf.write('{}.svs,{},{},{},{},{},{}\n'.format(line.split('.svs')[0],accuracy,precision,sensitivity,specificity,F1,IOU))
        f.close()
        '''
    mean_acc = mean_acc / count
    mean_sen = mean_sen / count
    mean_spe = mean_spe / count
    mean_iou = mean_iou / count
    mean_pre = mean_pre / count
    mean_f1 = mean_f1 / count
    print('mean accuracy:{},mean precision:{}, mean sensitivity:{},mean specificity:{},mean F1:{}, mean IOU:{}'.format(
        mean_acc, mean_pre, mean_sen, mean_spe, mean_f1, mean_iou))
    rf.write('{},{},{},{},{},{},{}\n'.format('average', mean_acc, mean_pre, mean_sen, mean_spe, mean_f1, mean_iou))
    print(
        str(final_result(list(label_mapping.keys())[-num_classes:], total_area_intersect, total_area_label, total_area_union, total_area_pred_label,
                         metric=['mIoU'],rf=rf)))
    print(
        str(final_result(list(label_mapping.keys())[-num_classes:], total_area_intersect, total_area_label, total_area_union, total_area_pred_label,
                         metric=['mDice'],rf=rf)))
    rf.close()


def get_mask_metrics(txt_file,num_classes):
    label_mapping = {
        'Q': 0,
        'NOR': 1,
        'HYP': 2,
        'DYS': 3,
        'CAR': 4,
    }
    result_file = txt_file

    ori_imgpath = '/media/ubuntu/Seagate Basic1/new_optim_data/test_datas/ann_dir'
    mask_save_file='/media/ubuntu/Seagate Basic/work_dirs/04-06/cls_5_pre'

    read_xls = xlrd.open_workbook('/media/ubuntu/Seagate Basic1/v3-test-train-validation-list.xlsx')
    xls_context = read_xls.sheets()[0]

    xml_files = []
    col_datas = xls_context.col_values(colx=0, start_rowx=1)

    for data in col_datas[1:]:
        if data != '':
            xml_files.append(str(data).replace('0F', 'OF'))  # xml lists


    rf = open(result_file, 'w')
    rf.write('{},{},{},{},{},{},{}\n'.format('name', 'accuracy', 'precision', 'sensitive', 'specificity', 'F1', 'IOU'))
    mean_acc = 0
    mean_sen = 0
    mean_spe = 0
    mean_iou = 0
    mean_pre = 0
    mean_f1 = 0
    count = 0
    total_area_intersect = np.zeros((num_classes,), dtype=np.float)
    total_area_union = np.zeros((num_classes,), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes,), dtype=np.float)
    total_area_label = np.zeros((num_classes,), dtype=np.float)
    res = dict()
    for file in xml_files:
        slide_file = glob(f'/media/ubuntu/Seagate Basic1/data-v3/{file.split("/")[-1].split(".")[0]}.svs') + glob(
            f'/media/ubuntu/Seagate Basic1/data-v3/{file.split("/")[-1].split(".")[0]}.tif')
        slide = open_slide(slide_file[0])
        w,h = slide.level_dimensions[0]
        img_size = (h,w)

        pre_gt = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        gt = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        row, col = 0, 0

        data_paths=glob(f'{mask_save_file}/{file.split(".")[0]}-*_mask.png')
        print(f'read xml file : {file} file patch num : {len(data_paths)}')
        for data_index in range(len(data_paths)):
            ori_img = np.array(cv2.imread(f"{ori_imgpath}/{file.split('.')[0]}-{data_index}.png"))
            patch_img=np.array(cv2.imread(f"{mask_save_file}/{file.split('.')[0]}-{data_index}_mask.png"))

            if patch_img.shape!=ori_img.shape:
                patch_img=cv2.resize(patch_img,(ori_img.shape[1],ori_img.shape[0]))
                print('resize patch img , new shape : {} ori img shape : {}'.format(patch_img.shape,ori_img.shape[:-1]))
            assert ori_img.shape==patch_img.shape,f'ori img patch shape : {ori_img.shape}  mask patch img shape : {patch_img.shape}'

            patch_size = patch_img.shape
            if row + patch_size[0] > h:
                patch_img = cv2.resize(patch_img, (h - row,patch_size[1]))
                patch_img = np.array(patch_img)

                ori_img=cv2.resize(ori_img,(h-row,patch_size[1]))
                ori_img=np.array(ori_img)
                print(
                    f'img size: {(h, w)} patch size: {patch_size} new patch size: {patch_img.shape} start_row: {row} patch size row over img size row')

            if col + patch_size[1] > w:
                patch_img = cv2.resize(patch_img, (patch_size[0],w - col))
                patch_img = np.array(patch_img)

                ori_img = cv2.resize(ori_img, ( patch_size[0],w - col))
                ori_img = np.array(ori_img)
                print(
                    f'img size: {(h, w)} patch size: {patch_size} new patch size: {patch_img.shape} start_col: {col} patch size col over img size col')

            patch_size = patch_img.shape

            pre_gt[row:row + patch_size[0], col:col + patch_size[1]] = patch_img[..., 0]
            gt[row:row + patch_size[0], col:col + patch_size[1]] = ori_img[..., 0]

            if col + patch_size[1] == w:
                row = row + patch_size[0]
                col = 0
            else:
                col = col + patch_size[1]

        """
        for point_dict in pre_slide_points:
            cls = list(point_dict.keys())[0]

            points = np.asarray([point_dict[cls]], dtype=np.int32)

            cv2.fillPoly(img=pre_gt, pts=points, color=(label_mapping[cls], label_mapping[cls], label_mapping[cls]))
        """

        if num_classes == 2:
            gt[gt != 0] = 1

        gt, pre_gt = np.array(gt), np.array(pre_gt)
        if gt.shape != pre_gt.shape:
            # print(gt.shape, pre_gt.shape)
            pre_gt = cv2.resize(pre_gt, (gt.shape[1], gt.shape[0]))
            pre_gt = np.array(pre_gt)

        if f'{file.split("/")[-1].split(".")[0]}' not in res.keys():
            res[f'{file.split("/")[-1].split(".")[0]}'] = dict()

        pre_gt = pre_gt[:, :]
        gt = gt[:, :]

        TP = 0
        FP = 0
        TN = 0
        FN = 0
        IOU = 0
        thresh = 0
        count = count + 1

        # pred_label = pre_gt // 128
        # label = gt // 128
        pred_label = pre_gt
        label = gt
        ignore_index = 255
        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]
        intersect = pred_label[pred_label == label]  # pre_true
        area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
        area_pred_label, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
        area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))

        area_union = area_pred_label + area_label - area_intersect  # all region
        total_area_intersect += area_intersect
        total_area_label += area_label
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        TP = area_intersect[1]
        if 'TP' not in res[f'{file.split("/")[-1].split(".")[0]}'].keys():
            res[f'{file.split("/")[-1].split(".")[0]}']['TP'] = int(TP)
        else:
            res[f'{file.split("/")[-1].split(".")[0]}']['TP'] = \
                res[f'{file.split("/")[-1].split(".")[0]}']['TP'] + int(TP)

        TN = area_intersect[0]
        if 'TN' not in res[f'{file.split("/")[-1].split(".")[0]}'].keys():
            res[f'{file.split("/")[-1].split(".")[0]}']['TN'] = int(TN)
        else:
            res[f'{file.split("/")[-1].split(".")[0]}']['TN'] = \
                res[f'{file.split("/")[-1].split(".")[0]}']['TN'] + int(TN)

        FP = area_pred_label[1] - area_intersect[1]
        if 'FP' not in res[f'{file.split("/")[-1].split(".")[0]}'].keys():
            res[f'{file.split("/")[-1].split(".")[0]}']['FP'] = int(FP)
        else:
            res[f'{file.split("/")[-1].split(".")[0]}']['FP'] = \
                res[f'{file.split("/")[-1].split(".")[0]}']['FP'] + int(FP)

        FN = area_pred_label[0] - area_intersect[0]
        if 'FN' not in res[f'{file.split("/")[-1].split(".")[0]}'].keys():
            res[f'{file.split("/")[-1].split(".")[0]}']['FN'] = int(FN)
        else:
            res[f'{file.split("/")[-1].split(".")[0]}']['FN'] = \
                res[f'{file.split("/")[-1].split(".")[0]}']['FN'] + int(FN)

        IOU = (TP / (area_union[1]) + TN / area_union[0]) / 2.
        if np.isnan(IOU):
            IOU = 0
        if 'IOU' not in res[f'{file.split("/")[-1].split(".")[0]}'].keys():
            res[f'{file.split("/")[-1].split(".")[0]}']['IOU'] = IOU
        else:
            res[f'{file.split("/")[-1].split(".")[0]}']['IOU'] = \
                res[f'{file.split("/")[-1].split(".")[0]}']['IOU'] + IOU

    print(res.keys())
    for name in res.keys():
        TP = res[name]['TP']
        TN = res[name]['TN']
        FP = res[name]['FP']
        FN = res[name]['FN']
        IOU = res[name]['IOU']

        accuracy = (TP + TN) / (TP + FP + FN + TN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        if TP == 0 and FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)
        if precision == 0 and sensitivity == 0:
            F1 = 0
        else:
            F1 = 2. * precision * sensitivity / (precision + sensitivity)
        mean_acc = mean_acc + accuracy
        mean_sen = mean_sen + sensitivity
        mean_spe = mean_spe + specificity
        mean_iou = mean_iou + IOU
        mean_pre = mean_pre + precision
        mean_f1 = mean_f1 + F1
        print('name:{},accuracy:{},precision:{},sensitivity:{},specificity:{},f1:{},IOU:{}'.format(name, accuracy,
                                                                                                   precision,
                                                                                                   sensitivity,
                                                                                                   specificity, F1,
                                                                                                   IOU))
        rf.write(
            '{}.svs/tif,{},{},{},{},{},{}\n'.format(name, accuracy, precision, sensitivity, specificity, F1,
                                                    IOU))

    mean_acc = mean_acc / count
    mean_sen = mean_sen / count
    mean_spe = mean_spe / count
    mean_iou = mean_iou / count
    mean_pre = mean_pre / count
    mean_f1 = mean_f1 / count
    print('mean accuracy:{},mean precision:{}, mean sensitivity:{},mean specificity:{},mean F1:{}, mean IOU:{}'.format(
        mean_acc, mean_pre, mean_sen, mean_spe, mean_f1, mean_iou))
    rf.write('{},{},{},{},{},{},{}\n'.format('average', mean_acc, mean_pre, mean_sen, mean_spe, mean_f1, mean_iou))
    print(
        str(final_result(num_classes, total_area_intersect, total_area_label, total_area_union, total_area_pred_label,
                         metric=['mIoU'],rf=rf)))
    print(
        str(final_result(num_classes, total_area_intersect, total_area_label, total_area_union, total_area_pred_label,
                         metric=['mDice'],rf=rf)))
    rf.close()


if __name__ == '__main__':

    """
    print('---------- latest pre multi cls metric results ------------------------')
    txt_file='/media/ubuntu/Seagate Basic/optim_data/test_datas/img_dir/latest_pre/multi_cls_metric_results.txt'
    pre_file='/media/ubuntu/Seagate Basic/optim_data/test_datas/img_dir/latest_pre'
    get_metrics(txt_file,pre_file,num_classes)
    txt_file = '/media/ubuntu/Seagate Basic/optim_data/test_datas/img_dir/latest_pre/two_cls_metric_results.txt'
    print('---------- latest pre two cls metric results ------------------------')
    get_metrics(txt_file,pre_file,2)

    print('---------- best pre multi cls metric results ------------------------')
    txt_file = '/media/ubuntu/Seagate Basic/optim_data/test_datas/img_dir/best_pre/multi_cls_metric_results.txt'
    pre_file = '/media/ubuntu/Seagate Basic/optim_data/test_datas/img_dir/best_pre'
    get_metrics(txt_file, pre_file, num_classes)
    txt_file = '/media/ubuntu/Seagate Basic/optim_data/test_datas/img_dir/best_pre/two_cls_metric_results.txt'
    print('---------- best pre two cls metric results ------------------------')
    get_metrics(txt_file, pre_file, 2)
    """

    # """
    pred_xml='/data/sdc/checkpoints/medicine_res/mix_scale_segformer_multi_cls_with_attn/15000/iter_240000/multi_pre'
    txt_file = f'{"/".join(pred_xml.split("/")[:-1])}/mix_wo_bg.txt'
    
    get_xml_metrics(txt_file,xml_path=pred_xml,num_classes=5)
    # get_mask_metrics(txt_file,num_classes=5)
    # """
    
    
    # cls_map={1:'NOR',2:'HYP',3:'DYS',4:'CAR'}
    # for base_path in ['/data/sdc/medicine_svs/ori_scale_datas/train_datas/ann_dir/*.png','/data/sdc/medicine_svs/multi_scale_datas/train_datas/ann_dir/*.png']:
    #     ann_statis=dict()
    #     for ann_file in tqdm(glob(base_path)):
    #         ann_file=cv2.imread(ann_file)
    #         for cls in np.unique(ann_file):
    #             if cls==0:
    #                 continue
    #             ann_statis[cls_map[cls]]=ann_statis.get(cls_map[cls],0)+np.sum(ann_file==cls)

    #     print(f'Path: {base_path}, ann_statis: {ann_statis}')

    """
    import matplotlib.pyplot as plt
    all_slide_files=glob('/media/ubuntu/Seagate Basic1/data-v3/*.svs')+glob('/media/ubuntu/Seagate Basic1/data-v3/*.tif')

    for file in all_slide_files:
        slide = openslide.open_slide(file)
        img_size = slide.level_dimensions[0]

        slide_region = slide.read_region((0, 0), 0, img_size).convert('RGB')
        slide_region = np.array(slide_region)  # size1,size2,channel  ==> size2,size1,channel

        plt.subplot(121)
        plt.imshow(cv2.resize(slide_region, (4096,4096)))

        for row in range(slide_region.shape[0]):
            for col in range(slide_region.shape[1]):
                pixel_channel=slide_region[row,col,:]
                if pixel_channel[0]>180 and pixel_channel[1]>180 and pixel_channel[2]>180:
                    slide_region[row,col,:]=(0,0,0)

        plt.subplot(122)
        plt.imshow(cv2.resize(slide_region, (4096, 4096)))
        plt.savefig(f'{file.split("/")[-1].split(".")[0]}.png')

        print(f'finsh file : {file.split("/")[-1].split(".")[0]}')
    """