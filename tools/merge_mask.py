import copy
import os
from glob import glob

import cv2
import numpy as np
import openslide


def merge_mask_to_xml():
    imgpath = '/media/ubuntu/Seagate Basic/merge_data/img_dir/pre/'
    test_lists = glob('/media/ubuntu/Seagate Basic1/data-v4/*.xml')

    for ori_name in test_lists:
        ori_name=ori_name.split('/')[-1].split('.')[0]
        svs_path = glob(f'/media/ubuntu/Seagate Basic1/data-v3/{ori_name}.svs') + glob(
            f'/media/ubuntu/Seagate Basic1/data-v3/{ori_name}.tif')
        print(f'read svs paths : {svs_path}  file name : {ori_name}')
        pre_mask_paths = glob(imgpath + f'{ori_name}-*_mask.png')  # split data path lists
        svs = openslide.open_slide(svs_path[0])
        w, h = svs.level_dimensions[0]
        overall_mask = np.tile(np.array([255]), (h, w))
        row, col = 0, 0
        for idx in range(len(pre_mask_paths)):
            ori_img = np.array(cv2.imread(f"{'/'.join(imgpath.split('/')[:-2])}/{ori_name}-{idx}.png"))
            ori_mask = np.array(cv2.imread(f"{'/'.join(imgpath.split('/')[:-3])}/ann_dir/{ori_name}-{idx}.png"))
            patch_img = np.array(cv2.imread(imgpath + f'{ori_name}-{idx}_mask.png'))  # read split data
            # patch_ori_img=np.array(cv2.imread(imgpath + f'{f_img_name}-{data_index}_mask_in_ori.png'))
            if patch_img.shape[:2] != ori_img.shape[:2]:
                patch_img = cv2.resize(patch_img, (ori_img.shape[1], ori_img.shape[0]))
                # patch_ori_img=cv2.resize(patch_ori_img,(ori_img.shape[1],ori_img.shape[0]))
                print(
                    'resize patch img , new shape : {} ori img shape : {}'.format(patch_img.shape, ori_img.shape[:-1]))
            assert ori_img.shape[:2] == patch_img.shape[:2], f'ori img patch shape : {ori_img.shape}  mask patch img shape : {patch_img.shape}'

            if ori_mask.shape[:2] != ori_img.shape[:2]:
                ori_mask = cv2.resize(ori_mask, (ori_img.shape[1], ori_img.shape[0]))
                # patch_ori_img=cv2.resize(patch_ori_img,(ori_img.shape[1],ori_img.shape[0]))
                print(
                    'resize patch img , new shape : {} ori img shape : {}'.format(ori_mask.shape, ori_img.shape[:-1]))
            assert ori_img.shape == patch_img.shape, f'ori img patch shape : {ori_img.shape} ori mask patch img shape : {ori_mask.shape}'

            patch_size = patch_img.shape

            if row + patch_size[0]>h:
                patch_img=cv2.resize(patch_img,(h-row,patch_size[1]))
                patch_img=np.array(patch_img)

                ori_mask=cv2.resize(ori_mask, (h - row, patch_size[1]))
                ori_mask = np.array(ori_mask)

                print(
                    f'img size: {(h, w)} patch size: {patch_size} new patch size: {patch_img.shape} start_row: {row} patch size row over img size row')

            if col + patch_size[1] > w:
                patch_img = cv2.resize(patch_img, (patch_size[0],w - col))
                patch_img = np.array(patch_img)

                ori_mask = cv2.resize(ori_mask, ( patch_size[0],w - col))
                ori_mask = np.array(ori_mask)
                print(
                    f'img size: {(h, w)} patch size: {patch_size} new patch size: {patch_img.shape} start_col: {col} patch size col over img size col')

            patch_size = patch_img.shape

            if len(patch_img.shape)==3:
                patch_img=patch_img[...,0]
            if len(ori_mask.shape)==3:
                ori_mask=ori_mask[...,0]

            overall_mask[row:row + patch_size[0], col:col + patch_size[1]] = np.multiply(patch_img,ori_mask)

            if col + patch_size[1] == w:
                row = row + patch_size[0]
                col = 0
            else:
                col = col + patch_size[1]

        print(f'process {ori_name} success')
        kernel = np.ones((7, 7), dtype=np.uint8)
        binary = cv2.morphologyEx((overall_mask).astype('uint8'), cv2.MORPH_OPEN, kernel, iterations=5)
        # os.makedirs(f'{imgpath}/new_xml/', exist_ok=True)
        os.makedirs(f'/media/ubuntu/Seagate Basic1/data-v4/new_xml/', exist_ok=True)
        get_contour_to_xml((binary[:, :]).astype('uint8'),
                                   f'/media/ubuntu/Seagate Basic1/data-v4/new_xml/{ori_name}.xml', )


CLASSES = ('Q', 'NOR', 'HYP', 'DYS', 'CAR')

def get_contour_to_xml(mask, save_path):

    # cv2.drawContours(image=svs_copy,contours=contours,contourIdx=-1,color=(255,0,0))
    # cv2.imwrite(f"{save_path.split('.xml')[0]}_mask_in_svs.png",svs_copy)

    file = open(save_path, 'w')
    file.write('<?xml version=\"1.0\"?>\n')
    file.write('<ASAP_Annotations>\n')
    file.write('\t<Annotations>\n')
    thresh = 15000
    # thresh = 0
    ann_count = 0
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
    thread_1=25
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


if __name__ == '__main__':
    merge_mask_to_xml()