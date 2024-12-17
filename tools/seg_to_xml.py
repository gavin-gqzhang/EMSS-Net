import xlrd
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
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)
# 40X level=0, 20X level=1, 10X level=2
def svsread(path,level):
    slide = open_slide(path)
    image = slide.read_region((0,0),level,slide.level_dimensions[level])
    image = np.array(image.convert("RGB"))
    slide.close()
    return image

def decrease_axis(con):
    ax = con[0][0][0]
    ay = con[0][0][1]
    thresh = 100
    con1 = []
    for j in range(len(con)):
        if j == 0:
           con1.append([ax,ay])
        elif j < len(con) - 1:
            if np.abs(ax - con[j][0][0]) > thresh or np.abs(ay - con[j][0][1]) > thresh:
                ax = con[j][0][0]
                ay = con[j][0][1]
                con1.append([ax, ay])
        else:
            con1.append([con[j][0][0],con[j][0][1]])
    con1.append([con[0][0][0] - 1,con[0][0][1] - 1])
    return con1

def nn_dis(con1,con2):
    dis0 = 100000000000000
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0
    con1 = decrease_axis(con1)
    con2 = decrease_axis(con2)
    for i  in range(len(con1)):
        for j in range(len(con2)):
            dis = np.sqrt(np.power(con1[i][0] - con2[j][0],2)+np.power(con1[i][1]-con2[j][1],2))
            if dis < dis0:
                dis0 = dis
                x0 = con1[i][0]
                y0 = con1[i][1]
                x1 = con2[j][0]
                y1 = con2[j][1]
    return x0,y0,x1,y1

def plot_cut(binary,axis):
    for i in range(len(axis)):
        x0 = axis[i][0]
        y0 = axis[i][1]
        x1 = axis[i][2]
        y1 = axis[i][3]
        seg = 20
        if x1 > x0:
            k = (y1 - y0) * 1.0 / (x1 - x0) * 1.0
            for x in range(x0-seg,x1+seg):
                y = int(k*(x-x0)+y0)
                binary[y-seg:y+seg,x-seg:x+seg]=0
        elif x1 < x0:
            k = (y1 - y0) * 1.0 / (x1 - x0) * 1.0
            for x in range(x1-seg,x0+seg):
                y = int(k*(x-x0)+y0)
                binary[y-seg:y+seg, x-seg:x+seg] = 0
        # else:
        #     if y1 > y0:
        #         binary[y0-seg:y1 + seg,:] = 0
        #     else:
        #         binary[y1 - seg:y0 + seg, :] = 0
    return binary

def find_nn_axis(contours,hierarchy,binary):
    thresh = 10000
    axis = []
    for i in range(len(contours)):
        con1 = contours[i]
        if cv2.contourArea(contours[i]) > thresh:
            if hierarchy[0][i][3] >= 0:
                con2 = contours[hierarchy[0][i][3]]
                x0,y0,x1,y1 = nn_dis(con1,con2)
                # print(x0,y0,x1,y1)
                axis.append([x0,y0,x1,y1])
    if len(axis)>0:
        binary = plot_cut(binary,axis)
    return binary


CLASSES = ('Q', 'NOR', 'HYP', 'DYS', 'CAR')
def contours_to_xml(binary,xml_name,level):
    if level==1:
        h,w = binary.shape
        binary = cv2.resize(np.array(binary).astype('uint8'), (w * 4, h * 4))
    if level ==2:
        h, w = binary.shape
        binary = cv2.resize(np.array(binary * 255).astype('uint8'), (w * 16, h * 16))
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    binary = find_nn_axis(contours, hierarchy, binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    file = open(xml_name,'w')
    file.write('<?xml version=\"1.0\"?>\n')
    file.write('<ASAP_Annotations>\n')
    file.write('\t<Annotations>\n')
    thresh = 10000
    ann_count = 0
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > thresh:
            print(CLASSES[binary[contours[i][0][0][1],contours[i][0][0][0]]])
            file.write('\t\t<Annotation Name=\"Annotation {:d}\" Type=\"Spline\" PartOfGroup=\"None\" '
                       'Color=\"#F4FA58\">\n'.format(ann_count))
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
                elif j < len(contours[i])-1:
                    if np.abs(ax-contours[i][j][0][0]) > thresh1 or np.abs(ay - contours[i][j][0][1]) > thresh1:
                        ax = contours[i][j][0][0]
                        ay = contours[i][j][0][1]
                        file.write(
                            '\t\t\t\t<Coordinate Order=\"{:d}\" X=\"{:.2f}\" Y=\"{:.2f}\" />\n'.format(count, ax, ay))
                        count = count + 1
                else:
                    file.write(
                        '\t\t\t\t<Coordinate Order=\"{:d}\" X=\"{:.2f}\" Y=\"{:.2f}\" />\n'.format(count, contours[i][j][0][0], contours[i][j][0][1]))
                    count = count + 1
                # print(contours[i][j][0])
            file.write('\t\t\t\t<Coordinate Order=\"{:d}\" X=\"{:.2f}\" Y=\"{:.2f}\" />\n'.format(count, contours[i][0][0][0]-1,
                                                                                               contours[i][0][0][1]-1))
            file.write('\t\t</Coordinates>\n')
            file.write('\t\t</Annotation>\n')
    file.write('\t</Annotations>\n')
    file.write('\t<AnnotationGroups />\n')
    file.write('</ASAP_Annotations>')
    file.close()


def get_contour_to_xml(mask,svs,save_path):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    binary = find_nn_axis(contours, hierarchy, mask)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image=svs_copy,contours=contours,contourIdx=-1,color=(255,0,0))
    # cv2.imwrite(f"{save_path.split('.xml')[0]}_mask_in_svs.png",svs_copy)

    file = open(save_path, 'w')
    file.write('<?xml version=\"1.0\"?>\n')
    file.write('<ASAP_Annotations>\n')
    file.write('\t<Annotations>\n')
    thresh = 10000
    ann_count = 0
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > thresh:
            Q_list=[]
            NOR_list=[]
            HYP_list=[]
            DYS_list=[]
            CAR_list=[]
            ax = contours[i][0][0][0]
            ay = contours[i][0][0][1]

            thresh1 = 100
            for j in range(len(contours[i])):
                if j == 0:
                    if CLASSES[mask[ay,ax]]=='Q':
                        Q_list.append([ax,ay])
                    elif CLASSES[mask[ay,ax]]=='NOR':
                        NOR_list.append([ax,ay])
                    elif CLASSES[mask[ay,ax]] == 'HYP':
                        HYP_list.append([ax,ay])
                    elif CLASSES[mask[ay,ax]] == 'DYS':
                        DYS_list.append([ax,ay])
                    else:
                        CAR_list.append([ax, ay])

                elif j < len(contours[i]) - 1:
                    if np.abs(ax - contours[i][j][0][0]) > thresh1 or np.abs(ay - contours[i][j][0][1]) > thresh1:
                        ax = contours[i][j][0][0]
                        ay = contours[i][j][0][1]
                        if CLASSES[mask[ay, ax]] == 'Q':
                            Q_list.append([ax, ay])
                        elif CLASSES[mask[ay, ax]] == 'NOR':
                            NOR_list.append([ax, ay])
                        elif CLASSES[mask[ay, ax]] == 'HYP':
                            HYP_list.append([ax, ay])
                        elif CLASSES[mask[ay, ax]] == 'DYS':
                            DYS_list.append([ax, ay])
                        else:
                            CAR_list.append([ax, ay])

                else:
                    ax=contours[i][j][0][0]
                    ay=contours[i][j][0][1]
                    if CLASSES[mask[ay,ax]]=='Q':
                        Q_list.append([ax,ay])
                    elif CLASSES[mask[ay,ax]]=='NOR':
                        NOR_list.append([ax,ay])
                    elif CLASSES[mask[ay,ax]] == 'HYP':
                        HYP_list.append([ax,ay])
                    elif CLASSES[mask[ay,ax]] == 'DYS':
                        DYS_list.append([ax,ay])
                    else:
                        CAR_list.append([ax, ay])

                # print(contours[i][j][0])
            ax=contours[i][0][0][0] - 1
            ay=contours[i][0][0][1] - 1
            if CLASSES[mask[ay, ax]] == 'Q':
                Q_list.append([ax, ay])
            elif CLASSES[mask[ay, ax]] == 'NOR':
                NOR_list.append([ax, ay])
            elif CLASSES[mask[ay, ax]] == 'HYP':
                HYP_list.append([ax, ay])
            elif CLASSES[mask[ay, ax]] == 'DYS':
                DYS_list.append([ax, ay])
            else:
                CAR_list.append([ax, ay])

            count = 0
            for i in range(len(Q_list)):
                [ax,ay]=Q_list[i]
                if i==0:
                    file.write(
                        '\t\t<Annotation Name=\"Annotation {:d}\" Type=\"Spline\" PartOfGroup=\"Q\" Color=\"#aa55ff\">\n'.format(
                            ann_count))
                    file.write('\t\t\t<Coordinates>\n')
                    ann_count=ann_count+1

                file.write(
                    '\t\t\t\t<Coordinate Order=\"{:d}\" X=\"{:.2f}\" Y=\"{:.2f}\" />\n'.format(count, ax, ay))

                count = count + 1
                if i==len(Q_list)-1:
                    file.write('\t\t</Coordinates>\n')
                    file.write('\t\t</Annotation>\n')

            count = 0
            for i in range(len(NOR_list)):
                [ax, ay] = NOR_list[i]
                if i == 0:
                    file.write(
                        '\t\t<Annotation Name=\"Annotation {:d}\" Type=\"Spline\" PartOfGroup=\"NOR\" Color=\"#64FE2E\">\n'.format(
                            ann_count))
                    file.write('\t\t\t<Coordinates>\n')
                    ann_count = ann_count + 1

                file.write(
                    '\t\t\t\t<Coordinate Order=\"{:d}\" X=\"{:.2f}\" Y=\"{:.2f}\" />\n'.format(count, ax, ay))

                count = count + 1
                if i == len(NOR_list) - 1:
                    file.write('\t\t</Coordinates>\n')
                    file.write('\t\t</Annotation>\n')

            count = 0
            for i in range(len(HYP_list)):
                [ax, ay] = HYP_list[i]
                if i == 0:
                    file.write(
                        '\t\t<Annotation Name=\"Annotation {:d}\" Type=\"Spline\" PartOfGroup=\"HYP\" Color=\"#0000ff\">\n'.format(
                            ann_count))
                    file.write('\t\t\t<Coordinates>\n')
                    ann_count = ann_count + 1

                file.write(
                    '\t\t\t\t<Coordinate Order=\"{:d}\" X=\"{:.2f}\" Y=\"{:.2f}\" />\n'.format(count, ax, ay))

                count = count + 1
                if i == len(HYP_list) - 1:
                    file.write('\t\t</Coordinates>\n')
                    file.write('\t\t</Annotation>\n')

            count = 0
            for i in range(len(DYS_list)):
                [ax, ay] = DYS_list[i]
                if i == 0:
                    file.write(
                        '\t\t<Annotation Name=\"Annotation {:d}\" Type=\"Spline\" PartOfGroup=\"DYS\" Color=\"#ffff00\">\n'.format(
                            ann_count))
                    file.write('\t\t\t<Coordinates>\n')
                    ann_count = ann_count + 1

                file.write(
                    '\t\t\t\t<Coordinate Order=\"{:d}\" X=\"{:.2f}\" Y=\"{:.2f}\" />\n'.format(count, ax, ay))

                count = count + 1
                if i == len(DYS_list) - 1:
                    file.write('\t\t</Coordinates>\n')
                    file.write('\t\t</Annotation>\n')

            count = 0
            for i in range(len(CAR_list)):
                [ax, ay] = CAR_list[i]
                if i == 0:
                    file.write(
                        '\t\t<Annotation Name=\"Annotation {:d}\" Type=\"Spline\" PartOfGroup=\"CAR\" Color=\"#ff0000\">\n'.format(
                            ann_count))
                    file.write('\t\t\t<Coordinates>\n')
                    ann_count = ann_count + 1

                file.write(
                    '\t\t\t\t<Coordinate Order=\"{:d}\" X=\"{:.2f}\" Y=\"{:.2f}\" />\n'.format(count, ax, ay))

                count=count+1
                if i == len(CAR_list) - 1:
                    file.write('\t\t</Coordinates>\n')
                    file.write('\t\t</Annotation>\n')

    file.write('\t</Annotations>\n')
    file.write('\t<AnnotationGroups />\n')
    file.write('</ASAP_Annotations>')
    file.close()


def image_to_xml():
    # file_name = 'results_20X_lscc'
    # imgpath = '/home/ubuntu/ssd/data/20211206/seg/HNSC/{}/'.format(file_name)
    imgpath = '/media/ubuntu/Seagate Basic/optim_data/test_datas/img_dir/pre/'
    # data_all = glob(imgpath + '*_mask.png')
    level = 0
    read_xls=xlrd.open_workbook('/media/ubuntu/Seagate Basic1/data-v3/test-train-validation-list.xlsx')
    xls_context=read_xls.sheets()[0]

    test_lists=[]
    col_datas = xls_context.col_values(colx=0, start_rowx=1)

    for data in col_datas[1:]:
       if data!='':
          test_lists.append(str(data).replace('0F','OF'))  # xml lists

    test_lists=['OF-5-E.xml','OF-6-E.xml','OF-10-E.xml','OF-12-2-E.xml','OF-13-E.xml','OF-17-E.xml','OF-30-E.xml','OF-46-E.xml','OF-56-E.xml','OF-80-E.xml','W-6b.xml','W-14.xml','W-20.xml']

    for f_img_name in test_lists:
        f_img_name=f_img_name.split('.')[0]  # overall data name
        svs_path=glob(f'/media/ubuntu/Seagate Basic1/data-v3/{f_img_name}.svs')+glob(f'/media/ubuntu/Seagate Basic1/data-v3/{f_img_name}.tif')
        data_paths=glob(imgpath+f'{f_img_name}-*_mask.png')  #  split data path lists
        svs=open_slide(svs_path[0])
        img_size=svs.level_dimensions[level]
        img_size=(img_size[1],img_size[0])
        print(f'{svs_path[0]} shape: {img_size}')
        overall_mask=np.zeros(img_size)
        row,col=0,0
        for split_data in data_paths:
            patch_img = np.array(cv2.imread(split_data))  # read split data
            patch_size=patch_img.shape
            # print('start point :{}  crop size :{} img_size :{}'.format(f'({row},{col})',patch_size,img_size))
            if row+patch_size[0]>img_size[0]:
                patch_img=cv2.resize(patch_img,(img_size[0]-row,patch_size[1]))
                print(
                    f'img size: {img_size} patch size: {patch_size} new patch size: {patch_img.shape} start_row: {row} patch size row over img size row')
                patch_size = patch_img.shape
            if col+patch_size[1]>img_size[1]:
                patch_img=cv2.resize(patch_img,(patch_size[0],img_size[1]-col))
                print(
                    f'img size: {img_size} patch size: {patch_size} new patch size: {patch_img.shape} start_col: {col} patch size col over img size col')

            overall_mask[row:row+patch_size[0],col:col+patch_size[1]]=patch_img[...,0]
            if col+patch_size[1]==img_size[1]:
                row=row+patch_size[0]
                col=0
            else:
                col=col+patch_size[1]

        # h,w,_ = results.shape

        final_results = cv2.resize(overall_mask, (img_size[1],img_size[0]))
        results = final_results

        # print(final_results.shape)
        # cv2.imwrite(data.replace('/results/', '/fine_xml/').replace('.svs', '.png'), results)
        kernel = np.ones((7, 7), dtype=np.uint8)
        binary = cv2.morphologyEx((results).astype('uint8'),cv2.MORPH_OPEN,kernel,iterations=5)
        # binary = results
        # print(np.unique(binary))
        # cv2.imwrite(data.replace('/results/', '/fine_xml/').replace('.svs', '.png'), (binary).astype('uint8'))
        # contours_to_xml((binary[:,:,0]//128).astype('uint8'), image_path.replace('.png','.xml'), level)
        # contours_to_xml((binary[:,:,0]).astype('uint8'), image_path.replace('.png','.xml'), level)
        # print(binary.shape)
        os.makedirs(f'{imgpath}/xml/',exist_ok=True)
        get_contour_to_xml((binary[:,:]).astype('uint8'),svs,f'{imgpath}/xml/{f_img_name}.xml')


if __name__ == '__main__':
    image_to_xml()