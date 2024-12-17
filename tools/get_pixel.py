import numpy as np
from glob import glob
import xlrd
import xml.etree.ElementTree as ET
import os,cv2
import openslide

label_mapping={
    'Q':0,
    'NOR':1,
    'HYP':2,
    'DYS':3,
    'CAR':4,
}

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


def open_slide_to_mask(slide_file,xml_file):
    slide=openslide.open_slide(slide_file)
    img_size=slide.level_dimensions[0]

    slide_region = slide.read_region((0, 0), 0, img_size).convert('RGB')
    slide_region = np.array(slide_region)  # size1,size2,channel  ==> size2,size1,channel

    img_size=(img_size[1],img_size[0])

    slide_points=xml_to_region(xml_file)

    mask=np.zeros((img_size[0],img_size[1]),dtype=np.uint8)
    # mask=np.tile(np.array([255],dtype=np.uint8),(img_size[0],img_size[1]))

    # ori_img=copy.deepcopy(slide_region)

    for point_dict in slide_points:
        cls=list(point_dict.keys())[0]

        points=np.asarray([point_dict[cls]],dtype=np.int32)

        cv2.fillPoly(img=mask, pts=points, color=(label_mapping[cls],label_mapping[cls],label_mapping[cls]))


    return mask


if __name__ == '__main__':
    read_xls = xlrd.open_workbook('/media/ubuntu/Seagate Basic1/pixel.xlsx')
    xls_context = read_xls.sheets()[0]
    col_datas = xls_context.col_values(colx=0, start_rowx=1)

    for col in col_datas:
        xml_path=f'/media/ubuntu/Seagate Basic1/data-v4/{col}'

        svs_path=f'/media/ubuntu/Seagate Basic1/data-v3/{col.split(".")[0]}.*'
        svs=glob(svs_path)[0]

        mask=open_slide_to_mask(svs,xml_path)

        cls_1 = np.sum(mask == 1)
        cls_2 = np.sum(mask == 2)
        cls_3 = np.sum(mask == 3)
        cls_4 = np.sum(mask == 4)

        print(f'xml file : {col}  Nor : {cls_1} HYP : {cls_2} DYS : {cls_3}  CAR : {cls_4}')

