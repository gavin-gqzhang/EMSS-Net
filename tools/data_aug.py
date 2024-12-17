import xlrd
import numpy as np
from glob import glob
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


def open_slide_to_mask(slide_file,xml_file,split,resized=0,overlap=False,cls_name=None):
    slide=openslide.open_slide(slide_file)
    img_size=slide.level_dimensions[0]

    slide_region = slide.read_region((0, 0), 0, img_size).convert('RGB')
    slide_region = np.array(slide_region)  # size1,size2,channel  ==> size2,size1,channel

    img_size=(img_size[1],img_size[0])

    slide_points=xml_to_region(xml_file)

    mask=np.zeros((img_size[0],img_size[1]),dtype=np.uint8)

    # ori_img=copy.deepcopy(slide_region)

    for point_dict in slide_points:
        cls=list(point_dict.keys())[0]

        if cls.upper()!=cls_name.upper():
            continue

        points=np.asarray([point_dict[cls]],dtype=np.int32)

        cv2.fillPoly(img=mask, pts=points, color=(label_mapping[cls],label_mapping[cls],label_mapping[cls]))

    slide_region[mask==0]=(255,255,255)

    img_dir = '/media/ubuntu/Seagate Basic1/enhance_data/enhance_{}/img_dir/{}-{}-{}'.format(cls_name, cls_name,
                                                                                             xml_file.split('/')[
                                                                                                 -1].split('.')[0],
                                                                                             resized)
    ann_dir = '/media/ubuntu/Seagate Basic1/enhance_data/enhance_{}/ann_dir/{}-{}-{}'.format(cls_name, cls_name,
                                                                                             xml_file.split('/')[
                                                                                                 -1].split('.')[0],
                                                                                             resized)

    if resized != 0:
        ratio = resized / min(img_size)
        img_size = (int(img_size[1] * ratio), int(img_size[0] * ratio))
        mask = cv2.resize(mask, img_size)
        slide_region = cv2.resize(slide_region, img_size)

    os.makedirs(f'{"/".join(img_dir.split("/")[:-1])}',exist_ok=True)
    os.makedirs(f'{"/".join(ann_dir.split("/")[:-1])}',exist_ok=True)



    num_key,drop_win=crop_split(slide_region,mask,img_size,img_dir,ann_dir,crop_size=(1024,1024),split=split,overlap=overlap)

    print(f'process {slide_file} success , resize ratio :{ratio if resized!=0 else 0} , resize :{img_size}, classes : {len(slide_points)} , threshold: 0.05 save patch : {num_key} drop patch : {drop_win}')


    return mask



def crop_split(slide_region,mask_region,img_size,img_dir,ann_dir,crop_size=(2048,2048),split='train',overlap=False):
    start_point=(0,0)
    # region_size=(int(img_size[0]/num_path),int(img_size[1]/num_path))
    split_x=int(img_size[0]/crop_size[0])
    split_y=int(img_size[1]/crop_size[1])
    redun_x,redun_y=False,False
    if img_size[0]%crop_size[0]!=0:
        split_x=split_x+1
        redun_x=True
    if img_size[1]%crop_size[1]!=0:
        split_y=split_y+1
        redun_y=True

    num_key=0
    drop_win=0
    if split=='test':
        for dimension_0 in range(split_x):
            for dimension_1 in range(split_y):
                if dimension_1==split_y-1 and dimension_0!=split_x-1:
                    img_patch = slide_region[start_point[0]:start_point[0] + crop_size[0],
                                start_point[1]:, :]
                    mask_patch = mask_region[start_point[0]:start_point[0] + crop_size[0],
                                 start_point[1]:]
                elif dimension_1==split_y-1 and dimension_0==split_x-1:
                    img_patch = slide_region[start_point[0]:, start_point[1]:, :]
                    mask_patch = mask_region[start_point[0]:, start_point[1]:]
                elif dimension_0==split_x-1 and dimension_1!=split_y-1:
                    img_patch = slide_region[start_point[0]:,start_point[1]:start_point[1] + crop_size[1], :]
                    mask_patch = mask_region[start_point[0]:,start_point[1]:start_point[1] + crop_size[1]]
                else:
                    img_patch = slide_region[start_point[0]:start_point[0] + crop_size[0],
                                start_point[1]:start_point[1] + crop_size[1], :]
                    mask_patch = mask_region[start_point[0]:start_point[0] + crop_size[0],
                                 start_point[1]:start_point[1] + crop_size[1]]

                # cv2.imwrite(f'../data/{dimension_0+dimension_1}.png',img_patch)
                # img_patch.save(f'../data/{num_key}.png')
                cv2.imwrite(f'{img_dir}-{num_key}.png', img_patch)
                cv2.imwrite(f'{ann_dir}-{num_key}.png', mask_patch)
                num_key = num_key + 1

                start_point = (start_point[0], start_point[1] + crop_size[1])


            start_point = (start_point[0] + crop_size[0], 0)


        return num_key, drop_win


    for dimension_0 in range(split_x):
        for dimension_1 in range(split_y):
            img_patch=slide_region[start_point[0]:start_point[0]+crop_size[0],start_point[1]:start_point[1]+crop_size[1],:]
            mask_patch=mask_region[start_point[0]:start_point[0]+crop_size[0],start_point[1]:start_point[1]+crop_size[1]]

            # cv2.imwrite(f'../data/{dimension_0+dimension_1}.png',img_patch)
            # img_patch.save(f'../data/{num_key}.png')
            if np.sum(mask_patch!=0)/(crop_size[0]*crop_size[1])>0.05:
                # for slide_row in range(img_patch.shape[0]):
                #     for slide_col in range(img_patch.shape[1]):
                #         pixel_channel = slide_region[slide_row, slide_col, :]
                #         if pixel_channel[0] > 180 and pixel_channel[1] > 180 and pixel_channel[2] > 180:
                #             mask_patch[slide_row, slide_col] = 255

                cv2.imwrite(f'{img_dir}-{num_key}.png',img_patch)
                cv2.imwrite(f'{ann_dir}-{num_key}.png',mask_patch)
                num_key = num_key + 1
            else:
                drop_win=drop_win+1
            # print(f'save {num_key}th img patch and mask patch success.... start_point:{start_point} region_size:{region_size}')

            if dimension_1==split_y-2 and redun_y:
                start_point=(start_point[0],img_size[1]-crop_size[1])
            else:
                start_point=(start_point[0],start_point[1]+int((crop_size[1])/2) if overlap else start_point[1]+crop_size[1])

        if dimension_0==split_x-2 and redun_x:
            start_point=(img_size[0]-crop_size[0],0)
        else:
            start_point=(start_point[0]+int((crop_size[0])/2) if overlap else start_point[0]+crop_size[0],0)

    return num_key,drop_win




if __name__ == '__main__':
    read_xls = xlrd.open_workbook('/media/ubuntu/Seagate Basic1/nor dys car enhance .xlsx')
    xls_context = read_xls.sheets()[0]

    aug_data = dict()
    for col in [5,6,7]:
        datas=[]
        col_datas=xls_context.col_values(colx=col, start_rowx=0)
        for data in col_datas[1:]:
            if data!='':
                datas.append(str(data).replace('0F', 'OF'))
        aug_data[col_datas[0]]=datas
        del datas

    print(f'Enhance Data list : {aug_data}')

    files = glob('/media/ubuntu/Seagate Basic1/data-v3/*.svs') + glob('/media/ubuntu/Seagate Basic1/data-v3/*.tif')
    svs_files=dict()
    for cls_name in aug_data.keys():
        svs_files[cls_name]=[]
        for file in files:
            file_name = file.split('/')[-1].split('.')[0] + '.xml'
            if file_name in aug_data[cls_name]:
                svs_files[cls_name].append(file)

    svs_files=dict(car=['P-18.svs','P-19.svs','V-93.svs','V-110.svs'])
    # print(f'Enhance file list : {svs_files}')
    resize_list = [ 15360]
    for resize in resize_list:
        for cls_name in svs_files.keys():
            print(f'------- Enhance {cls_name.upper()} datas , resize : {resize} -------\n')
            for file in svs_files[cls_name]:
                # xml_file = '/media/ubuntu/Seagate Basic1/data-v4/' + file.split('/')[-1].split('.')[0] + '.xml'

                root = '/media/ubuntu/Seagate Basic1/CANCER-supplement/'
                xml_file = root + file.split('.')[0] + '.xml'
                file=root+file
                mask = open_slide_to_mask(file, xml_file, 'train', resize, overlap=True,cls_name=cls_name)
