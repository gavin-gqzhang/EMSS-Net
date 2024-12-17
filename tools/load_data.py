import copy
import random
import xml.etree.ElementTree as ET
import os,cv2
import openslide
from PIL import ImageDraw, Image
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
import xlrd
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as transforms
import multiprocessing

label_mapping={
    'Q':0,
    'NOR':1,
    'HYP':2,
    'DYS':3,
    'CAR':4,
}

label_color={
    "NOR":(135,206,235), #yellow
    "HYP":(255,128,0), # blue
    "DYS":(135,38,87), # purple
    "CAR":(255,0,0) # red
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


def open_slide_to_mask(slide_file,xml_file,split,resized=0,overlap=False):
    slide=openslide.open_slide(slide_file)
    img_size=slide.level_dimensions[0]

    slide_region = slide.read_region((0, 0), 0, img_size).convert('RGB')
    slide_region = np.array(slide_region)  # size1,size2,channel  ==> size2,size1,channel

    img_size=slide_region.shape[:2]

    slide_points=xml_to_region(xml_file)

    mask=np.zeros((img_size[0],img_size[1]),dtype=np.uint8)
    # mask=np.tile(np.array([255],dtype=np.uint8),(img_size[0],img_size[1]))

    # ori_img=copy.deepcopy(slide_region)

    for point_dict in slide_points:
        cls=list(point_dict.keys())[0]

        points=np.asarray([point_dict[cls]],dtype=np.int32)

        cv2.fillPoly(img=mask, pts=points, color=(label_mapping[cls],label_mapping[cls],label_mapping[cls]))


    img_dir = '/media/ubuntu/Seagate Basic/nor_car_data/{}_datas/img_dir/{}-{}'.format(split,
                                                                              slide_file.split('/')[-1].split('.')[0],resized)
    ann_dir = '/media/ubuntu/Seagate Basic/nor_car_data/{}_datas/ann_dir/{}-{}'.format(split,
                                                                                slide_file.split('/')[-1].split('.')[0],resized)

    if resized != 0:
        ratio = resized / min(img_size)
        img_size = (int(img_size[1] * ratio),int(img_size[0] * ratio))
        mask = cv2.resize(mask, img_size)
        slide_region = cv2.resize(slide_region, img_size)

    os.makedirs(f'{"/".join(img_dir.split("/")[:-1])}',exist_ok=True)
    os.makedirs(f'{"/".join(ann_dir.split("/")[:-1])}',exist_ok=True)


    num_key,drop_win=crop_split(slide_region,mask,img_size,img_dir,ann_dir,crop_size=(1024,1024),split=split,overlap=overlap)

    print(f'process {slide_file} success , overlap : {overlap} , resize ratio :{ratio if resized!=0 else 0} , resize :{img_size}, classes : {len(slide_points)} , threshold: 0.05 save patch : {num_key} drop patch : {drop_win}')


    return mask


def rotation_openslide_to_mask(base_path,slide_file,xml_file,split,resized=0,angles=[],crop_size=(1024,1024),drop_rate=0.35,overlap=False):
    img_dir = '{}/{}_datas/img_dir'.format(base_path, split)
    ann_dir = '{}/{}_datas/ann_dir'.format(base_path, split)

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    slide=openslide.open_slide(slide_file)
    img_size=slide.level_dimensions[0]

    slide_region = slide.read_region((0, 0), 0, img_size).convert('RGB')
    slide_region = np.array(slide_region)  # size1,size2,channel  ==> size2,size1,channel

    img_size=slide_region.shape[:2]

    slide_points=xml_to_region(xml_file)

    mask=np.zeros((img_size[0],img_size[1]),dtype=np.uint8)

    for point_dict in slide_points:
        cls=list(point_dict.keys())[0]

        points=np.asarray([point_dict[cls]],dtype=np.int32)

        cv2.fillPoly(img=mask, pts=points, color=(label_mapping[cls],label_mapping[cls],label_mapping[cls]))
        
    """
    rgb_mask = np.tile(np.array([255], dtype=np.uint8), (img_size[0], img_size[1], 3))
    resized_img=cv2.resize(slide_region,(1024,1024))

    rgb_mask[mask==1]=list(label_color.values())[0]
    rgb_mask[mask==2]=list(label_color.values())[1]
    rgb_mask[mask==3]=list(label_color.values())[2]
    rgb_mask[mask==4]=list(label_color.values())[3]

    resized_mask=cv2.resize(rgb_mask,(1024,1024))

    cv2.imwrite('origin_rgb.png',resized_img)
    cv2.imwrite('origin_mask.png',resized_mask)
    """

    if resized != 0:
        ratio = resized / min(img_size)
        img_size = (int(img_size[1] * ratio),int(img_size[0] * ratio))
        mask = cv2.resize(mask, img_size)
        slide_region = cv2.resize(slide_region, img_size)

    img_name, ann_name = f'{img_dir}/{slide_file.split("/")[-1].split(".")[0]}-{resized}-0', f'{ann_dir}/{slide_file.split("/")[-1].split(".")[0]}-{resized}-0'

    num_key,drop_win=crop_split(slide_region,mask,img_size,img_name,ann_name,crop_size=crop_size,split=split,overlap=overlap,drop_rate=drop_rate)
    print(
        f'process {slide_file} success, overlap : {overlap}, resize ratio :{ratio if resized != 0 else 0}, resize :{img_size}, drop rate: {drop_rate}, rotated angle: 0, classes : {len(slide_points)}, save patch : {num_key}, drop patch : {drop_win}, mask classes: {np.unique(mask)}')

    if len(angles)!=0:
        to_pil=transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage()
        ])

        for angle in angles:
            try:
                rotated_img = to_pil(slide_region).rotate(angle, expand=True, fillcolor=255)
                rotated_img = np.array(rotated_img)

                # cv2.imwrite('rotation_rgb.png',cv2.resize(rotated_img,(1024,1024)))

                rotated_mask = to_pil(mask).rotate(angle, expand=True, fillcolor=00)
                rotated_mask = np.array(rotated_mask)

                img_name, ann_name = f'{img_dir}/{slide_file.split("/")[-1].split(".")[0]}-{resized}-{angle}', f'{ann_dir}/{slide_file.split("/")[-1].split(".")[0]}-{resized}-{angle}'

                num_key, drop_win = crop_split(rotated_img, rotated_mask, img_size, img_name, ann_name, crop_size=crop_size,
                                               split=split, overlap=overlap, drop_rate=drop_rate)
                print(
                    f'process {slide_file} success, overlap : {overlap}, resize ratio :{ratio if resized != 0 else 0}, resize :{img_size}, drop rate: {drop_rate}, rotated angle: {angle}, classes : {len(slide_points)}, save patch : {num_key}, drop patch : {drop_win}, mask classes: {np.unique(mask)}')
            except RuntimeError as e:
                print(f'process {slide_file} error, error info: {e}')
                continue

    """
    rgb_mask = np.tile(np.array([255], dtype=np.uint8), rotated_img.shape)
    rgb_mask[rotated_mask == 1] = list(label_color.values())[0]
    rgb_mask[rotated_mask == 2] = list(label_color.values())[1]
    rgb_mask[rotated_mask == 3] = list(label_color.values())[2]
    rgb_mask[rotated_mask == 4] = list(label_color.values())[3]
    
    cv2.imwrite('rotation_mask.png',cv2.resize(rgb_mask,(1024,1024)))
    
    """

    return mask


def patch_split(slide_region,mask_region,img_size,img_dir,ann_dir,num_path=16):
    start_point=(0,0)
    region_size=(int(img_size[0]/num_path),int(img_size[1]/num_path))

    num_key=0
    for dimension_0 in range(num_path):
        for dimension_1 in range(num_path):
            img_patch=slide_region[start_point[0]:start_point[0]+region_size[0],start_point[1]:start_point[1]+region_size[1],:]
            mask_patch=mask_region[start_point[0]:start_point[0]+region_size[0],start_point[1]:start_point[1]+region_size[1]]

            # cv2.imwrite(f'../data/{dimension_0+dimension_1}.png',img_patch)
            # img_patch.save(f'../data/{num_key}.png')
            cv2.imwrite(f'{img_dir}-{num_key}.png',img_patch)
            cv2.imwrite(f'{ann_dir}-{num_key}.png',mask_patch)
            # print(f'save {num_key}th img patch and mask patch success.... start_point:{start_point} region_size:{region_size}')

            start_point=(start_point[0],start_point[1]+region_size[1])

            if dimension_1==num_path-2:
                temp=region_size
                region_size=(region_size[0],img_size[1]-start_point[1])
            num_key=num_key+1
        region_size=temp
        start_point=(start_point[0]+region_size[0],0)
        if dimension_0==num_path-2:
            region_size=(img_size[0]-start_point[0],region_size[1])


def crop_split(slide_region,mask_region,img_size,img_dir,ann_dir,crop_size=(2048,2048),split='train',overlap=False,drop_rate=0.35):
    start_point=(0,0)
    # region_size=(int(img_size[0]/num_path),int(img_size[1]/num_path))
    step_x,step_y=crop_size[0]//4,crop_size[0]//4
    split_x=(img_size[0]-crop_size[0]//step_x)+1
    split_y=(img_size[1]-crop_size[1]//step_x)+1
    redun_x,redun_y=False,False
    if img_size[0]-(split_x-1)*step_x>crop_size[0]:
        split_x=split_x+1
        redun_x=True
    if img_size[1]-(split_y-1)*step_y>crop_size[1]:
        split_y=split_y+1
        redun_y=True

    num_key=0
    drop_win=0
    if split=='test':
        for dimension_0 in tqdm(range(split_x)):
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
                # for slide_row in range(img_patch.shape[0]):
                #     for slide_col in range(img_patch.shape[1]):
                #         pixel_channel = slide_region[slide_row, slide_col, :]
                #         if pixel_channel[0] > 180 and pixel_channel[1] > 180 and pixel_channel[2] > 180:
                #             mask_patch[slide_row, slide_col] = 255

                cv2.imwrite(f'{img_dir}-{num_key}.png', img_patch)
                cv2.imwrite(f'{ann_dir}-{num_key}.png', mask_patch)
                num_key = num_key + 1

                start_point = (start_point[0], start_point[1] + crop_size[1])


            start_point = (start_point[0] + crop_size[0], 0)


        return num_key, drop_win


    for dimension_0 in tqdm(range(split_x)):
        for dimension_1 in range(split_y):
            img_patch=slide_region[start_point[0]:start_point[0]+crop_size[0],start_point[1]:start_point[1]+crop_size[1],:]
            mask_patch=mask_region[start_point[0]:start_point[0]+crop_size[0],start_point[1]:start_point[1]+crop_size[1]]

            if np.sum(mask_patch!=0)/(crop_size[0]*crop_size[1])>drop_rate:
                # for slide_row in range(img_patch.shape[0]):
                #     for slide_col in range(img_patch.shape[1]):
                #         pixel_channel = slide_region[slide_row, slide_col, :]
                #         if pixel_channel[0] > 180 and pixel_channel[1] > 180 and pixel_channel[2] > 180:
                #             mask_patch[slide_row, slide_col] = 255ni

                cv2.imwrite(f'{img_dir}-{num_key}.png',img_patch)
                cv2.imwrite(f'{ann_dir}-{num_key}.png',mask_patch)
                num_key = num_key + 1
            else:
                drop_win=drop_win+1
            # print(f'save {num_key}th img patch and mask patch success.... start_point:{start_point} region_size:{region_size}')

            if dimension_1==split_y-1 and redun_y:
                start_point=(start_point[0],img_size[1]-crop_size[1])
            else:
                start_point=(start_point[0],start_point[1]+int((crop_size[1])/2) if overlap else start_point[1]+crop_size[1])

        if dimension_0==split_x-1 and redun_x:
            start_point=(img_size[0]-crop_size[0],0)
        else:
            start_point=(start_point[0]+int((crop_size[0])/2) if overlap else start_point[0]+crop_size[0],0)

    return num_key,drop_win


def data_ana():
    ann_imgs = glob(f'/media/ubuntu/Seagate Basic/nor_car_data/train_datas/ann_dir/*.png')
    cls_0,cls_1, cls_2, cls_3, cls_4 = 0, 0, 0, 0,0
    for ann in tqdm(ann_imgs):
        mask = np.array(cv2.imread(ann))
        cls_0+=np.sum(mask==0)
        cls_1 += np.sum(mask == 1)
        cls_4 += np.sum(mask == 4)
    print(
        f'original train data patch , Background:{cls_0} NOR:{cls_1} CAR:{cls_4}')

    """
    ann_imgs = glob(f'/media/ubuntu/Seagate Basic1/enhance_data/*/ann_dir/*.png')
    for ann in tqdm(ann_imgs):
        mask = np.array(cv2.imread(ann))
        cls_1 += np.sum(mask == 1)
        cls_2 += np.sum(mask == 2)
        cls_3 += np.sum(mask == 3)
        cls_4 += np.sum(mask == 4)

    print(
        f'enhanced train data cls_1:{cls_1} cls_2:{cls_2} cls_3:{cls_3} cls_4:{cls_4}')

    """

    val_ann_imgs = glob(f'/media/ubuntu/Seagate Basic/nor_car_data/val_datas/ann_dir/*png')
    cls_0,cls_1, cls_2, cls_3, cls_4 = 0, 0, 0, 0,0
    for ann in tqdm(val_ann_imgs):
        mask = np.array(cv2.imread(ann))
        cls_0+=np.sum(mask==0)
        cls_1 += np.sum(mask == 1)
        cls_4 += np.sum(mask == 4)
    print(
        f'val data patch , Background:{cls_0} NOR:{cls_1} CAR:{cls_4}')

    """
    test_ann_imgs = glob(f'/media/ubuntu/Seagate Basic/nor_car_data/test_datas/ann_dir/*png')
    cls_0,cls_1, cls_2, cls_3, cls_4 = 0, 0, 0, 0,0
    for ann in tqdm(test_ann_imgs):
        mask = np.array(cv2.imread(ann))
        cls_0+=np.sum(mask==0)
        cls_1 += np.sum(mask == 1)
        cls_4 += np.sum(mask == 4)
    print(
        f'test data patch , cls_0:{cls_0} cls_1:{cls_1} cls_4:{cls_4}')
    """

    """
    test_ann_imgs = glob(f'/media/ubuntu/Seagate Basic1/new_optim_data/test_datas/ann_dir/*png')
    test_cls_1, test_cls_2, test_cls_3, test_cls_4 = 0, 0, 0, 0
    for ann in tqdm(test_ann_imgs):
        mask = np.array(cv2.imread(ann))
        test_cls_1 += np.sum(mask == 1)
        test_cls_2 += np.sum(mask == 2)
        test_cls_3 += np.sum(mask == 3)
        test_cls_4 += np.sum(mask == 4)
        # if np.sum(mask==1)>0:
        #     test_cls_1+=1
        # if np.sum(mask==2)>0:
        #     test_cls_2+=1
        # if np.sum(mask==3)>0:
        #     test_cls_3+=1
        # if np.sum(mask==4)>0:
        #     test_cls_4+=1

    print(
        f'test data patch cls_1:{test_cls_1} cls_2:{test_cls_2} cls_3:{test_cls_3} cls_4:{test_cls_4}')
    """


def enhance_train_split():
    overlap = True
    resize_list=[0,8192]
    read_xls = xlrd.open_workbook('/media/ubuntu/Seagate Basic1/nor dys car enhance .xlsx')
    xls_context = read_xls.sheets()[0]

    enhance_list=dict()
    for col_num in [5,6,7]:
        col_datas = xls_context.col_values(colx=col_num, start_rowx=0)
        enhance_list[col_datas[0]]=[]
        for data in col_datas[1:]:
                if data != '':
                    enhance_list[col_datas[0]].append(str(data).replace('0F', 'OF'))
    error_files=[]
    for enhance_cls in enhance_list.keys():
        for file in enhance_list[enhance_cls]:
            xml_file = '/media/ubuntu/Seagate Basic1/data-v4/' + file
            svs_file = (glob(f'/media/ubuntu/Seagate Basic1/data-v3/{file.split(".")[0]}.tif')+glob(f'/media/ubuntu/Seagate Basic1/data-v3/{file.split(".")[0]}.svs'))[0]

            for resized in resize_list:
                try:
                    slide = openslide.open_slide(svs_file)
                except:
                    error_files.append(file)
                    print(f'ERROR File : {file} svs file : {svs_file}')
                    continue
                img_size = slide.level_dimensions[0]

                slide_region = slide.read_region((0, 0), 0, img_size).convert('RGB')
                slide_region = np.array(slide_region)  # size1,size2,channel  ==> size2,size1,channel

                img_size = (img_size[1], img_size[0])

                slide_points = xml_to_region(xml_file)

                mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)

                for point_dict in slide_points:
                    cls = list(point_dict.keys())[0]
                    if cls.upper()!=enhance_cls.upper():
                        continue

                    points = np.asarray([point_dict[cls]], dtype=np.int32)

                    cv2.fillPoly(img=mask, pts=points, color=(label_mapping[cls], label_mapping[cls], label_mapping[cls]))


                img_dir = '/media/ubuntu/Seagate Basic1/enhance_data/enhance_{}/img_dir/{}-{}-{}'.format(enhance_cls,enhance_cls,xml_file.split('/')[-1].split('.')[0],resized)
                ann_dir = '/media/ubuntu/Seagate Basic1/enhance_data/enhance_{}/ann_dir/{}-{}-{}'.format(enhance_cls,enhance_cls,xml_file.split('/')[-1].split('.')[0],resized)
                if resized != 0:
                    ratio = resized / min(img_size)
                    img_size = (int(img_size[1] * ratio), int(img_size[0] * ratio))
                    mask = cv2.resize(mask, img_size)
                    slide_region = cv2.resize(slide_region, img_size)

                os.makedirs(f'{"/".join(img_dir.split("/")[:-1])}', exist_ok=True)
                os.makedirs(f'{"/".join(ann_dir.split("/")[:-1])}', exist_ok=True)

                num_key, drop_win = crop_split(slide_region, mask, img_size, img_dir, ann_dir, crop_size=(1024, 1024),
                                               split='train', overlap=overlap)

                print(
                    f'process {svs_file} success , overlap : {overlap} , resize ratio :{ratio if resized != 0 else 0} , resize :{img_size}, classes : {len(slide_points)} , threshold: 0.05 save patch : {num_key} drop patch : {drop_win}')

    print(f'Error files :{error_files}')


def multi_process_data(thread=1):
    overlap=True
    read_xls=xlrd.open_workbook('/data/sdc/medicine_svs/v3-test-train-validation-list.xlsx')
    xls_context=read_xls.sheets()[0]

    train_lists=[]
    test_lists=[]
    validation_lists=[]
    for col_num in [0,2,4]:
        col_datas = xls_context.col_values(colx=col_num, start_rowx=0)
        if col_datas[0]=='train':
            for data in col_datas[1:]:
                if data!='':
                    train_lists.append(str(data).replace('0F','OF'))
        if col_datas[0]=='test':
            for data in col_datas[1:]:
                if data!='':
                    test_lists.append(str(data).replace('0F','OF'))
        if col_datas[0]=='validation':
            for data in col_datas[1:]:
                if data!='':
                    validation_lists.append(str(data).replace('0F','OF'))

    files = glob('/data/sdc/medicine_svs/data-v3/*.svs') + glob('/data/sdc/medicine_svs/data-v3/*.tif')

    train_files,test_files,val_files=[],[],[]
    for file in files:
        file_name=file.split('/')[-1].split('.')[0]+'.xml'
        if file_name in train_lists:
            train_files.append(file)
        if file_name in test_lists:
            test_files.append(file)
        if file_name in validation_lists:
            val_files.append(file)

    print(f'Number of train files : {len(train_files)} , Number of test files : {len(test_files)} , Number of val files : {len(val_files)}')

    resize_list = [0,8196,25600]
    drop_rate=0.5
    save_base_path='/data/sdc/medicine_svs/crop_data'
    angles=[30,60,90,120,150,180,210,240,270,300,330]
    
    if thread==1:
        for file in train_files:
            single_sample_process(file,save_base_path,resize_list,angles,drop_rate,overlap)
    else:
        multi_thread_params=[]
        for file in train_files[:thread]:
            multi_thread_params.append((file,save_base_path,resize_list,angles,drop_rate,overlap))
            
        pool = multiprocessing.Pool(processes=thread)

        results = pool.starmap(single_sample_process, multi_thread_params)

        pool.close()
        pool.join()

def single_sample_process(file,save_base_path,resize_list,angles,drop_rate,overlap):
    for resize in resize_list:
        xml_file = '/data/sdc/medicine_svs/data-v4/' + file.split('/')[-1].split('.')[0] + '.xml'
        svs_file = file

        if resize==25600:
            crop_size=(1536,1536)
        else:
            crop_size = (1024, 1024)
        # mask = open_slide_to_mask(save_base_path,svs_file, xml_file, 'train', resize,overlap=overlap)
        mask = rotation_openslide_to_mask(save_base_path,svs_file,xml_file,split='train',resized=resize,angles=angles,crop_size=crop_size,drop_rate=drop_rate,overlap=overlap)


if __name__ == '__main__':
    '''
    data_list=['OF-8-E.svs','OF-15-E.svs','OF-32-E.tif','OF-65-E.svs','OF-74-E.tif','OF-79-E.tif','W-19a.svs']
    for data in data_list:
        slide = openslide.open_slide('/media/ubuntu/Seagate Basic1/data-v3/{}'.format(data))
        img_size = slide.level_dimensions[0]
        slide_region = slide.read_region((0, 0), 0, img_size).convert('RGB')
        slide_region = np.array(slide_region)  # size1,size2,channel  ==> size2,size1,channel

        slide_points = xml_to_region('/media/ubuntu/Seagate Basic1/data-v3/{}.xml'.format(data.split('.')[0]))

        mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        # mask=np.tile(np.array([255],dtype=np.uint8),(img_size[0],img_size[1]))
        copy_region=copy.copy(slide_region)

        for point_dict in slide_points:
            cls = list(point_dict.keys())[0]
            points = np.asarray([point_dict[cls]], dtype=np.int32)

            if cls == 'None':
                cls = 'NOR'

            cv2.fillPoly(img=slide_region, pts=points, color=label_color[cls])

        slide_region = cv2.resize(slide_region, (4096, 4096))
        copy_region = cv2.resize(copy_region, (4096, 4096))

        for row in range(4096):
            for col in range(4096):
                if copy_region[row, col, 0] >= 175 and copy_region[row, col, 0] <= 245:
                    if copy_region[row, col, 2] >= 175 and copy_region[row, col, 2] <= 225:
                        if copy_region[row, col, 1] >= 55 and copy_region[row, col, 1] <= 180:
                            copy_region[row, col, :] = [0, 0, 0]

        plt.subplot(121)
        plt.imshow(slide_region)
        plt.subplot(122)
        plt.imshow(copy_region)
        plt.savefig('../{}.png'.format(data.split('.')[0]))
        plt.show()
    '''

    """
    import shutil
    load_imgs = glob('/media/ubuntu/Seagate Basic1/0429_datas/train_datas/img_dir/*.png')
    
    new_img_dir="/media/ubuntu/Seagate Basic1/0429_datas/train_datas/without_resize/img_dir"
    new_ann_dir = "/media/ubuntu/Seagate Basic1/0429_datas/train_datas/without_resize/ann_dir"
    for img_file in load_imgs:
        resize_num=int(img_file.split("/")[-1].split('.png')[0].split('-')[-3])
        
        assert resize_num in [0,8196,15360]
        if resize_num !=0:
            continue

        base_name=os.path.basename(img_file)
        ann_file=f'/media/ubuntu/Seagate Basic1/0429_datas/train_datas/ann_dir/{base_name}'
        
        if not os.path.exists(new_img_dir):
            os.makedirs(new_img_dir,exist_ok=True)
        if not os.path.exists(new_ann_dir):
            os.makedirs(new_ann_dir, exist_ok=True)
        new_img_file,new_ann_file=f'{new_img_dir}/{base_name}',f'{new_ann_dir}/{base_name}'
        shutil.copy(img_file,new_img_file)
        shutil.copy(ann_file, new_ann_file)
        
    raise

    # xml_file='/media/ubuntu/Seagate Basic/data-v3/OF-73-E.xml'
    # svs_file='/media/ubuntu/Seagate Basic/data-v3/OF-73-E.tif'
    # mask=open_slide_to_mask(svs_file,xml_file)

    # data_ana()
    # enhance_train_split()
    """
    
    """
    files = glob('/media/ubuntu/Seagate Basic/data-v3/*.svs') + glob('/media/ubuntu/Seagate Basic/data-v3/*.tif')
    train_files=files
    rand_index=random.sample(range(len(files)),30)
    test_index, val_index=rand_index[:20],rand_index[20:]
    test_files=[files[i] for i in test_index]
    val_files=[files[i] for i in val_index]
    for file in test_files+val_files:
        train_files.remove(file)
    
    print(f'total files num: {len(files)} train files num : {len(train_files)} test files num : {len(test_files)} val files num : {len(val_files)}')
    

    read_xls = xlrd.open_workbook('/media/ubuntu/Seagate Basic1/data-v3/test-train-validation-list.xlsx')
    xls_context = read_xls.sheets()[0]
    test_lists = []
    col_datas=xls_context.col_values(colx=0, start_rowx=1)
    for data in col_datas:
        if data!="":
            test_lists.append(data)

    files = glob('/media/ubuntu/Seagate Basic1/data-v3/*.svs') + glob('/media/ubuntu/Seagate Basic1/data-v3/*.tif')

    test_files=[]
    for file in files:
        file_name = file.split('/')[-1].split('.')[0] + '.xml'
        if file_name in test_lists:
            test_files.append(file)

    print(test_files)

    cls_1,cls_2,cls_3,cls_4,cls_5=0,0,0,0,0
    for file in test_files:
        xml_file = '/media/ubuntu/Seagate Basic1/data-v3/' + file.split('/')[-1].split('.')[0] + '.xml'
        svs_file = file
        mask = open_slide_to_mask(svs_file, xml_file, 'test')
        cls_1 = cls_1 + np.sum(mask == 1)
        cls_2 = cls_2 + np.sum(mask == 2)
        cls_3 = cls_3 + np.sum(mask == 3)
        cls_4 = cls_4 + np.sum(mask == 4)
        cls_5 = cls_5 + np.sum(mask == 5)

    print(f'Q:{cls_1}  NOR:{cls_2}  HYP:{cls_3}  DYS:{cls_4}  CAR:{cls_5}')
    """

    # ------- load patch data -----------
    # print(f'Using multi thread process data, thread numbers: {multiprocessing.cpu_count()}')
    # multi_process_data(multiprocessing.cpu_count())
    
    # ------- delete error images -----------
    # del_imgs=glob('/data/sdc/medicine_svs/crop_data/train_datas/img_dir/C13-E-t-25600-*.png')
    # for del_img in tqdm(del_imgs):
    #     os.remove(del_img)
    #     os.remove(del_img.replace('/img_dir/','/ann_dir/'))
    
    # -------- check files --------
    for file in glob('/data/sdc/medicine_svs/ori_scale_datas/train_datas/img_dir/*.png'):
        if not os.path.exists(file.replace('/img_dir/','/ann_dir/')):
            os.remove(file)
            print(f'remove file: {file}')
        
    """
    # ------- copy original data -----------
    import shutil
    read_xls=xlrd.open_workbook('/data/sdc/medicine_svs/v3-test-train-validation-list.xlsx')
    xls_context=read_xls.sheets()[0]

    train_lists=[]
    col_datas = xls_context.col_values(colx=2, start_rowx=0)
    if col_datas[0]=='train':
        for data in col_datas[1:]:
            if data!='':
                train_lists.append(str(data).replace('0F','OF'))
                
    base_path,goal_path='/data/sdc/medicine_svs/crop_data/train_datas/img_dir','/data/sdc/medicine_svs/ori_scale_datas/train_datas/img_dir'
    
    os.makedirs(goal_path,exist_ok=True)
    os.makedirs(goal_path.replace('/img_dir/','/ann_dir/'),exist_ok=True)
    
    unprocessed_files=[]
    for each_file in tqdm(train_lists):
        ori_files=glob(f'{base_path}/{os.path.splitext(each_file)[0]}-0-0-[0-9]*.png')

        if len(ori_files)==0:
            slide_file=glob(f'/data/sdc/medicine_svs/data-v3/{os.path.splitext(each_file)[0]}.svs')+glob(f'/data/sdc/medicine_svs/data-v3/{os.path.splitext(each_file)[0]}.tif')
            if len(slide_file)!=0:
                assert len(slide_file)==1,print(slide_file)
                unprocessed_files.append(slide_file[0])
            else:
                print(f'Not found {os.path.splitext(each_file)[0]} files.')

        for each_ori_file in ori_files:
            shutil.copy(each_ori_file,goal_path)
            shutil.copy(each_ori_file.replace('/img_dir/','/ann_dir/'),goal_path.replace('/img_dir/','/ann_dir/'))
    
    multi_thread_params=[]
    for file in unprocessed_files:
        multi_thread_params.append((file,'/data/sdc/medicine_svs/ori_scale_datas',[0],[],0.5,True))
        
    pool = multiprocessing.Pool(processes=len(unprocessed_files))

    results = pool.starmap(single_sample_process, multi_thread_params)

    pool.close()
    pool.join()
    """
    
    """
    datas=glob('/media/ubuntu/Seagate Basic/TCGA/pre/*_mask.png')
    svs = openslide.open_slide('/media/ubuntu/Seagate Basic/TCGA-CV-6962-01Z-00-DX1.7BF8A1EF-06D7-4DC1-98F3-0A845744A90B.svs')
    w, h = svs.level_dimensions[0]
    overall_mask = np.tile(np.array([255.0]), (h, w))
    # overall_mask_in_ori = np.tile(np.array([255]), (h, w,3))
    row, col = 0, 0

    for data_index in range(len(datas)):
        patch_img = np.array(cv2.imread(f'/media/ubuntu/Seagate Basic/TCGA/pre/{data_index}_mask.png'))  # read split data
        mask_in_ori=np.array(cv2.imread(f'/media/ubuntu/Seagate Basic/TCGA/pre/{data_index}_mask_in_ori.png'))

        patch_size = patch_img.shape
        # print('start point :{}  crop size :{} img_size :{}'.format(f'({row},{col})',patch_size,img_size))
        if row + patch_size[0] > h:
            patch_img = cv2.resize(patch_img, ( patch_size[1],h - row))
            patch_img = np.array(patch_img,dtype='float')

            mask_in_ori = cv2.resize(mask_in_ori, (patch_size[1], h - row))
            mask_in_ori = np.array(mask_in_ori, dtype='float')

            print(
                f'img size: {(h, w)} patch size: {patch_size} new patch size: {patch_img.shape} start_row: {row} patch size row over img size row')

        if col + patch_size[1] > w:
            patch_img = cv2.resize(patch_img, ( w - col,patch_size[0]))
            patch_img = np.array(patch_img,dtype='float')

            mask_in_ori = cv2.resize(mask_in_ori, ( w - col,patch_size[0]))
            mask_in_ori = np.array(mask_in_ori,dtype='float')

            print(
                f'img size: {(h, w)} patch size: {patch_size} new patch size: {patch_img.shape} start_col: {col} patch size col over img size col')

        patch_size = patch_img.shape
        # print(patch_size,overall_mask[row:row + patch_size[0], col:col + patch_size[1]].shape,row,col,overall_mask.shape)
        if overall_mask[row:row + patch_size[0], col:col + patch_size[1]].shape !=patch_size[:2]:
            overall_mask_shape=overall_mask[row:row + patch_size[0], col:col + patch_size[1]].shape
            print(overall_mask_shape,patch_img.shape)
            patch_img=cv2.resize(patch_img, (overall_mask_shape[1],overall_mask_shape[0]))
            mask_in_ori=cv2.resize(mask_in_ori, (overall_mask_shape[1],overall_mask_shape[0]))


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
    file = open('/media/ubuntu/Seagate Basic/TCGA.xml', 'w')
    file.write('<?xml version=\"1.0\"?>\n')
    file.write('<ASAP_Annotations>\n')
    file.write('\t<Annotations>\n')
    thresh = 15000
    ann_count = 0
    contours, hierarchy=cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

    # '''
    """

