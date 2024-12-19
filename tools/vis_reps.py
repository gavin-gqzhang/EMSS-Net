import cv2,os
import copy
from mmseg.apis import inference_segmentor, init_segmentor
import numpy as np
from glob import glob
import openslide
from tools.load_data import xml_to_region

label_mapping = {
    'Q': 0,
    'NOR': 1,
    'HYP': 2,
    'DYS': 3,
    'CAR': 4,
}


def crop_pre(model,slide_region,gt_mask, img_size,save_path, crop_size=(1024,1024)):
    
    # region_size=(int(img_size[0]/num_path),int(img_size[1]/num_path))
    split_x = int(img_size[0] / crop_size[0])
    split_y = int(img_size[1] / crop_size[1])
    if img_size[0] % crop_size[0] != 0:
        split_x = split_x + 1
    if img_size[1] % crop_size[1] != 0:
        split_y = split_y + 1

    start_point,count = (0, 0),1
    for dimension_0 in range(split_x):
        for dimension_1 in range(split_y):
            if dimension_1 == split_y - 1 and dimension_0 != split_x - 1:
                img_patch = slide_region[start_point[0]:start_point[0] + crop_size[0],
                            start_point[1]:, :]
                gt_patch=gt_mask[start_point[0]:start_point[0] + crop_size[0],
                            start_point[1]:, :]
                result = inference_segmentor(model, img_patch, f'{save_path}/{count}_reps')  # shape:(segmentor_rtn,batch_size,h,w)
                result=result[-1][0]
                if result.shape != img_patch.shape[:-1]:
                    # print(
                    #     f'result shape : {result.shape}, img_patch shape : {img_patch.shape[:-1]} resized shape : {cv2.resize(result.astype("float"),(img_patch.shape[1],img_patch.shape[0])).shape}')
                    result = cv2.resize(result.astype('float'), (img_patch.shape[1], img_patch.shape[0]))
                # rtn_mask[start_point[0]:start_point[0] + crop_size[0],
                # start_point[1]:] = result
                
                os.makedirs(f'{save_path}/img_patch',exist_ok=True)
                cv2.imwrite(f'{save_path}/img_patch/{count}.png',img_patch)
                
                os.makedirs(f'{save_path}/pre_patch',exist_ok=True)
                cv2.imwrite(f'{save_path}/pre_patch/{count}.png',result)
                
                os.makedirs(f'{save_path}/gt_patch',exist_ok=True)
                cv2.imwrite(f'{save_path}/gt_patch/{count}.png',gt_patch)
                count=count+1
                

            elif dimension_1 == split_y - 1 and dimension_0 == split_x - 1:
                img_patch = slide_region[start_point[0]:, start_point[1]:, :]
                gt_patch=gt_mask[start_point[0]:, start_point[1]:, :]
                result = inference_segmentor(model, img_patch, f'{save_path}/{count}_reps')  # shape:(segmentor_rtn,batch_size,h,w)
                result=result[-1][0]
                if result.shape != img_patch.shape[:-1]:
                    # print(
                    #     f'result shape : {result.shape}, img_patch shape : {img_patch.shape} resized shape : {cv2.resize(result.astype("float"),(img_patch.shape[1],img_patch.shape[0])).shape}')
                    result = cv2.resize(result.astype('float'), (img_patch.shape[1], img_patch.shape[0]))
                # rtn_mask[start_point[0]:, start_point[1]:] = result
                
                os.makedirs(f'{save_path}/img_patch',exist_ok=True)
                cv2.imwrite(f'{save_path}/img_patch/{count}.png',img_patch)
                
                os.makedirs(f'{save_path}/pre_patch',exist_ok=True)
                cv2.imwrite(f'{save_path}/pre_patch/{count}.png',result)
                
                os.makedirs(f'{save_path}/gt_patch',exist_ok=True)
                cv2.imwrite(f'{save_path}/gt_patch/{count}.png',gt_patch)
                count=count+1

            elif dimension_0 == split_x - 1 and dimension_1 != split_y - 1:
                img_patch = slide_region[start_point[0]:, start_point[1]:start_point[1] + crop_size[1], :]
                gt_patch=gt_mask[start_point[0]:, start_point[1]:start_point[1] + crop_size[1], :]
                result = inference_segmentor(model, img_patch, f'{save_path}/{count}_reps')  # shape:(segmentor_rtn,batch_size,h,w)
                result=result[-1][0]
                if result.shape != img_patch.shape[:-1]:
                    # print(
                    #     f'result shape : {result.shape}, img_patch shape : {img_patch.shape} resized shape : {cv2.resize(result.astype("float"),(img_patch.shape[1],img_patch.shape[0])).shape}')
                    result = cv2.resize(result.astype('float'), (img_patch.shape[1], img_patch.shape[0]))
                # rtn_mask[start_point[0]:, start_point[1]:start_point[1] + crop_size[1]] = result
                
                os.makedirs(f'{save_path}/img_patch',exist_ok=True)
                cv2.imwrite(f'{save_path}/img_patch/{count}.png',img_patch)
                
                os.makedirs(f'{save_path}/pre_patch',exist_ok=True)
                cv2.imwrite(f'{save_path}/pre_patch/{count}.png',result)
                
                os.makedirs(f'{save_path}/gt_patch',exist_ok=True)
                cv2.imwrite(f'{save_path}/gt_patch/{count}.png',gt_patch)
                count=count+1

            else:
                img_patch = slide_region[start_point[0]:start_point[0] + crop_size[0],
                            start_point[1]:start_point[1] + crop_size[1], :]
                gt_patch=gt_mask[start_point[0]:start_point[0] + crop_size[0],
                            start_point[1]:start_point[1] + crop_size[1], :]
                result = inference_segmentor(model, img_patch, f'{save_path}/{count}_reps')  # shape:(segmentor_rtn,batch_size,h,w)
                result=result[-1][0]
                if result.shape != img_patch.shape[:-1]:
                    # print(
                    #     f'result shape : {result.shape}, img_patch shape : {img_patch.shape} resized shape : {cv2.resize(result.astype("float"),(img_patch.shape[1],img_patch.shape[0])).shape}')
                    result = cv2.resize(result.astype('float'), (img_patch.shape[1], img_patch.shape[0]))
                # rtn_mask[start_point[0]:start_point[0] + crop_size[0],
                # start_point[1]:start_point[1] + crop_size[1]] = result

                os.makedirs(f'{save_path}/img_patch',exist_ok=True)
                cv2.imwrite(f'{save_path}/img_patch/{count}.png',img_patch)
                
                os.makedirs(f'{save_path}/pre_patch',exist_ok=True)
                cv2.imwrite(f'{save_path}/pre_patch/{count}.png',result)
                
                os.makedirs(f'{save_path}/gt_patch',exist_ok=True)
                cv2.imwrite(f'{save_path}/gt_patch/{count}.png',gt_patch)
                count=count+1

            start_point = (start_point[0], start_point[1] + crop_size[1])

        start_point = (start_point[0] + crop_size[0], 0)



if __name__=="__main__":
    config='/home/dell/zgq/medicine_code/local_configs/pathformer/B5/pathformer.b5.1024x1024.cancerseg.160k.py'
    checkpoint=""
    
    model = init_segmentor(config, checkpoint, device='cuda:0')
    slide_base_path,xml_base_path='',''
    save_base_path=''
    
    for slide_name in ['OF-73-E.tif','OF-74-E.tif','OF-28-E.svs','W-10.svs']:
        slide_file=f'{slide_base_path}/{slide_name}'
        xml_file=glob(f'{xml_base_path}/{slide_name.split(".")[0]}.xml')[0]
        
        slide = openslide.open_slide(slide_file)
        img_size = slide.level_dimensions[0]
        
        slide_region = slide.read_region((0, 0), 0, img_size).convert('RGB')

        slide_region = np.array(slide_region)  # size1,size2,channel  ==> size2,size1,chann

        img_size = (img_size[1], img_size[0])
        
        gt_slide_points = xml_to_region(xml_file)

        gt_mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)

        for point_dict in gt_slide_points:
            cls=list(point_dict.keys())[0]

            points=np.asarray([point_dict[cls]],dtype=np.int32)

            cv2.fillPoly(img=gt_mask, pts=points, color=(label_mapping[cls],label_mapping[cls],label_mapping[cls]))

        save_path=f'{save_base_path}/{slide_name.split(".")[0]}/'
        mask_lists = crop_pre(model, slide_region, gt_mask, img_size,save_path)


        