from glob import glob
import os

if __name__=="__main__":
    
    sec_imgs=glob('/data/sdc/medicine_svs/ori_scale_datas/train_datas/img_dir/*.png')
    
    forth_imgs=glob('/data/sdc/medicine_svs/multi_scale_datas/train_datas/img_dir/*.png')
    
    forth_dict,sec_dict=dict(),dict()
    for img in forth_imgs:
        basename=os.path.basename(img)
        
        scale=basename.split('.')[0].split('-')[-3]
        
        if scale not in forth_dict.keys():
            forth_dict[scale]=0
        forth_dict[scale]+=1
    
    for img in sec_imgs:
        basename=os.path.basename(img)
        
        scale=basename.split('.')[0].split('-')[-3]
        if scale not in sec_dict.keys():
            sec_dict[scale]=0
        sec_dict[scale]+=1
    
    print(forth_dict)
    print(sec_dict)