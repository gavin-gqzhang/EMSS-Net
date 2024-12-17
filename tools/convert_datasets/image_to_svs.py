import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=pow(2,63).__str__()
import time
import numpy as np
import tifffile
import cv2

def gen_im(size_hw,image):
    im_i = 0
    im_j = 0
    # im_ok = list()
    while True:
        im = image[size_hw[0]*im_j:size_hw[0]*(im_j+1),size_hw[1]*im_i:size_hw[1]*(im_i+1),:]
        im = cv2.putText(im,str(im_i),(size_hw[1],size_hw[0]),cv2.FONT_HERSHEY_PLAIN,6,color=(0,0,0),thickness=3)
        if size_hw[1]*(im_i+1) < np.shape(image)[1]:
            im_i += 1
        else:
            im_i = 0
            if size_hw[0]*(im_j+1) < np.shape(image)[0]:
                im_j += 1
        # im_ok.append(im)
        yield im
    # return im_ok

svs_desc = 'Aperio Image Library Shichao\nABC |AppMag={mag}|Filename={filename}|MPP={mpp}'
label_desc = 'Aperio Image Library Shichao\nlabel {W}x{H}'
macro_desc = 'Aperio Image Library Shichao\nmacro {W}x{H}'

def gen_pyramid_tiff(filename,out_file, res = '20X'):

    image = cv2.imread(filename)

    if len(np.shape(image))==0:
        print(f'ERROR File : {filename}')
        return
    print(np.shape(image))

    h = np.shape(image)[0]
    w = np.shape(image)[1]

    width = 1024
    height = 1024

    thumbnail_im = cv2.resize(image,(width,height))#np.zeros([512,512,3], dtype=np.uint8)
    thumbnail_im = cv2.putText(thumbnail_im,'thumbnail',(thumbnail_im.shape[1], thumbnail_im.shape[0]), cv2.FONT_HERSHEY_PLAIN,6,color=(255,0,0),thickness=3)

    label_im = cv2.resize(image,(width*4,height*4))#np.zeros([512,512,3],dtype=np.uint8)
    label_im = cv2.putText(label_im,'label',(label_im.shape[1],label_im.shape[0]),cv2.FONT_HERSHEY_PLAIN,6,color=(0,255,0),thickness=3)

    macro_im = cv2.resize(image,(width*4,height*4))#np.zeros([512,512,3],dtype=np.uint8)
    macro_im = cv2.putText(macro_im,'macro',(macro_im.shape[1],macro_im.shape[0]),cv2.FONT_HERSHEY_PLAIN,6,color=(0,0,255),thickness=3)

    tile_hw = np.int64([width,height])

    h = h // tile_hw[0] * tile_hw[0]
    w = w // tile_hw[1] * tile_hw[1]

    if res == '40X':
        multi_hw = np.int64([(h, w), (h // 4, w // 4), (h // 16, w // 16), (h // 64, w // 64)])  # 40X
    elif res == '20X':
        multi_hw = np.int64([(h*4, w*4), (h, w), (h // 4, w // 4), (h // 16, w // 16)]) # 20X
    elif res == '10X':
        multi_hw = np.int64([(h * 16, w * 16), (h*4, w*4), (h, w), (h // 4, w // 4)])  # 20X

    mpp = 0.25

    mag = 40

    with tifffile.TiffWriter(out_file,bigtiff=True) as tif:
        thw = tile_hw.tolist()
        compression = ['JPEG',95,dict(outcolorspace='YCbCr')]
        kwargs = dict(subifds=0, photometric='rgb',planarconfig='CONTIG',compression=compression,dtype=np.uint8,metadata=None)

        for i, hw in enumerate(multi_hw):
            # if i > 0:
            #     tile_hw = tile_hw // 4
            resolution = [hw[0], hw[1], 'CENTIMETER']
            gen = gen_im(tile_hw,cv2.cvtColor(cv2.resize(image,(hw[1],hw[0])),cv2.COLOR_BGR2RGB))
            hw = hw.tolist()

            if i==0:
                desc = svs_desc.format(mag=mag,filename=filename,mpp=mpp)
                tif.write(data=gen,shape=(*hw,3), tile=thw[::-1], resolution=resolution, description=desc,**kwargs)
            else:
                tif.write(data=gen,shape=(*hw,3),tile=thw[::-1],resolution=resolution,description='',**kwargs)
        tif.write(data=thumbnail_im, description='', **kwargs)
        tif.write(data=label_im,subfiletype=1,description=label_desc.format(W=label_im.shape[1],H=label_im.shape[0]),**kwargs)
        tif.write(data=macro_im,subfiletype=9,description=macro_desc.format(W=macro_im.shape[1],H=macro_im.shape[0]),**kwargs)
    tif.close()

from glob import glob
t1 = time.perf_counter()
im_path = '/media/ubuntu/Seagate Basic/WSI/'
save_path = '/media/ubuntu/Seagate Basic/SVS/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
res = '40X'
image_name = glob(im_path + '*.tif')
print(image_name)
for data in image_name:
    print(data)
    gen_pyramid_tiff(data,save_path+data.split('/')[-1].replace('.tif','.svs'), res = res)
print(time.perf_counter()-t1)
