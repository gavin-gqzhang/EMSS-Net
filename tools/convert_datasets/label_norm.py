import numpy as np
import scipy.misc
import os
import os.path as osp
from glob import glob
import mmcv

data = glob('/home/ubuntu/ssd/data/20211206/seg/HNSC/pyramid_slide/20X/labels/val/image/*.png')
# data = glob('/home/shichao/ssd/data/cityscapes/gtFine/train/aachen/*labelIds.png')
print(data)
count = 0
for file in data:
    print(count)
    count = count + 1
    image = scipy.misc.imread(file)
    if(len(np.shape(image))) == 3:
        image = image[:,:,0]
        if max(image[image>=0]) > 1:
            mmcv.imwrite(image[:, :] // 128, osp.join(file))
        else:
            mmcv.imwrite(image, osp.join(file))
    else:
        if max(image[image>=0]) > 1:
            mmcv.imwrite(image[:, :] // 128, osp.join(file))
