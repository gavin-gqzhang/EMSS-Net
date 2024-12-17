import os

impath = '/home/ubuntu/ssd/data/20211206/seg/HNSC/ESCC/'
for root,dirs,files in os.walk(impath):
    if len(files)>1 and '.txt' in files[1]:
        print(files)
        os.system('mv {} {}'.format(root + '/' + files[0], impath))
    if len(files)==1 and '.parcel' not in files[0]:
        print(root)
        os.system('mv {} {}'.format(root+'/'+files[0],impath))
        # os.system('rm -r {}'.format(root))