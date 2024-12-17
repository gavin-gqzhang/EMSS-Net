import multiresolutionimageinterface as mir
import numpy as np
reader = mir.MultiResolutionImageReader()
train_svs_path = '/home/ubuntu/ssd/data/20211206/seg/HNSC/'
file = 'test'
with open(train_svs_path+'{}.txt'.format(file),'r') as f:
    for line_all in f.readlines():
        line = line_all.split(' ')[0].strip()
        print(line)
        mr_image = reader.open(train_svs_path+'results_xml/'+line)
        annotation_list = mir.AnnotationList()
        xml_repository = mir.XmlRepository(annotation_list)
        xml_repository.setSource(train_svs_path+'results_xml/'+line.replace('.svs','.xml'))
        xml_repository.load()
        annotation_mask = mir.AnnotationToMask()
        annotation_mask.convert(annotation_list, train_svs_path+'test_xml/'+line.replace('.svs','.tif'), mr_image.getDimensions(), mr_image.getSpacing())
