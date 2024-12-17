import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CancersegDataset(CustomDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    # CLASSES = ('Background', 'Foreground')
    CLASSES = ('Background', 'NOR', 'HYP', 'DYS', 'CAR')
    # CLASSES = ('Background','Q', 'NOR', 'HYP', 'DYS', 'CAR')

    # PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    #            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    #            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    #            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    #            [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    # PALETTE = [[255,255,255], [135,206,235], [255,128,0], [135,38,87], [176,23,31]]
    # PALETTE=[white,gray,blue,orange,strawberry color, red]
    PALETTE = [[255,255,255], [135,206,235], [255,128,0], [135,38,87], [255,0,0]]
    # PALETTE=[white,yellow,light_blue,purple, red]

    def __init__(self, **kwargs):
        super(CancersegDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)
