# Originally written by Kazuto Nakashima
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from lightningvae.data import DoubleViewPairDataset
class MuiltivwDataset(BaseDataSet):
    """
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """
    def __init__(self, **kwargs):
        self.num_classes = 3
        self.palette = palette.get_voc_palette(self.num_classes)
        super(MuiltivwDataset, self).__init__(**kwargs)

    def _set_files(self):
        def data_len_filter(comm_name,frame_len_paris):
            if len(frame_len_paris)<2:
                return frame_len_paris[0]>10
            return min(*frame_len_paris)>10
        self.mvbdata = DoubleViewPairDataset(self.root.strip(),
					    segmentation= True,
                                            transform_frames= None,
					    number_views=1,
					    filter_func=data_len_filter)

    def __len__(self):
        # return len(self.mvbdata)
        return 8*10

    def _load_data(self, index):
        s = self.mvbdata[index]
        label = s["seg"]
        image =s["frames views " + str(0)]
        image = np.asarray(image, dtype=np.float32)
        label = np.asarray(label, dtype=np.int32)
        return image, label, s["id"]

class MVB(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = MuiltivwDataset(**kwargs)
        super(MVB, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

