# Originally written by Kazuto Nakashima
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import json
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

from multiview.video.datasets import ViewPairDataset
class MuiltivwDataset(BaseDataSet):
    """
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """
    def __init__(self, number_views, view_idx_labled, view_idx_adapt,**kwargs):
        # self.num_classes = 3
        self.num_classes = 2+10
        self.palette = palette.get_voc_palette(self.num_classes)
        self.number_views = number_views
        self.view_idx_adapt = view_idx_adapt
        self.view_idx_labled = view_idx_labled
        self.max_matching_len = 255
        if not isinstance(view_idx_adapt,int):
            raise ValueError('view_idx_adapt: {}'.format(view_idx_adapt))
        self.view_key_img = "frames views " + str(self.view_idx_labled)
        self.view_key_seg = "seg "+str(self.view_idx_labled)
        self.view_key_img_adapt = "frames views " + str(self.view_idx_adapt)
        # self.view_key_seg = "seg "+str(self.view_idx_adapt)
        assert isinstance(view_idx_adapt, int) and isinstance(number_views, int)
        super(MuiltivwDataset, self).__init__(**kwargs)
        print('data dir {}, view adapt {}, num views'.format(self.root, view_idx_adapt, number_views))
        self.match_dir=os.path.join(self.root,'../../superglue')

    def _set_files(self):
        def data_len_filter(comm_name,frame_len_paris):
            if len(frame_len_paris)<2:
                return frame_len_paris[0]>10
            return min(*frame_len_paris)>10
        self.mvbdata = ViewPairDataset(self.root.strip(),
					    segmentation= True,
                                            transform_frames= None,
					    number_views=self.number_views,
					    filter_func=data_len_filter)

    def __len__(self):
        return len(self.mvbdata)

    def _pad_match(self,mkpts0):
        mkpts0_padded=np.zeros((self.max_matching_len,2))
        mkpts0=np.array(mkpts0)
        max_l = min(self.max_matching_len-1,mkpts0.shape[0])
        try:
            mkpts0_padded[:max_l] = mkpts0[:max_l]
        except:
            print('mkpts0.shape[0]: {}'.format(mkpts0.shape[0]))
            print('max_l: {}'.format(max_l))


        return mkpts0_padded, mkpts0.shape[0]

    def _load_data(self, index):
        s = self.mvbdata[index]
        label = s[self.view_key_seg]
        image = s[self.view_key_img]
        image = np.asarray(image, dtype=np.float32)
        label = np.asarray(label, dtype=np.int32)

        image_adapt = s[self.view_key_img_adapt]
        image_adapt = np.asarray(image_adapt, dtype=np.float32)

        cm=s["common name"]
        frame_idx=s["frame index"]
        view_i= min(self.view_idx_labled,self.view_idx_adapt)
        view_j= max(self.view_idx_labled,self.view_idx_adapt)
        i, j = frame_idx,frame_idx
        key_match = 'view{}:frame{}->view{}:frame{}'.format(view_i,i,view_j,j)
        match_file = os.path.join(self.match_dir, cm+".txt")
        with open(match_file) as f:
            data_match = json.load(f)[key_match]
        # matches
        mkpts0, m_cnt = self._pad_match(data_match['mkpts0'])
        mkpts1,m_cnt1 = self._pad_match(data_match['mkpts1'])
        assert m_cnt==m_cnt1
        return image, label,image_adapt, mkpts0,mkpts1,m_cnt

class MVB(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False,
                    number_views=1, view_idx_labled=None, view_idx_adapt=None):

        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]
        assert view_idx_labled is not None, "set view idx in config"
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

        self.dataset = MuiltivwDataset(number_views,view_idx_labled,view_idx_adapt,**kwargs)
        super(MVB, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

