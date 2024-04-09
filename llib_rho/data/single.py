
from __future__ import division

import torch
import os.path as osp
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
import pickle
import json 

from llib.utils.image.augmentation import (
    crop, flip_img, flip_pose, flip_kp, transform, rot_aa
)
from llib.models.regressors.bev.utils import img_preprocess, bbox_preprocess
from llib_rho.data.preprocess.behave import Behave

class SingleDataset(Dataset):
    def __init__(
             self,
             dataset_cfg,
             dataset_name,
             augmentation,
             image_processing,
             split='train',
             body_model_type='smplx',
        ):
        """
        Base Dataset Class for optimization.
        Parameters
        ----------
        dataset_cfg: cfg
            config file of dataset
        dataset_name: str
            name of dataset (e.g. flickrci3ds)
        image_processing: cfg
            config file of image processing
        split: str
            split of dataset (train, val, test)
        body_model_type: str
            type of body model
        """
        super(SingleDataset, self).__init__()

        self.augmentation = augmentation
        self.image_processing = image_processing
        self.body_model_type = body_model_type
        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.split = split 

        if self.split != 'train':
            self.augmentation.use = False

        # Human parameters
        self.num_pose_params = 72
        self.num_shape_params = 10
        self.num_global_orient_params = 3
        self.num_transl_params = 3
        self.num_gt_kpts = 24
        self.num_op_kpts = 25

        # Object parameters
        self.obj_num_global_orient_params = 3
        self.obj_num_transl_params = 3

        self.IMGRES = self.image_processing.resolution

        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.split = split
        self.body_model_type = body_model_type

        self.img_dir = osp.join(
            dataset_cfg.data_folder, dataset_cfg.image_folder
        )

        # self.normalize_img = Normalize(
        #     mean=self.image_processing.normalization_mean,
        #     std=self.image_processing.normalization_std
        # )

        self.data = self.load_data()
        self.len = len(self.data)

    def load_data(self):
        dataset = Behave(
            **self.dataset_cfg, 
            split=self.split,
            body_model_type=self.body_model_type
        ).load()
        return dataset
    
    def to_tensors(self, target):
        for k, v in target.items():
            if isinstance(v, np.ndarray):
                target[k] = torch.from_numpy(v)
            elif isinstance(v, list):
                target[k] = torch.tensor(v)
            elif isinstance(v, dict):
                target[k] = self.to_tensors(v)
        return target
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.get_single_item(index)
    
    def get_single_item(self, index):

        item = self.data[index]
        return self.to_tensors(item)
    