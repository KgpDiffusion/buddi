import argparse
import torch 
import numpy as np
import cv2
import os
import sys
import smplx
import pickle
import random
import trimesh
import imageio
from tqdm import tqdm
import os.path as osp
from loguru import logger as guru
from omegaconf import OmegaConf 
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from llib_rho.bodymodels.build import build_bodymodel 
from llib_rho.cameras.build import build_camera
from llib_rho.data.build import build_optimization_datasets
from llib_rho.visualization.renderer import Pytorch3dRenderer
from llib_rho.visualization.utils import *
from llib_rho.logging.logger import Logger
from llib_rho.defaults.main import (
    config as default_config,
    merge as merge_configs
)
from llib_rho.models.build import build_model
from llib_rho.models.diffusion.build import build_diffusion
from llib_rho.method.hhc_diffusion.train_module import TrainModule
from llib_rho.method.hhc_diffusion.eval_module import EvalModule
from .fit_module import HHCSOpti
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import multiprocessing

SEED = 238492
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

class MaskLossStats(torch.nn.Module):
    def __init__(self,):

        # init call
        super(MaskLossStats, self).__init__()

        self.data_dir = Path('/home/shubhikg/data_intercap/')
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.output_dir = self.data_dir / ".." / "buddi" / "debug_output"
        os.makedirs(self.output_dir, exist_ok=True)

        self.object_names = ['backpack', 'basketball', 'boxlarge', 'boxtiny', 'boxlong',
                                'boxsmall', 'boxmedium', 'chairblack', 'chairwood', 'monitor',
                                'keyboard', 'plasticcontainer', 'stool', 'tablesquare', 'toolbox',
                                'suitcase', 'tablesmall', 'yogamat', 'yogaball', 'trashbin']
        
        self.object_names = ['obj01', 'obj02', 'obj03', 'obj04', 'obj05', 'obj06', 'obj07', 'obj08', 'obj09', 'obj10']
        
        self.label_to_id = {name: i for i, name in enumerate(self.object_names)}

        self.H, self.W = 1080, 1920 #1536, 2048

        self.cam_intrinsic = np.array([
                            [976.2120971679688, 0.0, 1017.9580078125],
                            [0.0, 976.0467529296875, 787.3128662109375],
                            [0.0, 0.0, 1.0]])
        
        ## initialize dict
        self.res=dict()
        splits=['val','test']
        for split in splits:
            self.res[split] = dict()
            for object in self.object_names:
                self.res[split][object] = []

        # deep copy res to stats dict
        import copy
        self.stats = copy.deepcopy(self.res)

    def read_metadata(self,split='val'):

        item={}
        data_dir = self.data_dir / split

        meta_data_file = data_dir / "metadata.pkl"
        with open(meta_data_file, "rb") as f:
            meta_data = pickle.load(f)

        for idx, fileID in enumerate(meta_data.keys()):

            item[fileID] = {}
            image_path = data_dir / "images" / f"{fileID}.jpg"
            obj_mask_path = data_dir / "masks" / f"{fileID}_obj.png"

            item[fileID]['image_path'] = image_path
            item[fileID]['obj_mask'] =  obj_mask_path
            item[fileID]['data'] = meta_data[fileID]
            item[fileID]['fileID'] = fileID

        return item
        
    def fast_storage(self,split, obj_mask_path, obj_name):

        # load metadata
        gt_obj_mask = cv2.imread(str(obj_mask_path), cv2.IMREAD_GRAYSCALE)
        ## assert 2d shape
        assert len(gt_obj_mask.shape) == 2
        assert gt_obj_mask.shape[0] == self.H and gt_obj_mask.shape[1] == self.W, f"Shape of mask is {gt_obj_mask.shape}"

        # count non zero pixels in mask
        self.res[split][obj_name].append(np.count_nonzero(gt_obj_mask))


    def store_data(self, split):

        """ Test the interpenetration loss"""
        metadata = self.read_metadata(split)


        # multiprocessing
        num_items = -1
        for fileID in tqdm(list(metadata.keys())[:num_items]):

            # multiprocessing
            obj_mask_path = metadata[fileID]['obj_mask']
            obj_name = metadata[fileID]['data'][1]
            self.fast_storage(split,obj_mask_path,obj_name)

    def compute_statistics(self):

        self.store_data('val')
        self.store_data('test')

        # plot frequency histogram of mask sizes for each object
        for object in self.object_names:
            if not len(self.res['val'][object]):
                continue
            # store new figure each time
            plt.figure()
            plt.hist(self.res['val'][object], bins=50, alpha=0.5, label='val')
            plt.hist(self.res['test'][object], bins=50, alpha=0.5, label='test')
            # plot mean and std
            plt.axvline(np.mean(self.res['val'][object]), color='k', linestyle='dashed', linewidth=1)
            plt.axvline(np.mean(self.res['test'][object]), color='k', linestyle='dashed', linewidth=1)
            # plt.axvline(np.mean(self.res['val'][object]) + np.std(self.res['val'][object]), color='r', linestyle='dashed', linewidth=1)
            # plt.axvline(np.mean(self.res['val'][object]) - np.std(self.res['val'][object]), color='r', linestyle='dashed', linewidth=1)
            plt.legend(loc='upper right')
            plt.title(f"Object: {object}")
            # save plot
            plt.savefig(f"/home/shubhikg/exp/buddi/temp/{object}_intercap.png")

        ## compute statistics
        for split in ['val','test']:
            for object in self.object_names:
                self.stats[split][object] = {}
                self.stats[split][object]['mean'] = 0
                self.stats[split][object]['std'] = 0
                self.stats[split][object]['percent_1.5'] = 0
                self.stats[split][object]['percent_1'] = 0
                self.stats[split][object]['percent_2'] = 0

                if len(self.res[split][object]):
                    print(f"Object: {object}, Split: {split}")
                    self.stats[split][object]['mean'] = np.mean(np.array(self.res[split][object]))
                    self.stats[split][object]['std'] = np.std(np.array(self.res[split][object]))
                    self.stats[split][object]['percent_1.5'] = np.sum(np.array(self.res[split][object]) > (self.stats[split][object]['mean'] 
                                                                - 1.5*self.stats[split][object]['std']))/len(self.res[split][object])
                    self.stats[split][object]['percent_1'] = np.sum(np.array(self.res[split][object]) > (self.stats[split][object]['mean'] 
                                                                - 1*self.stats[split][object]['std']))/len(self.res[split][object])
                    self.stats[split][object]['percent_2'] = np.sum(np.array(self.res[split][object]) > (self.stats[split][object]['mean'] 
                                                                - 2*self.stats[split][object]['std']))/len(self.res[split][object])
        
        # Save stats in a json file
        with open("/home/shubhikg/exp/buddi/temp/stats_intercap.json", "w") as f:
            import json
            json.dump(self.stats, f)


if __name__ == '__main__':


    loss_obj = MaskLossStats()
    loss_obj.compute_statistics()



