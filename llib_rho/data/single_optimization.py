
from __future__ import division

import os
import torch
import os.path as osp
from torch.utils.data import Dataset
import numpy as np
from llib_rho.data.preprocess.behave import Behave

class SingleOptiDataset(Dataset):

    def __init__(
             self,
             dataset_cfg,
             dataset_name,
             image_processing,
             split='train',
             body_model_type='smplx',
             use_hands=False,
             use_face=False,
             use_face_contour=False,
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

        super(SingleOptiDataset, self).__init__()

        self.image_processing = image_processing
        self.body_model_type = body_model_type
        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.split = split
        self.init_method = 'bev' 

        self.num_pose_params = 72
        self.num_shape_params = 10
        self.num_global_orient_params = 3
        self.num_transl_params = 3
        self.num_gt_kpts = 24
        self.num_op_kpts = 25

        self.kpts_idxs = np.arange(0,25)
        self.use_hands = use_hands
        self.use_face = use_face
        self.use_face_contour = use_face_contour

        if use_hands:
            self.kpts_idxs = np.concatenate([self.kpts_idxs, np.arange(25, 25 + 2 * 21)])
        if use_face:
            self.kpts_idxs = np.concatenate([self.kpts_idxs, np.arange(67, 67 + 51)])
        if use_face_contour:
            self.kpts_idxs = np.concatenate([self.kpts_idxs, np.arange(67 + 51, 67 + 51 + 17)])

        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.split = split
        self.body_model_type = body_model_type

        self.img_dir = osp.join(
            dataset_cfg.data_folder, dataset_cfg.image_folder
        )

        self.data = self.load_data()
        self.len = len(self.data)


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_data(self):
        dataset, mesh_v, mesh_f = Behave(
                **self.dataset_cfg, 
                split=self.split,
                body_model_type=self.body_model_type,
                get_mesh=True
            ).load() 
        self.mesh_vertices = mesh_v
        self.mesh_faces = mesh_f   
        return dataset
        
    def to_tensors(self, target):
        for k, v in target.items():
            if isinstance(v, np.ndarray):
                target[k] = torch.from_numpy(v).float()
            elif isinstance(v, list):
                target[k] = torch.tensor(v).float()
            elif isinstance(v, dict):
                target[k] = self.to_tensors(v).float()
            elif isinstance(v, torch.Tensor):
                target[k] = v.float()
        return target


    def get_single_item(self, index):

        item = self.data[index]

        img_height = item['img_height']
        img_width = item['img_width']
        
        gen_target = {
            #'images': input_image,
            'imgpath': item['imgpath'],
            'imgname_fn': item['imgname'],
            'imgname_fn_out': item['imgname'].replace('.jpg', '_')+ "pred",
            'img_height': img_height,
            'img_width': img_width,
            'sample_index': index,
        }

        cam_target = {
            'pitch': np.array(item['cam_rot'][0]),
            'yaw': np.array(item['cam_rot'][1]),
            'roll': np.array(item['cam_rot'][2]),
            'tx': np.array(item['cam_transl'][0]),
            'ty': np.array(item['cam_transl'][1]),
            'tz': np.array(item['cam_transl'][2]),
            'fl': np.array(item['fl']),
            'ih': np.array(item['img_height']),
            'iw': np.array(item['img_width']),
        }
       
        if 'bev_smpl_vertices_root_trans' in item.keys():
            bev_smpl_vertices = item['bev_smpl_vertices_root_trans'] # Tentative: Global frame vertices. Translated using cam and root.
        else:
            bev_smpl_vertices = np.zeros((1, 6890, 3))

        op_keypoints = item[f'openpose'] 
        if 'vitpose' not in item.keys():
            vitpose_keypoints = np.zeros((1, 25, 3))
        else:
            vitpose_keypoints = item[f'vitpose']

        #### SELECT FINAL KEYPLOINTS ####   
        if self.use_hands:
            final_keypoints = item[f'vitposeplus']
            mask = item['vitposeplus_human_idx'] == -1 # use openpose for missing humans
            final_keypoints[mask] = item['openpose'][mask]
        else:
            final_keypoints = item[f'vitpose']
            # add toe keypoints
            ankle_thres = 5.0
            right_ankle_residual = np.sum((final_keypoints[:,11,:] - op_keypoints[:,11,:])**2, axis=1)
            ram = right_ankle_residual < ankle_thres
            final_keypoints[ram,22,:] = op_keypoints[ram,22,:]
            left_ankle_residual = np.sum((final_keypoints[:,14,:] - op_keypoints[:,14,:])**2, axis=1)
            lam = left_ankle_residual < ankle_thres
            final_keypoints[lam,19,:] = op_keypoints[lam,19,:] 
        human_obj_target = { 
            'global_orient': item[f'{self.init_method}_global_orient'],
            'body_pose': item[f'{self.init_method}_body_pose'],
            'transl': item[f'{self.init_method}_transl'],
            'betas': item[f'{self.init_method}_betas'],
            'vertices': item[f'{self.init_method}_vertices'],
            'bev_smpl_vertices': bev_smpl_vertices,
            'op_keypoints': op_keypoints[:,self.kpts_idxs,:],
            'vitpose_keypoints': vitpose_keypoints[:,self.kpts_idxs,:], 
            'keypoints': final_keypoints[:,self.kpts_idxs,:], 
            'gender': item['gender'],
            'obj_name': item['obj_name'],
            'obj_embeddings': item['obj_embeddings'],
            'orient_obj': item[f'{self.init_method}_orient_obj'],
            'transl_obj': item[f'{self.init_method}_transl_obj'],
            'obj_vertices': self.mesh_vertices[item['obj_name']].astype(np.float32),
            'obj_faces': self.mesh_faces[item['obj_name']].astype(np.float32)
        }

        gt_human_obj_target = {
            'gt_betas': item['pgt_betas'],
            'gt_global_orient': item['pgt_global_orient'],
            'gt_body_pose': item['pgt_body_pose'],
            'gt_transl': item['pgt_transl'],
            'gt_orient_obj': item['pgt_orient_obj'],
            'gt_transl_obj': item['pgt_transl_obj'],
        }
                
        target = {**gen_target, **cam_target, **human_obj_target, **gt_human_obj_target}

        target = self.to_tensors(target)
        
        return target
