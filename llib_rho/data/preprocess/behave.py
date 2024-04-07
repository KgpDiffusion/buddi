import os.path as osp 
import json 
import torch
import numpy as np
import os
import cv2
import smplx
import math
import pickle
import trimesh
from tqdm import tqdm

from llib_rho.cameras.perspective import PerspectiveCamera
from llib_rho.bodymodels.utils import smpl_to_openpose
from loguru import logger as guru
from llib_rho.data.preprocess.utils.shape_converter import ShapeConverter

KEYPOINT_COST_TRHESHOLD = 0.008

import torch
import torch.nn as nn

class Behave():
    
    BEV_FOV = 60

    def __init__(
        self,
        data_folder,
        image_folder='images',
        bev_folder='bev',
        openpose_folder='openpose',
        split='train',
        body_model_type='smplx',
        vitpose_folder='vitpose',
        vitposeplus_folder='vitposeplus',
        pseudogt_folder='gt',
        **kwargs,
    ):  

        self.data_folder = data_folder
        self.split = split
        self.imgnames = os.listdir(os.path.join(data_folder, split, image_folder))

        self.body_model_type = body_model_type
        self.image_folder = osp.join(self.data_folder, self.split, image_folder)
        self.openpose_folder = osp.join(self.data_folder, self.split, openpose_folder)
        self.bev_folder = osp.join(self.data_folder, self.split, bev_folder)
        self.vitpose_folder = osp.join(self.data_folder, self.split, vitpose_folder)
        self.pseudogt_folder = osp.join(self.data_folder, self.split, pseudogt_folder)
        self.has_pseudogt = False if pseudogt_folder == '' else True

        self.num_verts = 10475 if self.body_model_type == 'smplx' else 6890
        self.shape_converter_smpl = ShapeConverter(inbm_type='smpl', outbm_type='smplx')

        # create body model to get bev root translation from pose params
        self.body_model = self.shape_converter_smpl.outbm

    def process_bev(self, bev_human_idx, bev_data, image_size):

        smpl_betas = bev_data['smpl_betas'][bev_human_idx][:10]
        smpl_body_pose = bev_data['smpl_thetas'][bev_human_idx][3:]
        smpl_global_orient = bev_data['smpl_thetas'][bev_human_idx][:3]

        smplx_betas = self.shape_converter_smpl.forward(torch.from_numpy(smpl_betas).unsqueeze(0))
       

        cam_trans = bev_data['cam_trans'][bev_human_idx]
        smpl_joints = bev_data['joints'][bev_human_idx]
        smpl_vertices = bev_data['verts'][bev_human_idx]
        smpl_joints_2d = bev_data['pj2d_org'][bev_human_idx]

        data = {
            'bev_smpl_global_orient': smpl_global_orient,
            'bev_smpl_body_pose': smpl_body_pose,
            'bev_smpl_betas': smpl_betas,
            'bev_betas': smplx_betas.float().unsqueeze(0),
            'bev_cam_trans': cam_trans,
            'bev_smpl_joints': smpl_joints,
            'bev_smpl_vertices': smpl_vertices,
            'bev_smpl_joints_2d': smpl_joints_2d,
        }
        
        height, width = image_size

        # hacky - use smpl pose parameters with smplx body model
        # not perfect, but close enough. SMPL betas are not used with smpl-x.
        if self.body_model_type == 'smplx':
            body_pose = data['bev_smpl_body_pose'][:63]
            global_orient = data['bev_smpl_global_orient']
            betas = data['bev_betas']

        
        bev_cam_trans = torch.from_numpy(data['bev_cam_trans'])
        bev_camera = PerspectiveCamera(
            rotation=torch.tensor([[0., 0., 180.]]),
            translation=torch.tensor([[0., 0., 0.]]),
            afov_horizontal=torch.tensor([self.BEV_FOV]),
            image_size=torch.tensor([[width, height]]),
            batch_size=1,
            device='cpu'
        )

        bev_vertices = data['bev_smpl_vertices']
        bev_root_trans = data['bev_smpl_joints'][[45,46],:].mean(0)
        bev_vertices_root_trans = bev_vertices - bev_root_trans[np.newaxis,:] \
            + bev_cam_trans.numpy()[np.newaxis,:]
        data['bev_smpl_vertices_root_trans'] = bev_vertices_root_trans
        
        smplx_update = {
            'bev_global_orient': [],  # 1, 3
            'bev_body_pose': [],  # 1, 63
            'bev_transl': [],  # 1, 3
            'bev_keypoints': [],
            'bev_vertices': [], # 1, 10475, 3
        }

        h_global_orient = torch.from_numpy(global_orient).float().unsqueeze(0) # 1, 3
        smplx_update['bev_global_orient'].append(h_global_orient)
        
        h_body_pose = torch.from_numpy(body_pose).float().unsqueeze(0)  # 1, 63
        smplx_update['bev_body_pose'].append(h_body_pose)

        h_betas = torch.from_numpy(
            betas
        ).float().unsqueeze(0)

        body = self.body_model(
            global_orient=h_global_orient,
            body_pose=h_body_pose,
            betas=h_betas,
        )

        root_trans = body.joints.detach()[:,0,:]
        transl = -root_trans.to('cpu') + bev_cam_trans.to('cpu')
        smplx_update['bev_transl'].append(transl)

        body = self.body_model(
            global_orient=h_global_orient,
            body_pose=h_body_pose,
            betas=h_betas,
            transl=transl,
        )

        keypoints = bev_camera.project(body.joints.detach())
        smplx_update['bev_keypoints'].append(keypoints.detach())

        vertices = body.vertices.detach().to('cpu')
        smplx_update['bev_vertices'].append(vertices)

        for k, v in smplx_update.items():
            smplx_update[k] = torch.cat(v, dim=0)

        data.update(smplx_update)

        return data

    def load_single_image(self, imgname):
        img_path = osp.join(self.image_folder, f'{imgname}')
        bev_path = osp.join(self.bev_folder, f'{imgname[:-4]}.npz')
        vitpose_path = osp.join(self.vitpose_folder, f'{imgname[:-4]}_keypoints.json')
        openpose_path = osp.join(self.openpose_folder, f'{imgname[:-4]}.json')

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        bev_data = np.load(bev_path, allow_pickle=True)['results'][()]
        vitpose_data = json.load(open(vitpose_path, 'r'))['people']
        if not os.path.exists(openpose_path):
            guru.warning(f'Openpose file does not exist; using ViTPose keypoints only.')
            op_data = vitpose_data
        else:
            op_data = json.load(open(openpose_path, 'r'))['people']

        height, width = img.shape[:2]
        # camera translation was already applied to mesh, so we can set it to zero.
        cam_transl = [0., 0., 0.] 
        # camera rotation needs 180 degree rotation around z axis, because bev and
        # pytorch3d camera coordinate systems are different            
        cam_rot = [0., 0., 180.]
        afov_radians = (self.BEV_FOV / 2) * math.pi / 180
        focal_length_px = (max(width, height)/2) / math.tan(afov_radians)

        image_data_template = {
            'imgname': imgname,
            'imgpath': img_path,
            'image': img,
            'img_height': height,
            'img_width': width,
            'cam_transl': cam_transl,
            'cam_rot': cam_rot,
            'fl': focal_length_px,
            'afov_horizontal': self.BEV_FOV,
        }

        human_idx = 0 # We only have one human in our data
        human_data = self.process_bev(human_idx, bev_data, (height, width))

        # process OpenPose keypoints
        kpts = op_data[human_idx]
        # body + hands
        body = np.array(kpts['pose_keypoints_2d'] + \
            kpts['hand_left_keypoints_2d'] + kpts['hand_right_keypoints_2d']
        ).reshape(-1,3)
        # face 
        face = np.array(kpts['face_keypoints_2d'],
            dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]
        contour = np.array(kpts['face_keypoints_2d'],
            dtype=np.float32).reshape([-1, 3])[:17, :]
        # final openpose
        op_kpts = np.expand_dims(np.concatenate([body, face, contour], axis=0), axis=0)  # 1, 135, 3

        # process Vitpose keypoints
        kpts = vitpose_data[human_idx]
        # body + hands
        body = np.array(kpts['pose_keypoints_2d'] + \
            kpts['hand_left_keypoints_2d'] + kpts['hand_right_keypoints_2d']
        ).reshape(-1,3)
        # face 
        face = np.array(kpts['face_keypoints_2d'],
            dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]
        contour = np.array(kpts['face_keypoints_2d'],
            dtype=np.float32).reshape([-1, 3])[:17, :]
        # final openpose
        vitpose_kpts = np.expand_dims(np.concatenate([body, face, contour], axis=0), axis=0)  # 1, 135, 3

        human_data['vitpose'] = vitpose_kpts
        human_data['openpose'] = op_kpts
        for k, v in human_data.items():

            # if k in [
            #     'bev_global_orient', 'bev_body_pose', 'bev_transl', 
            #     'bev_keypoints', 'bev_vertices'
            # ]:
            #     v = v[0]
            human_data[k] = v.clone() 

        image_data_template.update(human_data)

        if self.has_pseudogt:
            gt_path = osp.join(self.image_folder, imgname[:-3] + 'pkl')
            gt_fits = pickle.load(
                open(gt_path, 'rb'))
            
            pgt_data = {
                'pgt_betas': gt_fits['betas'],
                'pgt_global_orient': gt_fits['global_orient'],
                'pgt_body_pose': gt_fits['pose'],
                'pgt_transl': gt_fits['trans'],
                'pgt_orient_obj': gt_fits['obj_angle'],  # 1, 3
                'pgt_transl_obj': gt_fits['obj_trans']   # 1, 3
            }
            image_data_template.update(pgt_data)

        return image_data_template

    def load(self):
        data = []
        for imgname in tqdm(self.imgnames):
            data += self.load_single_image(imgname)
        return data