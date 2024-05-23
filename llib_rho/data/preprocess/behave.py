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
import random 
import torchvision.models as models
import torchvision
from llib_rho.cameras.perspective import PerspectiveCamera
from llib_rho.bodymodels.utils import smpl_to_openpose
from loguru import logger as guru
from llib_rho.data.preprocess.utils.shape_converter import ShapeConverter
from pytorch3d.transforms import matrix_to_axis_angle

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
        pose_folder='pose_pred',
        resnet_folder='resnet_feat',
        get_mesh=False,
        **kwargs,
    ):  

        self.data_folder = data_folder
        self.split = split
        self.get_mesh = get_mesh
        # self.imgnames = os.listdir(os.path.join(data_folder, split, image_folder))
        self.imgnames = []
        for img in os.listdir(os.path.join(data_folder, split, pseudogt_folder)):
            img = img[:-4]
            self.imgnames.append(img + ".jpg")
        # self.imgnames = random.choices(self.imgnames, k=100)

        assert body_model_type in ['smpl', 'smplh', 'smplx'], "Can only handle smpl, smplh and smplx body model."
        self.body_model_type = body_model_type

        self.image_folder = osp.join(self.data_folder, self.split, image_folder)
        self.cropped_image_folder = osp.join(self.data_folder, self.split, 'cropped_images')
        self.openpose_folder = osp.join(self.data_folder, self.split, openpose_folder)
        self.bev_folder = osp.join(self.data_folder, self.split, bev_folder)
        self.vitpose_folder = osp.join(self.data_folder, self.split, vitpose_folder)
        self.pose_folder = osp.join(self.data_folder, self.split, pose_folder)
        self.resnet_folder = osp.join(self.data_folder, self.split, resnet_folder)
        self.pseudogt_folder = osp.join(self.data_folder, self.split, pseudogt_folder)
        self.has_pseudogt = False if pseudogt_folder == '' else True

        self.num_verts = 10475 if self.body_model_type == 'smplx' else 6890
        self.shape_converter_smpl = ShapeConverter(inbm_type='smpl', outbm_type=self.body_model_type)

        # create body model to get bev root translation from pose params
        self.body_model = self.shape_converter_smpl.outbm
        if self.body_model_type == 'smplh':
            model_folder = osp.join('essentials', 'body_models')
            self.male_body_model = smplx.create(model_path=model_folder, model_type='smplh', gender='male')
            self.female_body_model = smplx.create(model_path=model_folder, model_type='smplh', gender='female')

        # Initialize pretrained resnet 18
        # self.resnet18 = models.resnet18(pretrained=True)
        # self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        # self.resnet18.train() # freeze the model

        meta_data_path = os.path.join(data_folder, split, 'metadata.pkl')
        self.meta_data = pickle.load(open(meta_data_path, 'rb'))

        self.obj_embeddings = {}
        obj_embeddings = np.load(os.path.join(data_folder, 'obj_embeddings.npy'), allow_pickle=True)
        for key in obj_embeddings.item().keys():
            self.obj_embeddings[key] = np.expand_dims(obj_embeddings.item()[key], axis=0).astype(np.float32)

        if self.get_mesh:
            self.mesh_vertices = {}
            self.mesh_faces = {}
            with open(os.path.join(data_folder, "ref_hoi.pkl"), "rb") as f:
                x = pickle.load(f)
                for obj_name in x['templates'].keys():
                    data = x['templates'][obj_name]
                    verts = data['verts']
                    faces = data['faces']
                    self.mesh_vertices[obj_name] = verts
                    self.mesh_faces[obj_name] = faces

        self.gender_one_hot = {}
        self.gender_one_hot['male'] = np.array([[1, 0]], dtype=np.float32)
        self.gender_one_hot['female'] = np.array([[0, 1]], dtype=np.float32)

    def process_bev(self, bev_human_idx, bev_data, image_size, gender):

        smpl_betas = bev_data['smpl_betas'][bev_human_idx][:10]
        smpl_body_pose = bev_data['smpl_thetas'][bev_human_idx][3:]
        smpl_global_orient = bev_data['smpl_thetas'][bev_human_idx][:3]

        if self.body_model_type == 'smplx':
            new_betas = self.shape_converter_smpl.forward(torch.from_numpy(smpl_betas).unsqueeze(0))
        else:
            new_betas = torch.from_numpy(smpl_betas).unsqueeze(0)
       

        cam_trans = bev_data['cam_trans'][bev_human_idx]
        smpl_joints = bev_data['joints'][bev_human_idx]
        smpl_vertices = bev_data['verts'][bev_human_idx]
        smpl_joints_2d = bev_data['pj2d_org'][bev_human_idx]

        data = {
            'bev_smpl_global_orient': smpl_global_orient,
            'bev_smpl_body_pose': smpl_body_pose,
            'bev_smpl_betas': smpl_betas,
            'bev_betas': new_betas.float().squeeze(0).cpu().numpy(),
            'bev_cam_trans': cam_trans,
            'bev_smpl_joints': smpl_joints,
            'bev_smpl_vertices': smpl_vertices,
            'bev_smpl_joints_2d': smpl_joints_2d,
        }
        
        height, width = image_size

        # hacky - use smpl pose parameters with smplx body model
        # not perfect, but close enough. SMPL betas are not used with smpl-x.
        if self.body_model_type == 'smplx' or self.body_model_type == 'smplh' or self.body_model_type == 'smpl':
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

        if self.body_model_type != 'smplh':
            body_model = self.body_model
        else:
            if gender == 'male':
                body_model = self.male_body_model
            else:
                body_model = self.female_body_model

        body = body_model(
            global_orient=h_global_orient,
            body_pose=h_body_pose,
            betas=h_betas,
        )

        root_trans = body.joints.detach()[:,0,:]
        transl = -root_trans.to('cpu') + bev_cam_trans.to('cpu')
        smplx_update['bev_transl'].append(transl)


        body = body_model(
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
            smplx_update[k] = torch.cat(v, dim=0).float()

        data.update(smplx_update)

        return data
    
    def preprocess_image(self, img_cropped: np.ndarray):
        """ Preprocess image for ResNet18.
            1. Apply transforms to cropped image
         
         Returns torch.Tensor of shape (3, 224, 224)"""
        
        INPUT_IMG_SIZE = 224

        assert img_cropped.shape == (INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3), f"Image shape is {img_cropped.shape}"

        # torchvision transforms
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img_tensor = transforms(img_cropped.permute(2,0,1))

        return img_tensor

    def load_single_image(self, imgname):
        imgID = os.path.basename(imgname)
        img_path = osp.join(self.image_folder, f'{imgname}')
        # img_cropped_path = osp.join(self.cropped_image_folder, f'{imgname}')

        bev_path = osp.join(self.bev_folder, f'{imgname[:-4]}.npz')
        vitpose_path = osp.join(self.vitpose_folder, f'{imgname[:-4]}_keypoints.json')
        openpose_path = osp.join(self.openpose_folder, f'{imgname[:-4]}.json')
        resnet_path = osp.join(self.resnet_folder, f'{imgname[:-4]}.pkl')

        # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # img_cropped = cv2.cvtColor(cv2.imread(img_cropped_path), cv2.COLOR_BGR2RGB)

        # preprocess image for resnet
        # input_resnet = self.preprocess_image(img_cropped) # 224, 224, 3 tensor
        # resnet_feat = self.resnet18(input_resnet.unsqueeze(0))
        # assert resnet_feat.shape == (1, 512), f"Resnet feature shape is {resnet_feat.shape}"

        # sanity check bev path
        if not os.path.exists(bev_path):
            return None
        
        bev_data = np.load(bev_path, allow_pickle=True)['results'][()]

        if len(bev_data.keys()) == 0:
            return None
        
        pose_path = osp.join(self.pose_folder, f'{imgname[:-4]}.json')
        pose_data = json.load(open(pose_path, 'r'))

        # load resnet data
        resnet_feat = pickle.load(open(resnet_path, 'rb'))
        assert resnet_feat.shape == (512,), f"Resnet feature shape is {resnet_feat.shape}"

        bev_data['bev_betas'] = np.expand_dims(bev_data['bev_betas'], axis=0).astype(np.float32)
        obj_rot = torch.from_numpy(np.expand_dims(pose_data['R_pred'], axis=0))
        bev_data['bev_orient_obj'] = matrix_to_axis_angle(obj_rot).numpy().astype(np.float32)
        bev_data['bev_transl_obj'] = np.array(pose_data['t_pred']).reshape(1, 3).astype(np.float32) + bev_data['bev_transl']
        # bev_data['bev_orient_obj'] = np.array([[0.0, 0.0, 0.0]]).astype(np.float32)
        # bev_data['bev_transl_obj'] = bev_data['bev_transl']
        
        # vitpose_data = json.load(open(vitpose_path, 'r'))['people']
        # if not os.path.exists(openpose_path):
        #     guru.warning(f'Openpose file does not exist; using ViTPose keypoints only.')
        #     op_data = vitpose_data
        # else:
        #     op_data = json.load(open(openpose_path, 'r'))['people']
        vitpose_data = np.zeros((1, 135, 3))
        vitpose_data[:, :, -1] = 1
        op_data = vitpose_data

        height, width = [1536, 2048] # img.shape[:2]
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
            # 'image': img,
            'resnet_feat': resnet_feat.reshape(1, 512),
            'img_height': height,
            'img_width': width,
            'cam_transl': cam_transl,
            'cam_rot': cam_rot,
            'fl': focal_length_px,
            'afov_horizontal': self.BEV_FOV,
        }

        # Get gender of human
        gender = self.meta_data[imgname[:-4]][0]
        image_data_template['gender'] = self.gender_one_hot[gender] # 1, 2

        human_idx = 0 # We only have one human in our data
        # human_data = self.process_bev(human_idx, bev_data, (height, width), gender)
        human_data = {}
        human_data.update(bev_data)

        # get obj name and corresponding embeddings
        obj_name = self.meta_data[imgname[:-4]][1]
        obj_embeddings = self.obj_embeddings[obj_name]
        image_data_template['obj_name'] = obj_name
        image_data_template['obj_embeddings'] = obj_embeddings  # 1, 256

        # # process OpenPose keypoints
        # kpts = op_data[human_idx]
        # # body + hands
        # body = np.array(kpts['pose_keypoints_2d'] + \
        #     kpts['hand_left_keypoints_2d'] + kpts['hand_right_keypoints_2d']
        # ).reshape(-1,3)
        # # face 
        # face = np.array(kpts['face_keypoints_2d'],
        #     dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]
        # contour = np.array(kpts['face_keypoints_2d'],
        #     dtype=np.float32).reshape([-1, 3])[:17, :]
        # # final openpose
        # op_kpts = np.expand_dims(np.concatenate([body, face, contour], axis=0), axis=0)  # 1, 135, 3
        op_kpts = op_data

        # # process Vitpose keypoints
        # kpts = vitpose_data[human_idx]
        # # body + hands
        # body = np.array(kpts['pose_keypoints_2d'] + \
        #     kpts['hand_left_keypoints_2d'] + kpts['hand_right_keypoints_2d']
        # ).reshape(-1,3)
        # # face 
        # face = np.array(kpts['face_keypoints_2d'],
        #     dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]
        # contour = np.array(kpts['face_keypoints_2d'],
        #     dtype=np.float32).reshape([-1, 3])[:17, :]
        # # final openpose
        # vitpose_kpts = np.expand_dims(np.concatenate([body, face, contour], axis=0), axis=0)  # 1, 135, 3
        vitpose_kpts = vitpose_data

        human_data['vitpose'] = vitpose_kpts
        human_data['openpose'] = op_kpts
        # for k, v in human_data.items():

        #     # if k in [
        #     #     'bev_global_orient', 'bev_body_pose', 'bev_transl', 
        #     #     'bev_keypoints', 'bev_vertices'
        #     # ]:
        #     #     v = v[0]
        #     human_data[k] = v.clone() 

        image_data_template.update(human_data)

        if self.has_pseudogt:
            gt_path = osp.join(self.pseudogt_folder, imgname[:-3] + 'pkl')
            gt_fits = pickle.load(
                open(gt_path, 'rb'))
            
            pose = np.expand_dims(gt_fits['pose'], axis=0).astype(np.float32)
            pgt_data = {
                'pgt_betas': np.expand_dims(gt_fits['betas'], axis=0).astype(np.float32),
                'pgt_global_orient': pose[:, :3],
                'pgt_body_pose': pose[:, 3:66],
                'pgt_transl': gt_fits['trans'].astype(np.float32),
                'pgt_orient_obj': np.expand_dims(gt_fits['obj_angle'], axis=0).astype(np.float32),  # 1, 3
                'pgt_transl_obj': np.expand_dims(gt_fits['obj_trans'], axis=0).astype(np.float32),   # 1, 3
            }
            image_data_template.update(pgt_data)
            # obj_transl = bev_data['bev_transl_obj'] - pgt_data['pgt_transl'] + bev_data['bev_transl']
            # image_data_template['bev_transl_obj'] = obj_transl

        return image_data_template

    def load(self):
        data = []
        for imgname in tqdm(self.imgnames):
            img_data = self.load_single_image(imgname)
            if img_data is not None:
                data.append(img_data)
        if self.get_mesh:
            return data, self.mesh_vertices, self.mesh_faces
        else:
            return data