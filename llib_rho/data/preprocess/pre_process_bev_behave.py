import os
import os.path as osp 
import torch
import numpy as np
import smplx
import math
import pickle
from tqdm import tqdm
import argparse

from llib_rho.cameras.perspective import PerspectiveCamera
from llib_rho.data.preprocess.utils.shape_converter import ShapeConverter

BEV_FOV = 60
 
def process_bev(bev_human_idx, bev_data, image_size, gender, body_model_type, shape_converter_smpl, body_model=None, male_body_model=None, female_body_model=None):

    smpl_betas = bev_data['smpl_betas'][bev_human_idx][:10]
    smpl_body_pose = bev_data['smpl_thetas'][bev_human_idx][3:]
    smpl_global_orient = bev_data['smpl_thetas'][bev_human_idx][:3]

    if body_model_type == 'smplx':
        new_betas = shape_converter_smpl.forward(torch.from_numpy(smpl_betas).unsqueeze(0))
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
    if body_model_type == 'smplx' or body_model_type == 'smplh' or body_model_type == 'smpl':
        body_pose = data['bev_smpl_body_pose'][:63]
        global_orient = data['bev_smpl_global_orient']
        betas = data['bev_betas']

    
    bev_cam_trans = torch.from_numpy(data['bev_cam_trans'])
    bev_camera = PerspectiveCamera(
        rotation=torch.tensor([[0., 0., 180.]]),
        translation=torch.tensor([[0., 0., 0.]]),
        afov_horizontal=torch.tensor([BEV_FOV]),
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

    if body_model_type != 'smplh':
        body_model = body_model
    else:
        if gender == 'male':
            body_model = male_body_model
        else:
            body_model = female_body_model

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/shubhikg/data/train', type=str)
    parser.add_argument('--output_body_model_type', default='smplh', type=str)
    args = parser.parse_args()

    input_folder_bev = os.path.join(args.data_path, "bev")
    output_folder_bev = os.path.join(args.data_path, "bev_process")
    os.makedirs(output_folder_bev, exist_ok=True)

    meta_data_path = os.path.join(args.data_path, 'metadata.pkl')
    meta_data = pickle.load(open(meta_data_path, 'rb'))
    height = 1536
    width = 2048
    output_body_model_type = args.output_body_model_type
    shape_converter_smpl = None # ShapeConverter(inbm_type='smpl', outbm_type=output_body_model_type)

    model_folder = osp.join('/home/shubhikg/exp/buddi/essentials', 'body_models')
    male_body_model = smplx.create(model_path=model_folder, model_type='smplh', gender='male')
    female_body_model = smplx.create(model_path=model_folder, model_type='smplh', gender='female')

    bev_names = os.listdir(input_folder_bev)
    for name in tqdm(bev_names):
        bev_path = osp.join(input_folder_bev, name)
        bev_data = np.load(bev_path, allow_pickle=True)['results'][()]

        human_idx = 0
        gender = meta_data[name[:-4]][0]
        human_data = {}
        if len(bev_data.keys()) > 0:
            human_data = process_bev(human_idx, bev_data, (height, width), gender, output_body_model_type,
                                    shape_converter_smpl, None, male_body_model, female_body_model)

            for key in human_data:
                val = human_data[key]
                if isinstance(val, torch.Tensor):
                    human_data[key] = val.numpy()
        else:
            print("No human detected")
        np.savez(os.path.join(output_folder_bev, name), results=human_data)