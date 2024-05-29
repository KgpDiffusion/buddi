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
from .loss_module import HHCOptiLoss
from .fit_module import HHCSOpti

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True 

DEBUG_IMG_NAMES = []


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-cfg', 
        type=str, dest='exp_cfgs', nargs='+', default=None, 
        help='The configuration of the experiment')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
        nargs='*', help='The configuration of the Detector') 
    parser.add_argument('--cluster_pid', 
        type=str, default=None, help='Cluster process id')
    parser.add_argument('--cluster_bs', 
        type=int, default=None, help='cluster batch size')

    cmd_args = parser.parse_args()

    cfg = merge_configs(cmd_args, default_config)

    return cfg, cmd_args

def main(cfg, cmd_args):

    is_cluster = cmd_args.cluster_pid is not None

    # cluster configuration
    if is_cluster:
        cpid = int(cmd_args.cluster_pid)
        cbs = int(cmd_args.cluster_bs)
        c_item_idxs = np.arange(cpid*cbs, cpid*cbs+cbs)
        guru.info(f'processing index: {c_item_idxs}')

    # save config file and create output folders
    logger = Logger(cfg)

    # make sure only one dataset is used
    assert len(cfg.datasets.train_names) <= 1, \
        "Only one dataset is supported for optimization. Hint: change config.datasets.train_names."

    # create dataloader
    FITTING_DATASETS = build_optimization_datasets(
        datasets_cfg=cfg.datasets,
        body_model_cfg=cfg.body_model, # necessary to load the correct contact maps
    )
    
    # Checkpoint 1 - Green flag!

    # configuration for optimization
    opti_cfg = cfg.model.optimization

    # load body models for human1 and human2
    body_model = build_bodymodel(
        cfg=cfg.body_model, 
        batch_size=1, 
        device=cfg.device
    )

    # build regressor used to predict diffusion params
    if opti_cfg.use_diffusion:
        diffusion_cfg = default_config.copy()
        diffusion_cfg.merge_with(OmegaConf.load(opti_cfg.pretrained_diffusion_model_cfg))
        #diffusion_logger = Logger(diffusion_cfg)
        regressor = build_model(diffusion_cfg.model.regressor).to(cfg.device)
        #checkpoint = torch.load(diffusion_logger.get_latest_checkpoint())
        checkpoint = torch.load(opti_cfg.pretrained_diffusion_model_ckpt)
        regressor.load_state_dict(checkpoint['model'], strict=False)
        diffusion = build_diffusion(**diffusion_cfg.model.diffusion)
        body_model = build_bodymodel(
            cfg=cfg.body_model, 
            batch_size=1, 
            device=cfg.device
        )
        diffusion_module = TrainModule(
            cfg=diffusion_cfg,
            train_dataset=None,
            val_dataset=None,
            diffusion=diffusion,
            model=regressor,
            criterion=None,
            evaluator=None,
            body_model=body_model,
            renderer=None,
        ).to(cfg.device)
    else:
        diffusion_module = None

    # create camera
    camera = build_camera(
        camera_cfg=cfg.camera,
        camera_type=cfg.camera.type,
        batch_size=cfg.batch_size,
        device=cfg.device
    ).to(cfg.device)


    # create validation/ evaluation metrics
    evaluator = EvalModule(
        eval_cfgs = cfg.evaluation,
        body_model_type = cfg.body_model.type,
    ).to(cfg.device)
    
    for ds in FITTING_DATASETS:
        if ds is None:
            continue

        guru.info(f'Processing {len(ds)} items from {ds.dataset_name}.')
        evaluator.reset()
        
        for item_idx in tqdm(range(len(ds))):
            if is_cluster:
                if item_idx not in c_item_idxs:
                    continue
            #try:
            if True:
                #guru.info(f'Processing item number {item_idx}')
                item = ds.get_single_item(item_idx)
                # check if item was already processed, if so, skip
                img_fn_out = item['imgname_fn_out']
                # keep to debug specific images
                
                if len(DEBUG_IMG_NAMES) > 0:
                    if img_fn_out not in DEBUG_IMG_NAMES:
                        continue
                

                out_fn_res = osp.join(logger.res_folder, f'{img_fn_out}.pkl')
                out_fn_img = osp.join(logger.img_folder, f'{img_fn_out}.png')
                # if osp.exists(out_fn_res) and osp.exists(out_fn_img):
                #     guru.info(f'Item {img_fn_out} was already processed. Skipping.')
                # else:
                #     guru.info(f'Processing item {img_fn_out} of index {item_idx}.')
                process_item(item_idx, cfg, item, logger, evaluator, diffusion_module, body_model, camera)

        evaluator.final_accumulate_step()
        for k, v in evaluator.accumulator.items():
            guru.info(f'{k}: {v}')
                

def process_item(item_idx, cfg, item, logger, evaluator, diffusion_module, body_model, camera):

    img_fn_out = item['imgname_fn_out']
    img_fn = item['imgname_fn']
    out_fn_res = osp.join(logger.res_folder, f'{img_fn_out}.pkl')
    out_fn_img = osp.join(logger.img_folder, f'{img_fn_out}.png')
    out_fn_mp4 = osp.join(logger.sum_folder, f'{img_fn_out}.mp4')
    out_fn_gif = osp.join(logger.sum_folder, f'{img_fn_out}.gif')

    # create vis directory for each filename
    vis_dir = osp.join(logger.img_folder, img_fn[:-4])
    os.makedirs(vis_dir, exist_ok=True)
    
    # create renderer (for overlay)
    renderer = Pytorch3dRenderer(
        cameras = camera.cameras,
        image_width=item['img_width'],
        image_height=item['img_height'],
    )

    opti_cfg = cfg.model.optimization
    # create losses
    criterion = HHCOptiLoss(
        losses_cfgs = opti_cfg.losses,
        body_model_type = cfg.body_model.type,
    ).to(cfg.device)

    # create optimizer module
    optimization_module = HHCSOpti(
        opti_cfg=opti_cfg,
        camera=camera,
        body_model=body_model,
        criterion=criterion,
        batch_size=cfg.batch_size,
        device=cfg.device,
        diffusion_module=diffusion_module,
        renderer=renderer,
    )


    # transform input item to human1, obj and camera dict
    human_data, cam_data, obj_data, gt_data = {}, {}, {}, {}
    for k, v in item.items():
        #TODO:shubhikg- ['joints'] not present in dataloader!
        if k in ['global_orient', 'body_pose', 'betas', 'transl', 'keypoints', 'op_keypoints', 'joints', 'gender']:
            human_data[k] = v.to(cfg.device)
        elif k in ['pitch', 'yaw', 'roll', 'tx', 'ty', 'tz', 'fl', 'ih', 'iw']:
            cam_data[k] = v.to(cfg.device)
        elif k in ['obj_name', 'obj_embeddings', 'orient_obj', 'transl_obj', 'obj_vertices', 'obj_faces']:
            if not isinstance(v, str):
                obj_data[k] = v.to(cfg.device)
            else:
                obj_data[k] = v
        elif k in ['gt_betas', 'gt_global_orient', 'gt_body_pose', 'gt_transl', 'gt_orient_obj', 'gt_transl_obj']:
            gt_data[k] = v.to(cfg.device)
        else:
            pass

    # get gt smpl.
    if human_data['gender'][0, 0] == 1:
        gt_smpl = diffusion_module.male_body_model(
            global_orient=gt_data['gt_global_orient'],
            body_pose=gt_data['gt_body_pose'],
            betas=gt_data['gt_betas'],
            transl=gt_data['gt_transl'],
        )
    else:
        gt_smpl = diffusion_module.female_body_model(
            global_orient=gt_data['gt_global_orient'],
            body_pose=gt_data['gt_body_pose'],
            betas=gt_data['gt_betas'],
            transl=gt_data['gt_transl'],
        )
    gt_obj_vertices_posed = obj_data['obj_vertices'] @ axis_angle_to_matrix(gt_data['gt_orient_obj'])[0].transpose(0, 1)  + gt_data['gt_transl_obj'][0]

    # optimize each item in dataset
    smpl_output_h1, obj_output = optimization_module.fit(
        init_human=human_data,
        init_camera=cam_data,
        init_obj=obj_data,
        item=item,
    )

    # guru.info(f'Optimization finished for {img_fn_out}. Saving results.')

    obj_vertices_posed = obj_data['obj_vertices'] @ axis_angle_to_matrix(obj_output.orient_obj)[0].transpose(0, 1)  + obj_output.transl_obj[0]
    verts_smpl = smpl_output_h1.vertices
    if human_data['gender'][0, 0] == 1:
        smplh_faces = torch.from_numpy(body_model[0].faces.astype(np.int32)).to(cfg.device)
    else:
        smplh_faces = torch.from_numpy(body_model[1].faces.astype(np.int32)).to(cfg.device)
    body_model_type = type(body_model[0]).__name__.lower().split('_')[0]
    obj_faces = obj_data['obj_faces']

    evaluator(
        est_smpl=smpl_output_h1, tar_smpl=gt_smpl,
        est_params=[obj_vertices_posed.detach()], tar_params=[gt_obj_vertices_posed], human_face=smplh_faces, object_face=[obj_faces],
        t_type='final_cond'
    )

    ## Save all rendered images
    renderings = optimization_module.renderings
    modes= ['pred_opti','one_step_pred']
    for mode in modes:
        images = []
        for stage in renderings.keys():
            for step in renderings[stage].keys():
                assert mode in renderings[stage][step].keys(), f"Mode {mode} not found in renderings keys {renderings[stage][step].keys()}"
                image = renderings[stage][step][mode]

                # write stage and step on rightmost part of image
                image = image.copy().astype(np.uint8)
                cv2.putText(image, f'ST={stage}, i={step}', (image.shape[1] - 800, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 3, cv2.LINE_AA)
                images.append(image)

        if len(images):
            # save
            out = np.vstack(images)
            out_fn_img = osp.join(vis_dir, f'{mode}.png')
            cv2.imwrite(out_fn_img, out)
    
    # print optimized values
    # print(f"Obj_output [Orient] for weight-> {opti_cfg.losses.diffusion_prior_rel_transl.weight}: ", obj_output.orient_obj)
    # print(f"Obj_output [Transl] for weight-> {opti_cfg.losses.diffusion_prior_rel_transl.weight}: ", obj_output.transl_obj)

    # if item_idx % 1 == 0 and item_idx < 150:
    #     print(f"Saving image ->{out_fn_img}")
    #     # visualize results
    #     renderer_newview = Pytorch3dRenderer(
    #         cameras = camera.cameras,
    #         image_width=200,
    #         image_height=300,
    #     )
    #     # create smpl model to get smpl faces
        
    #     vertices_methods = [[verts_smpl, obj_vertices_posed.unsqueeze(0)]]
    #     colors = [["light_blue1", "light_blue6"]]
    #     imgpath = item['imgpath']
    #     orig_img = cv2.imread(imgpath)[:,:,::-1].copy().astype(np.float32)
    #     IMG = add_alpha_channel(orig_img)
        
    #     # add keypoints to image
    #     IMGORIG = IMG.copy()
    #     h1pp = camera.project(smpl_output_h1.joints)
    #     vitpose_kpts = item['keypoints'][0]
        
    #     for idx, joint in enumerate(h1pp[0]):
    #         rand_col = np.random.randint(0, 256, 3)
    #         rand_col = tuple ([int(x) for x in rand_col])
    #         IMGORIG = cv2.circle(IMGORIG, (int(joint[0]), int(joint[1])), 3, (255, 255, 0), 2)
    #         IMGORIG = cv2.circle(IMGORIG, (int(vitpose_kpts[idx][0]), int(vitpose_kpts[idx][1])), 3, (255, 0, 0), 2)

    #     imgs_out = []
    #     for vidx, (verts, meshcol) in enumerate(zip(vertices_methods, colors)):
    #         IMG = IMGORIG.copy()
    #         renderer.update_camera_pose(
    #             camera.pitch.item(), camera.yaw.item(), camera.roll.item(), 
    #             camera.tx.item(), camera.ty.item(), camera.tz.item()
    #         )
    #         verts_hum = verts[0]
    #         verts_obj = verts[1]
    #         rendered_img = renderer.render(verts_hum, smplh_faces, verts_obj, obj_faces, colors=meshcol, body_model=body_model_type)
    #         color_image = rendered_img[0].detach().cpu().numpy() * 255
    #         overlay_image = overlay_images(IMGORIG.copy(), color_image)
    #         image_out = np.hstack((IMG, overlay_image))

    #         # now with different views
    #         vertex_transl_center = verts_hum.mean((0,1))

    #         verts_centered = verts_hum - vertex_transl_center
    #         verts_centered_obj = verts_obj - vertex_transl_center
            
    #         # y-axis rotation
    #         for yy in [45.0, 90.0, 135.0]:
    #             renderer_newview.update_camera_pose(0.0, yy, 180.0, 0.0, 0.2, 2.0)
    #             rendered_img = renderer_newview.render(
    #                 verts_centered,
    #                 smplh_faces,
    #                 verts_centered_obj,
    #                 obj_faces,
    #                 colors=meshcol,
    #                 body_model=body_model_type)
    #             color_image = rendered_img[0].detach().cpu().numpy() * 255
    #             scale = image_out.shape[0] / color_image.shape[0]
    #             newsize = (int(scale * color_image.shape[1]), int(image_out.shape[0]))
    #             color_image = cv2.resize(color_image, dsize=newsize)
    #             image_out = np.hstack((image_out, color_image))
            
    #         # bird view
    #         for pp in [270.0]:
    #             renderer_newview.update_camera_pose(pp, 0.0, 180.0, 0.0, 0.0, 2.0)
    #             rendered_img = renderer_newview.render(
    #                 verts_centered,
    #                 smplh_faces,
    #                 verts_centered_obj,
    #                 obj_faces,
    #                 colors=meshcol,
    #                 body_model=body_model_type)
    #             color_image = rendered_img[0].detach().cpu().numpy() * 255
    #             scale = image_out.shape[0] / color_image.shape[0]
    #             newsize = (int(scale * color_image.shape[1]), int(image_out.shape[0]))
    #             color_image = cv2.resize(color_image, dsize=newsize)
    #             image_out = np.hstack((image_out, color_image))
    #         imgs_out.append(image_out)

    #     # image_out = np.vstack((imgs_out[0], imgs_out[1]))
    #     image_out = imgs_out[0]
    #     cv2.imwrite(out_fn_img, image_out[...,[2,1,0,3]][...,:3])

    
if __name__ == "__main__":
    cfg, cmd_args = parse_args()
    main(cfg, cmd_args)
