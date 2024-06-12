# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn
import os
import numpy as np
from loguru import logger as guru
from llib_rho.optimizer.build import build_optimizer
from llib_rho.training.fitter import Stopper
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from llib_rho.visualization.utils import *
from llib_rho.visualization.renderer import Pytorch3dRenderer
import cv2

class ObjectBehave(nn.Module):
    def __init__(
        self,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        device: str = "cuda",
    ):
        super().__init__()
        self.register_parameter('orient_obj', nn.Parameter(rotation))
        self.register_parameter('transl_obj', nn.Parameter(translation))


class HHCSOpti(nn.Module):
    """HHC optimizes two meshes using image keypoints and discrete
    contact labels."""
    def __init__(self,
                 opti_cfg,
                 camera,
                 image_size,
                 body_model,
                 criterion,
                 batch_size=1,
                 device='cuda',
                 diffusion_module=None,
                 renderer=None
    ):
        super(HHCSOpti, self).__init__()

        # Save config file
        self.opti_cfg = opti_cfg
        self.batch_size = batch_size
        self.H, self. W = image_size
        self.device = device
        self.print_loss = opti_cfg.print_loss
        self.render_iters = opti_cfg.render_iters
        self.camera = camera

        self.male_body_model = body_model[0]
        self.female_body_model = body_model[1]
        self.body_model = None
        self.faces = torch.from_numpy(self.male_body_model.faces.astype(np.int32))

        self.diffusion_module = diffusion_module

        # human optimization
        self.criterion = criterion
        self.num_iters = opti_cfg.hhcs.max_iters

        # parameters to be optimized
        # only optimize betas in first stage
        self.optimizables = {
            0: [
                'body_model.transl',
                'body_model.betas',
                'body_model.body_pose',
                # 'body_model.global_orient',
                'obj.orient_obj',
                'obj.transl_obj',
            ],    
            1: [
                'body_model.transl',
                'body_model.body_pose',
                'obj.orient_obj',
                'obj.transl_obj',
            ],
        }       

        # TODO: ag6, shubhikg - In our case we have bev guidance so might as well uncomment it.
        # Can see visually the outputs of BEV rotation or measure quantitatively how far is BEV prediction from gt in global orient!
        # if bev guidance also optimize the body global orientation 
        # if len(self.diffusion_module.exp_cfg.guidance_params) > 0:
            # self.optimizables[0].extend([
                # 'body_model_h1.global_orient',
                # 'body_model_h2.global_orient',
            # ])
        

        # stop criterion 
        self.stopper = Stopper(
            num_prev_steps=opti_cfg.hhcs.num_prev_steps,
            slope_tol=opti_cfg.hhcs.slope_tol,
        )

        # initialize core renderer
        self.renderer = renderer

        # rendered images per iter
        if self.render_iters:
            
            self.renderer_newview = Pytorch3dRenderer(
            cameras = camera.cameras,
            image_width=200,
            image_height=300,
            )
            self.renderings = dict()

        ## Output SMPL dictionary
        self.total_steps_save=5
        self.save_at_steps=None

    def setup_optimiables(self, stage):
        
        self.final_params = [] 

        optimizer_type = self.opti_cfg.optimizer.type
        lr = stage_lr = eval(f'self.opti_cfg.optimizer.{optimizer_type}.lr')
        if stage in [1]:
            stage_lr = lr / 10

        # camera parameters
        for param_name, param in self.named_parameters():
            if param_name in self.optimizables[stage]:
                param.requires_grad = True
                self.final_params.append({'params': param, 'lr': stage_lr})
            else:
                param.requires_grad = False
            
    @torch.no_grad()
    def fill_params(self, init_human, init_cam, init_obj):
        """Fill the parameters of the human model and camera with the
        initial values."""

        device = self.male_body_model.betas.device
        gender_male_mask = init_human['gender'][0, 0] == 1
        if gender_male_mask:
            self.body_model = self.male_body_model.clone().to(device)
        else:
            self.body_model = self.female_body_model.clone().to(device)
        
        for param_name, param in self.body_model.named_parameters():
            # FOR SMPL-H they include all SMPL (global trans,orient,shape,pose) + (left_hand_pose+right_hand_pose)
            if param_name in init_human.keys():
                init_value = init_human[param_name][[0]].clone().detach().to(device).requires_grad_(True)
                param[:] = init_value

        # Maybe they'll not optimize it later on. Verify!
        for param_name, param in self.camera.named_parameters():
            if param_name in init_cam.keys():
                init_value = init_cam[param_name].clone().detach().unsqueeze(0).to(device).requires_grad_(True)
                param[:] = init_value

        self.camera.iw[:] = init_cam['iw']
        self.camera.ih[:] = init_cam['ih']

        self.obj = ObjectBehave(init_obj['orient_obj'], init_obj['transl_obj'])        


    def setup_optimizer(self, init_human, init_cam,init_obj, stage):
        """Setup the optimizer for the current stage / reset in stages > 0."""

        # in the first stage, set the SMPL-X parameters to the initial values        
        if stage == 0:
            self.fill_params(init_human, init_cam, init_obj)

        # pick the parameters to be optimized
        self.setup_optimiables(stage)

        # build optimizer
        self.optimizer = build_optimizer(
            self.opti_cfg.optimizer, 
            self.opti_cfg.optimizer.type,
            self.final_params
        )

    def print_losses(self, ld, stage=0, step=0, abbr=True):
        """Print the losses for the current stage."""
        total_loss = ld['total_loss'].item()
        out = f'Stage/step:{stage:2d}/{step:2} || Tl: {total_loss:.4f} || '
        for k, v in ld.items():
            if k != 'total_loss':
                kprint = ''.join([x[0] for x in k.split('_')]) if abbr else k
                if type(v) == torch.Tensor:
                    v = v.item()
                    out += f'{kprint}: {v:.4f} | '
        print(out)

    def convert_colored_image_to_mask(self, image, 
                                      lower_green=[40, 0,0], 
                                      upper_green=[150, 255, 255]):
        """ Convert colored image to mask.
            Args:   
                image (H,W,3): Colored image with object rendered in green
            Returns:
                mask: Mask of the object in the image"""
        # DO HSV masking for green color
        image = image[:,:,:3].astype(np.uint8)
        assert image.shape == (self.H, self.W, 3), f"Image shape is not correct {image.shape}"
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array(lower_green)
        upper_green = np.array(upper_green)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.merge((mask, mask, mask))
        # filter
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        assert mask.shape == (self.H, self.W), f"Mask shape is not correct {mask.shape}"
        return mask
    
    def render_obj_mask(self, init_obj, item, 
                            stage="", iter=None, 
                            color=['light_blue3', 'light_blue5'],):
        """Render the object binary mask"""
        imgname  = item['imgpath']
        image = cv2.imread(imgname).astype(np.float32)

        smpl_output = self.body_model()
        obj_output= self.obj
        obj_vertices_posed = init_obj['obj_vertices'] @ axis_angle_to_matrix(obj_output.orient_obj)[0].transpose(0, 1)  + obj_output.transl_obj[0]

        device = smpl_output.vertices.device
        verts_smpl = smpl_output.vertices
        smplh_faces = torch.from_numpy(self.body_model.faces.astype(np.int32)).to(device)

        body_model_type = type(self.body_model).__name__.lower().split('_')[0]
        obj_faces = init_obj['obj_faces']

        # create smpl model to get smpl faces
        vertices_methods = [[verts_smpl, obj_vertices_posed.unsqueeze(0)]]
        colors = [['light_blue1','light_green1']]

        imgs_out = []
        for vidx, (verts, meshcol) in enumerate(zip(vertices_methods, colors)):
            # from IPython import embed; embed()
            self.renderer.update_camera_pose(
                self.camera.pitch.item(), self.camera.yaw.item(), self.camera.roll.item(), 
                self.camera.tx.item(), self.camera.ty.item(), self.camera.tz.item()
            )
            verts_hum = verts[0]
            verts_obj = verts[1]
            rendered_img = self.renderer.render(verts_hum, smplh_faces, verts_obj, obj_faces, colors=meshcol, body_model=body_model_type)
            self.color_image = rendered_img[0][...,:3] * 255.0
            color_image_numpy = self.color_image.detach().cpu().numpy().astype(np.uint8)
            self.mask_ = torch.Tensor(self.convert_colored_image_to_mask(color_image_numpy)).to(self.device)/255.0
            self.color_image = torch.mean(self.color_image, dim=-1)/255.0
            self.mask_.requires_grad = True
            self.mask_.retain_grad()
            self.color_image.retain_grad()
            self.mask = self.color_image*(self.mask_)

        # self.mask[self.mask>0]=255.0 # binary self.mask
        self.mask = 1 - torch.exp(-100*self.mask)
        self.mask.retain_grad()

        assert self.mask.shape == (self.H, self.W), f"Mask shape is not correct {self.mask.shape}"
        return self.mask*255
        

    def render_current_estimate(self, init_obj, vis_out, item, 
                                stage="", iter=None, color=['light_blue3', 'light_blue5'],
                                prefix = 'pred_opti'):
        """Render the current estimates"""
        if iter in self.save_at_steps:
            with torch.no_grad():
                if vis_out is not None:
                    smpl_output = vis_out['one_step']['smpl_output']
                    obj_output = vis_out['one_step']['obj_output']
                    orient_obj = obj_output['orient_obj']
                    transl_obj = obj_output['transl_obj']
                    obj_vertices_posed = init_obj['obj_vertices'] @ axis_angle_to_matrix(orient_obj)[0].transpose(0, 1)  + transl_obj[0]
                else:
                    smpl_output = self.body_model()
                    obj_output= self.obj
                    obj_vertices_posed = init_obj['obj_vertices'] @ axis_angle_to_matrix(obj_output.orient_obj)[0].transpose(0, 1)  + obj_output.transl_obj[0]

                device = smpl_output.vertices.device
                verts_smpl = smpl_output.vertices
                smplh_faces = torch.from_numpy(self.body_model.faces.astype(np.int32)).to(device)
        
            body_model_type = type(self.body_model).__name__.lower().split('_')[0]
            obj_faces = init_obj['obj_faces']

            # create smpl model to get smpl faces
            vertices_methods = [[verts_smpl, obj_vertices_posed.unsqueeze(0)]]
            colors = [["light_blue1", "light_blue6"]]
            orig_img = cv2.imread(item['imgpath'])[:,:,::-1].copy().astype(np.float32)
            IMG = add_alpha_channel(orig_img)
            
            # add keypoints to image
            IMGORIG = IMG.copy()
            h1pp = self.camera.project(smpl_output.joints)
            vitpose_kpts = item['keypoints'][0]
            
            for idx, joint in enumerate(h1pp[0]):
                rand_col = np.random.randint(0, 256, 3)
                rand_col = tuple ([int(x) for x in rand_col])
                IMGORIG = cv2.circle(IMGORIG, (int(joint[0]), int(joint[1])), 3, (255, 255, 0), 2)
                IMGORIG = cv2.circle(IMGORIG, (int(vitpose_kpts[idx][0]), int(vitpose_kpts[idx][1])), 3, (255, 0, 0), 2)

            imgs_out = []
            for vidx, (verts, meshcol) in enumerate(zip(vertices_methods, colors)):
                IMG = IMGORIG.copy()
                self.renderer.update_camera_pose(
                    self.camera.pitch.item(), self.camera.yaw.item(), self.camera.roll.item(), 
                    self.camera.tx.item(), self.camera.ty.item(), self.camera.tz.item()
                )
                verts_hum = verts[0]
                verts_obj = verts[1]
                rendered_img = self.renderer.render(verts_hum, smplh_faces, verts_obj, obj_faces, colors=meshcol, body_model=body_model_type)
                color_image = rendered_img[0].detach().cpu().numpy() * 255
                overlay_image = overlay_images(IMGORIG.copy(), color_image)
                image_out = np.hstack((IMG, overlay_image))

                # now with different views
                vertex_transl_center = verts_hum.mean((0,1))

                verts_centered = verts_hum - vertex_transl_center
                verts_centered_obj = verts_obj - vertex_transl_center
                
                # y-axis rotation
                for yy in [45.0, 90.0, 135.0]:
                    self.renderer_newview.update_camera_pose(0.0, yy, 180.0, 0.0, 0.2, 2.0)
                    rendered_img = self.renderer_newview.render(
                        verts_centered,
                        smplh_faces,
                        verts_centered_obj,
                        obj_faces,
                        colors=meshcol,
                        body_model=body_model_type)
                    color_image = rendered_img[0].detach().cpu().numpy() * 255
                    scale = image_out.shape[0] / color_image.shape[0]
                    newsize = (int(scale * color_image.shape[1]), int(image_out.shape[0]))
                    color_image = cv2.resize(color_image, dsize=newsize)
                    image_out = np.hstack((image_out, color_image))
                
                # bird view
                for pp in [270.0]:
                    self.renderer_newview.update_camera_pose(pp, 0.0, 180.0, 0.0, 0.0, 2.0)
                    rendered_img = self.renderer_newview.render(
                        verts_centered,
                        smplh_faces,
                        verts_centered_obj,
                        obj_faces,
                        colors=meshcol,
                        body_model=body_model_type)
                    color_image = rendered_img[0].detach().cpu().numpy() * 255
                    scale = image_out.shape[0] / color_image.shape[0]
                    newsize = (int(scale * color_image.shape[1]), int(image_out.shape[0]))
                    color_image = cv2.resize(color_image, dsize=newsize)
                    image_out = np.hstack((image_out, color_image))
                imgs_out.append(image_out)

            # image_out = np.vstack((imgs_out[0], imgs_out[1]))
            image_out = imgs_out[0]
            joint_image = image_out[...,[2,1,0,3]][...,:3]
            self.renderings[stage][iter][str(prefix)] = joint_image

    def save_projected_objs(self, item, stage, i, prefix='proj_masks'):
        """ Save the projected object masks to the image plane."""
        proj_mask_pred = cv2.cvtColor((item['proj_obj_mask'].detach().cpu().numpy()).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        proj_mask_gt = cv2.cvtColor((item['gt_obj_mask'].detach().cpu().numpy()).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        proj_mask_pred[:,:,:2] = 0 # red
        proj_mask_gt[:,:,1:] = 0 # blue
        
        # do alpha blending
        out = cv2.addWeighted(proj_mask_pred, 0.5, proj_mask_gt, 0.5, 0)
        # put Text
        out = cv2.putText(out, f'Stage={stage}, i={i}', (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
        self.renderings[stage][i][str(prefix)] = out

    def check_grad(self,):
        """ check if grad is not None for useful params"""
        
        ## Object params
        for name, param in self.obj.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    print(f'Object param {name} has grad = ', param.grad.sum())
                else:
                    print(f'Object param {name} has no grad')

        ## Human params
        # for name, param in self.body_model.named_parameters():
        #     if param.requires_grad:
        #         if param.grad is not None:
        #             print(f'Human param {name} has grad = ', param.grad.sum())
        #         else:
        #             print(f'Human param {name} has no grad')

        # color imag
        if self.color_image.requires_grad:
            if self.color_image.grad is not None:
                print('Color image has grad= ', self.color_image.grad.sum())
            else:
                print('Color image has no grad')

        # mask 
        # if self.mask_.requires_grad:
        #     if self.mask_.grad is not None:
        #         print('Mask_ has grad= ', self.mask_.grad.sum())
        #     else:
        #         print('Mask_ has no grad')

        # mask 
        if self.mask.requires_grad:
            if self.mask.grad is not None:
                print('Mask  normal has grad= ', self.mask.grad.sum())
            else:
                print('Mask has no grad')

        # # self.criterion.proj_mask
        # if self.criterion.proj_mask.requires_grad:
        #     if self.criterion.proj_mask.grad is not None:
        #         print('Proj mask has grad= ', self.criterion.proj_mask.grad)
        #     else:
        #         print('Proj mask has no grad')

        # # check self.criterion.IOU_loss
        # if self.criterion.IOU_loss.requires_grad:
        #     if self.criterion.IOU_loss.grad is not None:
        #         print('IOU loss has grad= ', self.criterion.IOU_loss.grad)
        #     else:
        #         print('IOU loss has no grad')

        # check self.criterion.est_joints
        # if self.criterion.est_joints.requires_grad:
        #     if self.criterion.est_joints.grad is not None:
        #         print('Est joints has grad= ', self.criterion.est_joints.grad)
        #     else:
        #         print('Est joints has no grad')

    def optimize_params(
        self,
        #init_h1, 
        #init_h2, 
        init_human,
        init_camera,
        init_obj,
        item,
        stage,
        guidance_params,
        gender, 
        obj_embeddings
    ):  
        """Optimize the human parameters for the given stage."""

        # set the loss weights for the current stage
        self.criterion.set_weights(stage)

        num_iters = self.num_iters[stage]
        self.save_at_steps = [int(x) for x in np.linspace(0, num_iters, self.total_steps_save)]

        for i in range(self.num_iters[stage]):

            # initialize render output- False by default
            if self.render_iters:
                colors = {0: ['paper_blue', 'paper_red'], 1: ['paper_blue', 'paper_red']}
                if i in self.save_at_steps:
                    self.renderings[stage][i] = {}

            smpl_output = self.body_model()
            obj_output= self.obj
            camera = self.camera

            # we tried different approaches / noies levels when using the SDS loss
            if self.opti_cfg.use_diffusion:
                if self.opti_cfg.sds_type == "fixed":
                    # use fixed value for noise level t
                    t_i = self.opti_cfg.sds_t_fixed
                elif self.opti_cfg.sds_type == "range":
                    # sample random integer between range lower and upper bound
                    t_min, t_max = self.opti_cfg.sds_t_range
                    t_i = np.random.randint(t_min, t_max, 1)[0]
                elif self.opti_cfg.sds_type == "adaptive":
                    # change noise level based on iteration
                    p = (self.num_iters[stage] - (i+1)) / self.num_iters[stage]
                    pidx = int(np.where(np.array(self.opti_cfg.sds_t_adaptive_i) > p)[0][-1])
                    t_i = self.opti_cfg.sds_t_adaptive_t[pidx]
            else:
                # without SDS loss, set t to None
                t_i = None
            
            # if item['use_mask_loss']>0:
            #     print("Using mask loss")
            # else:
            #     print("Not using mask loss")

            if item['use_mask_loss']>0:
                mask = self.render_obj_mask(init_obj, item, stage, i) 
                item['proj_obj_mask'] = mask

            # compute all loss
            loss, loss_dict, vis_out = self.criterion(
                item,
                smpl_output, 
                obj_output, 
                camera,
                #init_h1, 
                #init_h2,
                init_human,
                init_camera,
                init_obj,
                gender, 
                obj_embeddings,
                use_diffusion_prior=self.opti_cfg.use_diffusion,
                diffusion_module=self.diffusion_module,
                t=t_i,
                guidance_params=guidance_params,
                use_mask_loss = item['use_mask_loss']
            )

            if self.print_loss:
                self.print_losses(loss_dict, stage, i)

            # render iters is false by default
            if self.render_iters and i in self.save_at_steps:
                # save vis
                if not item['use_mask_loss']>0:
                    mask = self.render_obj_mask(init_obj, item, stage, i) 
                    item['proj_obj_mask'] = mask

                self.save_projected_objs(item, stage, i)                
                self.render_current_estimate(init_obj, None, item, stage, i, colors[stage])
                # # render one step predictions
                self.render_current_estimate(init_obj,vis_out, item,stage, i, colors[stage], 
                                             prefix='one_step_pred')

            # optimizer step
            self.optimizer.zero_grad()
            if loss>0:
                loss.backward()
            # Check grad of relevant params
            # self.check_grad()
            self.optimizer.step()

            # break if stopping criterion is met
            stop_crit = self.stopper.check(loss.item())
            if stop_crit:
                break

    def fit(
        self, 
        init_human,
        init_camera,
        init_obj,
        item,
    ): 
        """Main fitting function running through all stages of optimization"""

        # we project the initial mesh to the image plane and use the keypoints 
        # if they're not visible in the image
        with torch.no_grad():
            self.fill_params(init_human, init_camera, init_obj)
            # project current bev points to camera!
            init_human['init_keypoints'] = torch.cat([
                self.camera.project(self.body_model().joints)], axis=0)

         # Get obj embeddings
        obj_name = init_obj['obj_name']
        obj_embeddings = init_obj['obj_embeddings']  # B, 256

        obj_vertices = [init_obj['obj_vertices']]

        # Get gender of the human
        gender = init_human['gender']  # B, 2

        # copy init human params for guidance
        #guidance_params = {k: v.clone().detach() for k, v in init_human.items()}
        guidance_params = {}
        if self.diffusion_module is not None:
            if len(self.diffusion_module.exp_cfg.guidance_params) > 0:
                dbs = self.diffusion_module.bs
                guidance_params = {
                    'orient': init_human['global_orient'].unsqueeze(0).repeat(dbs, 1, 1),
                    'pose': init_human['body_pose'].unsqueeze(0).repeat(dbs, 1, 1),
                    'shape': init_human['betas'].unsqueeze(0).repeat(dbs, 1, 1),
                    'transl': init_human['transl'].unsqueeze(0).repeat(dbs, 1, 1),
                    "orient_obj": init_obj["orient_obj"].unsqueeze(0).repeat(dbs, 1, 1),
                    "transl_obj": init_obj["transl_obj"].unsqueeze(0).repeat(dbs, 1, 1),
                }
                guidance_params = self.diffusion_module.cast_smpl(guidance_params, gender)
                guidance_params = self.diffusion_module.split_humans(guidance_params)


        def undo_orient_and_transl(diffusion_module, x_start_smpls, x_start_obj, target_rotation, target_transl):
            """ 
                ARGS:
                    diffusion_module- Used for inferencing on diffusion model
                    x_start_smpls - denoised SMPLs after DDIM inferencing
                    X_start_obj - denoised object rot and trans after ddim inferencing
                    target_rotation - bev estimate of rotation
                    target_transl- bev estimate of transl            
            """
            
            #orient, cam_rotation
            global_orient_h0 = x_start_smpls.global_orient.unsqueeze(1) # 1, 1, 3
            global_orient_obj = x_start_obj[0].unsqueeze(1) # 1, 1, 3
            param = torch.cat((global_orient_h0, global_orient_obj), dim=1) # 1, 2, 3
            param_rotmat = axis_angle_to_matrix(param) # 1, 2, 3, 3
            cam_rotation = torch.einsum('bml,bln->bmn', target_rotation, param_rotmat[:, 0, :, :].transpose(2, 1))
            new_orient = matrix_to_axis_angle(torch.einsum('bnm,bhml->bhnl', cam_rotation, param_rotmat))
            new_orient=new_orient[[0],:,:] # 1, 2, 3, 3

            if gender[0, 0] == 1:
                pelvis_orig = diffusion_module.male_body_model(betas=x_start_smpls.betas).joints[:,[0],:]
            else:
                pelvis_orig = diffusion_module.female_body_model(betas=x_start_smpls.betas).joints[:,[0],:]
            pelvis = torch.cat((
                pelvis_orig,
                pelvis_orig
            ), dim=1)

            transl_h0 = x_start_smpls.transl.unsqueeze(1) # 1, 1, 3
            transl_obj = x_start_obj[1].unsqueeze(1) # 1, 1, 3
            transl = torch.cat((transl_h0, transl_obj), dim=1) # 1, 2, 3
            root_transl = transl[:,[0],:] # 1, 1, 3
            cam_translation = (-1 * torch.einsum('bhn,bnm->bhm', target_transl + pelvis, cam_rotation)) + root_transl + pelvis
            xx = transl + pelvis - cam_translation
            new_transl = torch.einsum('bhn,bnm->bhm', xx, cam_rotation.transpose(2, 1)) - pelvis
            new_transl=new_transl[[0],:,:] # 1, 2, 3

            return new_orient[:, 0], new_transl[:, 0], new_orient[:, 1], new_transl[:, 1]

        ############ conditional sampling ##############
        if len(guidance_params) > 0:
            # guru.info('Start sampling unconditional')
            cond_ts = np.arange(1, self.diffusion_module.diffusion.num_timesteps, 100)[::-1]
            log_freq = cond_ts.shape[0] # no logging

            # x starts is basically final inferenced output from ddim!
            # obj_starts is basically denoised object vertices
            x_ts, x_starts, obj_ts, obj_starts = self.diffusion_module.sample_from_model(
                cond_ts, log_freq, guidance_params, gender, obj_embeddings, obj_vertices,
                store_obj_trans=True
            )
            # undo orient and transl
            init_rotation =  axis_angle_to_matrix(init_human['global_orient'][0]).detach().clone().repeat(dbs, 1, 1) # 1, 3, 3
            init_transl =  init_human['transl'][0].detach().clone().repeat(dbs, 1, 1)  # 1, 1, 3
            new_orient, new_transl, new_orient_obj, new_transl_obj = undo_orient_and_transl(
                self.diffusion_module, x_starts['final'], obj_starts['final'], init_rotation, init_transl
            )

            init_obj['orient_obj'] = new_orient_obj
            init_obj['transl_obj'] = new_transl_obj

            for param in ['global_orient', 'body_pose', 'betas', 'transl']:
                if param == 'global_orient':
                    init_human[param] = new_orient
                elif param == 'transl':
                    init_human[param] = new_transl
                else:
                    i_param = x_starts['final']
                    init_human[param] = eval(f'i_param.{param}').detach().clone()

            # we project the initial mesh to the image plane and use the keypoints 
            # if they're not visible in the image
            with torch.no_grad():
                self.fill_params(init_human, init_camera, init_obj)
                init_human['init_keypoints'] = torch.cat([
                    self.camera.project(self.body_model().joints)], axis=0) 


        # Optimization routine in multiple stages! 2 stages by default!
        for stage, _ in enumerate(range(len(self.num_iters))):
            if stage > 0:
                break
            guru.info(f'Starting with stage: {stage} \n')

            ## Initialize smpl outputs, obj_outputs
            if self.render_iters:
                self.renderings[stage] = dict()

            self.stopper.reset() # stopping criterion
            self.setup_optimizer(init_human, init_camera,init_obj, stage) # setup optimizer

            # clone the initial estimate and detach it from the graph since it'll be used
            # as initialization and as prior the optimization
            if stage > 0:
                init_human['body_pose'] = self.body_model.body_pose.detach().clone(),
                init_human['betas'] = self.body_model.betas.detach().clone(),
             
            # run optmization for one stage
            self.optimize_params(init_human, init_camera, init_obj,item, stage, guidance_params, gender, obj_embeddings)

        # Get final loss value and get full skinning
        with torch.no_grad():
            smpl_output_h1 = self.body_model()
            obj_output = self.obj

        return smpl_output_h1, obj_output
    
