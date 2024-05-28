import torch.nn as nn
import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from llib_rho.utils.threed.conversion import axis_angle_to_rotation6d,  rotation6d_to_axis_angle
from llib_rho.losses.l2 import L2Loss
from llib_rho.losses.build import build_loss
from llib_rho.utils.keypoints.gmfo import GMoF

class HHCOptiLoss(nn.Module):
    def __init__(
        self,
        losses_cfgs,
        body_model_type='smplh',
    ):
        super(HHCOptiLoss, self).__init__()

        self.cfg = losses_cfgs

        # when loss weigts are != 0, add loss as member variable
        for name, cfg in losses_cfgs.items():
            if name == 'debug':
                continue

            # add loss weight as member variable
            weight = cfg.weight
            setattr(self, name + '_weights', weight)
            
            # add criterion as member variable when weight != 0 exists
            if sum([x != 0 for x in cfg.weight]) > 0:

                function = build_loss(cfg, body_model_type)
                setattr(self, name + '_crit', function)

                # check if the criterion / loss is used in forward pass
                method = 'get_' + name + '_loss'
                assert callable(getattr(self, method)), \
                    f'Method {method} not implemented in HHCOptiLoss'

        self.set_weights(stage=0) # init weights with first stage

        self.robustifier = GMoF(rho=100.0)

        self.debug = []

    def set_weights(self, stage, default_stage=-1):

        for name, cfg in self.cfg.items():
            if name == 'debug':
                continue

            weight = getattr(self, name + '_weights')

            # use default stage value if weight for stage not specified
            weight_stage = default_stage if len(weight) <= stage else stage

            setattr(self, name + '_weight', weight[weight_stage])

    def get_keypoint2d_loss(self, vitpose, openpose, init_bev, est_joints, bs, num_joints, device):
        """Some keypoint processing to merge OpenPose and ViTPose keypoints."""
        
        gt_keypoints = vitpose #init['vitpose_keypoints'].unsqueeze(0).to(device)
        op_keypoints = openpose #init['op_keypoints'].unsqueeze(0).to(device)
        bs, nk, _ = gt_keypoints.shape

        # add openpose foot tip (missing in vitpose)
        ankle_joint = [11, 14]
        """
        ankle_thres = 5.0
        right_ankle_residual = torch.sum((gt_keypoints[:,11,:] - op_keypoints[:,11,:])**2)
        if right_ankle_residual < ankle_thres:
            gt_keypoints[:,22,:] = op_keypoints[:,22,:]
        left_ankle_residual = torch.sum((gt_keypoints[:,14,:] - op_keypoints[:,14,:])**2)
        if left_ankle_residual < ankle_thres:
            gt_keypoints[:,19,:] = op_keypoints[:,19,:]
        """

        # use initial (BEV) keypoints if detected ankle joints are missing/low confidence (e.g. when image is cropped)
        mask_init = (gt_keypoints < .2)[0,:,2]
        init_bev = init_bev # init['init_keypoints'] 
        init_keypoints = torch.cat([init_bev.double(), 0.5 * torch.ones(bs, nk, 1).to(device)], dim=-1)
        if mask_init[ankle_joint[0]] == 1:
            gt_keypoints[:,ankle_joint[0],:] = init_keypoints[:,ankle_joint[0],:]
            gt_keypoints[:,22:25,:] = init_keypoints[:,22:25,:]
        if mask_init[ankle_joint[1]] == 1:
            gt_keypoints[:,ankle_joint[1],:] = init_keypoints[:,ankle_joint[1],:]
            gt_keypoints[:,19:22,:] = init_keypoints[:,19:22,:]

        gt_keypoints[:, [0, 1, 15, 16, 17, 18], 2] = 0

        if gt_keypoints.shape[-1] == 3:
            gt_keypoints_conf = gt_keypoints[:, :, 2]
            gt_keypoints_vals = gt_keypoints[:, :, :2]
        else:
            gt_keypoints_vals = gt_keypoints
            gt_keypoints_conf = torch.ones([bs, num_joints], device=device)
        
        # normalize keypoint loss by bbox size 
        valid_kpts = gt_keypoints_vals[gt_keypoints_conf > 0]
        xmin, ymin = valid_kpts.min(0)[0]
        xmax, ymax = valid_kpts.max(0)[0]
        bbox_size = max(ymax-ymin, xmax-xmin)
        gt_keypoints_vals = gt_keypoints_vals / bbox_size * 512
        est_joints = est_joints / bbox_size * 512

        # robistify keypoints
        #residual = (gt_keypoints_vals - projected_joints) ** 2
        #rho = 100 ** 2
        #robust_residual = gt_keypoints_conf.unsqueeze(-1) * rho * \
        #                torch.div(residual, residual + rho)
        #keypoint2d_loss = torch.mean(robust_residual) * self.keypoint2d_weight

        # comput keypoint loss
        keypoint2d_loss = self.keypoint2d_crit(
            gt_keypoints_vals, est_joints, gt_keypoints_conf
        ) * self.keypoint2d_weight

        return keypoint2d_loss

    def get_shape_prior_loss(self, betas):
        shape_prior_loss = self.shape_prior_crit(
            betas, y=None) * self.shape_prior_weight
        return shape_prior_loss

    def get_pose_prior_loss(self, pose):
        pose_prior_loss = torch.sum(self.pose_prior_crit(
            pose)) * self.pose_prior_weight
        return pose_prior_loss

    def get_init_pose_loss(self, init_pose, est_body_pose, device):
        
        if len(init_pose.shape) == 1:
            init_pose = init_pose.unsqueeze(0)
        
        init_pose_prior_loss = self.init_pose_crit(
            init_pose, est_body_pose
        ) * self.init_pose_weight

        return init_pose_prior_loss

    def get_init_shape_loss(self, init_shape, est_shape, device):
        init_shape_loss = self.init_pose_crit(
            init_shape, est_shape
            ) * self.init_shape_weight
        return init_shape_loss

    def get_init_transl_loss(self, init_transl, est_transl, device):
        init_transl_loss = self.init_pose_crit(
            init_transl, est_transl
            ) * self.init_transl_weight
        return init_transl_loss

    def get_diffusion_prior_orient_loss(self, global_orient_diffused, global_orient_current):
        global_orient_loss = self.diffusion_prior_global_crit(
            global_orient_diffused, global_orient_current) * \
                self.diffusion_prior_global_weight
        return global_orient_loss

    def get_diffusion_prior_pose_loss(self, body_pose_diffused, body_pose_current):
        body_pose_loss = self.diffusion_prior_body_crit(
            body_pose_diffused, body_pose_current) * \
                self.diffusion_prior_body_weight
        return body_pose_loss
    
    def get_diffusion_prior_pose_obj_loss(self, body_pose_diffused, body_pose_current):
        body_pose_loss = self.diffusion_prior_body_crit(
            body_pose_diffused, body_pose_current) * \
                self.diffusion_prior_body_weight
        return body_pose_loss
    
    def get_diffusion_prior_shape_loss(self, betas_diffused, betas_current):
        betas_loss = self.diffusion_prior_shape_crit(
            betas_diffused, betas_current) * \
                self.diffusion_prior_shape_weight
        return betas_loss
    
    def get_diffusion_prior_scale_loss(self, betas_diffused, betas_current):
        betas_loss = self.diffusion_prior_shape_crit(
            betas_diffused, betas_current) * \
                self.diffusion_prior_shape_weight
        return betas_loss

    def get_diffusion_prior_transl_loss(self, transl_diffused, transl_current):
        transl_loss = self.diffusion_prior_transl_crit(
            transl_diffused, transl_current) * \
                self.diffusion_prior_transl_weight
        return transl_loss
    
    def get_interpenetration_loss(self,human_vertices:torch.Tensor, obj_vertices:torch.Tensor):
        """ Computes the interpenetration loss between the predicted human and object.
            The object and human mesh vertices should be defined for the same frame of reference.
            
            ARGS:
                human_vertices(torch.Tensor) - shape(bs,num_smpl_vertices,3)
                obj_vertices(torch.Tensor) - shape(bs,1024,3)
            RETURNS:
                penetration_loss(torch.float64)
            """
        
    def undo_orient_and_transl(self, diffusion_module, x_start_smpls, x_start_obj, target_rotation, target_transl, gender):
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

    def forward_diffusion(
        self,
        diffusion_module, # the diffusion module
        t, # noise level
        smpl_output, # the current estimate of person
        obj_output, # the current estimate of object
        guidance_params, # the initial estimate of person a and b
        gender, 
        obj_embeddings,
    ):
        """The SDS loss or L_diffusion as we define it in the paper"""

        ld = {} # store losses in dict for printing
        device = diffusion_module.cfg.device

        # take the current estimate of the optimization
        x_start_smpls = [smpl_output]

        # fix because optimization data loader does not do the flipping anymore
        #if smpl_output_h1.transl[0,0] > smpl_output_h2.transl[0,0]:
        #    x_start_smpls = [smpl_output_h2, smpl_output_h1]

        dbs = diffusion_module.bs

        # Run a diffuse-denoise step. To do this, we use torch.no_grad() to
        # ensure that the gradients are not propagated through the diffusion
        with torch.no_grad():
            # first, we need to transform the current estimate of the optimization to 
            # BUDDI's format
            init_rotation =  axis_angle_to_matrix(x_start_smpls[0].global_orient).detach().clone().repeat(dbs, 1, 1) # 1, 3, 3
            init_transl =  x_start_smpls[0].transl.detach().clone().repeat(dbs, 1, 1)  # 1, 1, 3
            x = {
                'orient': axis_angle_to_rotation6d(x_start_smpls[0].global_orient.unsqueeze(1)).repeat(dbs, 1, 1), # 1, 1, 6
                'pose': axis_angle_to_rotation6d(x_start_smpls[0].body_pose.unsqueeze(1).view(1, 1, -1, 3)).view(1, 1, -1).repeat(dbs, 1, 1),   # 1, 21, 6
                'shape': torch.cat((x_start_smpls[0].betas,), dim=-1).unsqueeze(1).repeat(dbs, 1, 1),  # 1, 1, 10
                'transl': x_start_smpls[0].transl.unsqueeze(1).repeat(dbs, 1, 1),  # 1, 1, 3
                'orient_obj': axis_angle_to_rotation6d(obj_output.orient_obj.unsqueeze(1)).repeat(dbs, 1, 1),  # 1, 1, 6
                'transl_obj': obj_output.transl_obj.unsqueeze(1).repeat(dbs, 1, 1)  # 1, 1, 3
            }

            # if len(diffusion_module.exp_cfg.guidance_params) > 0:
            #     guidance_params = {
            #         'orient': init_human['global_orient'].unsqueeze(0).repeat(dbs, 1, 1),
            #         'pose': init_human['body_pose'].unsqueeze(0).repeat(dbs, 1, 1),
            #         'shape': torch.cat((init_human['betas'], init_human['scale'].unsqueeze(1)), dim=-1).unsqueeze(0).repeat(dbs, 1, 1),
            #         'transl': init_human['transl'].unsqueeze(0).repeat(dbs, 1, 1)
            #     }
            #     guidance_params = diffusion_module.cast_smpl(guidance_params)
            #     guidance_params = diffusion_module.split_humans(guidance_params)
            # else:
            #     guidance_params = {} # no guidance params are used here
            x = diffusion_module.reset_orient_and_transl(x, gender) # use relative translation

            # run the diffusion (diffuse parameters and use BUDDI to denoise them)
            t = torch.tensor([t] * diffusion_module.bs).to(diffusion_module.cfg.device)
            diffusion_output = diffusion_module.diffuse_denoise(x=x, y=guidance_params, t=t, obj=obj_embeddings, gender=gender)
            denoised_smpls = diffusion_output['denoised_smpls']
            rot_denoise = rotation6d_to_axis_angle(diffusion_output['model_prediction']['orient_obj']) # 1, 3
            trans_denoise = diffusion_output['model_prediction']['transl_obj'] # 1, 3

            # now we need to bring the estimates back into the format of the original optimization
            new_orient, new_transl, new_orient_obj, new_transl_obj = self.undo_orient_and_transl(
                diffusion_module, denoised_smpls, [rot_denoise, trans_denoise], init_rotation, init_transl, gender)
            x_end_smpls = diffusion_module.get_smpl({
                'orient': axis_angle_to_rotation6d(new_orient).repeat(dbs, 1, 1),
                'pose': torch.cat([
                    axis_angle_to_rotation6d(denoised_smpls.body_pose.view(dbs,-1,3)).unsqueeze(1)], dim=1), 
                'shape': torch.cat([
                    torch.cat((denoised_smpls.betas,), dim=-1).unsqueeze(1)], dim=1), 
                'transl': new_transl.repeat(dbs, 1, 1)
            }, gender)

        # Regularize the orientation and pose of the two people
        if self.diffusion_prior_orient_weight > 0:
            ld['regularize_h0_orient'] = self.diffusion_prior_orient_weight * \
                torch.norm(x_start_smpls[0].global_orient[[0]] - x_end_smpls.global_orient[[0]].detach())

        # Regularize the pose of the two people
        if self.diffusion_prior_pose_weight > 0:
            ld['regularize_h0_pose'] = self.diffusion_prior_pose_weight * \
                    torch.norm(x_start_smpls[0].body_pose[[0]] - x_end_smpls.body_pose[[0]].detach())
        if self.diffusion_prior_pose_obj_weight > 0:
            ld['regularize_obj_pose'] = self.diffusion_prior_pose_obj_weight * \
                    torch.norm(obj_output.orient_obj - new_orient_obj.detach())
        
        # Regularize the relative translation of the two people
        if self.diffusion_prior_transl_weight > 0:
            # t1 - t0 = t2 - t1
            diffusion_dist = new_transl_obj.detach() - x_end_smpls.transl[[0]].detach()
            curr_dist = obj_output.transl_obj - x_start_smpls[0].transl[[0]]
            ld['regularize_h0_obj_transl'] = self.diffusion_prior_transl_weight * \
                                                                torch.norm(diffusion_dist - curr_dist)

        # Regularize the shape of the two people
        if self.diffusion_prior_shape_weight > 0:
            ld['regularize_h0_shape'] = self.diffusion_prior_shape_weight * \
                torch.norm(x_start_smpls[0].betas[[0]] - x_end_smpls.betas[[0]].detach())
    
        # Sum the loss terms
        diffusion_loss = torch.tensor([0.0]).to(device)
        for k, v in ld.items():
            diffusion_loss += v 

        # average the losses over batch
        ld_out = {}
        for k, v in ld.items():
            if type(v) == torch.Tensor:
                ld_out[k] = v.mean()

        # final loss value
        diffusion_loss = sum(ld_out.values())
        ld_out['total_sds_loss'] = diffusion_loss

        return diffusion_loss, ld_out


    def forward_fitting(
        self, 
        smpl_output, # the current estimate of person
        obj_output, # the current estimate of object
        camera, # camera
        #init_h1, # the initial estimate of person a (from BEV) 
        #init_h2, # the initial estimate of person b (from BEV)
        init_human, # the initial estimate of person a and b 
        init_camera, # BEV camera
        init_obj
    ):
        bs, num_joints, _ = smpl_output.joints.shape  # B=1, N, 3
        device = smpl_output.joints.device

        ld = {} # store losses in dict for printing

        #init_h1_betas = init_h1['betas'].unsqueeze(0).to(device)
        #init_h2_betas = init_h2['betas'].unsqueeze(0).to(device)

        # project 3D joinst to 2D
        projected_joints_h = camera.project(smpl_output.joints)
        #TODO: ag6 - Project dense object and obtain object masks maybe?

        # keypoint losses for human
        ld['keypoint2d_losses'] = 0.0
        if self.keypoint2d_weight > 0:
            ld['keypoint2d_losses'] += self.get_keypoint2d_loss(
                init_human['keypoints'][[0]],  # initial vit keypoints
                init_human['op_keypoints'][[0]], # initial openpose keypoints
                init_human['init_keypoints'][[0]],  # initial diffusion cond DDIM keypoints  
                projected_joints_h, bs, num_joints, device)

        # pose prior losses for each human w.r.t init
        ld['init_pose_losses'] = 0.0
        if self.init_pose_weight > 0:
            ld['init_pose_losses'] += self.get_init_pose_loss(
                init_human['body_pose'][[0]], smpl_output.body_pose, device)

        # shape prior losses for each human w.r.t init
        ld['init_shape_losses'] = 0.0
        if self.init_shape_weight > 0:
            ld['init_shape_losses'] += self.get_init_shape_loss(
                init_human['betas'][[0]], smpl_output.betas, device)
         
        # trans prior losses for human w.r.t init
        ld['init_transl_losses'] = 0.0
        if self.init_transl_weight > 0:
            ld['init_transl_losses'] += self.get_init_transl_loss(
                init_human['transl'][[0]], smpl_output.transl, device)
        
        ##TODO:ag6 - Think about how to add interpenetration loss between obj and human here!
        # contact loss between two humans
        # ld['hhc_contact_loss'] = 0.0
        # if self.hhc_contact_weight:
        #     ld['hhc_contact_loss'] += self.get_hhc_contact_loss(
        #         contact_map, smpl_output_h1.vertices, smpl_output_h2.vertices)

        # contact loss between two humans
        # ld['hhc_contact_general_loss'] = 0.0
        # if self.hhc_contact_general_weight:
        #     ld['hhc_contact_general_loss'] += self.get_hhc_contact_general_loss(
        #         smpl_output_h1.vertices, smpl_output_h2.vertices)

        ## TODO:ag6- Add object specific losses below!
        ############
        ############
        ############
        ## Filter it and only do it for large enough masks!
        
        # average the losses over batch
        ld_out = {}
        for k, v in ld.items():
            if type(v) == torch.Tensor:
                ld_out[k] = v.mean()

        # final loss value
        fitting_loss = sum(ld_out.values())
        ld_out['total_fitting_loss'] = fitting_loss

        return fitting_loss, ld_out


    def forward(
        self, 
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
        use_diffusion_prior=False,
        diffusion_module=None,
        t=None,
        guidance_params={},
    ): 
        """
        Compute all losses in the current optimization iteration.
        The current estimate is smpl_output_h1/smpl_output_h2, which
        we pass to the L_fitting and L_diffusion modules. The final
        loss is the sum of both losses.
        """ 
        
        # fitting losses (keypoints, pose / shape prior etc.)
        fitting_loss, fitting_ld_out = self.forward_fitting(
            smpl_output, 
            obj_output,
            camera,
            #init_h1, 
            #init_h2,
            init_human,
            init_camera,
            init_obj
        )

        # diffusion prior loss / sds loss / BUDDI loss
        if use_diffusion_prior:
            sds_loss, sds_ld_out = self.forward_diffusion(
                diffusion_module,
                t,
                smpl_output, 
                obj_output,
                guidance_params, # for bev conditioning
                gender, 
                obj_embeddings,
            )

        # update loss dict and sum up losses
        if use_diffusion_prior:
            total_loss = fitting_loss + sds_loss
            ld_out = {**fitting_ld_out, **sds_ld_out}
        else:
            total_loss = fitting_loss
            ld_out = fitting_ld_out
        
        ld_out['total_loss'] = total_loss

        return total_loss, ld_out