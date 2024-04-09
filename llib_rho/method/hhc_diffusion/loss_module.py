import torch.nn as nn
import numpy as np
import torch
from llib.losses.l2 import L2Loss
from llib.losses.build import build_loss
from llib.utils.threed.conversion import batch_rodrigues

class LossModule(nn.Module):
    def __init__(
        self,
        losses_cfgs,
        body_model_type='smplx'
    ):
        super(LossModule, self).__init__()

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

    def set_weights(self, stage, default_stage=-1):

        for name, cfg in self.cfg.items():
            if name == 'debug':
                continue
            
            weight = getattr(self, name + '_weights')

            # use default stage value if weight for stage not specified
            if len(weight) <= stage: 
                stage = default_stage

            setattr(self, name + '_weight', weight[stage])

    def get_keypoint2d_loss(self, gt_keypoints, projected_joints, bs, num_joints, device):
        raise NotImplementedError

    def get_shape_prior_loss(self, betas):
        raise NotImplementedError

    def get_pose_prior_loss(self, pose, betas):
        raise NotImplementedError

    def get_init_pose_loss(self, init_pose, est_body_pose, device):
        raise NotImplementedError
    
    def get_pseudogt_obj_pose_loss(self, init_pose, est_pose, device):
        bs = init_pose.shape[0]
        init_pose_rotmat = batch_rodrigues(init_pose.view(bs, -1, 3).reshape(-1, 3)).view(bs, -1, 3, 3)
        est_body_pose_rotmat = batch_rodrigues(est_pose.reshape(bs, -1, 3).reshape(-1,3)).reshape(bs, -1, 3, 3)
        mseloss = nn.MSELoss().to('cuda')
        init_pose_prior_loss = (
            (init_pose_rotmat - est_body_pose_rotmat)**2
        ).sum((1,2,3)).mean() * self.pseudogt_obj_pose_weight
        return init_pose_prior_loss
    
    def get_pseudogt_obj_transl_loss(self, init_transl, est_transl, device):
        init_transl = init_transl
        est_transl = est_transl
        pgt_transl_loss = self.pseudogt_obj_transl_crit(
            init_transl, est_transl) * self.pseudogt_obj_transl_weight
        return pgt_transl_loss

    def get_pseudogt_pose_loss(self, init_pose, est_body_pose, device):
        """Pose prior loss (pushes to pseudo-ground truth pose)"""
        bs = init_pose.shape[0]
        init_pose_rotmat = batch_rodrigues(init_pose.view(bs, -1, 3).reshape(-1, 3)).view(bs, -1, 3, 3)
        est_body_pose_rotmat = batch_rodrigues(est_body_pose.reshape(bs, -1, 3).reshape(-1,3)).reshape(bs, -1, 3, 3)
        mseloss = nn.MSELoss().to('cuda')
        init_pose_prior_loss = (
            (init_pose_rotmat - est_body_pose_rotmat)**2
        ).sum((1,2,3)).mean() * self.pseudogt_pose_weight
        return init_pose_prior_loss

    def get_pseudogt_shape_loss(self, init_shape, est_shape, device):
        """Shape parameter loss (pushes to pseudo-ground truth shape)"""
        pgt_shape_loss = self.pseudogt_shape_crit(
            init_shape, est_shape) * self.pseudogt_shape_weight
        return pgt_shape_loss

    def get_pseudogt_v2v_loss(self, init_verts, est_verts, device):
        """3D v2v loss (pushes to pseudo-ground truth shape)"""
        pgt_v2v_loss = ((init_verts - est_verts)**2).sum(-1).mean(-1).mean() \
             * self.pseudogt_v2v_weight
        return pgt_v2v_loss

    def get_pseudogt_j2j_loss(self, init_joints, est_joints, device):
        """3D joint-to-joint loss (pushes to pseudo-ground truth shape)"""
        pgt_j2j_loss = ((init_joints - est_joints)**2).sum(-1).mean(-1).mean() \
             * self.pseudogt_v2v_weight
        return pgt_j2j_loss

    def get_pseudogt_transl_loss(self, init_transl, est_transl, device):
        """Translation loss (pushes to pseudo-ground truth translation)"""
        init_transl = init_transl
        est_transl = est_transl
        pgt_transl_loss = self.pseudogt_transl_crit(
            init_transl, est_transl) * self.pseudogt_transl_weight
        return pgt_transl_loss

    def zero_loss_dict(self):
        ld = {} 
        # ld['shape_prior_loss_0'] = 0.0
        # ld['shape_prior_loss_1'] = 0.0
        # ld['pose_prior_loss_0'] = 0.0
        # ld['pose_prior_loss_1'] = 0.0
        ld['pseudogt_pose_losses_0'] = 0.0
        ld['pseudogt_global_orient_losses_0'] = 0.0
        ld['pseudogt_shape_losses_0'] = 0.0
        ld['pseudogt_scale_losses_0'] = 0.0
        ld['pseudogt_transl_losses_0'] = 0.0
        ld['pseudogt_v2v_losses_0'] = 0.0
        ld['pseudogt_j2j_losses_0'] = 0.0
        ld['pseudogt_obj_pose_losses'] = 0.0
        ld['pseudogt_obj_transl_losses'] = 0.0
        
        return ld

    def forward(
        self, 
        est_smpl, # estimated smpl
        tar_smpl, # target smpl
        est_params, # estimated parameters
        tar_params, # target parameters
    ):  

        bs, num_joints, _ = est_smpl.joints.shape
        device = est_smpl.joints.device

        ld = self.zero_loss_dict() # store losses in dict


        # human zero loss
        hidx=0
        h = f'_{hidx}'
        tar_smpl, est_smpl = [tar_smpl], [est_smpl]
        # # shape prior loss
        # if self.shape_prior_weight > 0:
        #     ld['shape_prior_loss'+h] += self.get_shape_prior_loss(
        #         est_smpl[hidx].betas)

        # # pose prior loss
        # if self.pose_prior_weight > 0:
        #     ld['pose_prior_loss'+h] += self.get_pose_prior_loss(
        #         est_smpl[hidx].body_pose, est_smpl[hidx].betas)
        
        # pose prior losses for each human
        if self.pseudogt_pose_weight > 0:
            ld['pseudogt_pose_losses'+h] += self.get_pseudogt_pose_loss(
                tar_smpl[hidx].body_pose, est_smpl[hidx].body_pose, device)
            ld['pseudogt_global_orient_losses'+h] += self.get_pseudogt_pose_loss(
                tar_smpl[hidx].global_orient, est_smpl[hidx].global_orient, device)

        # pose prior losses for each human
        if self.pseudogt_shape_weight > 0:
            # concat scale and betas
            ld['pseudogt_shape_losses'+h] += self.get_pseudogt_shape_loss(
                tar_smpl[hidx].betas, est_smpl[hidx].betas, device)

        # pose prior losses for each human
        if self.pseudogt_transl_weight > 0:
            ld['pseudogt_transl_losses'+h] += self.get_pseudogt_transl_loss(
                tar_smpl[hidx].transl, est_smpl[hidx].transl, device)

        # vertex to vertex losses for each human
        if self.pseudogt_v2v_weight > 0:
            ld['pseudogt_v2v_losses'+h] += self.get_pseudogt_v2v_loss(
                tar_smpl[hidx].vertices, est_smpl[hidx].vertices, device)
        
        # joint to joint losses for each human
        if self.pseudogt_j2j_weight > 0:
            ld['pseudogt_j2j_losses'+h] += self.get_pseudogt_j2j_loss(
                tar_smpl[hidx].joints, est_smpl[hidx].joints, device)
            
        #Object rotation loss
        if self.pseudogt_obj_pose_weight > 0:
            ld['pseudogt_obj_pose_losses'] += self.get_pseudogt_obj_pose_loss(
                tar_params['orient_obj'], est_params['orient_obj'].unsqueeze(1), device)
            
        # Object Translation loss
        if self.pseudogt_obj_transl_weight > 0:
            ld['pseudogt_obj_transl_losses'] += self.get_pseudogt_obj_transl_loss(
                tar_params['transl_obj'], est_params['transl_obj'].unsqueeze(1), device)

        # average the losses over batch
        ld_out = {}
        for k, v in ld.items():
            if type(v) == torch.Tensor:
                ld_out[k] = v.mean()

        # final loss value
        total_loss = sum(ld_out.values())
        # set breakpoint if total_loss is nan
        if torch.isnan(total_loss):
            import ipdb; ipdb.set_trace()
        ld_out['total_loss'] = total_loss

        return total_loss, ld_out
