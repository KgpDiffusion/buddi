import torch
import torch.nn as nn
import numpy as np
from torch.functional import F
import math
import cv2
from llib_rho.models.diffusion.resample import UniformSampler
from collections import namedtuple
import loguru as guru
from llib_rho.utils.threed.conversion import (
    axis_angle_to_rotation6d,
    rotation6d_to_axis_angle,
)
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    euler_angles_to_matrix,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)

#Tokens = namedtuple(
#    "Tokens",
#    ["orient", "pose", "betas", "scale", "transl"]
#)


class TrainModule(nn.Module):
    def __init__(
        self,
        cfg,
        train_dataset,
        val_dataset,
        diffusion,
        model,
        criterion,
        evaluator,
        body_model,
        renderer,
    ):
        super().__init__()
        """
        Takes SMPL parameters as input and outputs SMPL parameters as output.
        """
        self.cfg = cfg
        self.exp_cfg = cfg.model.regressor.experiment
        self.tra_cfg = cfg.model.regressor.diffusion_transformer
        self.bs = cfg.batch_size
        self.nh = 1  # number of humans
        self.human_params = ["orient", "pose", "shape", "transl"] # shape = concat((betas, scale))

        # check if guidance params are full bev or not
        self.is_full_bev_guidance = False
        if 'bev' in self.exp_cfg.guidance_params:
            self.is_full_bev_guidance = True


        self.train_ds = train_dataset
        self.val_ds = val_dataset

        self.criterion = criterion

        self.evaluator = evaluator

        self.male_body_model = body_model[0]
        self.female_body_model = body_model[1]
        self.body_model_type = type(self.male_body_model).__name__.lower().split('_')[0]
        face_tensor = torch.from_numpy(self.male_body_model.faces.astype(np.int32))
        self.register_buffer("faces_tensor", face_tensor)

        self.diffusion = diffusion

        self.trainable_params = ["model"]
        self.add_module("model", model)

        self.schedule_sampler = UniformSampler(diffusion)


        self.meshes_to_render = {
            'single_step_random_t': ["input", "input_noise", "input_with_guidance","sampled_0"],
            'single_step_25_t': ["input", "input_noise", "input_with_guidance", "sampled_0"],
            'single_step_50_t': ["input", "input_noise", "input_with_guidance", "sampled_0"],
            'single_step_75_t': ["input", "input_noise", "input_with_guidance", "sampled_0"],
            'unconditional': ['sampled_0'],
            'conditional': ['sampled_0', "input_with_guidance"],
        }
        
        self.meshcols = {
            "input": ["light_blue1", "light_blue6"],
            "input_noise": ["light_red1", "light_red6"],
            "input_with_guidance": ["light_green1", "light_green6"],
            "sampled_0": ["light_yellow1", "light_yellow6"],
        }

        self.renderer = renderer

        # self.register_buffer('unit_rotation', torch.eye(3).repeat(self.bs, 1, 1))
        self.register_buffer(
            "unit_rotation",
            euler_angles_to_matrix(torch.tensor([math.pi, 0, 0]), "XYZ").repeat(
                self.bs, 1, 1
            ),
        )

        self.checks()

        self.orient_dim = 6 if self.exp_cfg.rotrep == "sixd" else 3
        self.pose_dim = 21 * 6 if self.exp_cfg.rotrep == "sixd" else 21 * 3
        self.shape_dim = 10
        self.transl_dim = 3
        self.orient_obj_dim = 6 if self.exp_cfg.rotrep == "sixd" else 3
        self.transl_obj_dim = 3 

        if train_dataset is not None:
            self.obj_vertices = {}
            self.obj_faces = {}
            for obj_name, val in train_dataset.mesh_vertices.items():
                self.obj_vertices[obj_name] = torch.from_numpy(val.astype(np.float32)).to(self.cfg.device)
                self.obj_faces[obj_name] = torch.from_numpy(train_dataset.mesh_faces[obj_name].astype(np.float32)).to(self.cfg.device)

    def checks(self):
        # 1) check if probs are set correclty for guidance random noise
        x = self.exp_cfg.guidance_all_nc + self.exp_cfg.guidance_no_nc
        assert x <= 1, "Guidance params all and no noise sum should be <= 1."

        assert not (
            not self.exp_cfg.relative_transl and self.exp_cfg.relative_orient
        ), "Relative translation should be True when relative orient is True."

        print("Using guidance params: ", self.exp_cfg.guidance_params)
        print("All checks passed.")

    def prep_contact_map(self, contact_map):
        # change layout of contact map to token number
        return contact_map.view(self.bs, -1)

    def prep_global_orient(
        self,
        param,
        rotrep="sixd",
        relative=True,
        to_unit_rotation=True,
        target_rotation=None,
    ):

        # concatenate global orientation and poses of two humans
        # param = torch.cat((global_orient_h0, global_orient_h1), dim=1)
        if to_unit_rotation:
            target_rotation = self.unit_rotation

        if relative:
            param_rotmat = axis_angle_to_matrix(param)
            # T = (RR)C, (RR)=TCˆt
            RR = torch.einsum(
                "bml,bln->bmn",
                target_rotation,
                param_rotmat[:, 0, :, :].transpose(2, 1),
            )
            # RR = torch.einsum('bnm,ml->bnl',param_rotmat[:, 0, :, :].transpose(2, 1), self.unit_rotation)
            # batch multiply target rotation with input rotation
            param_rotmat = torch.einsum("bnm,bhml->bhnl", RR, param_rotmat)
            # param_rotmat = torch.einsum('bhnm,bml->bhnl', param_rotmat, RR)
            param = matrix_to_axis_angle(param_rotmat)
        else:
            RR = None  # self.unit_rotation.repeat(self.bs, 1, 1)

        if rotrep == "aa":
            param = param
        elif rotrep == "sixd":
            param = param.view(self.bs, 2, -1, 3)
            param = axis_angle_to_rotation6d(param).view(self.bs, 2, -1)
        else:
            raise ValueError("Invalid rotation representation.")

        return param, RR

    def prep_body_pose(self, param, rotrep="sixd"):
        # concatenate global orientation and poses of two humans

        if rotrep == "aa":
            param = param
        if rotrep == "sixd":
            param = param.view(self.bs, 1, -1, 3)
            param = axis_angle_to_rotation6d(param).view(self.bs, 1, -1) # B, 1, 126 
        else:
            raise ValueError("Invalid rotation representation.")

        return param

    def prep_translation(
        self,
        transl,
        relative=True,
        pelvis=None,
        cam_rotation=None,
        to_unit_transl=True,
        target_transl=None,
    ):
        if to_unit_transl:
            target_transl = torch.zeros_like(transl)

        if relative:
            if cam_rotation is not None:
                # transl = torch.cat((transl_h0, transl_h1), dim=1)
                root_transl = transl[:, [0], :]
                xx = target_transl + pelvis
                yy = root_transl + pelvis
                cam_translation = (
                    -1 * torch.einsum("bhn,bnm->bhm", xx, cam_rotation)
                ) + yy
                xx = transl + pelvis - cam_translation
                transl = (
                    torch.einsum("bhn,bnm->bhm", xx, cam_rotation.transpose(2, 1))
                    - pelvis
                )
            else:
                transl[:, 1, :] -= transl[:, 0, :]
                transl[:, 0, :] = 0.0

        return transl


    def get_params(self, batch, prefix=""):
        """ 
        Get parameters from batch.
        prefix: '' (uses bev params) or 'pseudogt_' uses optimized meshes
        """
        
        assert prefix in ['pgt', 'bev']
            
        return dict(
            orient=batch[f"{prefix}_global_orient"],
            pose=batch[f"{prefix}_body_pose"],
            betas=batch[f"{prefix}_betas"],
            scale=batch[f"{prefix}_scale"],
            transl=batch[f"{prefix}_transl"],
        )

    def update_camera_params(self, batch):
        for param_name, param in self.camera.named_parameters():
            param.requires_grad = False
            if param_name in batch.keys():
                init_value = batch[param_name].clone().detach().unsqueeze(-1)
                param[:] = init_value
        self.camera.iw[:] = batch["iw"].unsqueeze(-1)
        self.camera.ih[:] = batch["ih"].unsqueeze(-1)

    def get_smpl(self, params, gender):
        """SMPL forward pass from parameters."""

        # unpack params if provided
        orient = params["orient"]
        pose = params["pose"]
        shape = params["shape"]
        transl = params["transl"]

        # unpack pose params
        if self.exp_cfg.rotrep == "sixd":
            bs = pose.shape[0]  # use curernt batch size in case of partial batch
            orient = rotation6d_to_axis_angle(orient)  # bs, 1, 3
            pose = rotation6d_to_axis_angle(pose.view(bs, self.nh, -1, 6)).view(
                bs, self.nh, -1
            )

        # forward human 1
        gender_male_mask = gender[:, 0] == 1
        gender_female_mask = gender[:, 1] == 1

        smpl_male_h0 = self.male_body_model(
            global_orient=orient[:, 0],
            body_pose=pose[:, 0],
            betas=shape[:, 0],
            transl=transl[:, 0],
        )

        smpl_female_h0 = self.female_body_model(
            global_orient=orient[:, 0],
            body_pose=pose[:, 0],
            betas=shape[:, 0],
            transl=transl[:, 0],
        )

        keys = smpl_male_h0._fields
        for key in keys:
            male_val = eval('smpl_male_h0.' + key)[gender_male_mask]
            female_val = eval('smpl_female_h0.' + key)[gender_female_mask]
            bs = male_val.shape[0] + female_val.shape[0]
            combined_val = torch.ones((bs,) + male_val.shape[1:], dtype=male_val.dtype, device=male_val.device)
            combined_val[gender_male_mask] = male_val
            combined_val[gender_female_mask] = female_val
            smpl_male_h0 = smpl_male_h0._replace(**{key: combined_val})

        return smpl_male_h0

    def preprocess_batch(self, batch, in_data=None, clone=False):
        """
        Preprocess batch for training and return all parameters used in diffusion training.
        """

        # experiment setup
        if in_data is None:
            prefix = self.exp_cfg.in_data  # pgt or bev
        else:
            prefix = in_data
        
        out = {
            "orient": batch[f"{prefix}_global_orient"],
            "pose": batch[f"{prefix}_body_pose"],
            "shape": batch[f"{prefix}_betas"],
            "transl": batch[f"{prefix}_transl"],
            "orient_obj": batch[f"{prefix}_orient_obj"],
            "transl_obj": batch[f"{prefix}_transl_obj"],
        }

        # clone parameters if clone is true 
        if clone:
            for pp in ['orient', 'pose', 'shape', 'transl', 'orient_obj', 'transl_obj']:
                out[pp] = out[pp].clone()

        return out

    def reset_orient_and_transl(
        self,
        params,
        gender,
        to_unit_rotation=True,
        target_rotation=None,
        to_unit_transl=True,
        target_transl=None,
        relative_orient=None,
        relative_transl=None,
    ):
        """"
        Reset orientation and translation parameters to target or unit orient and transl.
        Useful e.g. in samling process to reset orientation and translation after each iteration.
        """

        rotrep = self.exp_cfg.rotrep  # rotation representation
        relative_orient = (
            self.exp_cfg.relative_orient if relative_orient is None else relative_orient
        )
        relative_transl = (
            self.exp_cfg.relative_transl if relative_transl is None else relative_transl
        )
        target_rotation = self.unit_rotation if to_unit_rotation else target_rotation
        target_transl = (
            torch.zeros_like(params["transl"]) if to_unit_transl else target_transl
        )

        if relative_orient:
            if rotrep == "aa":
                param_rotmat = axis_angle_to_matrix(params["orient"])
            elif rotrep == "sixd":
                param_rotmat = rotation_6d_to_matrix(params["orient"])

            cam_rotation = torch.einsum(
                "bml,bln->bmn",
                target_rotation,
                param_rotmat[:, 0, :, :].transpose(2, 1),
            )
            param_rotmat = torch.einsum("bnm,bhml->bhnl", cam_rotation, param_rotmat)

            if rotrep == "aa":
                orient = matrix_to_axis_angle(param_rotmat)
            elif rotrep == "sixd":
                orient = matrix_to_rotation_6d(param_rotmat)  # .view(self.bs, 2, -1)
            params["orient"] = orient
        else:
            cam_rotation = None  # self.unit_rotation.repeat(self.bs, 1, 1)

        pelvis = None
        gender_male_mask = gender[:, 0] == 1
        gender_female_mask = gender[:, 1] == 1
        if cam_rotation is not None:
            male_pelvis = torch.cat(
                (
                    self.male_body_model(
                        betas=params["shape"][gender_male_mask, [0], :10],
                    ).joints[:, [0], :],  # only use first 10 betas
                ),
                dim=1,
            )
            female_pelvis = torch.cat(
                (
                    self.female_body_model(
                        betas=params["shape"][gender_female_mask, [0], :10],
                    ).joints[:, [0], :],  # only use first 10 betas
                ),
                dim=1,
            )
            
            bs = male_pelvis.shape[0] + female_pelvis.shape[0]
            pelvis = torch.ones((bs,) + male_pelvis.shape[1:], dtype=male_pelvis.dtype, device=male_pelvis.device)
            pelvis[gender_male_mask] = male_pelvis
            pelvis[gender_female_mask] = female_pelvis # B, 1, N_j, 3

        if relative_transl:
            transl = params["transl"]
            if cam_rotation is not None:
                root_transl = transl[:, [0], :]
                xx = target_transl + pelvis
                yy = root_transl + pelvis
                cam_translation = (
                    -1 * torch.einsum("bhn,bnm->bhm", xx, cam_rotation)
                ) + yy
                xx = transl + pelvis - cam_translation
                transl = (
                    torch.einsum("bhn,bnm->bhm", xx, cam_rotation.transpose(2, 1))
                    - pelvis
                )
            else:
                transl[:, 1, :] -= transl[:, 0, :]
                transl[:, 0, :] = 0.0

            params["transl"] = transl

        return params

    def cast_smpl(self, params, gender):
        """ Bring SMPL parameters to the correct format. 
        params: dict of SMPL parameters, with keys: (orient, pose, shape, transl)
                of dims (bs, nh, (3, 63, 11, 3))
        """

        rotrep = self.exp_cfg.rotrep  # rotation representation
        relative_orient = (
            self.exp_cfg.relative_orient
        )  # absolute or relative global orientation
        relative_transl = (
            self.exp_cfg.relative_transl
        )  # absolute or relative translation

        concat_orient = torch.concat((params['orient'],params['orient_obj']), dim=1) # expected shape (B,2,3)
        concat_transl = torch.concat((params['transl'],params['transl_obj']), dim=1) # expected shape (B,2,3)
        orient, cam_rotation = self.prep_global_orient(
            concat_orient, rotrep, relative=relative_orient
        )
        
        gender_male_mask = gender[:, 0] == 1
        gender_female_mask = gender[:, 1] == 1

        pelvis = None
        if cam_rotation is not None:
            male_pelvis = torch.cat(
                (
                    self.male_body_model(
                        betas=params["shape"][gender_male_mask, [0], :10],
                    ).joints[:, [0], :],  # only use first 10 betas
                ),
                dim=1,
            )
            female_pelvis = torch.cat(
                (
                    self.female_body_model(
                        betas=params["shape"][gender_female_mask, [0], :10],
                    ).joints[:, [0], :],  # only use first 10 betas
                ),
                dim=1,
            )
            
            bs = male_pelvis.shape[0] + female_pelvis.shape[0]
            pelvis = torch.ones((bs,) + male_pelvis.shape[1:], dtype=male_pelvis.dtype, device=male_pelvis.device)
            pelvis[gender_male_mask] = male_pelvis
            pelvis[gender_female_mask] = female_pelvis # B, 1, N_j, 3
 
        transl = self.prep_translation(
            concat_transl,
            relative=relative_transl,
            pelvis=pelvis,
            cam_rotation=cam_rotation,
        )

        pose = self.prep_body_pose(params["pose"], rotrep)

        orient_obj = orient[:, [1], :]
        transl_obj = transl[:, [1], :]

        # update params with new values
        new_values = {"orient": orient[:, [0], :], "pose": pose, 
                      "transl": transl[:, [0], :], "orient_obj": orient_obj, 
                      "transl_obj": transl_obj}
        
        params.update(new_values)

        return params

    def get_guidance_params(
        self,
        batch,
        guidance_param_nc=None,
        guidance_all_nc=None,
        guidance_no_nc=None,
        guidance_params=None,
        clone=False
    ):
        """
        :param batch (dictionary from train/val set)
        :param guidance_param_nc (optional float) probability of masking out any input parameter in batch
        :param guidance_all_nc (optional float) probaiblity of masking out all of the parameters in batch (unconditional generation)
        :param guidance_no_nc (optional float) probability of not masking out anything (pass guidance parameters as is). should sum up to 1 with guidance_all_nc.
        """
        guidance = {}

        guidance_param_nc = (
            self.exp_cfg.guidance_param_nc
            if guidance_param_nc is None
            else guidance_param_nc
        )
        guidance_all_nc = (
            self.exp_cfg.guidance_all_nc if guidance_all_nc is None else guidance_all_nc
        )
        guidance_no_nc = (
            self.exp_cfg.guidance_no_nc if guidance_no_nc is None else guidance_no_nc
        )
        guidance_params = (
            self.exp_cfg.guidance_params if guidance_params is None else guidance_params
        )

        # with some prob we mask all guidance parameters. In this case set noise chance to 1.0.
        # torch.rand(1) samples from [0,1)
        all_or_none = torch.rand(self.bs)
        noise_chance = self.exp_cfg.guidance_param_nc * torch.ones(self.bs)
        cond1 = all_or_none <= guidance_all_nc
        cond2 = (all_or_none > guidance_all_nc) & (all_or_none <= guidance_all_nc + guidance_no_nc)
        noise_chance[cond1] = 1.0
        noise_chance[cond2] = 0.0
        null_value = 0.0

        if self.is_full_bev_guidance:
            gender = batch['gender'][:, 0]  # B, 2
            guidance = self.cast_smpl(self.preprocess_batch(batch, in_data='bev', clone=clone), gender)
            guidance = self.split_humans(guidance)
        elif len(guidance_params) == 0:
            pass
        else:
            raise NotImplementedError

        # set guidance param to None if prob allows it
        for kk in guidance.keys():
            guidance[kk][torch.rand(self.bs) < noise_chance] = null_value      

        return guidance

    def get_gt_params(self, batch):
        target = {}
        for param_name in self.human_params:
            target[param_name] = batch[param_name]
        return target

    def split_humans(self, x, keep_dim=False):
        """
        Split model params of for [BS, NH, D] into [BS, D] for each human.
        If keep_dim=True, the output will be [BS, 1, D] for each human.
        """
        out = {}
        for pp in x.keys():
            v = x[pp]
            if pp in self.human_params:
                for ii in range(self.nh):
                    value = v[:, ii] if not keep_dim else v[:, [ii]]
                    out.update({f"{pp}_h{ii}": value})
            else:
                out.update({pp: v[:, 0] if not keep_dim else v[:, [0]]})
        return out

    def concat_humans(self, x):
        """
        Concatenate model params of for [BS, D] for each human into [BS, NH, D].
        """
        out = {}
        for pp in self.human_params:
            params = [x[f"{pp}_h{ii}"].unsqueeze(1) for ii in range(self.nh)]
            out.update({pp: torch.cat(params, dim=1)})
        return out

    def merge_params(self, target, update):
        """Merge update into target."""
        out = {}
        for pp in target.keys():
            value = target[pp].clone()
            if pp in self.human_params:
                for ii in range(self.nh):
                    if f"{pp}_h{ii}" in update.keys():
                        value[:, ii] = update[f"{pp}_h{ii}"]
            else:
                value[:, 0] = update[pp]
            out.update({pp: value})
        return out

    def sampling_loop(self, x, y, ts, obj, obj_vertices, gender, inpaint=None, log_steps=1, return_latent_vec=False, eta=1.0):
        """Sample from diffusion model."""

        sbs = x['orient_h0'].shape[0]
        x_ts, x_starts, x_latent = {}, {}, {}  # x_ts is mesh with noise, x_starts the denoised mesh
        obj_ts, obj_starts = {}, {}
        # setup inpaint params if provided 

        last_step = ts[-1]
        for ii_idx,  ii in enumerate(ts):

            if inpaint is not None:
                for k in inpaint['mask'].keys():
                    x[k][inpaint['mask'][k]] = inpaint['values'][k][inpaint['mask'][k]]

            # reset orientation and translation
            #x = self.reset_orient_and_transl(x)

            # timestep
            t = torch.tensor([ii] * self.bs).to(self.cfg.device)
            if ii_idx + 1 < len(ts):
                prev_t = torch.tensor([ts[ii_idx + 1]] * self.bs).to(self.cfg.device)
            else:
                prev_t = None
            # add noise to params
            #diffused_params = {}
            input_noise = {}

            pred = self.model(
                x=x,  # diffused_tokens_dict,
                timesteps=self.diffusion._scale_timesteps(t),
                guidance=y,
                gender=gender,
                obj=obj,
                return_latent_vec=return_latent_vec,
            )

            if return_latent_vec:
                pred, latent_vec = pred

            # predicted tokens to params and smpl bodies
            denoised_params = self.concat_humans(pred)
            denoised_smpls = self.get_smpl(denoised_params, gender)

            # get q(x_{t-1}| x_t, x_0)
            for k in pred.keys():
                sample, noise = self.diffusion.p_sample_ddim(pred[k], t, prev_t, x[k], eta=eta)
                x[k] = sample
                input_noise[k] = noise

            diffused_params = self.concat_humans(x)
            diffused_smpl = self.get_smpl(diffused_params, gender)

            diffusion_output = {
                "denoised_params": pred,
                "denoised_smpls": denoised_smpls,
                "diffused_params": x,
                "diffused_smpls": diffused_smpl,
            }

            # update x_start_tokens
            #x = diffusion_output["denoised_smpls"]

            # log results
            if ii % log_steps == 0:
                x_ts[ii] = diffusion_output["diffused_smpls"]
                x_starts[ii] = diffusion_output["denoised_smpls"]
                
                # diffused object vertices
                obj_ts[ii] = []
                obj_starts[ii] = []
                rot = rotation_6d_to_matrix(diffusion_output['diffused_params']['orient_obj'])
                trans = diffusion_output['diffused_params']['transl_obj']

                rot_denoise = rotation_6d_to_matrix(diffusion_output['denoised_params']['orient_obj'])
                trans_denoise = diffusion_output['denoised_params']['transl_obj']
                for obj_idx, vertices in enumerate(obj_vertices):
                    # diffused object vertices
                    new_postion = vertices @ rot[obj_idx].transpose(0, 1) + trans[obj_idx]
                    obj_ts[ii].append(new_postion)

                    # denoised object vertices
                    denoise_position = vertices @ rot_denoise[obj_idx].transpose(0, 1) + trans_denoise[obj_idx]
                    obj_starts[ii].append(denoise_position)

                if return_latent_vec:
                    x_latent[ii] = diffusion_output["model_latent_vec"]

            if ii == last_step:
                x_starts["final"] = diffusion_output["denoised_smpls"]
                x_ts["final"] = diffusion_output["diffused_smpls"]

                obj_ts["final"] = []
                obj_starts["final"] = []
                rot = rotation_6d_to_matrix(diffusion_output['diffused_params']['orient_obj'])
                trans = diffusion_output['diffused_params']['transl_obj']

                rot_denoise = rotation_6d_to_matrix(diffusion_output['denoised_params']['orient_obj'])
                trans_denoise = diffusion_output['denoised_params']['transl_obj']
                for obj_idx, vertices in enumerate(obj_vertices):
                    # diffused object vertices
                    new_postion = vertices @ rot[obj_idx].transpose(0, 1) + trans[obj_idx]
                    obj_ts["final"].append(new_postion)

                    # denoised object vertices
                    denoise_position = vertices @ rot_denoise[obj_idx].transpose(0, 1) + trans_denoise[obj_idx]
                    obj_starts["final"].append(denoise_position)

        if return_latent_vec:
            return x_ts, x_starts, obj_ts, obj_starts, x_latent
        else:
            return x_ts, x_starts, obj_starts, obj_ts


    def sampling_loop_orig(self, x, y, ts, inpaint=None, log_steps=1, return_latent_vec=False):
        """Sample from diffusion model."""

        sbs = x[0].body_pose.shape[0]
        x_ts, x_starts, x_latent = {}, {}, {}  # x_ts is mesh with noise, x_starts the denoised mesh

        last_step = ts[-1]
        for ii in ts:

            x = {
                "orient": torch.cat(
                    [
                        axis_angle_to_rotation6d(x[0].global_orient.unsqueeze(1)),
                        axis_angle_to_rotation6d(x[1].global_orient.unsqueeze(1)),
                    ],
                    dim=1,
                ),
                "pose": torch.cat(
                    [
                        axis_angle_to_rotation6d(
                            x[0].body_pose.unsqueeze(1).view(sbs, 1, -1, 3)
                        ).view(sbs, 1, -1),
                        axis_angle_to_rotation6d(
                            x[1].body_pose.unsqueeze(1).view(sbs, 1, -1, 3)
                        ).view(sbs, 1, -1),
                    ],
                    dim=1,
                ),
                "shape": torch.cat(
                    [
                        torch.cat((x[0].betas, x[0].scale), dim=-1).unsqueeze(1),
                        torch.cat((x[1].betas, x[1].scale), dim=-1).unsqueeze(1),
                    ],
                    dim=1,
                ),
                "transl": torch.cat(
                    [x[0].transl.unsqueeze(1), x[1].transl.unsqueeze(1)], dim=1
                ),
            }

            # do the inpainting
            if inpaint is not None:
                for k, mm in inpaint["mask"].items():
                    x[k][mm] = inpaint["values"][k][mm]

            # reset orientation and translation
            x = self.reset_orient_and_transl(x)

            # timestep
            t = torch.tensor([ii] * self.bs).to(self.cfg.device)

            # diffusion forward (add noise) and backward (remove noise) process
            diffusion_output = self.diffuse_denoise(
                x=x, y=y, t=t, return_latent_vec=return_latent_vec)

            if inpaint is not None:
                denoised_params = diffusion_output["denoised_params"]
                for k, mm in inpaint["mask"].items():
                    denoised_params[k][mm] = inpaint["values"][k][mm]
                diffusion_output["denoised_smpls"] = self.get_smpl(denoised_params)

            # update x_start_tokens
            x = diffusion_output["denoised_smpls"]

            # log results
            if ii % log_steps == 0:
                x_ts[ii] = diffusion_output["diffused_smpls"]
                x_starts[ii] = diffusion_output["denoised_smpls"]
                
                if return_latent_vec:
                    x_latent[ii] = diffusion_output["model_latent_vec"]

            if ii == last_step:
                x_starts["final"] = diffusion_output["denoised_smpls"]
                x_ts["final"] = diffusion_output["diffused_smpls"]

        if return_latent_vec:
            return x_ts, x_starts, x_latent
        else:
            return x_ts, x_starts

    def sampling_loop_ddim(self, x, y, ts, inpaint=None, log_steps=100, return_latent_vec=False):
        """Sample from diffusion model."""

        sbs = x[0].body_pose.shape[0]
        x_ts, x_starts, x_latent = {}, {}, {}  # x_ts is mesh with noise, x_starts the denoised mesh

        last_step = ts[-1]

        x = {
            "orient": torch.cat(
                [
                    axis_angle_to_rotation6d(x[0].global_orient.unsqueeze(1)),
                    axis_angle_to_rotation6d(x[1].global_orient.unsqueeze(1)),
                ],
                dim=1,
            ),
            "pose": torch.cat(
                [
                    axis_angle_to_rotation6d(
                        x[0].body_pose.unsqueeze(1).view(sbs, 1, -1, 3)
                    ).view(sbs, 1, -1),
                    axis_angle_to_rotation6d(
                        x[1].body_pose.unsqueeze(1).view(sbs, 1, -1, 3)
                    ).view(sbs, 1, -1),
                ],
                dim=1,
            ),
            "shape": torch.cat(
                [
                    torch.cat((x[0].betas, x[0].scale), dim=-1).unsqueeze(1),
                    torch.cat((x[1].betas, x[1].scale), dim=-1).unsqueeze(1),
                ],
                dim=1,
            ),
            "transl": torch.cat(
                [x[0].transl.unsqueeze(1), x[1].transl.unsqueeze(1)], dim=1
            ),
        }

        # add noise to params
        diffused_params = {}
        input_noise = {}
        t = torch.tensor([999] * self.bs).to(self.cfg.device)
        for pp, v in x.items():
            input_noise[pp] = torch.randn_like(v)
            diffused_params[pp] = self.diffusion.q_sample(v, t, input_noise[pp])
        diffused_smpls = self.get_smpl(diffused_params)

        # merge input parameters with guidance (only used for visualization)        
        diffused_with_guidance_params = self.merge_params(diffused_params, y)
        diffused_with_guidance_smpls = self.get_smpl(diffused_with_guidance_params)

        # concatenate human parameters when using H0H1 token setup
        if self.exp_cfg.token_setup == "H0H1":
            split_dims = [diffused_params[pp].shape[-1] for pp in self.human_params]
            self.split_dims = split_dims
            diffused_params = {
                "human": torch.cat(
                    [diffused_params[pp] for pp in self.human_params], dim=-1
                )
            }

        # forward model / transformer
        #x = self.split_humans(diffused_params) 
        #x = diffused_params
        x = self.split_humans(diffused_params)
        for ii_idx, ii in enumerate(ts):
            """
            x = {
                "orient": torch.cat(
                    [
                        axis_angle_to_rotation6d(x[0].global_orient.unsqueeze(1)),
                        axis_angle_to_rotation6d(x[1].global_orient.unsqueeze(1)),
                    ],
                    dim=1,
                ),
                "pose": torch.cat(
                    [
                        axis_angle_to_rotation6d(
                            x[0].body_pose.unsqueeze(1).view(sbs, 1, -1, 3)
                        ).view(sbs, 1, -1),
                        axis_angle_to_rotation6d(
                            x[1].body_pose.unsqueeze(1).view(sbs, 1, -1, 3)
                        ).view(sbs, 1, -1),
                    ],
                    dim=1,
                ),
                "shape": torch.cat(
                    [
                        torch.cat((x[0].betas, x[0].scale), dim=-1).unsqueeze(1),
                        torch.cat((x[1].betas, x[1].scale), dim=-1).unsqueeze(1),
                    ],
                    dim=1,
                ),
                "transl": torch.cat(
                    [x[0].transl.unsqueeze(1), x[1].transl.unsqueeze(1)], dim=1
                ),
            }
            """

            # do the inpainting
            #if inpaint is not None:
            #    for k, mm in inpaint["mask"].items():
            #        x[k][mm] = inpaint["values"][k][mm]

            # reset orientation and translation
            #x = self.reset_orient_and_transl(x)
            #x['transl_h1'] -= x['transl_h0']
            #x['transl_h0'] = torch.zeros_like(x['transl_h0'])
            #transl[:, 1, :] -= transl[:, 0, :]
            #transl[:, 0, :] = 0.0
            
            # timestep
            t = torch.tensor([ii] * self.bs).to(self.cfg.device)
            if ii_idx + 1 < len(ts):
                prev_t = torch.tensor([ts[ii_idx + 1]] * self.bs).to(self.cfg.device)
            else:
                prev_t = None

            # diffusion forward (add noise) and backward (remove noise) process
            #diffusion_output = self.diffuse_denoise(
            #    x=x, y=y, t=t, return_latent_vec=return_latent_vec)

            pred = self.model(
                x=x,  # diffused_tokens_dict,
                timesteps=self.diffusion._scale_timesteps(t),
                guidance=y,
                return_latent_vec=return_latent_vec,
            )

            if return_latent_vec:
                pred, latent_vec = pred

            for pp, v in pred.items():
                pred_prev_t = self.diffusion.p_sample_ddim(v, t, prev_t, x[pp])
                x[pp] = pred_prev_t
            
            if self.diffusion.model_mean_type == "start_x":
                denoised_tokens = pred
            elif self.diffusion.model_mean_type == "epsilon":
                # in this case denoised_params are the predicted noise
                # we need to remove the noise from the input params
                denoised_tokens = {}
                for k, pred_noise in pred.items():
                    denoised_tokens[k] = x[k] - pred_noise
            else:
                raise NotImplementedError

        
            # split tokens when using H0H1 token setup
            """
            if self.exp_cfg.token_setup == "H0H1":
                for ii in range(self.nh):
                    token = denoised_tokens.pop(f"human_h{ii}")
                    params = torch.split(token, split_dims, -1)
                    for pp, vv in zip(self.human_params, params):
                        denoised_tokens[f"{pp}_h{ii}"] = vv

                    # split predicted noise
                    if self.diffusion.model_mean_type == "epsilon":
                        token = pred.pop(f"human_h{ii}")
                        params = torch.split(token, split_dims, -1)
                        for pp, vv in zip(self.human_params, params):
                            pred[f"{pp}_h{ii}"] = v
            """

            #if self.diffusion.model_mean_type == "epsilon":
            #    pred = self.concat_humans(pred)

            # predicted tokens to params and smpl bodies
            denoised_params = self.concat_humans(denoised_tokens)
            denoised_smpls = self.get_smpl(denoised_params)
            pred_prev_t_params = self.concat_humans(x)
            pred_prev_t_smpls = self.get_smpl(pred_prev_t_params)

            #if inpaint is not None:
                #denoised_params = diffusion_output["denoised_params"]
            #    for k, mm in inpaint["mask"].items():
            #        denoised_params[k][mm] = inpaint["values"][k][mm]
                #diffusion_output["denoised_smpls"] = self.get_smpl(denoised_params)

            # update x_start_tokens
            #x = diffusion_output["denoised_smpls"]

            # log results
            if ii % log_steps == 0:
                x_ts[ii] = pred_prev_t_smpls #diffusion_output["diffused_smpls"]
                x_starts[ii] = denoised_smpls #diffusion_output["denoised_smpls"]
                
                if return_latent_vec:
                    x_latent[ii] = latent_vec #diffusion_output["model_latent_vec"]

            if ii == last_step:
                x_starts["final"] = denoised_smpls #diffusion_output["denoised_smpls"]
                x_ts["final"] = pred_prev_t_smpls #diffusion_output["diffused_smpls"]

        if return_latent_vec:
            return x_ts, x_starts, x_latent
        else:
            return x_ts, x_starts

    def diffuse_denoise(self, x, y, t, obj, gender, noise=None, return_latent_vec=False):
        """Add noise to input and forward through diffusion model."""

        # add noise to params
        diffused_params = {}
        input_noise = {}
        for pp, v in x.items():
            input_noise[pp] = torch.randn_like(v) if noise is None else noise[pp]
            diffused_params[pp] = self.diffusion.q_sample(v, t, input_noise[pp])
        diffused_smpls = self.get_smpl(diffused_params, gender)

        # merge input parameters with guidance (only used for visualization)        
        diffused_with_guidance_params = self.merge_params(diffused_params, y)
        diffused_with_guidance_smpls = self.get_smpl(diffused_with_guidance_params, gender)


        # forward model / transformer
        x = self.split_humans(diffused_params)
        pred = self.model(
            x=x,  # diffused_tokens_dict,
            timesteps=self.diffusion._scale_timesteps(t),
            guidance=y,
            gender=gender,
            obj=obj,
            return_latent_vec=return_latent_vec,
        )

        if return_latent_vec:
            pred, latent_vec = pred

        if self.diffusion.model_mean_type == "start_x":
            denoised_tokens = pred
        elif self.diffusion.model_mean_type == "epsilon":
            # in this case denoised_params are the predicted noise
            # we need to remove the noise from the input params
            denoised_tokens = {}
            for k, pred_noise in pred.items():
                denoised_tokens[k] = x[k] - pred_noise
        else:
            raise NotImplementedError

        if self.diffusion.model_mean_type == "epsilon":
            pred = self.concat_humans(pred)

        # predicted tokens to params and smpl bodies
        denoised_params = self.concat_humans(denoised_tokens)
        denoised_smpls = self.get_smpl(denoised_params, gender)

        return {
            "denoised_params": denoised_params,
            "denoised_smpls": denoised_smpls,
            "diffused_with_guidance_smpls": diffused_with_guidance_smpls,
            'diffused_with_guidance_params': diffused_with_guidance_params,
            "diffused_smpls": diffused_smpls,
            "gt_noise": input_noise,
            "model_prediction": pred,
            'model_latent_vec': latent_vec if return_latent_vec else None,
            'x': x
        }
    
    def single_training_step(self, batch):
        """Implement a single training step."""

        # select and transform input (e.g. bev or pseudo gt)
        guidance_params = self.get_guidance_params(batch)

        # Get obj embeddings
        obj_name = batch['obj_name']
        obj_embeddings = batch['obj_embeddings'][:, 0]  # B, 256

        # Get gender of the human
        gender = batch['gender'][:, 0]  # B, 2

        # overwrite batch with corret input
        batch = self.cast_smpl(self.preprocess_batch(batch), gender)

        # target / gt params
        # target_params = self.get_gt_params(batch)
        target_smpls = self.get_smpl(batch, gender)

        # sample t
        t, weights = self.schedule_sampler.sample(self.bs, "cuda")

        # diffusion forward (add noise) and backward (remove noise) process
        diffusion_output = self.diffuse_denoise(x=batch, y=guidance_params, t=t, obj=obj_embeddings, gender=gender)

        # compute custom loss for diffusion model predicting x_start
        if self.diffusion.model_mean_type == "start_x":
            total_loss, loss_dict = self.criterion(
                est_smpl=diffusion_output["denoised_smpls"],
                tar_smpl=target_smpls,
                est_params=diffusion_output["model_prediction"],
                tar_params=batch,
            )
        elif self.diffusion.model_mean_type == "epsilon":
            gt_noise = diffusion_output["gt_noise"]
            pred_noise = diffusion_output["model_prediction"]
            loss_dict = {}
            for k, v in pred_noise.items():
                loss_dict[f"{k}"] = torch.mean((v - gt_noise[k]) ** 2)
            total_loss = sum(loss_dict.values())
        else:
            raise NotImplementedError

        # Visualization code
        imglabel = (
            t.tolist()
        )

        bs = batch['orient_obj'].shape[0]
        diffused_obj = []
        denoised_obj = []
        target_obj = []
        diffused_guid_obj = []

        for idx in range(bs):
            obj_vertices =  self.obj_vertices[obj_name[idx]]
            rot = rotation_6d_to_matrix(diffusion_output['x']['orient_obj'][[idx]])[0]
            new_postion = obj_vertices @ rot.transpose(0, 1) + diffusion_output['x']['transl_obj'][[idx]]
            diffused_obj.append(new_postion)

            rot = rotation_6d_to_matrix(diffusion_output['model_prediction']['orient_obj'][[idx]])[0]
            new_postion = obj_vertices @ rot.transpose(0, 1) + diffusion_output['model_prediction']['transl_obj'][[idx]]
            denoised_obj.append(new_postion)

            rot = rotation_6d_to_matrix(batch['orient_obj'][idx])[0]
            new_postion = obj_vertices @ rot.transpose(0, 1) + batch['transl_obj'][idx]
            target_obj.append(new_postion)

            guid = diffusion_output['diffused_with_guidance_params']
            rot = rotation_6d_to_matrix(guid['orient_obj'][idx])[0]
            new_postion = obj_vertices @ rot.transpose(0, 1) + guid['transl_obj'][idx]
            diffused_guid_obj.append(new_postion)

        # diffused_obj = torch.stack(diffused_obj, dim=0)
        # denoised_obj = torch.stack(denoised_obj, dim=0)
        # target_obj = torch.stack(target_obj, dim=0)

        if len(guidance_params) > 0:
            diffused_output_with_guidance_for_rendering_smpl = diffusion_output["diffused_with_guidance_smpls"]
        else:
            diffused_output_with_guidance_for_rendering_smpl = None
        output_dict = { 
            "images": { 
                "single_step_random_t": [
                    self.get_tb_image_data(
                        diffusion_output["denoised_smpls"],
                        diffusion_output["diffused_smpls"],
                        diffused_output_with_guidance_for_rendering_smpl,
                        target_smpls,
                        0,
                        diffused_obj,
                        denoised_obj,
                        target_obj,
                        diffused_guid_obj,
                        obj_name
                    ), imglabel]
            }
        }

        return total_loss, loss_dict, output_dict

    def sample_from_model(self, timesteps, log_freq, guidance_params={}, gender=None, curr_obj_embeddings= None,
                          obj_vertices=None, batch_size=None):

        """Sample from the diffusion model without conditioning starting from noise"""

        assert curr_obj_embeddings is not None, "Please specfiy object embeddings"
        bs = self.cfg.batch_size if batch_size is None else batch_size
        device = self.cfg.device

        # create random noise to start with
        noise = {
            "orient": torch.randn([bs, 1, self.orient_dim]).to(device),
            "pose": torch.randn([bs, 1, self.pose_dim]).to(device),
            "shape": torch.randn([bs, 1, self.shape_dim]).to(device),
            "transl": torch.randn([bs, 1, self.transl_dim]).to(device),
            "orient_obj": torch.randn([bs, 1, self.orient_obj_dim]).to(device),
            "transl_obj": torch.randn([bs, 1, self.transl_obj_dim]).to(device),
        }
        if gender is None:
            gender = torch.nn.functional.one_hot(torch.randint(0, 2, (bs,)).to(device), num_classes=2)

        noise_params = self.split_humans(noise)

        # run diffusion sampling loop
        x_ts, x_starts, obj_ts, obj_starts = self.sampling_loop(
            x=noise_params, y=guidance_params, ts=timesteps, obj=curr_obj_embeddings, gender=gender,
            obj_vertices=obj_vertices, log_steps=log_freq, return_latent_vec=False
        )

        return x_ts, x_starts, obj_ts, obj_starts
    
    @torch.no_grad()
    def single_validation_step(self, batch, batch_idx):
        """Implement the full validation precedure. Use val_dataset."""

        if batch_idx == 0:
            self.evaluator.tb_output = {
                'images': {}
            }

        # get guidance params - do not add noise to params at validation
        guidance_params_no_noise = self.get_guidance_params(
            batch, guidance_param_nc=0.0, guidance_all_nc=0.0, guidance_no_nc=1.0, clone=True
        )
        guidance_params_all_noise = self.get_guidance_params(
            batch, guidance_param_nc=0.0, guidance_all_nc=1.0, guidance_no_nc=0.0, clone=True
        )

        # Get obj embeddings
        obj_name = batch['obj_name']
        obj_embeddings = batch['obj_embeddings'][:, 0]  # B, 256

        # Get gender of the human
        gender = batch['gender'][:, 0]  # B, 2

        # select and transform input (e.g. bev or pseudo gt)
        batch = self.cast_smpl(self.preprocess_batch(batch), gender)

        # target / gt params
        # target_params = self.get_gt_params(batch)
        target_smpls = self.get_smpl(batch, gender)

        # get guidance params - do not add noise to params at validation
        # guidance_params = self.get_guidance_params(
            # batch, guidance_param_nc=0.0, guidance_all_nc=0.0, guidance_no_nc=1.0
        # )

        if batch_idx == 0:
            obj_vertices = []
            for name in obj_name:
                curr_obj_vertices = self.obj_vertices[name]
                obj_vertices.append(curr_obj_vertices)

            ############ unconditional sampling ##############
            # guru.info('Start sampling unconditional')
            uncond_ts = np.arange(1, self.diffusion.num_timesteps, 10)[::-1]
            log_freq = 10
            x_ts, x_starts, obj_ts, obj_starts = self.sample_from_model(
                uncond_ts, log_freq, {}, gender, obj_embeddings, obj_vertices 
            )

            self.evaluator.tb_output['images']['unconditional'] = [
                self.get_tb_image_data(
                    x_starts['final'], 
                    None, 
                    None, 
                    None,
                    0,
                    None,
                    obj_starts['final'],
                    None,
                    None,
                    obj_name),
                ['final_uncond'] * x_starts['final'].vertices.shape[0]]

            ############ conditional sampling ##############
            if len(guidance_params_no_noise) > 0:
                # guru.info('Start sampling unconditional')
                cond_ts = np.arange(1, self.diffusion.num_timesteps, 10)[::-1]
                log_freq = 10
                x_ts, x_starts, obj_ts, obj_starts = self.sample_from_model(
                    cond_ts, log_freq, guidance_params_no_noise, gender, obj_embeddings, obj_vertices
                )

                guidance_params_no_noise_params = self.concat_humans(guidance_params_no_noise)
                for pp in guidance_params_no_noise.keys():
                    if pp not in guidance_params_no_noise_params:
                        guidance_params_no_noise_params[pp] = guidance_params_no_noise[pp].unsqueeze(1)
                guidance_params_no_noise_smpls = self.get_smpl(guidance_params_no_noise_params, gender)

                diffused_guid_obj = []
                bs = x_starts['final'].vertices.shape[0]
                for idx in range(bs):
                    obj_vertices =  self.obj_vertices[obj_name[idx]]
                    rot = rotation_6d_to_matrix(guidance_params_no_noise_params['orient_obj'][idx])[0]
                    new_postion = obj_vertices @ rot.transpose(0, 1) + guidance_params_no_noise_params['transl_obj'][idx]
                    diffused_guid_obj.append(new_postion)

                self.evaluator.tb_output['images']['conditional'] = [
                    self.get_tb_image_data(x_starts['final'], None, guidance_params_no_noise_smpls, None, 0, None, obj_starts['final'], None, diffused_guid_obj, obj_name),
                    ['final_cond'] * bs
                ]

        ############ SAME AS TRAINING STEP ##############
        # add noise to params
        # for t_type in ['random', 25, 50, 75]:
            # if t_type == 'random'
        for t_type in ['single_step_random_t', '50']:
            target_params = {x: v.clone() for x, v in batch.items()}
            guidance_params_no_noise = {x: v.clone() for x, v in guidance_params_no_noise.items()}

            if t_type != 'single_step_random_t':
                t = (int(t_type) * torch.ones(self.bs).to(self.cfg.device)).to(torch.int64)
                t_type = 'single_step_' + t_type + '_t'
            else:
                t, weights = self.schedule_sampler.sample(self.bs, self.cfg.device)

            # diffusion forward (add noise) and backward (remove noise) process
            diffusion_output = self.diffuse_denoise(x=target_params, y=guidance_params_no_noise, t=t, obj=obj_embeddings, gender=gender)

            # compute custom loss for diffusion model predicting x_start
            if self.diffusion.model_mean_type == "start_x":
                total_loss, loss_dict = self.criterion(
                    est_smpl=diffusion_output["denoised_smpls"],
                    tar_smpl=target_smpls,
                    est_params=diffusion_output["model_prediction"],
                    tar_params=batch,
                )
            elif self.diffusion.model_mean_type == "epsilon":
                gt_noise = diffusion_output["gt_noise"]
                pred_noise = diffusion_output["model_prediction"]
                loss_dict = {}
                for k, v in pred_noise.items():
                    loss_dict[f"{k}"] = torch.mean((v - gt_noise[k]) ** 2)
                total_loss = sum(loss_dict.values())
            else:
                raise NotImplementedError

            bs = batch['orient_obj'].shape[0]
            diffused_obj = []
            denoised_obj = []
            target_obj = []
            obj_face = []
            diffused_guid_obj = []

            for idx in range(bs):
                obj_vertices =  self.obj_vertices[obj_name[idx]]
                rot = rotation_6d_to_matrix(diffusion_output['x']['orient_obj'][[idx]])[0]
                new_postion = obj_vertices @ rot.transpose(0, 1) + diffusion_output['x']['transl_obj'][[idx]]
                diffused_obj.append(new_postion)

                rot = rotation_6d_to_matrix(diffusion_output['model_prediction']['orient_obj'][[idx]])[0]
                new_postion = obj_vertices @ rot.transpose(0, 1) + diffusion_output['model_prediction']['transl_obj'][[idx]]
                denoised_obj.append(new_postion)

                rot = rotation_6d_to_matrix(batch['orient_obj'][idx])[0]
                new_postion = obj_vertices @ rot.transpose(0, 1) + batch['transl_obj'][idx]
                target_obj.append(new_postion)

                guid = diffusion_output['diffused_with_guidance_params']
                rot = rotation_6d_to_matrix(guid['orient_obj'][idx])[0]
                new_postion = obj_vertices @ rot.transpose(0, 1) + guid['transl_obj'][idx]
                diffused_guid_obj.append(new_postion)                

                obj_face.append(self.obj_faces[obj_name[idx]])

            if len(guidance_params_no_noise) > 0:
                diffused_output_with_guidance_for_rendering_smpl = diffusion_output["diffused_with_guidance_smpls"]
            else:
                diffused_output_with_guidance_for_rendering_smpl = None

            # run evaluation
            if t_type == 'single_step_50_t':
                self.evaluator(
                    est_smpl=diffusion_output["denoised_smpls"], tar_smpl=target_smpls,
                    est_params=denoised_obj, tar_params=target_obj, human_face=self.faces_tensor, object_face=obj_face,
                    t_type=t_type
                )                
                self.evaluator.accumulate("total_loss", total_loss.cpu().numpy()[None])

            if batch_idx == 0:
                imglabel = (t.tolist())
                self.evaluator.tb_output['images'][t_type] = [
                        self.get_tb_image_data(
                            diffusion_output["denoised_smpls"],
                            diffusion_output["diffused_smpls"],
                            diffused_output_with_guidance_for_rendering_smpl,
                            target_smpls,
                            0,
                            diffused_obj,
                            denoised_obj,
                            target_obj,
                            diffused_guid_obj,
                            obj_name
                        ),
                        imglabel,
                ]


    ##############################################################################################
    ############################# CREATE OUTPUT DATA FOR TENSORBOARD #############################
    ##############################################################################################
    def get_tb_image_data(
        self, sampled_smpls, input_noise_smpls=None, input_with_guidance=None, input_smpls=None, timestep=0,
        diffused_obj=None, denoised_obj=None, target_obj=None, guid_obj=None, obj_name=None,
    ):
        out = {}
        out[f"h0_sampled_{timestep}"] = sampled_smpls
        if input_smpls is not None:
            out[f"h0_input"] = input_smpls
        if input_noise_smpls is not None:
            out[f"h0_input_noise"] = input_noise_smpls
        if input_with_guidance is not None:
            out[f"h0_input_with_guidance"] = input_with_guidance
        if denoised_obj is not None:
            out[f"obj_sampled_{timestep}"] = denoised_obj
        if diffused_obj is not None:
            out[f"obj_input_noise"] = diffused_obj
        if target_obj is not None:
            out[f"obj_input"] = target_obj
        if guid_obj is not None:
            out[f"obj_input_with_guidance"] = guid_obj
        if obj_name is not None:
            out["obj_name"] = obj_name
        return out

    def get_tb_histogram_data(self, params):
        histogram_data = {}
        for i in range(2):
            human_hist = {
                f"human{i}/orient": params.orient[:, i, :],
                f"human{i}/pose": params.pose[:, i, :],
                f"human{i}/betas": params.betas[:, i, :],
                f"human{i}/scale": params.scale[:, i, :],
                f"human{i}/transl": params.transl[:, i, :],
            }
            histogram_data.update(human_hist)
        return histogram_data

    def render_one_method(
        self,
        batch_size,
        verts_h0,
        verts_obj,
        body_model_type,
        meshcol,
        faces_tensor,
        obj_faces_tensor,
        view_to_row,
        method_idx=0,
        timesteps=None,
        vertex_transl_center=None,
    ):

        num_images_per_row = len(view_to_row.keys())
        ih, iw = self.renderer.ih, self.renderer.iw
        row_width = num_images_per_row * iw

        if self.final_image_out is None:
            self.final_image_out = np.zeros((batch_size * ih, row_width, 4))

        def stage_to_idx(col_idx, row_idx, ih, iw, row_width, method_idx=0):
            c0, c1 = col_idx * ih, (col_idx + 1) * ih
            r0, r1 = row_idx * iw, (row_idx + 1) * iw
            r0 = r0 + method_idx * row_width
            r1 = r1 + method_idx * row_width
            return c0, c1, r0, r1

        for idx in range(batch_size):

            if timesteps is not None:
                timestep = "t=" + str(timesteps[idx])
            else:
                timestep = ""

            vh0 = verts_h0[idx]
            verts = torch.cat([vh0], dim=0)
            obj = verts_obj[idx]
            verts_obj0 = torch.cat([obj], dim=0)

            if vertex_transl_center is None:
                vertex_transl_center = verts.mean((0, 1))
                # vertex_transl_center_obj = verts_obj0.mean((0, 1))
            else:
                if not vertex_transl_center.shape == torch.Size([3]):
                    vertex_transl_center = vertex_transl_center[idx]
                    # vertex_transl_center_obj = vertex_transl_center[idx]
            verts_centered = verts - vertex_transl_center
            verts_centered_obj = verts_obj0 - vertex_transl_center

            for yy in [-20, 20]:
                self.renderer.update_camera_pose(0.0, yy, 180.0, 0.0, 0.2, 2.0)
                rendered_img = self.renderer.render(
                    verts_centered,
                    faces_tensor,
                    verts_centered_obj,
                    obj_faces_tensor[idx],
                    colors=meshcol,
                    body_model=body_model_type,
                )
                color_image = rendered_img[0].detach().cpu().numpy() * 255

                c0, c1, r0, r1 = stage_to_idx(
                    idx, view_to_row[yy], ih, iw, row_width, method_idx
                )
                self.final_image_out[c0:c1, r0:r1, :] = color_image

            # bird view
            for pp in [270]:
                self.renderer.update_camera_pose(pp, 0.0, 180.0, 0.0, 0.0, 2.0)
                rendered_img = self.renderer.render(
                    verts_centered,
                    faces_tensor,
                    verts_centered_obj,
                    obj_faces_tensor[idx],
                    colors=meshcol,
                    body_model=body_model_type,
                )
                color_image = rendered_img[0].detach().cpu().numpy() * 255

                # add black text to image showing timestep
                color_image = cv2.putText(
                    color_image,
                    timestep,
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

                c0, c1, r0, r1 = stage_to_idx(
                    idx, view_to_row[pp], ih, iw, row_width, method_idx
                )
                self.final_image_out[c0:c1, r0:r1, :] = color_image

    def render_output(self, output_all, max_images=1):
        """Implement logging of training step."""

        imgname_img = {}

        for output_key, output in output_all.items():
            output, timesteps = output

            num_methods = len(self.meshes_to_render[output_key])
            view_to_row = {
                -20: 0,
                20: 1,
                270: 2,
            }  # mapping between rendering view and row index in image (per method)
            num_views = len(view_to_row.keys())
            ih, iw = self.renderer.ih, self.renderer.iw
            self.final_image_out = np.zeros(
                (max_images * ih, num_methods * num_views * iw, 4)
            )

            # render meshes for outputs
            for idx, name in enumerate(self.meshes_to_render[output_key]):
                if f"h0_{name}" in output.keys():
                    verts_h0 = [
                        output[f"h0_{name}"].vertices[[iidx]].detach()
                        for iidx in range(max_images)
                    ]

                    verts_obj = [
                        output[f"obj_{name}"][iidx].detach().unsqueeze(0)
                        for iidx in range(max_images)
                    ]

                    faces_obj = [
                        self.obj_faces[output[f"obj_name"][iidx]]
                        for iidx in range(max_images)
                    ]
                    
                    self.render_one_method(
                        max_images,
                        verts_h0,
                        verts_obj,
                        self.body_model_type,
                        self.meshcols[name],
                        self.faces_tensor,
                        faces_obj,
                        view_to_row,
                        idx,
                        timesteps,
                    )

            imgname_img[f'renderings/{output_key}'] = self.final_image_out

        return imgname_img
