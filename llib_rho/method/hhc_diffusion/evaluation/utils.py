import os.path as osp
import torch
import torch
import os

from tqdm import tqdm
import numpy as np
import pickle
import cv2
import json
import sys
import trimesh

from llib_rho.models.build import build_model
from llib_rho.logging.logger import Logger
from llib_rho.bodymodels.build import build_bodymodel
from llib_rho.cameras.build import build_camera
from llib_rho.visualization.renderer import Pytorch3dRenderer
from llib_rho.models.diffusion.build import build_diffusion
from llib_rho.method.hhc_diffusion.train_module import TrainModule


from torch.utils.data import DataLoader
from llib_rho.data.build import build_datasets


from llib_rho.visualization.diffusion_eval import create_hist_of_errors_gif, save_gif
from llib_rho.utils.io.utils import dict_to_device
from llib_rho.utils.metrics.angles import angle_error
from llib_rho.utils.threed.distance import pcl_pcl_pairwise_distance
from llib_rho.utils.threed.intersection import winding_numbers
from llib_rho.utils.metrics.contact import MaxIntersection
from llib_rho.utils.metrics.diffusion import GenerationGraph
from llib_rho.utils.threed.conversion import axis_angle_to_rotation6d
from pytorch3d.transforms import axis_angle_to_matrix
import matplotlib.pyplot as plt


def setup_diffusion_module(cfg, cmd_args):
    # create logger and write vconfig file
    cfg.logging.logger = 'tensorbaord' # if not set, config stored in wandb might be overwritten
    logger = Logger(cfg)

    # build regressor used to predict diffusion params
    regressor = build_model(cfg.model.regressor).to(cfg.device)
    latest_checkpoint_path = logger.get_latest_checkpoint()
    if latest_checkpoint_path is not None:
        checkpoint_folder = "/".join(latest_checkpoint_path.split("/")[:-1])
        if cmd_args.checkpoint_name == "latest":
            checkpoint_path = latest_checkpoint_path
        elif cmd_args.checkpoint_name == "best":
            checkpoint_fn = [x for x in os.listdir(checkpoint_folder) if "best" in x][0]
            checkpoint_path = osp.join(checkpoint_folder, checkpoint_fn)
        else:
            checkpoint_path = osp.join(checkpoint_folder, cmd_args.checkpoint_name)
    else:  # use the checkpoint path directly
        assert os.path.isfile(cmd_args.checkpoint_name)
        checkpoint_path = cmd_args.checkpoint_name

    checkpoint = torch.load(checkpoint_path)
    regressor.load_state_dict(checkpoint["model"], strict=False)
    regressor.eval()

    # build diffusion process
    diffusion = build_diffusion(**cfg.model.diffusion)

    # load body models for human1 and human2
    body_model = build_bodymodel(
        cfg=cfg.body_model, batch_size=cfg.batch_size, device=cfg.device,
    )

    # create optimizer module
    diffusion_module = TrainModule(
        cfg=cfg,
        train_dataset=None,
        val_dataset=None,
        diffusion=diffusion,
        model=regressor,
        criterion=None,
        evaluator=None,
        body_model=body_model,
        renderer=None,
    ).to(cfg.device)

    return diffusion_module


def setup_gt_dataset(cfg, dataset_name='behave', shuffle=False, drop_last=False):
    # create datasets
    
    train_dataset, val_dataset = build_datasets(
        datasets_cfg=cfg.datasets,
        body_model_type=cfg.body_model.type,  # necessary to load the correct contact maps
    )

    ds = val_dataset[dataset_name] #train_dataset if train_dataset is not None else val_dataset
    
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=False,
        drop_last=drop_last,
    )
    return ds, loader


def sample_unconditional(diffusion_module, timesteps, obj_embeddings, mesh, log_steps, **kwargs):
    """Sample from the diffusion model without conditioning starting from noise"""

    bs = diffusion_module.cfg.batch_size
    device = diffusion_module.cfg.device

    # create random noise to start with
    noise = {
        "orient": torch.randn([bs, 1, diffusion_module.orient_dim]).to(device),
        "pose": torch.randn([bs, 1, diffusion_module.pose_dim]).to(device),
        "shape": torch.randn([bs, 1, diffusion_module.shape_dim]).to(device),
        "transl": torch.randn([bs, 1, diffusion_module.transl_dim]).to(device),
        "orient_obj": torch.randn([bs, 1, diffusion_module.orient_obj_dim]).to(device),
        "transl_obj": torch.randn([bs, 1, diffusion_module.transl_obj_dim]).to(device),
    }
    gender = torch.nn.functional.one_hot(torch.randint(0, 2, (bs,)).to(device), num_classes=2)
    # noise_smpls = diffusion_module.get_smpl(noise, gender)

    noise_params = diffusion_module.split_humans(noise)

    curr_obj_embeddings = torch.from_numpy(obj_embeddings['chairblack']).float().to(device)
    curr_obj_embeddings = curr_obj_embeddings.expand(bs, -1)

    curr_obj_vertices = torch.from_numpy(mesh.vertices.astype(np.float32)).to(device)
    curr_obj_faces = torch.from_numpy(mesh.faces.astype(np.float32)).to(device)

    # run diffusion sampling loop
    eta = kwargs.get("eta", 0.0)
    x_ts, x_starts, obj_ts, obj_starts = diffusion_module.sampling_loop(
        x=noise_params, y={}, ts=timesteps, obj=curr_obj_embeddings, gender=gender,
        obj_vertices=curr_obj_vertices, log_steps=log_steps, return_latent_vec=False, eta=eta
    )

    return x_ts, x_starts, obj_ts, obj_starts, curr_obj_faces

def sample_unconditional_latent(diffusion_module, timesteps, log_steps, **kwargs):
    """Sample from the diffusion model without conditioning starting from noise"""

    bs = diffusion_module.cfg.batch_size
    device = diffusion_module.cfg.device

    # create random noise to start with
    noise = {
        "orient": torch.randn([bs, 2, diffusion_module.orient_dim]).to(device),
        "pose": torch.randn([bs, 2, diffusion_module.pose_dim]).to(device),
        "shape": torch.randn([bs, 2, diffusion_module.shape_dim]).to(device),
        "transl": torch.randn([bs, 2, diffusion_module.transl_dim]).to(device),
    }
    noise_smpls = diffusion_module.get_smpl(noise)

    # run diffusion sampling loop
    eta = kwargs.get("eta", 0.0)
    x_ts, x_starts, x_latent = diffusion_module.sampling_loop(
        x=noise_smpls, y={}, ts=timesteps, log_steps=log_steps, return_latent_vec=True, eta=eta
    )

    return x_ts, x_starts, x_latent

def sample_conditional_with_inpainting(
    diffusion_module, timesteps, log_steps, conditions, condition_params, **kwargs
):
    bs = diffusion_module.cfg.batch_size
    device = diffusion_module.cfg.device

    # create random noise to start with
    noise = {
        "orient": torch.randn([bs, 2, diffusion_module.orient_dim]).to(device),
        "pose": torch.randn([bs, 2, diffusion_module.pose_dim]).to(device),
        "shape": torch.randn([bs, 2, diffusion_module.shape_dim]).to(device),
        "transl": torch.randn([bs, 2, diffusion_module.transl_dim]).to(device),
    }

    noise_smpls = diffusion_module.get_smpl(noise)

    guidance = diffusion_module.get_guidance_params(
        conditions["values"],
        guidance_param_nc=0.0,
        guidance_all_nc=0.0,
        guidance_no_nc=1.0,
        guidance_params=condition_params,
    )
    guidance = move_to(guidance, device)

    # run diffusion sampling loop
    eta = kwargs.get("eta", 0.0)
    x_ts, x_starts = diffusion_module.sampling_loop(
        x=noise_smpls, y=guidance, ts=timesteps, inpaint=None, log_steps=log_steps, eta=eta
    )

    return x_ts, x_starts


def sample_unconditional_with_inpainting(
    diffusion_module, timesteps, log_steps, conditions, **kwargs
):
    bs = diffusion_module.cfg.batch_size
    device = diffusion_module.cfg.device

    # create random noise to start with
    noise = {
        "orient": torch.randn([bs, 2, diffusion_module.orient_dim]).to(device),
        "pose": torch.randn([bs, 2, diffusion_module.pose_dim]).to(device),
        "shape": torch.randn([bs, 2, diffusion_module.shape_dim]).to(device),
        "transl": torch.randn([bs, 2, diffusion_module.transl_dim]).to(device),
    }
    conditions = move_to(conditions, device)

    # do the inpainting
    for k, mm in conditions["mask"].items():
        noise[k][mm] = conditions["values"][k][mm]

    noise_smpls = diffusion_module.get_smpl(noise)

    # run diffusion sampling loop
    eta = kwargs.get("eta", 0.0)
    x_ts, x_starts = diffusion_module.sampling_loop(
        x=noise_smpls, y={}, ts=timesteps, inpaint=conditions, log_steps=log_steps, eta=eta
    )

    return x_ts, x_starts


def batch_sample(
    num_batches,
    diffusion_module,
    timesteps,
    log_steps,
    conditions=None,
    condition_params=None,
    sampling_function=True,
    return_latent_vec=False,
    eta=0.0,
):
    """Wrapper around sampling to sample multiple times and concatenate the results"""

    def add_timesteps(current, current_obj, t, final):
        if t not in final.keys():
            final[t] = {}
            final[t]["vertices"] = current.vertices
            final[t]["vertices_obj"] = current_obj
        else:
            final[t]["vertices"] = torch.cat((final[t]["vertices"], current.vertices), dim=0)
            final[t]["vertices_obj"] = torch.cat((final[t]["vertices_obj"], current_obj), dim=0)
        return final

    x_obj_ts_all, x_obj_starts_all, x_latents_all = {}, {}, {}
    if conditions is not None:
        num_batches = len(conditions)

    read_obj_embeddings = np.load("/home/shubhikg/data_subset/train/obj_embeddings.npy", allow_pickle=True)
    obj_embeddings = {}
    for key in read_obj_embeddings.item().keys():
            obj_embeddings[key] = np.expand_dims(read_obj_embeddings.item()[key], axis=0).astype(np.float32)

    with open("/home/shubhikg/data_subset/train/ref_hoi.pkl", "rb") as f:
            x = pickle.load(f)
            data = x['templates']['chairblack']
            verts = data['verts']
            faces = data['faces']
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    for bi in tqdm(range(num_batches)):
        cond = conditions[bi] if conditions is not None else None
        sampling_output = sampling_function(
            diffusion_module=diffusion_module,
            timesteps=timesteps,
            conditions=cond,
            condition_params=condition_params,
            obj_embeddings=obj_embeddings,
            mesh=mesh,
            log_steps=log_steps,
            eta=eta,
        )
        if return_latent_vec:
            x_ts, x_starts, obj_ts, obj_starts, obj_face_tensor, x_latents = sampling_output
        else:
            x_ts, x_starts, obj_ts, obj_starts, obj_face_tensor = sampling_output

        for t in x_ts.keys():
            x_obj_ts_all = add_timesteps(x_ts[t], obj_ts[t], t, x_obj_ts_all)
            x_obj_starts_all = add_timesteps(x_starts[t], obj_starts[t], t, x_obj_starts_all)
        if return_latent_vec:
            for t in x_latents.keys():
                if return_latent_vec:
                    if t not in x_latents_all.keys():
                        x_latents_all[t] = x_latents[t]
                    else:
                        x_latents_all[t] = torch.cat([x_latents_all[t], x_latents[t]], dim=0)

    if return_latent_vec:
        return x_obj_ts_all, x_obj_starts_all, obj_face_tensor, x_latents_all
    else:
        return x_obj_ts_all, x_obj_starts_all, obj_face_tensor


def move_to(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [move_to(x, device) for x in obj]
    return obj
