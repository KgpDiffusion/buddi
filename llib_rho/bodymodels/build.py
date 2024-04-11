import os.path as osp
import smplx
from llib_rho.bodymodels.build_smpl_family import smpl_cfg_to_args
from .utils import *
from llib_rho.bodymodels.smplx_ours import SMPLX_Ours
from llib_rho.bodymodels.smplh_ours import SMPLH_Ours

def build_joint_mapper(joint_mapper_type, joint_mapper_cfg):
    """ 
    Build joint mapper.
    Parameters
    ----------
    joint_mapper_type: str
        type of joint mapper (smpl_to_openpose)
    joint_mapper_cfg: cfg
        config file (see defaults)
    """
    if joint_mapper_type == 'smpl_to_openpose':
        cfg = joint_mapper_cfg.smpl_to_openpose
        joint_map = smpl_to_openpose(**cfg)
    else:
        raise NotImplementedError

    joint_mapper = JointMapper(joint_map)
    
    return joint_mapper

def build_bodymodel(
    cfg, 
    bodymodel_type=None, 
    batch_size=1, 
    device='cuda'
):
    """
    Build SMPL model. If cfg.age == 'adult', build SMPL-X model and if
    cfg.age == 'kid', build SMPL-A model.
    Parameters
    ----------
    cfg: cfg
        config file of body_model
    bodymodel_type: str, optional
        type of body model (smpl, smplh, smplx)
    batch_size: int, optional
        batch size
    device: str, optional
        device to use
    """

    if bodymodel_type is None:
        bodymodel_type = cfg.type

    model_cfg = eval(f'cfg.{bodymodel_type}')

    if bodymodel_type in ['smpl', 'smplh', 'smplx']:
        # transform smpl config parameters to dict
        smpl_args = smpl_cfg_to_args(cfg, batch_size)

        # create joint mapper
        joint_mapper_cfg = model_cfg.init.joint_mapper
        if joint_mapper_cfg.use:
            joint_mapper = build_joint_mapper(
                joint_mapper_cfg.type, joint_mapper_cfg)
            smpl_args['joint_mapper'] = joint_mapper
            if eval(f'joint_mapper_cfg.{joint_mapper_cfg.type}.use_face_contour'):
                smpl_args['use_face_contour'] = True 

        # create smpl model
        if smpl_args['age'] == 'adult':
            mt = smpl_args.pop('model_type')
            mp = smpl_args.pop('model_path')
            smpl_args['model_path'] = osp.join(mp, mt)
            if bodymodel_type == 'smplx':
                # Same male and female body model as we load neutral body model here.
                body_model = [SMPLX_Ours(**smpl_args).to(device)]*2
            elif bodymodel_type == 'smplh':
                smpl_args['gender'] = 'male'
                male_body_model = SMPLH_Ours(**smpl_args).to(device)
                smpl_args['gender'] = 'female'
                female_body_model = SMPLH_Ours(**smpl_args).to(device)
                body_model = [male_body_model, female_body_model]
            else:
                raise NotImplementedError   
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return body_model 
