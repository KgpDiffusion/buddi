from smplx import SMPLX
import torch
import torch.nn as nn
from typing import Optional
from collections import namedtuple

TensorOutput = namedtuple('TensorOutput',
                          ['vertices', 'joints', 'betas',
                          'expression', 
                          'global_orient', 'body_pose', 'left_hand_pose',
                           'right_hand_pose', 'jaw_pose', 'transl', 'full_pose',
                           'v_shaped'])

class SMPLX_Ours(SMPLX):
    def __init__(self,
        **kwargs
    ):
        """ 
        SMPL-XA Model, which extends SMPL-X to children and adults.
        Parameters
        ----------
        kwargs:
            Same as SMPL-X   
        """
        super(SMPLX_Ours, self).__init__(**kwargs)

    def name(self) -> str:
        return 'SMPL-X'

    def forward(
        self,
        betas: Optional[torch.Tensor] = None,
        transl: Optional[torch.Tensor] = None,
        global_orient: Optional[torch.Tensor] = None,
        body_pose: Optional[torch.Tensor] = None,
        **kwargs
    ):

        betas = betas if betas is not None else self.betas
        transl = transl if transl is not None else self.transl

        body_pose = body_pose if body_pose is not None else self.body_pose
        
        global_orient = global_orient if global_orient is not None else self.global_orient

        # Figure out why smpla is passed to super SMPLX
        body = super(SMPLX_Ours, self).forward(
            betas=betas, 
            transl=transl,
            global_orient=global_orient,
            body_pose=body_pose,
            **kwargs
        )

        output = TensorOutput(vertices=body.vertices,
                             joints=body.joints,
                             betas=betas,
                             expression=body.expression,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             left_hand_pose=body.left_hand_pose,
                             right_hand_pose=body.right_hand_pose,
                             jaw_pose=body.jaw_pose,
                             v_shaped=body.v_shaped,
                             transl=transl,
                             full_pose=body.full_pose)

        return output