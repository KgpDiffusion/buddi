from smplx import SMPLH
import torch
import torch.nn as nn
from typing import Optional
from collections import namedtuple

TensorOutput = namedtuple('TensorOutput',
                          ['vertices', 'joints', 'betas', 
                          'global_orient', 'body_pose', 'left_hand_pose',
                           'right_hand_pose', 'transl'])

class SMPLH_Ours(SMPLH):
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
        super(SMPLH_Ours, self).__init__(**kwargs)
        self.kwargs = kwargs

    def name(self) -> str:
        return 'SMPL-H'
    
    def clone(self):
        # Create a new instance and clone necessary attributes
        new_instance = SMPLH_Ours(**self.kwargs)
        return new_instance

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
        body = super(SMPLH_Ours, self).forward(
            betas=betas, 
            transl=transl,
            global_orient=global_orient,
            body_pose=body_pose,
            **kwargs
        )

        output = TensorOutput(vertices=body.vertices,
                             joints=body.joints,
                             betas=betas,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             left_hand_pose=body.left_hand_pose,
                             right_hand_pose=body.right_hand_pose,
                             transl=transl)

        return output