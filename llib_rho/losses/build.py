import torch.nn as nn
from .l2 import L2Loss


class Placeholder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        return None


def build_loss(loss_cfg, body_model_type='smplx'):
    loss_type = loss_cfg.type
    if loss_type == 'l2':
        loss = L2Loss(**loss_cfg)
    elif loss_type == '':
        loss = Placeholder()
    else:
        raise ValueError(f'Loss {loss_type} not implemented')
    return loss
