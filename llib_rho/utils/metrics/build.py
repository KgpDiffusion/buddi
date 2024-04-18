from llib_rho.utils.metrics.alignment import * 
from llib_rho.utils.metrics.points import PointError
from llib_rho.utils.metrics.chamfer import ChamferError

def build_metric(cfg):
    if cfg.name == 'PointError':
        return PointError(**cfg)
    if cfg.name == 'ChamferError':
        return ChamferError(**cfg)
    else:
        raise ValueError(f'Unknown metric type: {cfg.name}')