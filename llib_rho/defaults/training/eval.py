import os
import os.path as osp
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import List, Tuple
from dataclasses import dataclass, field

@dataclass
class PointError:
    name: str = 'PointError'
    alignment: str = 'root'

@dataclass
class ChamferError:
    name: str = 'ChamferError'
    alignment: str = 'procrustes'

@dataclass 
class Evaluation:
    checkpoint_metric: str = 'total_loss' # this is the value which will be added to the checkpoint filename

    # the metrics computed during validation and evaluation
    metrics: List[str] = field(default_factory=lambda: []) #['v2v', 'mpjpe', 'pa_mpjpe', 'pairwise_pa_chamfer'])
    per_person_metrics: List[str] = field(default_factory=lambda: ['v2v', 'mpjpe', 'pa_mpjpe'])

    # metrics
    v2v: PointError = PointError(alignment='root')
    mpjpe: PointError = PointError(alignment='root')
    pa_mpjpe: PointError = PointError(alignment='procrustes')
    pairwise_pa_chamfer: ChamferError = ChamferError(alignment='procrustes')

conf = OmegaConf.structured(Evaluation)