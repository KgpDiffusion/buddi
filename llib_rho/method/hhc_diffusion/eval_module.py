import torch.nn as nn
import numpy as np
import torch
from llib_rho.utils.metrics.build import build_metric

class EvalModule(nn.Module):
    def __init__(
        self,
        eval_cfgs,
        body_model_type='smplx'
    ):
        super(EvalModule, self).__init__()

        self.cfg = eval_cfgs

        self.per_person_metrics = self.cfg.per_person_metrics # ['v2v', 'mpjpe', 'pa_mpjpe']
        self.num_humans = 1

        # metrics for reconstruction        
        self.metrics = self.cfg.metrics

        self.all_metrics = self.metrics

        self.accumulator = {} # dict to store all the metrics
        self._init_metrics()     

        self.tb_output = None
    
    def reset(self):
        self.accumulator = {}
        self._init_metrics()

    def _init_metrics(self):
        # add member variables for each metric
        for name in self.all_metrics:
            metric = build_metric(self.cfg[name])
            setattr(self, f'{name}_func', metric)   

        # fill accumulator dict
        for name in self.all_metrics:
            if name in self.per_person_metrics:
                for i in range(self.num_humans): 
                    self.accumulator[f'{name}_{i}'] = np.array([])
            else:
                self.accumulator[f'{name}_hum'] = np.array([])
                self.accumulator[f'{name}_obj'] = np.array([])       

        # add total_loss to accumulator
        self.accumulator['total_loss'] = np.array([])       

    def accumulate(self, metric_name, array):
        curr_array = self.accumulator[metric_name]
        self.accumulator[metric_name] = np.concatenate((curr_array, array), axis=0) \
            if curr_array.size else array


    def forward(
        self, 
        est_smpl, # estimated smpl
        tar_smpl, # target smpl
        est_params, # estimated parameters
        tar_params, # target parameters
        human_face,
        object_face,
        t_type=''
    ):  
        est_smpl = [est_smpl]
        tar_smpl = [tar_smpl]
        for name in self.metrics:

            metric = getattr(self, f'{name}_func') # metric class / function
            if name in self.per_person_metrics:
                for i in range(self.num_humans):
                    if name == 'v2v':
                        in_points, tar_points = est_smpl[i].vertices, tar_smpl[i].vertices
                    elif name in ['mpjpe', 'pa_mpjpe']:
                        in_points, tar_points = est_smpl[i].joints, tar_smpl[i].joints                   
                    errors = metric(in_points.detach().cpu().numpy(), tar_points.detach().cpu().numpy())
                    self.accumulate(f'{name}_{i}', errors)
            else:
                if name == 'pairwise_pa_chamfer':
                    in_points = torch.cat([est_smpl[i].vertices for i in range(self.num_humans)], dim=1)
                    tar_points = torch.cat([tar_smpl[i].vertices for i in range(self.num_humans)], dim=1)
                    errors, errors_obj = metric(in_points.detach().cpu().numpy(), tar_points.detach().cpu().numpy(),
                                                est_params, tar_params, human_face, object_face)
                    self.accumulate(f'{name}_hum', errors)
                    self.accumulate(f'{name}_obj', errors_obj)
                
    def final_accumulate_step(self):
        # add metric to tensorboard
        accumulator_keys = list(self.accumulator.keys())
        for key in accumulator_keys:
            self.accumulator[key] = torch.tensor(self.accumulator[key]).mean()