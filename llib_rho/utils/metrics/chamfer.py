import torch 
import torch.nn as nn
import numpy as np
from typing import NewType, List, Union, Tuple, Optional
from .alignment import build_alignment 
import trimesh
from sklearn.neighbors import NearestNeighbors

class ChamferError(nn.Module):
    def __init__(
        self,
        name,
        alignment,
    ):
        super(ChamferError, self).__init__()

        self.align = build_alignment(alignment)

        self.sample_num = 6000
        self.sample_obj_num = 10000

    def forward(self, input_points, target_points, input_obj_points, target_obj_points, human_face, obj_face):
        error = self.chamfer_error(input_points, target_points, input_obj_points, target_obj_points, human_face, obj_face)
        return error

    def chamfer_error(self, input_points, target_points, input_obj_points, target_obj_points, human_face, obj_face):
        ''' Calculate chamfer error
        Parameters
        ----------
            input_points: numpy.array, BxPx3
                The estimated points
            target_points: numpy.array, BxPx3
                The ground truth points
        '''
        bs = input_points.shape[0]
        tot_smpl_error = []
        tot_obj_error = []
        for i in range(bs):
            curr_target_points = target_points[i]
            curr_input_points = input_points[i]
            curr_obj_target_points = target_obj_points[i].cpu().numpy()
            curr_obj_input_points = input_obj_points[i].cpu().numpy()
            len_human_pts = curr_input_points.shape[0]

            comb_input_points = np.concatenate((curr_input_points, curr_obj_input_points), axis=0)
            comb_target_points = np.concatenate((curr_target_points, curr_obj_target_points), axis=0)
            comb_input_points, _ = self.align(comb_input_points, comb_target_points)
            curr_input_points = comb_input_points[0, :len_human_pts]
            curr_obj_input_points = comb_input_points[0, len_human_pts:]            

            smpl_samples = [self.surface_sampling(v, human_face.cpu().numpy(), self.sample_num) for v in [curr_target_points, curr_input_points]]
            obj_samples = [self.surface_sampling(v, obj_face[i].cpu().numpy(), self.sample_obj_num) for v in [curr_obj_target_points, curr_obj_input_points]]
            err_smpl = self.chamfer_distance(smpl_samples[0], smpl_samples[1])
            err_obj = self.chamfer_distance(obj_samples[0], obj_samples[1])
            tot_smpl_error.append(err_smpl)
            tot_obj_error.append(err_obj)
        return np.array(tot_smpl_error), np.array(tot_obj_error)

    def surface_sampling(self, verts, faces, sample_num):
        "sample points on the surface"
        m = self.to_trimesh(verts, faces)
        points = m.sample(sample_num)
        return points

    def to_trimesh(self, verts, faces):
        "psbody mesh to trimesh"
        trim = trimesh.Trimesh(verts, faces, process=False)
        return trim

    def chamfer_distance(self, x, y, metric='l2'):
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
        return chamfer_dist
