from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance

from .utils import *


class DanceAE_Metric(Metric):
    full_state_update = True

    def __init__(self, dist_sync_on_step=True, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "DanceAE_Metric scores"

        self.metrics = ["rot_l1loss", "rot_smoothl1loss", "xyz_l1loss"]
        self.add_state("rot_l1loss",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("rot_smoothl1loss",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("xyz_l1loss",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")

        # chached batches
        self.add_state("motion_rst", default=[], dist_reduce_fx=None)
        self.add_state("motion_ref", default=[], dist_reduce_fx=None)
        self.add_state("xyz_rst", default=[], dist_reduce_fx=None)
        self.add_state("xyz_ref", default=[], dist_reduce_fx=None)
        self.SmoothL1Loss = torch.nn.SmoothL1Loss(reduction='mean')
        self.L1Loss = torch.nn.L1Loss()

    def compute(self, sanity_flag):
        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # if in sanity check stage then jump
        if sanity_flag:
            return metrics

        # cat all embeddings
        motion_ref = torch.cat(self.motion_ref, axis=0)
        motion_rst = torch.cat(self.motion_rst, axis=0)
        xyz_ref = torch.cat(self.xyz_ref, axis=0)
        xyz_rst = torch.cat(self.xyz_rst, axis=0)
        metrics['rot_smoothl1loss'] = self.SmoothL1Loss(motion_ref, motion_rst)
        metrics['rot_l1loss'] = self.L1Loss(motion_ref, motion_rst)
        metrics['xyz_l1loss'] = self.L1Loss(xyz_ref, xyz_rst)

        return {**metrics}

    def update(
        self,
        motion_ref: Tensor,
        motion_rst: Tensor,
        xyz_ref: Tensor,
        xyz_rst: Tensor,
    ):

        # store all motion
        self.motion_ref.append(motion_ref)
        self.motion_rst.append(motion_rst)
        self.xyz_ref.append(xyz_ref)
        self.xyz_rst.append(xyz_rst)

        
        
        
        
        
