from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance

from .utils import *


class DanceDiffuse_Metric(Metric):
    full_state_update = True

    def __init__(self, cfg, dist_sync_on_step=True, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.name = "DanceAE_Metric scores"

        # diffuse_l1loss is diffuse generated key motion loss, corresponding to rot_l1loss
        self.metrics = ["value_mseLoss", "noise_mseLoss", "prior_mseLoss", "diffuse_l1loss"]
        self.add_state("value_mseLoss",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("noise_mseLoss",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("prior_mseLoss",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("diffuse_l1loss",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")

        # chached batches
        self.add_state("noise", default=[], dist_reduce_fx=None)
        self.add_state("noise_pred", default=[], dist_reduce_fx=None)
        self.add_state("noise_prior", default=[], dist_reduce_fx=None)
        self.add_state("noise_prior_pred", default=[], dist_reduce_fx=None)
        self.add_state("pred", default=[], dist_reduce_fx=None)
        self.add_state("latent", default=[], dist_reduce_fx=None)
        self.add_state("lat_t", default=[], dist_reduce_fx=None)
        self.add_state("lat_m", default=[], dist_reduce_fx=None)
        # self.SmoothL1Loss = torch.nn.SmoothL1Loss(reduction='mean')
        self.L1Loss = torch.nn.L1Loss()
        self.MseLoss = torch.nn.MSELoss()

    def compute(self, sanity_flag):
        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # if in sanity check stage then jump
        if sanity_flag:
            return metrics

        # cat all embeddings
        if self.cfg.TRAIN.ABLATION.PREDICT_EPSILON:
            noise = torch.cat(self.noise, axis=0)
            noise_pred = torch.cat(self.noise_pred, axis=0)
            metrics['noise_mseLoss'] = self.MseLoss(noise_pred, noise)
        else:       
            pred = torch.cat(self.pred, axis=0)
            latent = torch.cat(self.latent, axis=0)
            metrics['value_mseLoss'] = self.MseLoss(pred, latent)
        
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_prior = torch.cat(self.noise_prior, axis=0)
            noise_prior_pred = torch.cat(self.noise_prior_pred, axis=0)
            metrics['prior_mseLoss'] = self.MseLoss(noise_prior_pred, noise_prior)
            
        lat_t = torch.cat(self.lat_t, axis=0)
        lat_m = torch.cat(self.lat_m, axis=0)
        print("lat_t.shape", lat_t.shape)
        print("lat_m.shape", lat_m.shape)
        metrics['diffuse_l1loss'] = self.L1Loss(lat_t, lat_m)

        return {**metrics}

    def update(
        self,
        noise: Tensor,
        noise_pred: Tensor,
        noise_prior: Tensor,
        noise_prior_pred: Tensor,
        pred: Tensor,
        latent: Tensor,
        lat_t: Tensor,
        lat_m: Tensor,
    ):

        # store all motion
        self.noise.append(noise)
        self.noise_pred.append(noise_pred)
        
        self.noise_prior.append(noise_prior)
        self.noise_prior_pred.append(noise_prior_pred)
        
        self.pred.append(pred)
        self.latent.append(latent)

        self.lat_t.append(lat_t)
        self.lat_m.append(lat_m)
        
        
        
        
        
