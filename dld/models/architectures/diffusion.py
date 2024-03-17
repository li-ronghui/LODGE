from ast import If
import copy
import os
import pickle
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from p_tqdm import p_map
from pytorch3d.transforms import (axis_angle_to_quaternion,
                                  quaternion_to_axis_angle)
from tqdm import tqdm

from dld.data.utils.preprocess import ax_from_6v, quat_slerp
from .utils import extract, make_beta_schedule
from dld.data.render_joints.utils.motion_process import recover_from_ric
from dld.data.render_joints.smplfk import ax_to_6v


def identity(t, *args, **kwargs):
    return t

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        cfg, 
        model,
        normalizer,
        horizon,
        repr_dim,
        smplx_model,
        n_timestep=1000,
        schedule="linear",
        loss_type="l1",
        clip_denoised=True,
        predict_epsilon=True,
        guidance_weight=3,
        use_p2=False,
        cond_drop_prob=0.2,
        dis_model=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.horizon = horizon
        self.transition_dim = repr_dim
        self.model = model
        self.normalizer = normalizer
        self.ema = EMA(0.9999)
        self.master_model = copy.deepcopy(self.model)
        if dis_model is not None:
            self.dis_model=dis_model
            self.master_model_dis = copy.deepcopy(self.dis_model)
        

        self.cond_drop_prob = cond_drop_prob

        # make a SMPL instance for FK module
        self.smplx_fk = smplx_model

        betas = torch.Tensor(
            make_beta_schedule(schedule=schedule, n_timestep=n_timestep)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timestep = int(n_timestep)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon      # 设置为不预测噪声，直接预测x

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.guidance_weight = guidance_weight

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # p2 weighting
        self.p2_loss_weight_k = 1
        self.p2_loss_weight_gamma = 0.5 if use_p2 else 0
        self.register_buffer(
            "p2_loss_weight",
            (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -self.p2_loss_weight_gamma,
        )

        ## get loss coefficients and initialize objective
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def model_predictions(self, x, cond, genre, t, weight=None, clip_x_start = False):
        weight = weight if weight is not None else self.guidance_weight
        model_output = self.model.guided_forward(x, cond, genre, t, weight)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, genre, t):
        # guidance clipping
        if t[0] > 1.0 * self.n_timestep:
            weight = min(self.guidance_weight, 0)
        elif t[0] < 0.1 * self.n_timestep:
            weight = min(self.guidance_weight, 1)
        else:
            weight = self.guidance_weight

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.model.guided_forward(x, cond, genre, t, weight)
        )

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, cond, genre, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, cond=cond, genre=genre, t=t
        )
        noise = torch.randn_like(model_mean)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(noise.shape) - 1))
        )
        x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_out, x_start

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        # default to diffusion over whole timescale
        start_point = self.n_timestep if start_point is None else start_point
        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)

        if return_diffusion:
            diffusion = [x]

        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, _ = self.p_sample(x, cond, timesteps)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x
        
    @torch.no_grad()
    def ddim_sample(self, shape, cond, genre, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        x_start = None

        # for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):        # 
        for time, time_next in time_pairs: 
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, genre, time_cond, clip_x_start = self.clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
        return x
    
    @torch.no_grad()
    def long_ddim_sample(self, shape, cond, genre, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1
        
        if batch == 1:
            return self.ddim_sample(shape, cond, genre)

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        weights = np.clip(np.linspace(0, self.guidance_weight * 2, sampling_timesteps), None, self.guidance_weight)
        time_pairs = list(zip(times[:-1], times[1:], weights)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)
        
        assert batch > 1
        assert x.shape[1] % 2 == 0
        half = x.shape[1] // 2

        x_start = None

        for time, time_next, weight in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, genre, time_cond, weight=weight, clip_x_start = self.clip_denoised) 

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            if time > 0:
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:]
        return x

    @torch.no_grad()
    def inpaint_loop(
        self,
        shape,
        cond,
        genre,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, genre, timesteps)
            # enforce constraint between each denoising step
            value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
            x = value_ * mask + (1.0 - mask) * x

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x
        
    @torch.no_grad()
    def inpaint_soft_loop(
        self,
        shape,
        cond,
        genre,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels

        start_point = self.n_timestep if start_point is None else start_point
        print("start_point", start_point)
        for i in tqdm(reversed(range(0, start_point))):
            if int(i) > start_point*(1-self.cfg.soft):  # self.opt.hint:        # 数字越小控制soft hint效果越强
                # fill with i
                timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

                # sample x from step i to step i-1
                x, _ = self.p_sample(x, cond, genre, timesteps)      # p_sample
                # enforce constraint between each denoising step
                value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
                x = value_ * mask + (1.0 - mask) * x
                x[:, :4, :] = value[:, :4, :]  * mask[:, :4, :]  + (1.0 - mask[:, :4, :] ) * x[:, :4, :] 
                x[:, -4:, :] = value[:, -4:, :]  * mask[:, -4:, :]  + (1.0 - mask[:, -4:, :] ) * x[:, -4:, :]
            else:
                # fill with i
                timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
                # sample x from step i to step i-1
                x, _ = self.p_sample(x, cond, genre, timesteps)
                # enforce constraint between each denoising step
                # value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
                # x = value_ * mask + (1.0 - mask) * x
                x[:, :4, :] = value[:, :4, :]  * mask[:, :4, :]  + (1.0 - mask[:, :4, :] ) * x[:, :4, :] 
                x[:, -4:, :] = value[:, -4:, :]  * mask[:, -4:, :]  + (1.0 - mask[:, -4:, :] ) * x[:, -4:, :]
            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def long_inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        assert x.shape[1] % 2 == 0
        if batch_size == 1:
            # there's no continuation to do, just do normal
            return self.p_sample_loop(
                shape,
                cond,
                noise=noise,
                constraint=constraint,
                return_diffusion=return_diffusion,
                start_point=start_point,
            )
        assert batch_size > 1
        half = x.shape[1] // 2

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, timesteps)
            # enforce constraint between each denoising step
            if i > 0:
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:] 

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x
        

    @torch.no_grad()
    def inpaint_soft_ddim(
        self,
        shape,
        cond,
        genre,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        # device = self.betas.device
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1
        if batch == 1:
            return self.ddim_sample(shape, cond)
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        weights = np.clip(np.linspace(0, self.guidance_weight * 2, sampling_timesteps), None, self.guidance_weight)
        time_pairs = list(zip(times[:-1], times[1:], weights)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        assert batch > 1
        assert x.shape[1] % 2 == 0
        half = x.shape[1] // 2
        x_start = None

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels

        for time, time_next, weight in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, genre, time_cond, weight=weight, clip_x_start = self.clip_denoised) 

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            if time > 50*(1-self.cfg.soft):
                x = value * mask + (1.0 - mask) * x
            x[:, :4, :] = value[:, :4, :]  * mask[:, :4, :]  + (1.0 - mask[:, :4, :] ) * x[:, :4, :] 
            x[:, -4:, :] = value[:, -4:, :]  * mask[:, -4:, :]  + (1.0 - mask[:, -4:, :] ) * x[:, -4:, :]
        return x

    @torch.no_grad()
    def conditional_sample(
        self, shape, cond, constraint=None, *args, horizon=None, **kwargs
    ):
        """
            conditions : [ (time, state), ... ]
        """
        device = self.betas.device
        horizon = horizon or self.horizon

        return self.p_sample_loop(shape, cond, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample
    
    

    def smpl_loss(self, model_out_ori, target_ori, t):
        # full reconstruction loss
        loss = self.loss_fn(model_out_ori, target_ori, reduction="none")            # mse loss
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        # velocity loss
        target_v = target_ori[:, 1:, 4:] - target_ori[:, :-1, 4:]
        model_out_v = model_out_ori[:, 1:, 4:] - model_out_ori[:, :-1, 4:]
        v_loss = self.loss_fn(model_out_v, target_v, reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

        # FK loss
        b, s, c = model_out_ori.shape      
        # unnormalize
        if self.normalizer is not None:
            model_out_ori = self.normalizer.unnormalize(model_out_ori)
            target_ori = self.normalizer.unnormalize(target_ori)
        # split off contact from the rest
        model_contact, model_out = torch.split(model_out_ori, (4, model_out_ori.shape[2] - 4), dim=2)  # 前4维是foot contact
        target_contact, target = torch.split(target_ori, (4, target_ori.shape[2] - 4), dim=2)       # b, length, jxc
        
        # model_x为root position, model_q为rotation
        model_x = model_out[:, :, :3]   # root position
        model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))      # 以rot6d方式训练
        target_x = target[:, :, :3]
        target_q = ax_from_6v(target[:, :, 3:].reshape(b, s, -1, 6))
        b, s, nums, c_ = model_q.shape

        model_xp = self.smplx_fk.forward(model_q, model_x)
        target_xp = self.smplx_fk.forward(target_q, target_x)
        model_xp = model_xp.view(b, s, -1, 3)
        target_xp = target_xp.view(b, s, -1, 3) 

        fk_loss = self.loss_fn(model_xp, target_xp, reduction="none")
        fk_loss = reduce(fk_loss, "b ... -> b (...)", "mean")
        fk_loss = fk_loss * extract(self.p2_loss_weight, t, fk_loss.shape)

        # foot skate loss
        foot_idx = [7, 8, 10, 11]

        # find static indices consistent with model's own predictions
        static_idx = model_contact > 0.95  # N x S x 4
        model_feet = model_xp[:, :, foot_idx]  # foot positions (N, S, 4, 3)
        model_foot_v = torch.zeros_like(model_feet)
        model_foot_v[:, :-1] = (
            model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
        )  # (N, S-1, 4, 3)
        model_foot_v[~static_idx] = 0               # 不计算动态帧
        foot_loss = self.loss_fn(                   # 静态的foot，让它的速度为0
            model_foot_v, torch.zeros_like(model_foot_v), reduction="none"
        )
        foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")


        losses = (
            self.cfg.LOSS.LAMBDA_MSE  * loss.mean(),
            self.cfg.LOSS.LAMBDA_V    * v_loss.mean(),
            self.cfg.LOSS.LAMBDA_FK   * fk_loss.mean(),
            self.cfg.LOSS.LAMBDA_FOOT * foot_loss.mean(),
        )

        loss_dict = {}
        loss_dict.update({
                    "loss": sum(losses),
                    "mseloss": losses[0],
                    "Vloss": losses[1],
                    "fk_loss": losses[2],
                    "foot_loss": losses[3],
                })
        
        return loss_dict
    


    def smpl_loss_relative(self, model_out_ori, target_ori, t):
        # full reconstruction loss
        loss = self.loss_fn(model_out_ori, target_ori, reduction="none")            # mse loss
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)


        # velocity loss
        target_v = target_ori[:, 1:, 4:] - target_ori[:, :-1, 4:]
        model_out_v = model_out_ori[:, 1:, 4:] - model_out_ori[:, :-1, 4:]
        v_loss = self.loss_fn(model_out_v, target_v, reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

        # FK loss
        b, s, c = model_out_ori.shape      
        # unnormalize
        if self.normalizer is not None:
            model_out_ori = self.normalizer.unnormalize(model_out_ori)
            target_ori = self.normalizer.unnormalize(target_ori)
        # split off contact from the rest
        model_contact, model_out = torch.split(model_out_ori, (4, model_out_ori.shape[2] - 4), dim=2)  # 前4维是foot contact
        target_contact, target = torch.split(target_ori, (4, target_ori.shape[2] - 4), dim=2)       # b, length, jxc


        # model_x  is root position, model_q  is rotation
        model_x = model_out[:, :, :3]   # root position
        model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))      # 以rot6d方式训练
        target_x = target[:, :, :3]
        target_q = ax_from_6v(target[:, :, 3:].reshape(b, s, -1, 6))
        b, s, nums, c_ = model_q.shape

        model_xp = self.smplx_fk.forward(model_q, model_x)
        target_xp = self.smplx_fk.forward(target_q, target_x)
        model_xp = model_xp.view(b, s, -1, 3)
        target_xp = target_xp.view(b, s, -1, 3) 

        # model_xp[:,:,:,:] = model_xp[:,:,:,:] - model_xp[:,:,0:1,:]
        # target_xp[:,:,:,:] = target_xp[:,:,:,:] - target_xp[:,:,0:1,:]
        fk_loss = self.loss_fn(model_xp[:,:,:,:] - model_xp[:,:,0:1,:], target_xp[:,:,:,:] - target_xp[:,:,0:1,:], reduction="none")
        fk_loss = reduce(fk_loss, "b ... -> b (...)", "mean")
        fk_loss = fk_loss * extract(self.p2_loss_weight, t, fk_loss.shape)

        # foot skate loss
        foot_idx = [7, 8, 10, 11]       # l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx

        # find static indices consistent with model's own predictions
        static_idx = model_contact > 0.95  # N x S x 4
        model_feet = model_xp[:, :, foot_idx].clone()  # foot positions (N, S, 4, 3)
        model_foot_v = torch.zeros_like(model_feet)
        model_foot_v[:, :-1] = (
            model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
        )  # (N, S-1, 4, 3)
        model_foot_v_fc = model_foot_v.clone()
        model_foot_v[~static_idx] = 0               # 不计算动态帧
        foot_loss = self.loss_fn(                   # 静态的foot，让它的速度为0
            model_foot_v, torch.zeros_like(model_foot_v), reduction="none"
        )
        foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")

        if self.cfg.FineTune:
            if self.cfg.LOSS.LAMBDA_FC > 0.0:
                foot_y_ankle = model_xp[:, :, [7, 8], 1]
                foot_y_toe = model_xp[:, :, [10, 11], 1]
                ground_height = 0
                velocity_foot_normal = model_foot_v_fc[:, :, :, 1:2].clone()
                fc_mask_ankle = torch.unsqueeze((foot_y_ankle <= (0.08+ground_height)), dim=3).repeat(1, 1, 1, 3)     # ground height is 0
                fc_mask_teo = torch.unsqueeze((foot_y_toe <= (0.05+ground_height)), dim=3).repeat(1, 1, 1, 3)
                fc_mask_y = torch.cat([fc_mask_ankle, fc_mask_teo], dim=2)
                model_foot_v_fc[~fc_mask_y] = 0
                velocity_foot_normal_v = torch.zeros_like(velocity_foot_normal)
                velocity_foot_normal_v[:, :-1] = (velocity_foot_normal[:, 1:, :, :] - velocity_foot_normal[:, :-1, :, :]) 
                normal_mask = (velocity_foot_normal_v<0)
                model_foot_v_fc[:,:,:,1:2][~normal_mask] = 0        # double check here
                fc_loss = self.loss_fn(                   # 静态的foot，让它的速度为0
                    model_foot_v_fc, torch.zeros_like(model_foot_v_fc), reduction="none"
                )
                fc_loss = reduce(fc_loss, "b ... -> b (...)", "mean")
                fc_loss = fc_loss * extract(self.p2_loss_weight, t, fc_loss.shape)
                fc_loss = fc_loss.mean()
            else:
                fc_loss = 0.0


            model_x_v = model_x[:, 1:, :] - model_x[:, :-1, :]
            model_x_a = model_x_v[:, 1:, :] - model_x_v[:, :-1, :]
            target_x_v = target_x[:, 1:, :] - target_x[:, :-1, :]
            target_x_a = target_x_v[:, 1:, :] - target_x_v[:, :-1, :]

            # root_orient_loss = self.loss_fn(model_q[:,:,0,:] , target_q[:,:,0,:], reduction="none")
            # root_orient_loss = reduce(root_orient_loss, "b ... -> b (...)", "mean")
            # root_orient_loss = root_orient_loss * extract(self.p2_loss_weight, t, root_orient_loss.shape)

            transl_loss_v = self.loss_fn(model_x_v , target_x_v , reduction="none")
            transl_loss_v = reduce(transl_loss_v, "b ... -> b (...)", "mean")
            transl_loss_v = transl_loss_v * extract(self.p2_loss_weight, t, transl_loss_v.shape)
            transl_loss_a = self.loss_fn(model_x_a , target_x_a, reduction="none")
            transl_loss_a = reduce(transl_loss_v, "b ... -> b (...)", "mean")
            transl_loss_a = transl_loss_a * extract(self.p2_loss_weight, t, transl_loss_a.shape)
            trans_loss = 0.3*transl_loss_v.mean() + 0.6 * transl_loss_a.mean()

            if self.cfg.LOSS.LAMBDA_FK_V>0:
                model_xp_v = model_xp[:, 1:, :] - model_xp[:, :-1, :]
                target_xp_v = target_xp[:, 1:, :] - target_xp[:, :-1, :]
                fk_loss_v = self.loss_fn(model_xp_v , target_xp_v , reduction="none")
                fk_loss_v = reduce(fk_loss_v, "b ... -> b (...)", "mean")
                fk_loss_v = fk_loss_v * extract(self.p2_loss_weight, t, fk_loss_v.shape)
                fk_loss_v = fk_loss_v.mean()
            else:
                fk_loss_v = 0.0
            if self.cfg.LOSS.LAMBDA_FK_A>0:
                model_xp_a = model_xp_v[:, 1:, :] - model_xp_v[:, :-1, :]
                target_xp_a = target_xp_v[:, 1:, :] - target_xp_v[:, :-1, :]
                fk_loss_a = self.loss_fn(model_xp_a , target_xp_a, reduction="none")
                fk_loss_a = reduce(fk_loss_a, "b ... -> b (...)", "mean")
                fk_loss_a = fk_loss_a * extract(self.p2_loss_weight, t, fk_loss_a.shape)
                fk_loss_a = fk_loss_a.mean()
            else:
                fk_loss_a = 0.0



            losses = (
            self.cfg.LOSS.LAMBDA_MSE  * loss.mean(),
            self.cfg.LOSS.LAMBDA_V    * v_loss.mean(),
            self.cfg.LOSS.LAMBDA_FK   * fk_loss.mean(),
            self.cfg.LOSS.LAMBDA_FK_V   * fk_loss_v,
            self.cfg.LOSS.LAMBDA_FK_A   * fk_loss_a,
            self.cfg.LOSS.LAMBDA_FOOT * foot_loss.mean(),
            self.cfg.LOSS.LAMBDA_FC * fc_loss,
            self.cfg.LOSS.LAMBDA_TRANS * trans_loss,
            )
            loss_dict = {}
            loss_dict.update({
                        "loss": sum(losses),
                        "mseloss": losses[0],
                        "Vloss": losses[1],
                        "fk_loss": losses[2],
                        "fk_loss_v": losses[3],
                        "fk_loss_a": losses[4],
                        "foot_loss": losses[5],
                        "fc_loss": losses[6],
                        "trans_loss": losses[7],
                    })
        else:
            losses = (
            self.cfg.LOSS.LAMBDA_MSE  * loss.mean(),
            self.cfg.LOSS.LAMBDA_V    * v_loss.mean(),
            self.cfg.LOSS.LAMBDA_FK   * fk_loss.mean(),
            self.cfg.LOSS.LAMBDA_FOOT * foot_loss.mean(),
            )
            loss_dict = {}
            loss_dict.update({
                        "loss": sum(losses),
                        "mseloss": losses[0],
                        "Vloss": losses[1],
                        "fk_loss": losses[2],
                        "foot_loss": losses[3],
                    })
        
        return loss_dict

    def p_losses(self, x_start, cond, genre_id, t, isgen=False):
        noise = torch.randn_like(x_start)           
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)      # denoise x0

        if self.cfg.FineTune:           
            x_noisy[:, :4, :]  =  x_start[:, :4, :]
            x_noisy[:, -4:, :] =  x_start[:, -4:, :]
 
        x_recon = self.model(x_noisy, cond, genre_id, t, cond_drop_prob=self.cond_drop_prob)
        assert noise.shape == x_recon.shape

        model_out = x_recon
        if isgen:
            return model_out
        
        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        if self.cfg.LOSS.TYPE == 'smpl_loss':
            loss_dict = self.smpl_loss(model_out, target, t)
        elif self.cfg.LOSS.TYPE == 'smpl_loss_relative':
            loss_dict = self.smpl_loss_relative(model_out, target, t)
        else:
            raise("error of motion representation")
     
            

        if self.cfg.Discriminator:
            return loss_dict, model_out
        else:
            return loss_dict
            
            
    

    def loss(self, x, cond, genre_id, t_override=None, isgen=False):
        batch_size = len(x)
        if t_override is None:
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x.device).long()
        else:
            t = torch.full((batch_size,), t_override, device=x.device).long()
        return self.p_losses(x, cond, genre_id, t, isgen)

    def forward(self, x, cond, genre_id=None, t_override=None, isgen=False):
        return self.loss(x, cond, genre_id, t_override, isgen)

    def partial_denoise(self, x, cond, t):
        x_noisy = self.noise_to_t(x, t)
        return self.p_sample_loop(x.shape, cond, noise=x_noisy, start_point=t)

    def noise_to_t(self, x, timestep):
        batch_size = len(x)
        t = torch.full((batch_size,), timestep, device=x.device).long()
        return self.q_sample(x, t) if timestep > 0 else x
    
    def smplxmodel_fk(self, local_q, root_pos):      # input
        b, s, nums, c = local_q.shape
        local_q = local_q.view(b*s, -1)
        full_pose = self.smplx_model(
                    betas = torch.zeros([b*s, 10], device=local_q.device, dtype=torch.float32),
                    transl = root_pos.view(b*s, -1),        # global translation
                    global_orient = local_q[:, :3],
                    body_pose = local_q[:, 3:66],           # 21
                    jaw_pose = torch.zeros([b*s, 3], device=local_q.device, dtype=torch.float32),         # 1
                    leye_pose = torch.zeros([b*s,  3], device=local_q.device, dtype=torch.float32),        # 1
                    reye_pose= torch.zeros([b*s,  3], device=local_q.device, dtype=torch.float32),          # 1
                    left_hand_pose = local_q[:, 66:111],   # 15
                    right_hand_pose = local_q[:, 111:], # 15
                    expression = torch.zeros([b*s, 10], device=local_q.device, dtype=torch.float32),
                    return_verts = False
            )
        full_pose = full_pose.joints.view(b, s, -1, 3)   # b, s, 55, 3
        # full_q = full_q.view(b, s, nums, c_)
        # full_pose_52 = torch.concat((full_pose[:, :, :22, :], full_pose[:, :, 25:, :]), dim=2)
        
        return full_pose    #full_pose_52
        

    def render_sample(
        self,
        shape,
        cond,
        normalizer,
        epoch,
        render_out,
        fk_out=None,
        name=None,
        sound=True,
        mode="normal",
        noise=None,
        constraint=None,
        sound_folder="ood_sliced",
        start_point=None,
        render=True,
        genre=None,
        # do_normalize=False,
    ):
        if isinstance(shape, tuple):
            if mode == "inpaint":
                func_class = self.inpaint_loop
            elif mode == "inpaint_soft":
                func_class = self.inpaint_soft_loop
            elif mode == "inpaint_soft_ddim":
                func_class = self.inpaint_soft_ddim
            elif mode == "normal":
                func_class = self.ddim_sample
            elif mode == "long":
                func_class = self.long_ddim_sample
            else:
                assert False, "Unrecognized inference mode"
            samples = (
                func_class(
                    shape,
                    cond,
                    genre,
                    noise=noise,
                    constraint=constraint,
                    start_point=start_point,
                )
                .detach()
                .cpu()
            )
        else:
            samples = shape


        if self.cfg.FINEDANCE.nfeats != 263:
            if self.cfg.FINEDANCE.nfeats == 139 or self.cfg.FINEDANCE.nfeats==135:
                reshape_size = 66
            elif self.cfg.FINEDANCE.nfeats == 319 or self.cfg.FINEDANCE.nfeats==315:
                reshape_size = 156
            else:
                raise("error of nfeats")
            
            if self.cfg.Norm:
                samples = normalizer.unnormalize(samples)

            if samples.shape[2] == 319 or samples.shape[2] == 151 or samples.shape[2] == 139:                 # debug if samples.shape[2] == 151:    
                sample_contacts, samples = torch.split(
                    samples, (4, samples.shape[2] - 4), dim=2
                )
                sample_contacts = sample_contacts.to(cond.device)
            else:
                sample_contact = None
            # do the FK all at once
            b, s, c = samples.shape
            pos = samples[:, :, :3].to(cond.device)  # np.zeros((sample.shape[0], 3))
            q = samples[:, :, 3:].reshape(b, s, -1, 6)      # debug 24
            # go 6d to ax
            q = ax_from_6v(q).to(cond.device)
        else:
            print("nfeats is 263, do unnormalize")
            samples = normalizer.unnormalize(samples)
        
        if mode == "long":
            if self.cfg.FINEDANCE.nfeats == 263:
                pass
                print("need to be added")
                b,s,c = samples.shape
                assert c == 263
                assert s % 2 == 0
                half = s // 2
                pos = recover_from_ric(samples, 22).clone()
                # q = samples.clone()
                pos = pos.reshape(b,s,66)
                print("pos", pos.shape)

                if b > 1:
                    # if long mode, stitch position using linear interp
                    fade_out = torch.ones((1, s, 1)).to(pos.device)
                    fade_in = torch.ones((1, s, 1)).to(pos.device)
                    fade_out[:, half:, :] = torch.linspace(1, 0, half)[None, :, None].to(
                        pos.device
                    )
                    fade_in[:, :half, :] = torch.linspace(0, 1, half)[None, :, None].to(
                        pos.device
                    )
                    pos[:-1] *= fade_out
                    pos[1:] *= fade_in

                    full_pos = torch.zeros((s + half * (b - 1), 66)).to(pos.device)
                    idx = 0
                    for pos_slice in pos:
                        full_pos[idx : idx + s] += pos_slice
                        idx += half

                    full_pos = full_pos.unsqueeze(0)
                else:
                    full_pos = pos

                Path(fk_out).mkdir(parents=True, exist_ok=True)
                for num, (full_pos_one, filename) in enumerate(zip(full_pos, name)):
                    filename = os.path.basename(filename).split(".")[0]
                    outname = f"{epoch}_{num}_{filename}.npy"
                    np.save(f"{fk_out}/{outname}", full_pos_one)
            else:
                b, s, c1, c2 = q.shape
                assert s % 2 == 0
                half = s // 2
                if b > 1:
                    # if long mode, stitch position using linear interp

                    fade_out = torch.ones((1, s, 1)).to(pos.device)
                    fade_in = torch.ones((1, s, 1)).to(pos.device)
                    fade_out[:, half:, :] = torch.linspace(1, 0, half)[None, :, None].to(
                        pos.device
                    )
                    fade_in[:, :half, :] = torch.linspace(0, 1, half)[None, :, None].to(
                        pos.device
                    )

                    pos[:-1] *= fade_out
                    pos[1:] *= fade_in

                    full_pos = torch.zeros((s + half * (b - 1), 3)).to(pos.device)
                    idx = 0
                    for pos_slice in pos:
                        full_pos[idx : idx + s] += pos_slice
                        idx += half

                    # stitch joint angles with slerp
                    slerp_weight = torch.linspace(0, 1, half)[None, :, None].to(pos.device)

                    left, right = q[:-1, half:], q[1:, :half]
                    # convert to quat
                    left, right = (
                        axis_angle_to_quaternion(left),
                        axis_angle_to_quaternion(right),
                    )
                    merged = quat_slerp(left, right, slerp_weight)  # (b-1) x half x ...
                    # convert back
                    merged = quaternion_to_axis_angle(merged)

                    full_q = torch.zeros((s + half * (b - 1), c1, c2)).to(pos.device)
                    full_q[:half] += q[0, :half]
                    idx = half
                    for q_slice in merged:
                        full_q[idx : idx + half] += q_slice
                        idx += half
                    full_q[idx : idx + half] += q[-1, half:]

                    # unsqueeze for fk
                    full_pos = full_pos.unsqueeze(0)
                    full_q = full_q.unsqueeze(0)
                else:
                    full_pos = pos
                    full_q = q
                    
                
                if fk_out is not None:
                    outname = f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.pkl'  # f'{epoch}_{"_".join(name)}.pkl' #
                    Path(fk_out).mkdir(parents=True, exist_ok=True)
                    pickle.dump(
                        {
                            "smpl_poses": full_q.squeeze(0).reshape((-1, reshape_size)).cpu().numpy(),    # local rotations      # debug!!
                            "smpl_trans": full_pos.squeeze(0).cpu().numpy(),                    # root translation
                            # "full_pose": full_pose[0],                                          # 3d positions
                        },
                        open(os.path.join(fk_out, outname), "wb"),
                    )
                return

        if self.cfg.FINEDANCE.nfeats != 263:
            if fk_out is not None and mode != "long":
                Path(fk_out).mkdir(parents=True, exist_ok=True)
                # for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):
                for num, (sample_contact, qq, pos_, filename) in enumerate(zip(sample_contacts, q, pos, name)):
                    filename = os.path.basename(filename).split(".")[0]
                    outname = f"{epoch}_{num}_{filename}.npy"
                    qq_rot6d = ax_to_6v(qq).reshape(pos_.shape[0], -1)
                    print("sample_contact", sample_contact.shape)
                    print("pos_", pos_.shape)
                    print("qq_rot6d", qq_rot6d.shape)
                    result_data = torch.cat([sample_contact, pos_, qq_rot6d], dim=1).detach().cpu().numpy()
                    Path(fk_out).mkdir(parents=True, exist_ok=True)
                    np.save(os.path.join(fk_out, outname), result_data)
        else:
            if fk_out is not None and mode != "long":
                Path(fk_out).mkdir(parents=True, exist_ok=True)
                # for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):
                for num, (sample, filename) in enumerate(zip(samples, name)):
                    filename = os.path.basename(filename).split(".")[0]
                    outname = f"{epoch}_{num}_{filename}.npy"
                    np.save(f"{fk_out}/{outname}", sample)
