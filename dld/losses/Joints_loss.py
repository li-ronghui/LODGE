import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric
from .foot_contact import Foot_contact_loss
# from mld.data.humanml.scripts.motion_process import (qrot,
#                                                      recover_root_rot_pos)



class Joints_losses(Metric):
    """
    MLD Loss
    """

    def __init__(self, cfg):
        super().__init__(dist_sync_on_step=cfg.LOSS.DIST_SYNC_ON_STEP)

        # Save parameters
        # self.vae = vae
        self.cfg = cfg
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.stage = cfg.TRAIN.STAGE
        # try:
        #     self.stage = cfg.TRAIN.STAGE
        # expect:
        #     self.stage = "vae"

        losses = []
        losses.append("total")
        # diffusion loss
        if self.stage in ['diffusion', 'vae_diffusion']:
            # instance noise loss
            losses.append("inst__loss")
            losses.append("x__loss")
            if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
                # prior noise loss
                losses.append("prior__loss")

        if self.stage in ['vae', 'vae_diffusion']:
            # reconstruction loss
            losses.append("recons__trans_rot")
            losses.append("recons__verts_trans_rot")
            losses.append("recons__trans")
            losses.append("recons__verts_trans")
            losses.append("recons__joints")
            losses.append("recons__verts_joints")
            losses.append("foot__contact")

            losses.append("gen__feature")
            losses.append("gen__joints")

            # KL loss
            losses.append("kl__motion")
    
        if self.stage not in ['vae', 'diffusion', 'vae_diffusion']:
            raise ValueError(f"Stage {self.stage} not supported")
        

        for loss in losses:
            self.add_state(loss,
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
            # self.register_buffer(loss, torch.tensor(0.0))
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")
        self.losses = losses

        self._losses_func = {}
        self._params = {}
        
        
        for loss in losses:
            if loss.split('__')[0] == 'inst':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            elif loss.split('__')[0] == 'x':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            elif loss.split('__')[0] == 'prior':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_PRIOR
                
            if loss.split('__')[0] == 'kl':
                if cfg.LOSS.LAMBDA_KL != 0.0:
                    self._losses_func[loss] = KLLoss()
                    self._params[loss] = cfg.LOSS.LAMBDA_KL
                    
            elif loss in ['recons__trans_rot', "recons__verts_trans_rot"]:
                self._losses_func[loss] = torch.nn.L1Loss(              # 默认为SmoothL1loss
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_REC
            elif loss in ['recons__trans', "recons__verts_trans"]:
                self._losses_func[loss] = torch.nn.L1Loss(              # 默认为SmoothL1loss
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_TRANS
            elif loss in ['recons__joints', "recons__verts_joints"]:
                self._losses_func[loss] = torch.nn.L1Loss(              # 默认为SmoothL1loss
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_JOINT
            elif loss  == "foot__contact":
                self._losses_func["foot__contact"] = Foot_contact_loss()     # reduction='mean'
                self._params["foot__contact"] = cfg.LOSS.LAMBDA_FOOT
                
            elif loss.split('__')[0] == 'gen':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_GEN
            elif loss.split('__')[0] == 'latent':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_LATENT
            else:
                ValueError("This loss is not recognized.")
 

    def update(self, rs_set):
        total: float = 0.0
        # Compute the losses
        # Compute instance loss
        if self.stage in ["vae"]:
            if self.cfg.LOSS.LAMBDA_REC != 0:
                total += self._update_loss("recons__trans_rot", rs_set['m_rst'],           # 只计算了recons_trans_rot
                                            rs_set['m_ref'])
                total += self._update_loss("recons__verts_trans_rot", rs_set['m_rst'][:, 1:] - rs_set['m_rst'][:, :-1],          
                                            rs_set['m_ref'][:, 1:] - rs_set['m_ref'][:, :-1])
                
            if self.cfg.LOSS.LAMBDA_TRANS != 0:
                total += self._update_loss("recons__trans", rs_set['m_rst'][:,:,:3],           # 只计算了recons_trans
                                            rs_set['m_ref'][:,:,:3])
                total += self._update_loss("recons__verts_trans", rs_set['m_rst'][:, 1:, :3] - rs_set['m_rst'][:, :-1, :3],          
                                            rs_set['m_ref'][:, 1:, :3] - rs_set['m_ref'][:, :-1, :3])
                
            if self.cfg.LOSS.LAMBDA_JOINT != 0:
                total += self._update_loss("recons__joints", rs_set['xyz_rst'],           # 只计算了recons_trans_rot
                                        rs_set['xyz_ref'])
                total += self._update_loss("recons__verts_joints", rs_set['xyz_rst'][:, 1:] - rs_set['xyz_rst'][:, :-1],          
                                        rs_set['xyz_ref'][:, 1:] - rs_set['xyz_ref'][:, :-1])
            if self.cfg.LOSS.LAMBDA_FOOT != 0:
                total += self._update_loss("foot__contact", rs_set['xyz_rst'], None)        # None or rs_set['contact_rst']
        
        
            # total += self._update_loss("kl__motion", rs_set['dist_m'], rs_set['dist_ref'])

        if self.stage in ["diffusion", "vae_diffusion"]:
            # predict noise
            if self.predict_epsilon:
                total += self._update_loss("inst__loss", rs_set['noise_pred'],
                                           rs_set['noise'])
            # predict x
            else:
                total += self._update_loss("x__loss", rs_set['pred'],       # pred为预测的值， latent为gt
                                           rs_set['latent'])

            if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
                # loss - prior loss
                total += self._update_loss("prior__loss", rs_set['noise_prior'],
                                           rs_set['dist_m1'])

        if self.stage in ["vae_diffusion"]:
            # loss
            # noise+text_emb => diff_reverse => latent => decode => motion
            total += self._update_loss("gen__feature", rs_set['gen_m_rst'],
                                       rs_set['m_ref'])
            total += self._update_loss("gen__joints", rs_set['gen_joints_rst'],
                                       rs_set['joints_ref'])

        self.total += total.detach()
        self.count += 1

        return total

    def compute(self, split):
        count = getattr(self, "count")
        return {loss: getattr(self, loss) / count for loss in self.losses}

    def _update_loss(self, loss: str, outputs, inputs):
        # Update the loss
        val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("__")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name


class KLLoss:
    
    def __init__(self):
        pass

    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class KLLossMulti:

    def __init__(self):
        self.klloss = KLLoss()

    def __call__(self, qlist, plist):
        return sum([self.klloss(q, p) for q, p in zip(qlist, plist)])

    def __repr__(self):
        return "KLLossMulti()"
