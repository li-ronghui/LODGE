import inspect
import os
from re import L
import sys
import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torchmetrics import MetricCollection
import time
from dld.config import instantiate_from_config, get_obj_from_str
from os.path import join as pjoin
from dld.data.render_joints.smplfk import SMPLX_Skeleton
from dld.losses.Joints_loss import Joints_losses
from dld.models.modeltype.base import BaseModel
from dld.models.architectures.model import AdversarialLoss, DanceDiscriminator
from torch.optim import Optimizer
from pathlib import Path
from render import ax_from_6v,ax_to_6v


from .base import BaseModel
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance

def remove_padding(tensors, lengths):
    return [
        tensor[:tensor_length]
        for tensor, tensor_length in zip(tensors, lengths)
    ]


def swap_left_right(data):   
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    
    if data.shape[-1] == 22*6:
        device_ = data.device
        t,c= data.shape
        data = ax_from_6v(data.view(t,22,6))
    elif data.shape[-1] == 52*6:
        t,c= data.shape
        data = ax_from_6v(data.view(t,52,6))
    assert len(data.shape) == 3 and data.shape[-1] == 3
    pose = data.clone()
    
    # pose = data[:,1:,:].clone()
    tmp = pose[:, right_chain].clone()
    pose[:, right_chain] = pose[:, left_chain].clone()
    pose[:, left_chain] = tmp.clone()
    if pose.shape[1] > 24:
        tmp = pose[:, right_hand_chain].clone()
        pose[:, right_hand_chain] = pose[:, left_hand_chain].clone()
        pose[:, left_hand_chain] = tmp.clone()
        
    pose[:,:,1:3] *= -1
    return pose

def swap_left_right_ske(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data
    
class Local_Module(BaseModel):
    """
    Stage 2 diffusion
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.cfg = cfg
        self.stage = cfg.TRAIN.STAGE
        self.condition = cfg.model.condition        # 
        # self.is_vae = cfg.model.vae                 # True
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.nfeats = cfg.FINEDANCE.nfeats  #   # cfg.DATASET.NFEATS
        self.njoints = cfg.FINEDANCE.njoints           # cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncodp = cfg.model.guidance_uncondp
        self.datamodule = datamodule
        self.loss_ = {}

        if cfg.Norm:
            dataname = str(cfg.TRAIN.DATASETS[0])
            self.normalizer = torch.load(eval(f"cfg.DATASET.{dataname.upper()}.normalizer"))  
            # self.normalizer = instantiate_from_config(eval(f"cfg.DATASET.{dataname.upper()}.normalizer"))
        else:
            self.normalizer = None

        # self.DanceDecoder = instantiate_from_config(cfg.model.DanceDecoder)
        self.smplx_fk = SMPLX_Skeleton(Jpath='data/smplx_neu_J_1.npy', device=cfg.DEVICE) 
        self.DanceDecoder = get_obj_from_str(cfg.model.DanceDecoder["target"])(smplx_model=self.smplx_fk, normalizer=self.normalizer, genre_num=self.cfg.FINEDANCE.GENRE_NUM,**cfg.model.DanceDecoder.get("params", dict()))
        
        # self.dis_model = instantiate_from_config(cfg.model.DanceDiscriminator)
        self.dis_model = get_obj_from_str(cfg.model.DanceDiscriminator["target"])(genre_num=self.cfg.FINEDANCE.GENRE_NUM, **cfg.model.DanceDiscriminator.get("params", dict()))
        self.adv_loss = AdversarialLoss('hinge')
        self.l1_loss = torch.nn.L1Loss()

        self.diffusion = get_obj_from_str(cfg.model.diffusion["target"])(cfg=self.cfg, model=self.DanceDecoder, dis_model=self.dis_model, normalizer= self.normalizer, smplx_model=self.smplx_fk,  **cfg.model.diffusion.get("params", dict()))
    
       
       
        # if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
        #     self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
        #                            params=self.parameters())
        if cfg.TRAIN.OPTIM.TYPE.lower() == "adan":
            if self.cfg.Discriminator:
                self.optim_g  = Adan(self.DanceDecoder.parameters(), 
                                    lr=cfg.TRAIN.OPTIM.LR, weight_decay=0.02)
                self.optim_d  = Adan(self.dis_model.parameters(), 
                                    lr=cfg.TRAIN.OPTIM.LR, weight_decay=0.02)
            else:
                self.optimizer = Adan(self.DanceDecoder.parameters(), 
                                    lr=cfg.TRAIN.OPTIM.LR, weight_decay=0.02)
        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")
        
    def render_sample_ori(
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=False, setmode="normal", device=f'cuda:0'
    ):
        _, cond, wavname = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        shape = (render_count, self.cfg.FINEDANCE.full_seq_len, self.nfeats)
        cond = cond.to(device).float()
        self.diffusion.render_sample(
            shape,
            cond[:render_count],
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode=setmode,         
            fk_out=fk_out,
            render=render,
        )

    def render_sample(self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=False, setmode="normal", cons=None, device=None, Returnfull=False, soft_hint=False, genre=None):
        _, cond, wavname, orikey = data_tuple
        assert len(cond.shape) == 3
        # if render_count < 0:
        render_count = len(cond)
        shape = (render_count, self.cfg.length2, self.nfeats)
        cond = cond.to(device).float()
        if genre is not None:
            print("genre", genre)
            print("genre", genre.shape)
            genre = genre.repeat(shape[0]).squeeze()
            print("genre", genre)
            print("genre", genre.shape)

        if self.normalizer is not None:
            for idx_ in range(len(orikey)):
                orikey[idx_] = self.normalizer.normalize(torch.from_numpy(orikey[idx_])).detach().cpu().numpy()

        if cond.shape[-1] == 35:
            if orikey[0].shape[-1] == 139:
                if soft_hint == 'dod':
                    constraint={}
                    constraint["value"] = torch.zeros(cond.shape[0], cond.shape[1], 139)
                    constraint["mask"] = torch.zeros(cond.shape[0], cond.shape[1], 139)
                    for i in range(cond.shape[0]):
                        if cond.shape[1] == 256:
                            mocond_1 = orikey[i][:4,:]
                            print("orikey i .shape", orikey[i].shape)
                            print("before is mocond_1[:, 4]", mocond_1[:, 4])
                            print("before is mocond_1[:, 6]", mocond_1[:, 6])
                            mocond_1[:, 4]  = mocond_1[:, 4] - mocond_1[:1, 4]  
                            mocond_1[:, 6]  = mocond_1[:, 6] - mocond_1[:1, 6]  
                            print("after is mocond_1[:, 4]", mocond_1[:, 4])
                            print("after is mocond_1[:, 6]", mocond_1[:, 6])
                            mocond_2 = orikey[i][-4:,:]
                            mocond_2[:, 4]  = mocond_2[:, 4] - mocond_2[:1, 4]  
                            mocond_2[:, 6]  = mocond_2[:, 6] - mocond_2[:1, 6]   
                            mid = torch.from_numpy(orikey[i][4:-4]).to(cond)
                            Mmid_pose = swap_left_right(mid[:, 7:])
                            print("Mmid_pose.shape", Mmid_pose.shape)
                            Mmid_pose = ax_to_6v(Mmid_pose).view(-1, 132)
                            print("Mmid_pose.shape", Mmid_pose.shape)
                            # Mmid_pose = Mmid_pose.view(Mmid_pose.shape[0], -1)
                            Mmid_root = mid[:, [4,5,6]].clone()
                            Mmid_root[:,0] *= -1
                            Mmid_foot = torch.cat( [mid[:, 1:2], mid[:, 0:1], mid[:, 3:4], mid[:, 2:3] ] , dim=-1)
                            Mmid = torch.cat([Mmid_foot, Mmid_root, Mmid_pose], dim=-1)
                            assert mid.shape[0] == 16
                            constraint["value"][i, :4, :] = torch.from_numpy(mocond_1).to(cond)
                            constraint["value"][i, -4:, :] = torch.from_numpy(mocond_2).to(cond)
                            constraint["value"][i, 28:36, :] = mid[:8]
                            constraint["value"][i, 60:68, :] = Mmid[:8]
                            constraint["value"][i, 156:164, :] = mid[-8:]
                            constraint["value"][i, 188:196, :] = Mmid[-8:]
                            constraint["mask"][i, :, :] = 0
                            constraint["mask"][i, 28:36, :4] =1
                            constraint["mask"][i, 28:36, 7:] =1
                            constraint["mask"][i, 60:68, :4] =1
                            constraint["mask"][i, 60:68, 7:] =1
                            constraint["mask"][i, 156:164, :4] =1
                            constraint["mask"][i, 156:164, 7:] =1
                            constraint["mask"][i, 188:196, :4] =1
                            constraint["mask"][i, 188:196, 7:] =1
                            # constraint["mask"][i, :4,:] = 1 
                            # constraint["mask"][i, -4:,:] = 1 
                            constraint["mask"][i, :4, :4] =1
                            constraint["mask"][i, :4:, 7:] =1
                            constraint["mask"][i, -4:, :4] =1
                            constraint["mask"][i, -4:, 7:] =1
                        elif cond.shape[1] == 128:
                            mocond_1 = orikey[i][:4,:]
                            print("orikey i .shape", orikey[i].shape)
                            # mocond_1[:, [4,6]]  = mocond_1[:, [4,6]] - mocond_1[:1, [4,6]]  
                            mocond_1[:, 4]  = mocond_1[:, 4] - mocond_1[:1, 4]  
                            mocond_1[:, 6]  = mocond_1[:, 6] - mocond_1[:1, 6]  
                            mocond_2 = orikey[i][-4:,:]
                            # mocond_2[:, [4,6]]  = mocond_2[:, [4,6]] - mocond_2[:1, [4,6]] 
                            mocond_2[:, 4]  = mocond_2[:, 4] - mocond_2[:1, 4]  
                            mocond_2[:, 6]  = mocond_2[:, 6] - mocond_2[:1, 6]   
                            mid = torch.from_numpy(orikey[i][4:-4]).to(cond)
                            # Mmid_pose = swap_left_right(mid[:, 7:])
                            # print("Mmid_pose.shape", Mmid_pose.shape)
                            # Mmid_pose = ax_to_6v(Mmid_pose).view(-1, 132)
                            # print("Mmid_pose.shape", Mmid_pose.shape)
                            # # Mmid_pose = Mmid_pose.view(Mmid_pose.shape[0], -1)
                            # Mmid_root = mid[:, 4:7].clone()
                            # Mmid_root[:,0] *= -1
                            # Mmid_foot = torch.cat( [mid[:, 1:2], mid[:, 0:1], mid[:, 3:4], mid[:, 2:3] ] , dim=-1)
                            # Mmid = torch.cat([Mmid_foot, Mmid_root, Mmid_pose], dim=-1)
                            print("orikey[i]", orikey[i].shape)
                            print("orikey len", len(orikey))
                            print("mid.shape", mid.shape)
                            # sys.exit(0)
                            assert mid.shape[0] == 16
                            constraint["value"][i, :4, :] = torch.from_numpy(mocond_1).to(cond)
                            constraint["value"][i, -4:, :] = torch.from_numpy(mocond_2).to(cond)
                            constraint["value"][i, 38:46, :] = mid[:8]
                            constraint["value"][i, 82:90, :] = mid[-8:]
                            constraint["mask"][i, :, :] = 0
                            # constraint["mask"][i, 38:46, :4] =1
                            # constraint["mask"][i, 38:46, 7:] =1
                            # constraint["mask"][i, 82:90, :4] =1
                            # constraint["mask"][i, 82:90, 7:] =1
                            # constraint["mask"][i, :4,:] = 1 
                            # constraint["mask"][i, -4:,:] = 1 
                            constraint["mask"][i, :4, :4] =1
                            constraint["mask"][i, :4:, 7:] =1
                            constraint["mask"][i, -4:, :4] =1
                            constraint["mask"][i, -4:, 7:] =1
                        
            else:
                constraint={}
                constraint["value"] = torch.zeros(cond.shape[0], cond.shape[1], 139)
                constraint["mask"] = torch.zeros(cond.shape[0], cond.shape[1], 139)
                # print("constraint[value].shape", constraint["value"].shape)
                # print("orikey.shape", orikey.shape)
                for i in range(cond.shape[0]):
                    mocond_1 = orikey[i][:4,:]
                    mocond_1[:, [4,6]]  = mocond_1[:, [4,6]] - mocond_1[:1, [4,6]]  
                    mocond_2 = orikey[i][-4:,:]
                    mocond_2[:, [4,6]]  = mocond_2[:, [4,6]] - mocond_2[:1, [4,6]]  
                    constraint["value"][i, :4, :] = torch.from_numpy(mocond_1).to(cond)
                    constraint["value"][i, -4:, :] = torch.from_numpy(mocond_2).to(cond)
                    constraint["mask"][i, :4,:] = 1 
                    constraint["mask"][i, -4:,:] = 1 
        else:
            raise("cond fea error!")
        # constraint = None
        self.diffusion.render_sample(
            shape,
            cond[:render_count],
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            constraint=constraint,
            mode=setmode,            
            fk_out=fk_out,
            render=render,
            genre=genre,
        )

    
    def allsplit_step(self, split: str, batch, batch_idx, epoch, optimizer_idx=None):     
        motion, cond, genre_id = batch
        genre_id = genre_id.squeeze()
        if split == 'train':
            if self.cfg.Discriminator:
                # (opt_g, opt_d) = self.optimizers()
                if optimizer_idx == 0:
                    self.loss_, model_out = self.diffusion(motion, cond, t_override=None, genre_id=genre_id)
                    # advloss
                    f_logit = self.dis_model(cond, model_out, genre_id)
                    self.loss_['l_adv_loss'] = 0.01 * self.adv_loss(f_logit, True, False)
                    self.loss_['loss'] +=  self.loss_['l_adv_loss']
         

                    # div_loss
                    b=motion.shape[0]
                    noise1 = torch.randn(b, 256).to(cond.device)
                    noise2 = torch.randn(b, 256).to(cond.device)
                    style1 = self.DanceDecoder.mapping(noise1, genre_id)
                    style2 = self.DanceDecoder.mapping(noise2, genre_id)
                    self.loss_['l_div_loss'] = 0.01 * self.l1_loss(noise1, noise2) / self.l1_loss(style1, style2)
                    self.loss_['loss'] +=  self.loss_['l_div_loss']

                    # style focusing loss
                    g_tensor = torch.arange(self.cfg.FINEDANCE.GENRE_NUM).long().to(cond.device)
                    genre_id_ = [g_tensor[g_tensor != id][torch.randperm(self.cfg.FINEDANCE.GENRE_NUM-1)[0]] for id in genre_id]
                    genre_id_ = torch.stack(genre_id_, dim=0)

                    dance_g_sfc = model_out
                    r_logit, f_logit = self.dis_model(cond, dance_g_sfc, genre_id, genre_id_)

                    l_sfc_real = self.adv_loss(r_logit, True, False)
                    l_sfc_fake = self.adv_loss(f_logit, False, False)

                    self.loss_['l_sfc_loss'] = 0.01 * (l_sfc_real + l_sfc_fake)/2
                    self.loss_['loss']  += self.loss_['l_sfc_loss']

                    return self.loss_

                elif optimizer_idx == 1:
                    model_out = self.diffusion(motion, cond, t_override=None, genre_id=genre_id, isgen=True)
                    self.loss_['dis_loss'] = self.dis_loss_cal(cond, motion, model_out, genre_id) * self.cfg.LOSS.LAMBDA_DIS 
                    return self.loss_['dis_loss']
            else:
                self.loss_ = self.diffusion(motion, cond, t_override=None)

            return self.loss_
        elif split in ['val', 'test']:
            # render_count = 2                # 渲染两个
            shape = motion.shape
            samples = self.diffusion.ddim_sample(
                    shape,
                    cond,
                    genre=genre_id,
                ).detach().cpu().numpy()
            
            if self.cfg.Discriminator:
                loss_dict, model_out = self.diffusion(motion, cond, t_override=None, genre_id=genre_id)
                loss_dict['dis_loss'] = self.dis_loss_cal(cond, motion, model_out, genre_id) * self.cfg.LOSS.LAMBDA_DIS 
            else:
                loss_dict = self.diffusion(motion, cond, t_override=None)

            outputs = samples, cond, loss_dict
            return outputs
            
    def dis_loss_cal(self,music,dance_r,dance_g,dance_id):
        b = music.shape[0]
        device = music.device


        loss, log_dict = 0.0, {}
        ##ground truth的是dance_r  生成的是dance_g
        r_logit = self.dis_model(music, dance_r, dance_id)
        f_logit = self.dis_model(music, dance_g.detach(), dance_id)

        l_dis_real = self.adv_loss(r_logit, True, True)
        l_dis_fake = self.adv_loss(f_logit, False, True)

        l_adv_loss = ((l_dis_real + l_dis_fake) / 2)
        return l_adv_loss
            
    def training_epoch_end(self, outputs):
        return self.allsplit_epoch_end("train", outputs)

    def validation_epoch_end(self, outputs):
        # print("in validation", self.trainer.current_epoch)
        if self.trainer.current_epoch % 50 == 0  or  (self.trainer.current_epoch+1) % 50 == 0:       
            self.save_npy(outputs[0][0], outputs[0][1], phase='val', epoch=self.trainer.current_epoch)
        return self.allsplit_epoch_end("val", outputs)

    def test_epoch_end(self, outputs):
        print("in test_epoch_end", self.trainer.current_epoch)
        self.save_npy(outputs[0][0], outputs[0][1], phase='test', epoch=self.trainer.current_epoch)
        self.cfg.TEST.REP_I = self.cfg.TEST.REP_I + 1

        return self.allsplit_epoch_end("test", outputs)
    
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        epoch = int(self.trainer.current_epoch)
        batch = self.get_input(batch, batch_idx)
        loss = self.allsplit_step("train", batch, batch_idx, epoch, optimizer_idx)
        # self.training_step_outputs.append(loss)

        return loss
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        # ema优化
        if args[3]==0:
            self.diffusion.ema.update_model_average(
                self.diffusion.master_model, self.diffusion.model
            )
        elif args[3]==1:
            self.diffusion.ema.update_model_average(
                self.diffusion.master_model_dis, self.diffusion.dis_model
            )
        return

    def validation_step(self, batch, batch_idx):
        epoch = int(self.trainer.current_epoch)
        # print("In validation_step! epoch is :", epoch)
        batch = self.get_input(batch, batch_idx)
        return self.allsplit_step("val", batch, batch_idx, epoch)

    def test_step(self, batch, batch_idx):
        epoch = int(self.trainer.current_epoch)
        print("In test_step! epoch is :", epoch)
        batch = self.get_input(batch, batch_idx)
        if len(self.times) *self.cfg.TEST.BATCH_SIZE % (100) > 0 and len(self.times) > 0:
            print(f"Average time per sample ({self.cfg.TEST.BATCH_SIZE*len(self.times)}): ", np.mean(self.times)/self.cfg.TEST.BATCH_SIZE)
        return self.allsplit_step("test", batch, batch_idx, epoch)

            
            
    def allsplit_epoch_end(self, split: str, outputs):
        avg_loss = {}
        if split == 'train':
            # initialt the dictory
            if self.cfg.Discriminator:      # gen 和 dis 有两个loss
                for key in outputs[0][0].keys():
                    avg_loss[key] = 0.0
                for key in outputs[0][1].keys():
                    avg_loss[key] = 0.0

                for output_li in outputs:
                    for output in output_li:
                        for key in output.keys():
                            avg_loss[key] += output[key]
            else:
                for key in outputs[0].keys():
                    avg_loss[key] = 0.0

                for output in outputs:
                    loss_dict = output
                    for key in loss_dict.keys():
                        avg_loss[key] += loss_dict[key]
            for key in avg_loss.keys():
                avg_loss[key] = avg_loss[key]/len(outputs)
        elif split in ["val", "test"]:
            # out_sample, total_loss, loss, v_loss, fk_loss, foot_loss = outputs[0]       
            # init
            out_sample, _, loss_dict = outputs[0]
            for key in loss_dict.keys():
                avg_loss[key] = 0.0
            for output in outputs:
                out_sample, _, loss_dict = output
                for key in loss_dict.keys():
                    avg_loss[key] += loss_dict[key]
            for key in avg_loss.keys():
                avg_loss[key] = avg_loss[key]/len(outputs)

        dico = {}
        if split in ["train", "val"]:
            for key in avg_loss.keys():
                dico[key + f'/{split}'] = avg_loss[key]
            
        if split in ["val", "test"]:
            for key in avg_loss.keys():
                dico['Metrics/' + key] = avg_loss[key]
          
        if split != "test":
            dico.update({
                "epoch": float(self.trainer.current_epoch),
                "step": float(self.trainer.current_epoch),
            })
        # don't write sanity check into log
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)
            
    def save_npy(self, samples, cond, phase, epoch):
        cfg = self.cfg
        # time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
        if phase == 'test':
            output_dir = Path(
                os.path.join(
                    cfg.FOLDER,
                    str(cfg.model.model_type),
                    str(cfg.NAME),
                    phase,
                    "samples_" + cfg.TIME,
                ))
        elif phase == 'val':
            output_dir = Path(
                os.path.join(
                    cfg.FOLDER,
                    str(cfg.model.model_type),
                    str(cfg.NAME),
                    phase + str(epoch),
                    "samples_" + cfg.TIME,
                ))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print("savedir: ", output_dir)
        else:
            pass
            print("Attention! output_dir is already exits!:", output_dir)
            # raise("Path already exits!")
            
        if cfg.TEST.SAVE_PREDICTIONS:
            # motion_ref = [i["m_ref"] for i in samples]
            motion_rst = torch.from_numpy(samples)
            if cfg.FINEDANCE.mode == 'double':
                motion_rst_a = motion_rst[..., :int(motion_rst.shape[-1]/2)]
                motion_rst_b = motion_rst[..., int(motion_rst.shape[-1]/2):]
            elif cfg.FINEDANCE.mode == 'double_react':
                motion_rst_a = cond[..., 35:]
                motion_rst_b = motion_rst

            if cfg.Norm:
                if cfg.FINEDANCE.mode == 'double':
                    motion_rst_a = self.normalizer.unnormalize(motion_rst_a)
                    motion_rst_b = self.normalizer.unnormalize(motion_rst_b)
                elif cfg.FINEDANCE.mode == 'double_react':
                    motion_rst_a = self.normalizer.unnormalize(motion_rst_a)
                    motion_rst_b = self.normalizer.unnormalize(motion_rst_b)
                else:
                    motion_rst = self.normalizer.unnormalize(motion_rst)

            if cfg.TEST.DATASETS[0].lower() in ["finedance",  "finedance_139cut",  "aistpp",  "aistpp_60FPS"]:
                for i in range(len(motion_rst)):
                    if cfg.TEST.REPLICATION_TIMES > 1:
                        if phase == 'test':
                            name = f"{str(i).zfill(3)}_{cfg.TEST.REP_I}"
                        elif phase == 'val':
                            name = f"{str(i).zfill(3)}_{str(epoch)}"
                    else:
                        name = f"{str(i).zfill(3)}.npy"
                    # save predictions results
                    npypath = os.path.join(output_dir, name)
                    # print(npypath)
                    np.save(npypath, motion_rst[i].detach().cpu().numpy())
            else:
                raise("cfg.TEST.DATASETS is not finedance!")
        
        
def exists(val):
    return val is not None
       
class Adan(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.02, 0.08, 0.01),
        eps=1e-8,
        weight_decay=0,
        restart_cond: callable = None,
    ):
        assert len(betas) == 3

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            restart_cond=restart_cond,
        )

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if exists(closure):
            loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            restart_cond = group["restart_cond"]

            for p in group["params"]:
                if not exists(p.grad):
                    continue

                data, grad = p.data, p.grad.data
                assert not grad.is_sparse

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["prev_grad"] = torch.zeros_like(grad)
                    state["m"] = torch.zeros_like(grad)
                    state["v"] = torch.zeros_like(grad)
                    state["n"] = torch.zeros_like(grad)

                step, m, v, n, prev_grad = (
                    state["step"],
                    state["m"],
                    state["v"],
                    state["n"],
                    state["prev_grad"],
                )
                if step > 0:
                    prev_grad = state["prev_grad"]
                    # main algorithm
                    m.mul_(1 - beta1).add_(grad, alpha=beta1)
                    grad_diff = grad - prev_grad
                    v.mul_(1 - beta2).add_(grad_diff, alpha=beta2)
                    next_n = (grad + (1 - beta2) * grad_diff) ** 2
                    n.mul_(1 - beta3).add_(next_n, alpha=beta3)

                # bias correction terms

                step += 1

                correct_m, correct_v, correct_n = map(
                    lambda n: 1 / (1 - (1 - n) ** step), (beta1, beta2, beta3)
                )

                # gradient step

                def grad_step_(data, m, v, n):
                    weighted_step_size = lr / (n * correct_n).sqrt().add_(eps)

                    denom = 1 + weight_decay * lr

                    data.addcmul_(
                        weighted_step_size,
                        (m * correct_m + (1 - beta2) * v * correct_v),
                        value=-1.0,
                    ).div_(denom)

                grad_step_(data, m, v, n)

                # restart condition

                if exists(restart_cond) and restart_cond(state):
                    m.data.copy_(grad)
                    v.zero_()
                    n.data.copy_(grad ** 2)

                    grad_step_(data, m, v, n)

                # set new incremented step

                prev_grad.copy_(grad)
                state["step"] = step

        return loss
