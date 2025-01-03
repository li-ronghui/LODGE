import inspect
import os
from re import L
import sys
import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torchmetrics import MetricCollection
import time, random
from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
import gc

from dld.config import instantiate_from_config, get_obj_from_str
from os.path import join as pjoin
from dld.data.render_joints.smplfk import do_smplxfk
from dld.data.render_joints.smplfk import SMPLX_Skeleton
from dld.models.modeltype.base import BaseModel
from torch.optim import Optimizer
from pathlib import Path

from .base import BaseModel
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance
from dld.data.render_joints.utils.motion_process import recover_from_ric

def remove_padding(tensors, lengths):
    return [
        tensor[:tensor_length]
        for tensor, tensor_length in zip(tensors, lengths)
    ]

def get_list(lists, keynum, full_seq_len):
    # 最终结果的列表
    final_result = []

    for lst in lists:
        for one in lst:
            if int(one) < 12 or int(one) > (full_seq_len-12):
                lst.remove(one)

        if len(lst) >= keynum:
            # 如果列表长度大于等于8，则均匀选择8个元素
            # lst = sorted(random.sample(lst, 8))
            interval = len(lst) // keynum
            ind = list(range(keynum))
            ind = [x * interval for x in ind]
            lst  = [lst[i] for i in ind]
        else:
            # 如果列表长度小于8，则从0到500之间均匀抽样，使得总共有8个元素
            additional_elements_count = keynum - len(lst)
            test_lst = lst.copy()
            test_lst.insert(0,8)
            test_lst.append(full_seq_len-8)
            for _ in range(additional_elements_count):
                # 寻找两个元素之间的差异最大的位置
                max_diff_index = 0
                max_diff = 0
                for i in range(len(test_lst) - 1):
                    diff = test_lst[i + 1] - test_lst[i]
                    if diff<10:
                        continue
                    if diff > max_diff:
                        max_diff = diff
                        max_diff_index = i
                # print(max_diff_index)
                # print(max_diff)

                # 计算两个元素之间的中间值
                mid_value = (test_lst[max_diff_index] + test_lst[max_diff_index + 1]) // 2

                # if mid_value < 12:
                #     mid_value = 12
                # if mid_value > (full_seq_len-12):
                #     mid_value = (full_seq_len-12)
                # print(mid_value)
                lst.insert(max_diff_index, mid_value)
                test_lst.insert(max_diff_index+1, mid_value)
        final_result.append(lst)

    return final_result

    
class Global_Module(BaseModel):
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
            # aaa = eval(f"cfg.DATASET.{dataname.upper()}.normalizer")
            # self.normalizer = instantiate_from_config(eval(f"cfg.DATASET.{dataname.upper()}.normalizer"))
            self.normalizer = torch.load(eval(f"cfg.DATASET.{dataname.upper()}.normalizer"))  
        else:
            self.normalizer = None

        # self.Loss_Module = instantiate_from_config(cfg.model.normalizer)

        self.DanceDecoder = instantiate_from_config(cfg.model.DanceDecoder)
        self.smplx_fk = SMPLX_Skeleton(Jpath='data/smplx_neu_J_1.npy', device=cfg.DEVICE) 
        self.diffusion = get_obj_from_str(cfg.model.diffusion["target"])(cfg=self.cfg, model=self.DanceDecoder, normalizer= self.normalizer, smplx_model=self.smplx_fk,  **cfg.model.diffusion.get("params", dict()))
       
       
        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
                                   params=self.parameters())
        elif cfg.TRAIN.OPTIM.TYPE.lower() == "adan":
            self.optimizer  = Adan(self.DanceDecoder.parameters(), 
                                   lr=cfg.TRAIN.OPTIM.LR, weight_decay=0.02)
        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")
        
        # self._losses = MetricCollection({
        #     split: Joints_losses(cfg=cfg)
        #     for split in ["losses_train", "losses_test", "losses_val"]
        # })
            
        # self.losses = {
        #     key: self._losses["losses_" + key]
        #     for key in ["train", "test", "val"]
        # }

    def render_sample_ori(
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=False, setmode="normal", device=f'cuda:0', genre=None,
    ):
        _, cond, wavname = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        print("render_count", render_count)
        print("cond", cond.shape)
        print("wavname", wavname)

        if self.cfg.FINEDANCE.full_seq_len == 1024:
            shape = (render_count, 104, self.nfeats)
        elif self.cfg.FINEDANCE.full_seq_len == 256:
            shape = (render_count, 56, self.nfeats)
        else: 
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
            mode=setmode,            # 这里设置        default is long
            fk_out=fk_out,
            render=render,
            genre=genre,
        )
    
    def allsplit_step(self, split: str, batch, batch_idx, epoch):     
        motion, cond, genre_id = batch
        genre_id = genre_id.squeeze()
        B,T,C = motion.shape
        if self.nfeats == 263:
            posi = recover_from_ric(motion, 22)
        elif self.nfeats == 139:
            posi = do_smplxfk(motion, self.smplx_fk)
        # print("posi", posi.shape)
        posi = posi.detach().cpu().numpy()
        mobeatlist = []
        for pos_num in range(posi.shape[0]):
            one_pos =  posi[pos_num]
            kinetic_vel = np.mean(np.sqrt(np.sum((one_pos[1:] - one_pos[:-1]) ** 2, axis=2)), axis=1)
            kinetic_vel = G(kinetic_vel, 5)
            motion_beats = argrelextrema(kinetic_vel, np.less)#.reshape(B,-1)
            motion_beats =  np.array(motion_beats[0]).astype(int).tolist()
            # if len(motion_beats) < 5: 
            #     nummmm += 1
            mobeatlist.append(motion_beats)

        # debug!!!
        middle_num = (self.cfg.FINEDANCE.full_seq_len // self.cfg.FINEDANCE.length_fi) *2
        mobeatlist = get_list(mobeatlist, middle_num, self.cfg.FINEDANCE.full_seq_len)
        del posi
        gc.collect()

        res = []
        # key_count = 0
        for key_i in range(B):
            bias = random.randint(-2, 2)
            keyidx = mobeatlist[key_i]
            motion_key = motion[key_i:key_i+1, :8, :]
            keynum = 0
            for oneidx in keyidx:
                if oneidx<8:
                    oneidx = 8
                elif oneidx>self.cfg.FINEDANCE.full_seq_len-8:
                    oneidx = self.cfg.FINEDANCE.full_seq_len-8
                
                keynum += 1
                keymo_ = motion[key_i:key_i+1, oneidx-4 + bias : oneidx+4 + bias, :]
                keymo_[:, :, 4]  = keymo_[:, :, 4]  - keymo_[:, :1, 4] 
                keymo_[:, :, 6]  = keymo_[:, :, 6]  - keymo_[:, :1, 6] 
                motion_key = torch.cat([motion_key,  keymo_], dim = 1)
                if keynum % 2 == 0 and keynum<len(keyidx):
                    keymo_ = motion[key_i:key_i+1, int(self.cfg.FINEDANCE.length_fi * ((keynum)/len(keyidx)) )-4 + bias : int(self.cfg.FINEDANCE.length_fi*((keynum)/len(keyidx)))+4 + bias, :]
                    keymo_[:, :, 4]  = keymo_[:, :, 4]  - keymo_[:, :1, 4] 
                    keymo_[:, :, 6]  = keymo_[:, :, 6]  - keymo_[:, :1, 6] 
                    motion_key =  torch.cat([motion_key, keymo_ ], dim = 1)
            #     print("keynum is {}, motionkey is {}".format(keynum, motion_key.shape) )
            # print("motion key 1", motion_key.shape)
            keymo_ = motion[key_i:key_i+1, -8:, :]
            keymo_[:, :, 4]  = keymo_[:, :, 4]  - keymo_[:, :1, 4] 
            keymo_[:, :, 6]  = keymo_[:, :, 6]  - keymo_[:, :1, 6] 
            motion_key = torch.cat([ motion_key, keymo_ ], dim =1)
            # print("motion key 2", motion_key.shape)
            res.append(motion_key)
        motion_key = torch.cat(res, dim = 0)
        del res
        gc.collect()
        # print("motion key", motion_key.shape)
        # sys.exit(0)
        motion = motion_key
        # self.horizon = horizon = motion_key.shape[1]

        if split == 'train':
            self.loss_ = self.diffusion(motion, cond, t_override=None, genre_id=genre_id)

            return self.loss_
        elif split in ['val', 'test']:
            # render_count = 2                # 渲染两个
            shape = motion.shape
            samples = self.diffusion.ddim_sample(
                    shape,
                    cond,
                    genre=genre_id,
                ).detach().cpu().numpy()
            
            loss_dict = self.diffusion(motion, cond, t_override=None, genre_id=genre_id)
            outputs = samples, cond, loss_dict

            return outputs
            
 
            
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
    
    
    def training_step(self, batch, batch_idx):
        epoch = int(self.trainer.current_epoch)
        batch = self.get_input(batch, batch_idx)
        loss = self.allsplit_step("train", batch, batch_idx, epoch)
        # self.training_step_outputs.append(loss)
        return loss

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

            if cfg.TEST.DATASETS[0].lower() in ["finedance",  "finedance_139cut", "aistpp", "aistpp_60fps"]:
                for i in range(len(motion_rst)):
                    # for bid in range(
                    #         min(cfg.TEST.BATCH_SIZE, samples[i].shape[0])):
                    #     keyid = keyids[i * cfg.TEST.BATCH_SIZE + bid]
                    #     gen_joints = samples[i][bid].cpu().numpy()
                    if cfg.FINEDANCE.mode in ['double', 'double_react']:
                        if cfg.TEST.REPLICATION_TIMES > 1:
                            if phase == 'test':
                                name_a = f"{str(i).zfill(3)}_{cfg.TEST.REP_I}" + "_a"
                                name_b = f"{str(i).zfill(3)}_{cfg.TEST.REP_I}" + "_b"
                            elif phase == 'val':
                                name_a = f"{str(i).zfill(3)}_{str(epoch)}"+ "_a"
                                name_b = f"{str(i).zfill(3)}_{str(epoch)}"+ "_b"
                        else:
                            name_a = f"{str(i).zfill(3)}_a.npy"
                            name_b = f"{str(i).zfill(3)}_b.npy"

                        # save predictions results
                        npypath_a = os.path.join(output_dir, name_a)
                        npypath_b = os.path.join(output_dir, name_b)
                        np.save(npypath_a, motion_rst_a[i].detach().cpu().numpy())
                        np.save(npypath_b, motion_rst_b[i].detach().cpu().numpy())
                    else:
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

            
# class EdgeLoss(Metric):
#     def __init__(self, cfg):
#         super().__init__(dist_sync_on_step=cfg.LOSS.DIST_SYNC_ON_STEP)
    
#     losses = []
#     losses.append("total")
    
#     losses.append("recons__loss")
#     losses.append("recons__v_loss")
#     losses.append("recons__fk_loss")
#     losses.append("foot__loss")
    
#     def update(self, rs_set):
#         total: float = 0.0
        
        
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
