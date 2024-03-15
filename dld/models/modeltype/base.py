import os
from pathlib import Path
import numpy as np
import torch
from pytorch_lightning import LightningModule
from dld.models.metrics import MMMetrics, DanceAE_Metric, DanceDiffuse_Metric
from os.path import join as pjoin
from collections import OrderedDict
# from dld.config import instantiate_from_config
import time

class BaseModel(LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_step_outputs = []
        self.times = []

    def __post_init__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())

        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable

    def get_input(self, batch_, batch_idx):
        if self.cfg.FINEDANCE.mode == "single":
            motion = batch_[0]
            if motion.shape[-1] != 263 and motion.shape[-1] != 266:
                if motion.shape[-1] == 319 or motion.shape[-1] == 139:
                    motion[:, :, [4,6]]  = motion[:, :, [4,6]] - motion[:, :1, [4,6]]           # The first 4 dimension are foot contact
                else:
                    motion[:, :, :3] = motion[:, :, :3] - motion[:, :1, :3]
            out = []
            if self.cfg.Norm:
                motions_ref_norm = self.normalizer.normalize(motion)
                out = [motions_ref_norm, batch_[1], batch_[2]]
                return out
            else:
                return [motion, batch_[1], batch_[2]]
        elif self.cfg.FINEDANCE.mode == "double_react":
            motion_a, motion_b, music = batch_

            if motion_a.shape[-1] != 263 and motion_a.shape[-1] != 266:
                if motion_a.shape[-1] == 319 or motion_a.shape[-1] == 139:
                    # motion_a[:, :, 4:7]  = motion_a[:, :, 4:7] - motion_a[:, :1, 4:7]           # The first 4 dimension are foot contact
                    # motion_b[:, :, 4:7]  = motion_b[:, :, 4:7] - motion_b[:, :1, 4:7] 
                    motion_a[:, :, 4]  = motion_a[:, :, 4] - motion_a[:, :1, 4]           # The first 4 dimension are foot contact
                    motion_b[:, :, 4]  = motion_b[:, :, 4] - motion_b[:, :1, 4] 
                    motion_a[:, :, 6]  = motion_a[:, :, 6] - motion_a[:, :1,6]           # The first 4 dimension are foot contact
                    motion_b[:, :, 6]  = motion_b[:, :, 6] - motion_b[:, :1, 6] 
                else:
                    # motion_a[:, :, :3] = motion_a[:, :, :3] - motion_a[:, :1, :3]
                    # motion_b[:, :, :3] = motion_b[:, :, :3] - motion_b[:, :1, :3]
                    motion_a[:, :, [0,1]] = motion_a[:, :, [0,1]] - motion_a[:, :1, [0,1]]
                    motion_b[:, :, [0,1]] = motion_b[:, :, [0,1]] - motion_b[:, :1, [0,1]]
            elif motion_a.shape[-1] == 266 or motion_a.shape[-1]==338:
                a_init = motion_a[:, :1, [0,2]]
                b_init = motion_b[:, :1, [0,2]]
                mid_init = (a_init + b_init) / 2.0
                motion_a[:, :1, [0,2]] = motion_a[:, :1, [0,2]] - mid_init
                motion_b[:, :1, [0,2]] = motion_b[:, :1, [0,2]] - mid_init

            out = []
            if self.cfg.Norm:
                motions_ref_norm_a = self.normalizer.normalize(motion_a)
                motions_ref_norm_b = self.normalizer.normalize(motion_b)
                cond = torch.cat([music, motions_ref_norm_a], dim=-1)
                out = [motions_ref_norm_b, cond]
                return out
            else:
                return batch_
        elif self.cfg.FINEDANCE.mode == "double":
            motion_a, motion_b, music = batch_
            if motion_a.shape[-1] == 266 or motion_a.shape[-1]==338:
                a_init = motion_a[:, :1, [0,2]]
                b_init = motion_b[:, :1, [0,2]]
                mid_init = (a_init + b_init) / 2.0
                motion_a[:, :1, [0,2]] = motion_a[:, :1, [0,2]] - mid_init
                motion_b[:, :1, [0,2]] = motion_b[:, :1, [0,2]] - mid_init
            out = []
            if self.cfg.Norm:
                motions_ref_norm_a = self.normalizer.normalize(motion_a)
                motions_ref_norm_b = self.normalizer.normalize(motion_b)
                motion = torch.cat([motions_ref_norm_a, motions_ref_norm_b], dim=-1)
                out = [motion, music]
                return out
            else:
                return batch_    
        

    def training_step(self, batch, batch_idx):
        batch = self.get_input(batch, batch_idx)
        loss = self.allsplit_step("train", batch, batch_idx)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.get_input(batch, batch_idx)
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        batch = self.get_input(batch, batch_idx)
        if len(self.times) *self.cfg.TEST.BATCH_SIZE % (100) > 0 and len(self.times) > 0:
            print(f"Average time per sample ({self.cfg.TEST.BATCH_SIZE*len(self.times)}): ", np.mean(self.times)/self.cfg.TEST.BATCH_SIZE)
        return self.allsplit_step("test", batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def allsplit_epoch_end(self, split: str, outputs):
        dico = {}

        if split in ["train", "val"]:
            losses = self.losses[split]
            loss_dict = losses.compute(split)
            losses.reset()
            dico.update({
                losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items() if not torch.isnan(value)
            })

        if split in ["val", "test"]:
            if self.trainer.datamodule.is_mm and "TM2TMetrics" in self.metrics_dict:        # datamodule为FineDanceDataModule
                metrics_dicts = ['MMMetrics']
            else:
                metrics_dicts = self.metrics_dict
            for metric in metrics_dicts:
                metrics_dict = getattr(
                    self,
                    metric).compute(sanity_flag=self.trainer.sanity_checking)
                # reset metrics
                getattr(self, metric).reset()
                dico.update({
                    f"Metrics/{metric}": value.item()
                    for metric, value in metrics_dict.items()
                })
        if split != "test":
            dico.update({
                "epoch": float(self.trainer.current_epoch),
                "step": float(self.trainer.current_epoch),
            })
        # don't write sanity check into log
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def training_epoch_end(self, outputs):
        return self.allsplit_epoch_end("train", outputs)
    # def on_train_epoch_end(self):
    #     epoch_average = torch.stack(self.training_step_outputs).mean()
    #     self.log("training_epoch_average", epoch_average)
    #     self.training_step_outputs.clear()  # free memory

    def validation_epoch_end(self, outputs):
        # # ToDo
        # # re-write vislization checkpoint?
        # # visualize validation
        # parameters = {"xx",xx}
        # vis_path = viz_epoch(self, dataset, epoch, parameters, module=None,
        #                         folder=parameters["folder"], writer=None, exps=f"_{dataset_val.dataset_name}_"+val_set) 
        if self.trainer.current_epoch % 20 == 0:       
            self.save_npy(outputs, phase='val', epoch=self.trainer.current_epoch)
        return self.allsplit_epoch_end("val", outputs)

    def test_epoch_end(self, outputs):
        self.save_npy(outputs, phase='test', epoch=self.trainer.current_epoch)
        self.cfg.TEST.REP_I = self.cfg.TEST.REP_I + 1

        return self.allsplit_epoch_end("test", outputs)

    def on_save_checkpoint(self, checkpoint):
        # pass
        # don't save clip to checkpoint
        state_dict = checkpoint['state_dict']
        clip_k = []
        for k, v in state_dict.items():
            if 'text_encoder' in k:
                clip_k.append(k)
        for k in clip_k:
            del checkpoint['state_dict'][k]

    def on_load_checkpoint(self, checkpoint):
        print(checkpoint['state_dict'].keys())
        # restore clip state_dict to checkpoint
        # clip_state_dict = self.text_encoder.state_dict()
        # new_state_dict = OrderedDict()
        # for k, v in clip_state_dict.items():
        #     new_state_dict['text_encoder.' + k] = v
        # for k, v in checkpoint['state_dict'].items():
        #     if 'text_encoder' not in k:
        #         new_state_dict[k] = v
        # checkpoint['state_dict'] = new_state_dict

    def load_state_dict(self, state_dict, strict=True):
        # load clip state_dict to checkpoint
        # clip_state_dict = self.text_encoder.state_dict()
        # new_state_dict = OrderedDict()
        # for k, v in clip_state_dict.items():
        #     new_state_dict['text_encoder.' + k] = v
        # for k, v in state_dict.items():
        #     if 'text_encoder' not in k:
        #         new_state_dict[k] = v
        # super().load_state_dict(new_state_dict, strict)
        super().load_state_dict(state_dict, strict)

    def configure_optimizers(self):
        if self.cfg.Discriminator:
            return self.optim_g, self.optim_d
        else:
            return {"optimizer": self.optimizer}

    def configure_metrics(self):
        for metric in self.metrics_dict:
            if metric == "DanceAE_Metric":      
                self.DanceAE_Metric = DanceAE_Metric(
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "DanceDiffuse_Metric":     
                self.DanceDiffuse_Metric = DanceDiffuse_Metric(
                    cfg=self.cfg, dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            else:
                raise NotImplementedError(
                    f"Do not support Metric Type {metric}")
        if "TM2TMetrics" in self.metrics_dict or "UncondMetrics" in self.metrics_dict:
            self.MMMetrics = MMMetrics(
                mm_num_times=self.cfg.TEST.MM_NUM_TIMES,
                dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
            )