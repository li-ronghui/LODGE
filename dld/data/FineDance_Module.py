import numpy as np
import torch, sys

# from mld.data.humanml.scripts.motion_process import (process_file,
#                                                      recover_from_ric)

# from .BaseData_Module import BASEDataModule
from .FineDance_dataset import FineDance_Smpl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .render_joints.smplfk import SMPLX_Skeleton, ax_from_6v


class FineDanceDataModule(pl.LightningDataModule):
    def __init__(self,
                 cfg,
                 batch_size,
                 num_workers,
                 name,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.name = name
        self.kwargs = kwargs
        self.is_mm = False
        self.smplx_fk = SMPLX_Skeleton(Jpath='/data2/lrh/project/dance/Lodge/lodge_pub/data/smplx_neu_J_1.npy', device=cfg.DEVICE)           # debug 这里的DEVICE？
        
        # self.save_hyperparameters(logger=False)
        # self.njoints = 52       # 55
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = FineDance_Smpl(args=self.cfg, istrain=True, dataname=self.name)
            self.valset = FineDance_Smpl(args=self.cfg, istrain=False, dataname=self.name)
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = FineDance_Smpl(args=self.cfg, istrain=False)
        
            
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.cfg.EVAL.BATCH_SIZE, num_workers=self.num_workers, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.cfg.TEST.BATCH_SIZE, num_workers=self.num_workers, shuffle=False, drop_last=True)


    def feats2joints(self, features):
        # mean = torch.tensor(self.hparams.mean).to(features)
        # std = torch.tensor(self.hparams.std).to(features)
        # features = features * std + mean
        if features.shape[2] == 315:    
            trans, rot6d = torch.split(features, (3, features.shape[2] - 3), dim=2)      # 前4维是foot contact
            b, s, c = rot6d.shape
            local_q_156 = ax_from_6v(rot6d.reshape(b, s, -1, 6)) 
            joints = self.smplx_fk.forward(local_q_156, trans)
            joints = joints.view(b, s, 55, 3)
            return joints
        else: 
            print("shape IS", features.shape)
            raise("feats2joints's input shape error!!!!")
        

    def mm_mode(self, mm_on=True):
        # random select samples for mm
        if mm_on:
            self.is_mm = True
            # self.test_dataset.name_list = []
            # self.test_dataset.name_list.append(self.name)
            # self.name_list = self.test_dataset.name_list
            # self.mm_list = np.random.choice(self.name_list,
            #                                 self.cfg.TEST.MM_NUM_SAMPLES,
            #                                 replace=False)
            # self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
    
if __name__ == "__main__":
    trainset = FineDance_Smpl(args={'a':1}, istrain=True)

