from os.path import join as pjoin

import numpy as np
# from .humanml.utils.word_vectorizer import WordVectorizer
# from .HumanML3D import HumanML3DDataModule
# from .Kit import KitDataModule
# from .Humanact12 import Humanact12DataModule
# from .Uestc import UestcDataModule
# from .utils import *
from .FineDance_Module import FineDanceDataModule

# map config name to module&path
dataset_module_map = {
    "finedance": FineDanceDataModule,
    "finedance_139cut": FineDanceDataModule,
    "aistpp": FineDanceDataModule,
    "aistpp_60fps": FineDanceDataModule,
}
# motion_subdir = {"FineDance": "new_joint_vecs"}


def get_datasets(cfg, logger=None, phase="train"):
    # get dataset names form cfg
    dataset_names = eval(f"cfg.{phase.upper()}.DATASETS")
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name.lower() in ["finedance", "finedance_139cut", "aistpp", "aistpp_60fps"]:
            dataset = dataset_module_map[dataset_name.lower()](
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                name=dataset_name,
            )
            datasets.append(dataset)
      
    # cfg.DATASET.NFEATS = datasets[0].nfeats
    # cfg.DATASET.NJOINTS = datasets[0].njoints
    return datasets
