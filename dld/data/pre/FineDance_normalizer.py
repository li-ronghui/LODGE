import glob
import os, sys
import re
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
sys.path.append(os.getcwd())
from dld.data.utils.preprocess import Normalizer
modir = '/data2/lrh/dataset/fine_dance/gound/mofea319'

data_li = []
for file in tqdm(os.listdir(modir)):
    if not file.split('.')[-1] == 'npy':
        continue
    filepath = os.path.join(modir, file)
    data = np.load(filepath)[:,:139]
    data = torch.from_numpy(data)
    for idx in range(data.shape[0]):
        data_li.append(data[idx].unsqueeze(0))
data_li = torch.cat(data_li, dim=0)
data_li_ori = data_li.clone()
Normalizer_ = Normalizer(data_li)
torch.save(Normalizer_, '/data2/lrh/project/dance/Lodge/lodge302/data/Normalizer.pth')


reNorm = torch.load('Normalizer.pth')
data_newnormed = reNorm.normalize(data_li)
data_newunnormed = reNorm.unnormalize(data_newnormed)
print(data_newnormed[0,:20])
print(data_newunnormed[0,:20])
print(data_li_ori[0,:20])