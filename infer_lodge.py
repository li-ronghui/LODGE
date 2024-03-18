import logging
import os, pickle
from omegaconf import OmegaConf

import sys
import time
from builtins import ValueError
from multiprocessing.sharedctypes import Value
from pathlib import Path
from functools import cmp_to_key
from tempfile import TemporaryDirectory
print("1")

import numpy as np
import torch, glob, random
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
       
from dld.config import parse_args
from dld.models.get_model import get_module
from dld.data.get_data import get_datasets
from dld.utils.logger import create_logger
from dld.data.utils.audio import slice_audio
from dld.data.utils.audio import extract as extract_music35
from concat_res import concat_res
from dld.data.render_joints.smplfk import ax_from_6v, ax_to_6v
from dld.data.FineDance_dataset import music2genre, Genres_fd


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
    elif data.shape[-1] == 55*6:
        right_chain = [2, 5, 8, 11, 14, 17, 19, 21, 24]
        left_chain = [1, 4, 7, 10, 13, 16, 18, 20, 23]
        left_hand_chain = [25, 26, 27, 37, 38, 39, 28, 29, 30, 34, 35, 36, 31, 32, 33]
        right_hand_chain = [46, 47, 48, 49, 50, 51, 43, 44, 45, 40, 41, 42, 52, 53, 54]
        t,c= data.shape
        data = ax_from_6v(data.view(t,55,6))
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



# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])


def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0

def load_modata(keymopath, device):
    if keymopath[-3:] == 'pkl':
        pkl_data = pickle.load(open(keymopath, "rb"))
        smpl_poses = pkl_data["smpl_poses"]
        T,C = smpl_poses.shape
        smpl_poses = smpl_poses.reshape(T, -1, 3)
        smpl_poses = torch.from_numpy(smpl_poses).to(device)
        smpl_rot = ax_to_6v(smpl_poses ).reshape(T, -1)
        smpl_trans = torch.from_numpy(pkl_data["smpl_trans"]).to(device)
        assert smpl_rot.shape[1] == 132

        modata = torch.cat( [ smpl_trans, smpl_rot], dim=1)
    elif keymopath[-3:] == 'npy':
        modata  = np.load(keymopath)

    print(modata.shape)

    return modata


def expand_list(lst):
    while len(lst) < 4:
        # 找到相邻两个数中差最大的两个数
        max_diff = 0
        index = 0
        for i in range(len(lst) - 1):
            diff = abs(lst[i+1] - lst[i])
            if diff > max_diff:
                max_diff = diff
                index = i

        # 取这两个数的平均数
        new_element = int((lst[index] + lst[index + 1]) / 2)

        # 将新元素插入列表
        lst.insert(index + 1, new_element)

    return lst

def select_items(lst):
    length = len(lst)
    if length > 4:
        # 从列表中均匀选取4个元素
        indices = [int(i * (length - 1) / 3) for i in range(4)]
        return [lst[index] for index in indices]
    else:
        # 选择列表的中间数
        # mid = length // 2
        return expand_list(lst)

def remove_greater_than(lst, threshold):
    # 使用列表推导保留小于或等于阈值的元素
    return [x for x in lst if x <= threshold]

def remove_close_to_multiples(lst, n):
    # 使用列表推导过滤掉与n的倍数差的绝对值小于8的元素
    return [x for x in lst if all(abs(x - i * n) >= 8 for i in range(1, (max(lst) // n) + 1))]


def split_list(lst, N):
    # 计算需要的子列表个数
    n = math.ceil(N / 256)

    # 初始化子列表
    sublists = [[] for _ in range(n)]

    # 遍历原始列表，分配元素
    for item in lst:
        # 确定当前元素应该在哪个子列表中
        index = min(item // 256, n - 1)
        sublists[index].append(item)

    return sublists


def test(cfg):
    count = 1
    length_co = cfg.length1
    length_fi = cfg.length2
    print("cfg.device", cfg.DEVICE)
    device = f"cuda:{cfg.DEVICE[0]}"

    print(device)

    fk_out = None

    dataset = get_datasets(cfg, logger=logger, phase="test")[0]
    # create mld model
    total_time = time.time()
    # cfg_coarse.model.model_type = 'Lodeg_Coarse_Module'
    model_coarse = get_module(cfg_coarse, dataset)
    logger.info("Loading checkpoints from {}".format(cfg.checkpoint1))
    state_dict = torch.load(cfg.checkpoint1,
                            map_location="cpu")["state_dict"]

    model_coarse.load_state_dict(state_dict, strict=True)
    logger.info("model {} loaded".format(cfg_coarse.model.model_type))
    model_coarse.to(device)
    model_coarse.eval()


    model_fine = get_module(cfg, dataset)
    logger.info("Loading checkpoints from {}".format(cfg.checkpoint2))
    state_dict = torch.load(cfg.checkpoint2,
                            map_location="cpu")["state_dict"]

    model_fine.load_state_dict(state_dict, strict=True)
    logger.info("model {} loaded".format(cfg.model.model_type))
    model_fine.to(device)
    model_fine.eval()

    
    # elif opt.wavdir != 'None':
    for file in os.listdir(music_dir):
        flag = 0
        if not file[:3] in test_list:
            continue

        file_name = file[:-4]
        mufile = os.path.join(music_dir, file)

        if cfg.DEMO.use_cached_features:     # cfg.DEMO.use_cached_features:
            music_fea_full = np.load(mufile)
        else:
            music_fea_full, peakidx = extract_music35(fpath=mufile)
        
        print("music_fea_full", music_fea_full.shape)
        local_num = music_fea_full.shape[0] // cfg.length2
        music_fea_full = music_fea_full[:local_num * cfg.length2]

        global_num = local_num // (int(cfg.length1/cfg.length2))
        if local_num % (int(cfg.length1/cfg.length2)) != 0:
            global_num += 1
        # flag = (int(cfg.length1/cfg.length2)) - ( local_num % (int(cfg.length1/cfg.length2)) )
        flag = local_num % (int(cfg.length1/cfg.length2)) 
 
        for gi in range(global_num):
            assert music_fea_full.shape[0] % cfg.length2 == 0
            if (gi+1)*cfg.length1 > music_fea_full.shape[0]:
                music_fea = music_fea_full[(-1)*cfg.length1 : ]
            else:
                music_fea = music_fea_full[gi*cfg.length1 : (gi+1)*cfg.length1]

            music_fea = torch.from_numpy(music_fea).to(device).unsqueeze(0)
            music_fea = music_fea.repeat(count, 1, 1)
            if gi == 0 :
                music_fea_cat = music_fea
                all_filenames_cat = [file_name + 'g' + str(gi).zfill(3) + 'g']*count
            else:
                music_fea_cat = torch.cat([music_fea_cat, music_fea], dim=0)
                all_filenames_cat = all_filenames_cat + [file_name + 'g' + str(gi).zfill(3) + 'g']*count

        fk_out = None
        fk_out = output_dir
            
        # if cfg.fullpt:
        print("music_fea_cat", music_fea_cat.shape)
        print("all_filenames_cat", all_filenames_cat)
        data_tuple = None, music_fea_cat, all_filenames_cat
        model_coarse.render_sample_ori(
             data_tuple, "global", output_dir, render_count=-1, fk_out=fk_out, render=cfg.DEMO.RENDER, setmode="normal", device=device
        )
        print("Done")
        print("OK")

        molist_cat = []
        modata13_cat = []
        all_localfilename_cat = []
        for rgi in range(global_num):       # rgi --> read global i
            music_fea_cat = music_fea_cat.reshape(-1, length_fi, 35)
            music_fea_cat = music_fea_cat[:local_num]
            keymopath = os.path.join(fk_out, "global_" + str(rgi) + "_" + file_name + 'g' + str(rgi).zfill(3) + 'g' + ".npy")
            modata_13 = load_modata(keymopath, device)
            print("modata_13", modata_13.shape)
            modata_13_temp = modata_13.copy()

            if rgi > 0 and rgi<(global_num-1):
                print("modata13_cat", len(modata13_cat))
                modata_13_temp[:8] = modata13_cat[-1][-8:]
            elif rgi > 0 and rgi == (global_num-1):
                if flag == 0:
                    print("global_num", global_num)
                    print("modata13_cat", len(modata13_cat))
                    modata_13_temp[:8] = modata13_cat[-1][-8:]
                else:
                    modata_13_temp = modata_13[modata_13.shape[0] -8 - ((8* (int(cfg.length1 / cfg.length2) - 1) )*flag) :]
                modata_13_temp[:8] = modata13_cat[-1][-8:]

            modata = modata_13_temp[4:-4]        # 12*8
            # modata = modata.reshape(-1, 8, modata.shape[-1])  
            print("music_fea.shape", music_fea.shape)
            print("modata.shape", modata.shape)

            scale =  8* (int(cfg.length1 / cfg.length2) - 1)       # int( modata.shape[0] /  int(opt.length1 / opt.length2) )
            molist = []
            for item in range(modata.shape[0] // scale):
                print("item", item)
                print("modata[item*scale : (item+1)*scale]", modata[item*scale : (item+1)*scale].shape)
                molist.append(modata[item*scale : (item+1)*scale])
                all_localfilename_cat = all_localfilename_cat + [file_name + 'g' + str(rgi).zfill(3) + 'g_' + 'l' + str(item).zfill(3)]
            print("len(molist)", len(molist))
            print("molist[i].shape", molist[0].shape)
            modata13_cat.append(modata_13)
            molist_cat += molist


        print("file ", file)
        genre = music2genre_[file[:3]]
        genre = np.array(Genres_fd[genre])
        genre = torch.from_numpy(genre).unsqueeze(0)

        data_tuple = None, music_fea_cat, all_localfilename_cat, molist_cat
        model_fine.render_sample(
                data_tuple, "dod", output_dir, render_count=-1, fk_out=fk_out, render=cfg.DEMO.RENDER, setmode=setmode, cons=molist, soft_hint='dod',  device=device, genre=genre     # inpaint_soft_ddim
                )
        

         
        


if __name__ == "__main__":
    # Select DDIM or DDPM
    # setmode = "inpaint_soft"      # Using DDPM. It takes times
    setmode = "inpaint_soft_ddim"   # Using DDIM. Spend less time and get a comparable performance

    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    cfg.length1 = 1024
    cfg.length2 = 256
    cfg.checkpoint1 = 'exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt'
    cfg.checkpoint2 = 'exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt'
    cfg_coarse =  OmegaConf.load('exp/Global_Module/FineDance_Global/global_train.yaml')
    music2genre_ = music2genre("data/finedance/label_json")
    music_dir = "data/finedance/music"  
    print("cfg.soft", cfg.soft)
    

    logger, final_output_dir = create_logger(cfg, phase="demo")
    temp_dir_list = []
    all_cond = []
    all_filenames = []
    print(cfg.checkpoint1.split('.')[0].split('=')[-1])

    print(cfg.checkpoint1.split('.')[0].split('=')[-1])
    output_dir = Path(os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "samples_dod_" + str(cfg.checkpoint1.split('.')[0].split('=')[-1]) + "_" + str(cfg.checkpoint2.split('.')[0].split('=')[-1]) + "_" + setmode + "_notranscontrol_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)

    command = ' '.join(sys.argv)
    with open(os.path.join(output_dir, 'command.txt'), 'a') as f:
        f.write(command)

    test_list = ["063", "193", "132", "143", "036", "098", "198", "130", "012", "120",  "179", "065", "137", "161", "092",  "037", "109", "204", "144", "211"]  
    test(cfg)
    concat_res(output_dir)
