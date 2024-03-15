import glob
import os
import pickle
import numpy as np
import argparse
from omegaconf import OmegaConf
import sys, glob
from pathlib import Path

from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_matrix, rotation_6d_to_matrix)
import torch
from render import ax_from_6v,ax_to_6v

def get_songlist(modir):
    songlist = []
    for file in os.listdir(modir):
        if file[-3:] not in [ 'npy', 'pkl']:
            continue
        if len(file.split('_')) < 7:
            song = file.split('_')[2].split("g")[0]
            if song not in songlist:
                songlist.append(song)
        else:
            names = file.split('_')[2:7]
            names.append(file.split('_')[7].split('g')[0])
            song = '_'.join(names)
            if song not in songlist:
                songlist.append(song)

    return songlist




def get_repeatnum(modir):
    num = 0
    for file in os.listdir(modir):
        if file[-3:] not in [ 'npy', 'pkl']:
            continue
        # print(file)
        if not "_r" in file:
            return 1
        else:
            if num < int(file.split('.')[0].split('_r')[-1]):
                num = int(file.split('.')[0].split('_r')[-1])
    return num+1


def concat_res(modir):
    songlist = get_songlist(modir)
    print(songlist)
    print(len(songlist))
    repeatnum = 1   # get_repeatnum(modir)
    print("repeatnum is {}".format(str(repeatnum) ))


    catdir = os.path.join(Path(modir), "concat" , 'npy')        
    if not os.path.exists(catdir):
        os.makedirs(catdir)

    quadir = os.path.join(Path(modir), "concat", "qua")      
    if not os.path.exists(quadir):
        os.makedirs(quadir)
    
    for song in songlist:
        one_song = sorted(glob.glob(os.path.join(modir, 'dod' + '*'+ song + '*')))
        print(one_song)
        print("total num", len(one_song))
        idx = 0
        total_num = len(one_song)

        for idx in range(total_num):
            gi = idx //4
            li = idx %4 
            print(os.path.join(modir, 'dod' + '*'+ song + 'g' + str(gi).zfill(3) + 'g_l' + str(li).zfill(3) + '.npy'))
            local_fineme = sorted(glob.glob(os.path.join(modir, 'dod' + '*'+ song + 'g' + str(gi).zfill(3) + 'g_l' + str(li).zfill(3)+ '.npy')))
            if len(local_fineme) == 1:
                local_fineme = local_fineme[0]
            print("local_fineme", local_fineme)

            if local_fineme[-3:] == 'pkl':
                pkl_data = pickle.load(open(local_fineme, "rb"))
                smpl_poses = pkl_data["smpl_poses"].reshape(-1, 22, 3)
                T, J, C = smpl_poses.shape
                smpl_poses = ax_to_6v(torch.from_numpy(smpl_poses)).detach().cpu().numpy().reshape(T, -1)
                modata = np.concatenate((pkl_data["smpl_trans"], smpl_poses), axis=1)
            elif local_fineme[-3:] == 'npy':
                modata = np.load(local_fineme)
            print(modata.shape)
            if idx == 0:
                dance = modata
            else:
                dance = np.concatenate((dance, modata), axis=0)
            print(idx)

        print("danceshape", dance.shape)
        np.save(os.path.join(catdir, song+'.npy'), dance)
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--modir", type=str, default='/data2/lrh/project/dance/Lodge/lodge302/experiments/Local_Module/FineDance_relative_Norm_GenreDis_bc190/samples_dod_inpaint_soft_ddim_2024-03-08-02-47-33') 
    
    args = parser.parse_args()
    args = OmegaConf.create(vars(args))

    concat_res(args.modir)
    



    