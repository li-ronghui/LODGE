import argparse
import os
from pydoc import doc
from cv2 import mean
import numpy as np
from pathlib import Path
import torch
import sys
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append(os.getcwd()) 
from dld.data.render_joints.smplfk import SMPLX_Skeleton, do_smplxfk, ax_to_6v, ax_from_6v


floor_height = 0


def vectorize_many(data):
    # given a list of batch x seqlen x joints? x channels, flatten all to batch x seqlen x -1, concatenate
    batch_size = data[0].shape[0]
    seq_len = data[0].shape[1]

    out = [x.reshape(batch_size, seq_len, -1).contiguous() for x in data]

    global_pose_vec_gt = torch.cat(out, dim=2)
    return global_pose_vec_gt

def set_on_ground(root_pos, local_q_156, smplx_model):
    # root_pos = root_pos[:, :] - root_pos[:1, :]
    length = root_pos.shape[0]
    # model_q = model_q.view(b*s, -1)
    # model_x = model_x.view(-1, 3)
    positions = smplx_model.forward(local_q_156, root_pos)
    positions = positions.view(length, -1, 3)   # bxt, j, 3
    
    l_toe_h = positions[0, 10, 1] - floor_height
    r_toe_h = positions[0, 11, 1] - floor_height
    if abs(l_toe_h - r_toe_h) < 0.02:
        height = (l_toe_h + r_toe_h)/2
    else:
        height = min(l_toe_h, r_toe_h)
    root_pos[:, 1] = root_pos[:, 1] - height

    return root_pos, local_q_156

def set_on_ground_139(data, smplx_model, ground_h=0):
    length = data.shape[0]
    assert len(data.shape) == 2
    assert data.shape[1] == 139
    positions = do_smplxfk(data, smplx_model)
    l_toe_h = positions[0, 10, 1] - floor_height
    r_toe_h = positions[0, 11, 1] - floor_height
    if abs(l_toe_h - r_toe_h) < 0.02:
        height = (l_toe_h + r_toe_h)/2
    else:
        height = min(l_toe_h, r_toe_h)
    data[:, 5] = data[:, 5] - (height -  ground_h)

    return data

def motion_feats_extract(moinputs_dir, mooutputs_dir, music_indir, music_outdir):

    device = "cpu"
    print("extracting")
    raw_fps = 30
    data_fps = 30
    data_fps <= raw_fps
    device = "cpu"
    smplx_model = SMPLX_Skeleton()

    os.makedirs(mooutputs_dir, exist_ok=True)
    os.makedirs(music_outdir, exist_ok=True)
        
    motions = sorted(glob.glob(os.path.join(moinputs_dir, "*.npy")))
    for motion in tqdm(motions):
        print(motion)
        data = np.load(motion)
        fname = os.path.basename(motion).split(".")[0]
        mname = fname if 'M' not in fname else fname[1:]
        music_fea = np.load(os.path.join(music_indir, mname+".npy"))
        if mname in ["010", "014"]:
            data = data[3:]
            music_fea = music_fea[3:]
        # The following dances, and the first few seconds of the movement is very monotonous, in order to avoid the impact on the network, we do not train this small part of the movement
        if mname == '004':
            data = data[8*30:]
            music_fea = music_fea[8*30:]
        if mname == '005':
            data = data[10*30:]
            music_fea = music_fea[10*30:]
        if mname == '067':
            data = data[6*30:]
            music_fea = music_fea[6*30:]
        if mname == '105':
            data = data[19*30:]
            music_fea = music_fea[19*30:]
        if mname == '110':
            data = data[14*30:]
            music_fea = music_fea[14*30:]
        if mname == '113':
            data = data[29*30:]
            music_fea = music_fea[29*30:]
        if mname == '153':
            data = data[52*30:]
            music_fea = music_fea[52*30:]
        if mname == '211':
            data = data[22*30:]
            music_fea = music_fea[22*30:]
        # 004:8
        # 005:10
        # 067:6
        # 105:19
        # 110:14
        # 113:29
        # 153:52
        # 211:22
        np.save(os.path.join(music_outdir, mname+".npy"), music_fea)

        if data.shape[1] == 315:
            pos = data[:, :3]   # length, c
            q = data[:, 3:]
        elif data.shape[1] == 319:
            pos = data[:, 4:7]   # length, c
            q = data[:, 7:]
        print("data.shape", data.shape)
        print("pos.shape", pos.shape)
        print("q.shape", q.shape)
        root_pos = torch.Tensor(pos).to(device) # 150, 3
        local_q = torch.Tensor(q).to(device).view(q.shape[0], 52, 6)    # 150, 165
        local_q = ax_from_6v(local_q)
        length = root_pos.shape[0]
        local_q = local_q.view(length, -1, 3)  
        print("local_q", local_q.shape)
        local_q_156 = local_q.view(length, 156)
        root_pos, local_q_156 = set_on_ground(root_pos, local_q_156, smplx_model)
        positions = smplx_model.forward(local_q_156, root_pos)
        positions = positions.view(length, -1, 3)   # bxt, j, 3

        # contacts

        feet = positions[:, (7, 8, 10, 11)]  # # 150, 4, 3
        contacts_d_ankle = (feet[:,:2,1] < 0.12).to(local_q_156)
        contacts_d_teo = (feet[:,2:,1] < 0.05).to(local_q_156)
        contacts_d = torch.cat([contacts_d_ankle, contacts_d_teo], dim=-1).detach().cpu().numpy()


        local_q_156 = local_q_156.view(length, 52, 3)  
        local_q_312 = ax_to_6v(local_q_156).view(length,312).detach().cpu().numpy()
        print("contacts_d.shape", contacts_d.shape)
        print("root_pos.shape", root_pos.shape)
        print("local_q_312.shape", local_q_312.shape)
        mofeats_input = np.concatenate( [contacts_d, root_pos, local_q_312] ,axis=-1)
        np.save(os.path.join(mooutputs_dir, fname+".npy"), mofeats_input)
        print("mofeats_input", mofeats_input.shape)
    return


if __name__ == "__main__":
    motion_feats_extract(#moinputs_dir='/data2/lrh/dataset/fine_dance/origin/motion_feature315', 
                        moinputs_dir='data/finedance/motion/', 
                        mooutputs_dir="data/finedance/mofea319/", 
                        music_indir="data/finedance/music_npy", 
                        # music_indir="/data2/lrh/dataset/fine_dance/origin/music_feature35_edge",
                        music_outdir="data/finedance/music_npynew/", )
