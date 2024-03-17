import numpy as np
import pickle 
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from scipy import linalg
import json, torch, sys
# kinetic, manual
import os, librosa
from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt 
sys.path.append(os.getcwd())
from dld.data.render_joints.smplfk import SMPLX_Skeleton, do_smplxfk



def get_mb(key, length=None):
    path = os.path.join(music_root, key)
    with open(path) as f:
        #print(path)
        sample_dict = json.loads(f.read())
        if length is not None:
            beats = np.array(sample_dict['music_array'])[:, 53][:][:length]
        else:
            beats = np.array(sample_dict['music_array'])[:, 53]

        beats = beats.astype(bool)
        beat_axis = np.arange(len(beats))
        beat_axis = beat_axis[beats]
    
        return beat_axis
    
def get_music_beat_fromwav(fpath, length):
    FPS = 30
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    # EPS = 1e-6
    data, _ = librosa.load(fpath, sr=SR)[:length]
    # print("loaded music data shape", data.shape)
    envelope = librosa.onset.onset_strength(y=data, sr=SR)  # (seq_len,)
    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH
    )
    start_bpm = librosa.beat.tempo(y=data)[0]
    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=SR,
        hop_length=HOP_LENGTH,
        start_bpm=start_bpm,
        tightness=100,
    )
    return beat_idxs


def get_music_beat_from_musicfea35(fpath, length):


    data = np.load(fpath)[:length]
    beat_idxs = data[-1]

    beats = beats.astype(bool)
    beat_axis = np.arange(len(beats))
    beat_axis = beat_axis[beats]
  
    return beat_idxs



def calc_db(keypoints, name=''):
    keypoints = np.array(keypoints).reshape(-1, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats, len(kinetic_vel)


def BA(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats))

def calc_ba_score(motionroot, musicroot):
    # gt_list = []
    ba_scores = []
    test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211",  "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]

    for pkl in os.listdir(motionroot):
        # print(pkl)
        if os.path.isdir(os.path.join(motionroot, pkl)):
            continue
        if pkl[:3] not in test_list:
            continue
        if pkl[-3:] == 'pkl':
            data = pickle.load(open(os.path.join(motionroot, pkl), "rb"))
            print(data.keys())
            model_q = torch.from_numpy(data['smpl_poses'] )   
            model_x = torch.from_numpy(data['smpl_trans'] )   
            print("model_q", model_q.shape)
            print("model_x", model_x.shape)
            model_q156 = torch.cat([model_q, torch.zeros([model_q.shape[0], 90])], dim=-1)
            with torch.no_grad():
                joint3d = smplx_model.forward(model_q156, model_x)[:,:24,:]
                print("joint3d", joint3d.shape)
        elif pkl[-3:] == 'npy':
            data = np.load(os.path.join(motionroot, pkl))
            assert len(data.shape) == 2
            if data.shape[1] == 139 or data.shape[1] == 319:
                data = data[:,:139]
                data = torch.from_numpy(data)  
            elif data.shape[1] == 135 or data.shape[1] == 315:
                data = data[:,:139]
                data = torch.from_numpy(data)  
                data = torch.cat([torch.zeros([data.shape[0], 4]).to(data),  data], dim=1) 
            # print(data.shape)
            assert data.shape[-1] == 139
            with torch.no_grad():
                joint3d = do_smplxfk(data, smplx_model)[:,:24,:]
        else:
            continue
        assert len(joint3d.shape) == 3      # T, J, 3
        joint3d = joint3d.reshape(joint3d.shape[0], 24*3).detach().cpu().numpy()
        roott = joint3d[:1, :3]
        joint3d = joint3d - np.tile(roott, (1, 24)) 
        joint3d = joint3d.reshape(-1, 24, 3)

        # joint3d = np.load(os.path.join(motionroot, pkl), allow_pickle=True).item()['pred_position'][:, :]
        dance_beats, length = calc_db(joint3d, pkl)        
        # music_beats = get_mb(pkl.split('.')[0] + '.json', length)
        music_beats = get_music_beat_fromwav(os.path.join(musicroot, pkl.split('.')[0] + '.wav'), joint3d.shape[0])

        ba_scores.append(BA(music_beats, dance_beats))
        
    return np.mean(ba_scores)

if __name__ == '__main__':
    music_root = "data/finedance/music_wav"
    smplx_model = SMPLX_Skeleton()
    pred_root = '/data2/lrh/project/dance/Lodge/lodge_pub/experiments/Local_Module/FineDance_FineTuneV2_Local/samples_dod_2999_299_inpaint_soft_ddim_notranscontrol_2024-03-16-04-29-01/concat/npy'

    print('Calculating pred metrics')
    print(pred_root)
    print(calc_ba_score(pred_root, music_root))
  