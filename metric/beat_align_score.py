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
        
        # fig, ax = plt.subplots()
        # ax.set_xticks(beat_axis, minor=True)
        # # ax.set_xticks([0.3, 0.55, 0.7], minor=True)
        # ax.xaxis.grid(color='deeppink', linestyle='--', linewidth=1.5, which='minor')
        # ax.xaxis.grid(True, which='minor')


        # print(len(beats))
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
        # 起始位置归0
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
    music_root = "/data2/lrh/dataset/fine_dance/origin/music"
    # aa = np.random.randn(39, 72)*
    # bb = np.random.randn(39, 72)*0.1
    # print(calc_fid(aa, bb))
    smplx_model = SMPLX_Skeleton()
    # gt_root = '/mnt/lustre/lisiyao1/dance/bailando/aist_features_zero_start'
    # pred_root = '/mnt/lustressd/lisiyao1/dance_experiements/experiments/sep_vqvae_root_global_vel_wav_acc_batch8/vis/pkl/ep000500'
    # pred_root = ''
    # pred_root = '/mnt/lustressd/lisiyao1/dance_experiements/experiments/music_ccgpt2_ac_ba_1e-5_freeze_droupout_beta0.5/vis/pkl/ep000005'
    
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_gpt_ds8_lbin512_c512_di3full/eval/pkl/ep000300'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_cross_cond_gpt_ds8_lbin512_c512_di3_init_0.01_beta0.9_full_dim768_3_9_9_ac_reward2_with_entropy_loss_alpha0.5_lr1e-4_no_pretrain/eval/pkl/ep000020'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_cross_cond_gpt_ds8_lbin512_c512_di3_init_0.01_beta0.9_full_dim768_3_9_9_ac_reward2_with_entropy_loss_alpha0.5_lr1e-4_no_pretrain/vis/pkl/ep000300'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_cross_cond_gpt_ds8_lbin512_c512_di3_init_0.01_beta0.9_full_dim768_wav_bsz_16_layer6/eval/pkl/ep000040'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/sep_vqvae_root_data_l1_d8_local_c512_di3_global_vel_full_beta0.9_1e-4_wav_beta0.5/eval/pkl/ep000300'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_cross_cond_gpt_ds8_lbin512_c512_di3_init_0.01_beta0.9_full_dim768_wav/eval/pkl/ep000300'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_cross_cond_gpt_ds8_lbin512_c512_di3_init_0.01_beta0.9_full_dim768_666_ac_reward2_with_entropy_loss_alpha0.5_lr1e-4_no_pretrain/vis/pkl/ep000080'
    # print('Calculating and saving features')

    # gt_root = '/data2/lrh/dataset/fine_dance/origin/motion_feature319'
    # pred_root = '/data2/lrh/project/dance/long/experiments/results/zero/1023edge139_256_35/bce_fc2/inferdodsoft/whole/test2/concat/npy'
    # pred_root = '/data2/lrh/project/dance/long/experiments/infer/clip8_139/FineDance_1007_1024_win128/1012/test1_soft_with4ctc'

    # pred_root = '/data2/lrh/project/dance/long/experiments/compare/mnet/data1min'
    # pred_root = '/data2/lrh/project/dance/long/experiments/compare/fact/data1min'
    # pred_root = 'experiments/compare/gpt/experiments/infer/clip8_139/FineDance_1007_1024_win128/1107/res1_hint0'
    # pred_root = 'experiments/compare/edge/edge_5_long/2175/npy'
    # pred_root = '/data2/lrh/project/dance/long/experiments/results/zero/1023edge139_256_35/bce_fc2/inferdodsoft/whole/res1_hint0_trans/concat/npy'
    # pred_root = '/data2/lrh/project/dance/long/experiments/results/zero/1023edge139_256_35/bce_fc2/inferdodsoft/whole/res1_hint0/concat/npy'
    # pred_root = '/data2/lrh/project/dance/long/experiments/results/zero/1023edge139_256_35/bce_fc2/inferdodsoft/whole/res1_hint100/concat/npy'


    # pred_root = '/data2/lrh/project/dance/long/experiments/results/zero/1023edge139_256_35/bce_fc2/inferdodsoft/whole/res1_hint0/concat/npy'
    # pred_root = '/data2/lrh/dataset/fine_dance/gound/mofea319/testset'
    # pred_root= "experiments/results/zero/1023edge139_256_35/bce_fc2/inferdodsoft/whole/res1_hint0_beat_nomirror/concat/npy"


    # pred_root = '/data2/lrh/project/dance/long/experiments/results/zero/1023edge139_256_35/bce_fc2/inferdodsoft/ddim_0/concat/npy'
    # pred_root = '/data2/lrh/project/dance/Lodge/lodge302/experiments/Local_Module/AFineDance_FineTuneV2_originweight_relative_Norm_GenreDis_bc190/samples_dod2999_299_inpaint_soft_notranscontrol_2024-03-11-21-05-27/concat/npy'
    # pred_root = '/data2/lrh/project/dance/Lodge/lodge302/experiments/Local_Module/AFineDance_FineTuneV2_originweight_relative_Norm_GenreDis_bc190/samples_dod2999_549_inpaint_soft_notranscontrol_2024-03-11-21-05-44/concat/npy/'
    # pred_root = 'experiments/Local_Module/AFineDance_FineTuneV2_originweight_relative_Norm_GenreDis_bc190_nofc/samples_dod2999_199_inpaint_soft_notranscontrol_2024-03-12-04-53-44/concat/npy'
    pred_root = 'dance_results/Local_Module/AFineDance_NoFootBlock_V2_originweight_relative_Norm_GenreDis_bc190/samples_dod2999_1499_inpaint_soft_notranscontrol_2024-03-14-05-01-56/concat/npy'
    # pred_root =  '/data2/lrh/dataset/fine_dance/gound/mofea319/'




    print('Calculating pred metrics')
    print(pred_root)
    print(calc_ba_score(pred_root, music_root))
    # print('Calculating gt metrics')
    # print(calc_ba_score(gt_root, music_root))

    # print('Calculating metrics')
    # print(gt_root)
    # print(pred_root)
    # print(quantized_metrics(pred_root, gt_root))