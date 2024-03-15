import sys
import numpy as np
import os
import torch
from visualize.joints2smpl.src import config
import smplx
from smplx import SMPLX
import h5py
from tqdm import tqdm
import argparse
from visualize.joints2smpl.src.smplify import SMPLify3D
from visualize.joints2smpl.src.smplify_smpl import SMPLify3D_smpl_t2m



class joints2smplx:
    def __init__(self, num_frames, device_id, num_joints, joint_category, cuda=True):
        self.device = torch.device("cuda:" + str(device_id) if cuda else "cpu")
        self.batch_size = num_frames
        self.num_joints = num_joints
        self.joint_category = joint_category
        # debug!!!      # 150
        self.num_smplify_iters = 150
        self.fix_foot = False
        print(config.SMPLX_MODEL_DIR)
        smplxmodel = smplx.create(config.SMPLX_MODEL_DIR, use_pca=False,
                                 model_type="smplx", gender="neutral", ext="npz",
                                 batch_size=self.batch_size).to(self.device)

        # ## --- load the mean pose as original ----
        smpl_mean_file = config.SMPL_MEAN_FILE

        file = h5py.File(smpl_mean_file, 'r')
        self.init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(self.device)
        self.init_mean_pose = self.init_mean_pose[:,:22*3]
        print("self.init_mean_pose", self.init_mean_pose.shape)

        self.init_mean_shape  = torch.zeros(self.batch_size, 10).to(self.device)
        self.init_mean_shape[:,1] = -1.2
        print("self.init_mean_shape", self.init_mean_shape.shape)
        self.cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(self.device)

        self.init_jaw_pose = torch.zeros(self.batch_size, 3).to(self.device)
        self.init_leye_pose = torch.zeros(self.batch_size, 3).to(self.device)
        self.init_reye_pose = torch.zeros(self.batch_size, 3).to(self.device)
        self.init_left_hand = torch.zeros(self.batch_size, 45).to(self.device)
        self.init_right_hand = torch.zeros(self.batch_size, 45).to(self.device)
        #

        # # #-------------initialize SMPLify
        self.smplify = SMPLify3D(smplxmodel=smplxmodel,
                            batch_size=self.batch_size,
                            joints_category=self.joint_category,
                            num_iters=self.num_smplify_iters,
                            device=self.device)


    def npy2smpl(self, npy_path):
        out_path = npy_path.replace('.npy', '_rot.npy')
        motions = np.load(npy_path, allow_pickle=True)[None][0]
        # print_batch('', motions)
        n_samples = motions['motion'].shape[0]
        # n_samples = 10
        all_thetas = []
        for sample_i in tqdm(range(n_samples)):
            print(sample_i)
            thetas, _ = self.joint2smpl(motions['motion'][sample_i].transpose(2, 0, 1))  # [nframes, njoints, 3]
            all_thetas.append(thetas.cpu().numpy())
        motions['motion'] = np.concatenate(all_thetas, axis=0)
        print('motions', motions['motion'].shape)

        print(f'Saving [{out_path}]')
        np.save(out_path, motions)
        exit()



    def joint2smpl(self, input_joints, init_params=None, origin_data=None):
        _smplify = self.smplify

        # run the whole seqs
        num_seqs = input_joints.shape[0]


        # joints3d = input_joints[idx]  # *1.2 #scale problem [check first]
        keypoints_3d = torch.Tensor(input_joints).to(self.device).float()

        # if idx == 0:
        if init_params is None:
            pred_betas = self.init_mean_shape
            pred_pose = self.init_mean_pose
            pred_cam_t = self.cam_trans_zero

            pred_jaw_pose = self.init_jaw_pose
            pred_leye_pose  = self.init_leye_pose
            pred_reye_pose  = self.init_reye_pose
            pred_left_hand  = self.init_left_hand
            pred_right_hand = self.init_right_hand 
        else:
            pred_betas = init_params['betas']
            pred_pose = init_params['pose']
            pred_cam_t = init_params['cam']

        if self.joint_category in ["AMASS", "SMPLX55"]:
            confidence_input = torch.ones(self.num_joints)
            # make sure the foot and ankle
            if self.fix_foot == True:
                confidence_input[7] = 1.5
                confidence_input[8] = 1.5
                confidence_input[10] = 1.5
                confidence_input[11] = 1.5
        else:
            print("Such category not settle down!")

        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
        new_opt_cam_t, new_opt_joint_loss = _smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            pred_jaw_pose.detach(),
            pred_leye_pose.detach(),
            pred_reye_pose.detach(), 
            pred_left_hand.detach(), 
            pred_right_hand.detach(),
            keypoints_3d,
            conf_3d=confidence_input.to(self.device),
            origin_data=origin_data,
            # seq_ind=idx
        )

        thetas = new_opt_pose.reshape(self.batch_size, self.num_joints, 3).view(self.batch_size, self.num_joints*3)
        # thetas = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(thetas))  # [bs, 24, 6]
        root_loc = torch.tensor(keypoints_3d[:, 0])  # [bs, 3]
        # root_loc = torch.cat([root_loc, torch.zeros_like(root_loc)], dim=-1).unsqueeze(1)  # [bs, 1, 6]
        thetas = torch.cat([root_loc, thetas], dim=1)   #.unsqueeze(0)  #.permute(0, 2, 3, 1)  # [1, 25, 6, 196]

        return thetas.clone().detach(), {'trans':keypoints_3d[:, 0], 'axis':new_opt_pose, 'pose': new_opt_joints[0, :self.num_joints].flatten().clone().detach(), 'betas': new_opt_betas.clone().detach(), 'cam': new_opt_cam_t.clone().detach()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='Blender file or dir with blender files')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    params = parser.parse_args()

    simplify = joints2smplx(device_id=params.device, cuda=params.cuda)

    if os.path.isfile(params.input_path) and params.input_path.endswith('.npy'):
        simplify.npy2smpl(params.input_path)
    elif os.path.isdir(params.input_path):
        files = [os.path.join(params.input_path, f) for f in os.listdir(params.input_path) if f.endswith('.npy')]
        for f in files:
            simplify.npy2smpl(f)