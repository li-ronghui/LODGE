from cgi import test
from turtle import left
import numpy as np
import torch
import os, sys
sys.path.append(os.getcwd())
from dld.data.render_joints.smplfk import do_smplxfk, ax_from_6v, ax_to_6v, SMPLX_Skeleton



def set_on_ground_139(data, smplx_model, ground_h=0):
    length = data.shape[0]
    assert len(data.shape) == 2
    assert data.shape[1] == 139
    positions = do_smplxfk(data, smplx_model)
    l_toe_h = positions[0, 10, 1]
    r_toe_h = positions[0, 11, 1]
    if abs(l_toe_h - r_toe_h) < 0.02:
        height = (l_toe_h + r_toe_h)/2
    else:
        height = min(l_toe_h, r_toe_h)
    data[:, 5] = data[:, 5] - (height -  ground_h)

    return data

def calc_foot_skating_ratio(data):
    assert len(data.shape) == 2
    smplx = SMPLX_Skeleton()
    if data.shape[1] == 139 or data.shape[1] == 319:
        data = torch.from_numpy(data[:,:139])
    elif data.shape[1] == 135 or data.shape[1] == 315:
        data = torch.from_numpy(data[:,:135])
        data = torch.cat([torch.zeros([data.shape[0], 4]).to(data), data], dim=-1)
        assert data.shape[1] == 139
    else:
        print(data.shape)
        raise("input data shape error!")
    data = set_on_ground_139(data, smplx, ground_height)
    with torch.no_grad():
        model_xp = do_smplxfk(data, smplx)

    l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
    relevant_joints = [l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx]
    pred_joint_xyz = model_xp[:, relevant_joints, :] 
    pred_vel = torch.zeros_like(pred_joint_xyz)
    pred_vel[:-1] = (
        pred_joint_xyz[1:, :, :] - pred_joint_xyz[:-1, :, :]
    )  # (S-1, 4, 3)
    left_foot_y_ankle = model_xp[ :, l_ankle_idx, 1]
    right_foot_y_ankle = model_xp[ :, r_ankle_idx, 1]
    left_foot_y_toe = model_xp[:, l_foot_idx, 1]
    right_foot_y_toe = model_xp[:, r_foot_idx, 1]

    print("pred_vel.shape", pred_vel.shape)
    left_fc_mask = (left_foot_y_ankle <= (ground_height +0.08)) & (left_foot_y_toe <= (ground_height +0.05))
    right_fc_mask = (right_foot_y_ankle <= (ground_height +0.08)) & (right_foot_y_toe <= (ground_height +0.05))
    # fc_mask_y = torch.cat([fc_mask_ankle, fc_mask_teo], dim=1).squeeze(0)
    left_pred_vel = torch.cat([pred_vel[:, 0:1, :], pred_vel[:, 2:3, :]], dim=1)
    right_pred_vel = torch.cat([pred_vel[:, 1:2, :], pred_vel[:, 3:4, :]], dim=1)
    # pred_vel[~fc_mask_v] = 0
    left_pred_vel[~left_fc_mask] = 0
    right_pred_vel[~right_fc_mask] = 0
    print("left_fc_mask", left_fc_mask.shape)
    left_static_num = torch.sum(left_fc_mask)
    print("left_static_num", left_static_num)
    right_static_num = torch.sum(right_fc_mask)
    print("right_static_num", right_static_num)
    # sys.exit(0)
    left_velocity_foot_tangent = torch.cat([left_pred_vel[:, :, 0:1], left_pred_vel[:, :, 2:3] ], dim=2)
    left_velocity_foot_tangent = torch.abs(torch.mean(left_velocity_foot_tangent, dim=-1))        # T, 4
    right_velocity_foot_tangent = torch.cat([right_pred_vel[:, :, 0:1], right_pred_vel[:, :, 2:3] ], dim=2)
    right_velocity_foot_tangent = torch.abs(torch.mean(right_velocity_foot_tangent, dim=-1))        # T, 4
    print("right_velocity_foot_tangent.shape", right_velocity_foot_tangent.shape)

    left_velocity_foot_tangent = left_velocity_foot_tangent > 0.01 #(0.05 / 30)          # 0.025     # 0.1/30
    right_velocity_foot_tangent = right_velocity_foot_tangent > 0.01 #(0.05 / 30)
    left_slide_frames = torch.any(left_velocity_foot_tangent, dim=-1)
    left_slide_num = torch.sum(left_slide_frames)
    left_ratio = left_slide_num / left_static_num
    left_ratio = left_ratio.item()
    print("left_ratio", left_ratio)

    right_slide_frames = torch.any(right_velocity_foot_tangent, dim=-1)
    right_slide_num = torch.sum(right_slide_frames)
    right_ratio = right_slide_num / right_static_num
    right_ratio = right_ratio.item()
    print("right_ratio", right_ratio)




    # static_ratio = (slide_num / static_num).item()
    # print("static_ratio", static_ratio)

    return left_ratio, right_ratio

# def calc_foot_skating_ratio(data):
#     assert len(data.shape) == 2
#     smplx = SMPLX_Skeleton()
#     if data.shape[1] == 139 or data.shape[1] == 319:
#         data = torch.from_numpy(data[:,:139])
#     elif data.shape[1] == 135 or data.shape[1] == 315:
#         data = torch.from_numpy(data[:,:135])
#         data = torch.cat([torch.zeros([data.shape[0], 4]).to(data), data], dim=-1)
#         assert data.shape[1] == 139
#     else:
#         print(data.shape)
#         raise("input data shape error!")
#     data = set_on_ground_139(data, smplx, -1.2)
#     with torch.no_grad():
#         model_xp = do_smplxfk(data, smplx)

#     # 地面loss 地面-1.2
#     l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
#     relevant_joints = [l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx]
#     pred_joint_xyz = model_xp[:, relevant_joints, :] 
#     pred_vel = torch.zeros_like(pred_joint_xyz)
#     pred_vel[:-1] = (
#         pred_joint_xyz[1:, :, :] - pred_joint_xyz[:-1, :, :]
#     )  # (S-1, 4, 3)
#     foot_y_ankle = pred_joint_xyz[ :, :2, 1]
#     foot_y_toe = pred_joint_xyz[:, 2:, 1]
#     print("pred_vel.shape", pred_vel.shape)
#     velocity_foot_normal = torch.linalg.norm(pred_vel[:, :, 1:2], axis=2)  # [B,t,4] 
#     velocity_foot_tangent = torch.cat([pred_vel[:, :, 0:1], pred_vel[:, :, 2:3] ], dim=2)
#     # fc_mask_v = torch.unsqueeze((velocity_foot_normal <= 0.01), dim=2).repeat(1, 1, 1, 3)
#     # fc_mask_ankle = torch.unsqueeze((foot_y_ankle <= (-1.2+0.05)), dim=2).repeat(1, 1, 1, 3)
#     # fc_mask_teo = torch.unsqueeze((foot_y_toe <= (-1.2+0.05)), dim=2).repeat(1, 1, 1, 3)
#     fc_mask_ankle = (foot_y_ankle <= (-1.2+0.05))
#     fc_mask_teo = (foot_y_toe <= (-1.2+0.05))
#     fc_mask_y = torch.cat([fc_mask_ankle, fc_mask_teo], dim=1).squeeze(0)
#     # pred_vel[~fc_mask_v] = 0
#     print("fc_mask_y", fc_mask_y.shape)
#     pred_vel[~fc_mask_y] = 0
#     static_num = torch.sum(torch.any(fc_mask_y.view(fc_mask_y.shape[0], 4), dim=-1))
#     print("static_num", static_num)
#     velocity_foot_tangent = torch.cat([pred_vel[:, :, 0:1], pred_vel[:, :, 2:3] ], dim=2)
#     velocity_foot_tangent = torch.abs(torch.mean(velocity_foot_tangent, dim=-1))        # T, 4
#     print("velocity_foot_tangent.shape", velocity_foot_tangent.shape)
#     velocity_foot_tangent = velocity_foot_tangent > 0.025
#     slide_frames = torch.any(velocity_foot_tangent, dim=-1)
#     slide_num = torch.sum(slide_frames)
#     ratio = slide_num / data.shape[0]
#     ratio = ratio.item()
#     print("ratio", ratio)
#     static_ratio = (slide_num / static_num).item()
#     print("static_ratio", static_ratio)

#     return ratio, static_ratio



if __name__ == '__main__':
    test_dir = '/data2/lrh/dataset/fine_dance/gound/mofea319/testset'
    # test_dir = '/data2/lrh/project/dance/long/experiments/compare/fact/data1min'
    # test_dir = '/data2/lrh/project/dance/long/experiments/compare/mnet/data1min'
    # test_dir = '/data2/lrh/dataset/fine_dance/origin/motion_feature319/testset'
    # test_dir = 'experiments/compare/edge/edge_5_long/2175/npy'
    # test_dir = 'experiments/compare/gpt/experiments/infer/clip8_139/FineDance_1007_1024_win128/1107/res1_hint0'
    # test_dir = '/data2/lrh/project/dance/long/experiments/results/zero/1023edge139_256_35/bce_fc2/inferdodsoft/whole/res1_hint0/concat/npy'
    # test_dir = 'experiments/results/zero/1023edge139_256_35/bce_fc2/inferdodsoft/whole/res1_hint200/concat/npy'

    # test_dir = '/data2/lrh/project/dance/long/experiments/results/zero/1023edge139_256_35/bce_fc2/inferdodsoft/ddim_10/concat/npy'
    # test_dir = '/data2/lrh/project/dance/Lodge/lodge302/experiments/Local_Module/AFineDance_FineTuneV2_originweight_relative_Norm_GenreDis_bc190/samples_dod2999_549_inpaint_soft_notranscontrol_2024-03-11-21-05-44/concat/npy'
    # test_dir  = '/data2/lrh/project/dance/Lodge/lodge302/experiments/Local_Module/AFineDance_FineTuneV2_originweight_relative_Norm_GenreDis_bc190/samples_dod2999_299_inpaint_soft_notranscontrol_2024-03-11-21-05-27/concat/npy'
    # test_dir = 'experiments/Local_Module/AFineDance_FineTuneV2_originweight_relative_Norm_GenreDis_bc190_nofc/samples_dod2999_199_inpaint_soft_notranscontrol_2024-03-12-04-53-44/concat/npy'
    test_dir = 'dance_results/Local_Module/AFineDance_NoFootBlock_V2_originweight_relative_Norm_GenreDis_bc190/samples_dod2999_1499_inpaint_soft_notranscontrol_2024-03-14-05-01-56/concat/npy'

    ground_height = 0
    left_ratio_list = []
    right_ratio_list = []
    ratio_list = []
    for file in os.listdir(test_dir):
        if file[-3:] != 'npy':
            continue
        filepath = os.path.join(test_dir, file)
        data = np.load(filepath)  
        left_ratio, right_ratio = calc_foot_skating_ratio(data)
        left_ratio_list.append(left_ratio)
        right_ratio_list.append(right_ratio)
        
    print("final ")
    print("left_ratio:", np.mean(left_ratio_list))
    print("right_ratio:", np.mean(right_ratio_list))
    print("ratio:", (np.mean(right_ratio_list) + np.mean(left_ratio_list))/2 )
    print('test_dir', test_dir)
