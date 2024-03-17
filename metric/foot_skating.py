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




if __name__ == '__main__':
    test_dir = '/data2/lrh/project/dance/Lodge/lodge_pub/experiments/Local_Module/FineDance_FineTuneV2_Local/samples_dod_2999_299_inpaint_soft_ddim_notranscontrol_2024-03-16-04-29-01/concat/npy'

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
