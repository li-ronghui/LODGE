from enum import Flag
import sys
import numpy as np
import torch
import os, pickle
from render_aist import ax_from_6v, ax_to_6v, quat_from_6v, quat_to_6v, quat_from_ax
def quat_slerp(x, y, a):
    """
    Performs spherical linear interpolation (SLERP) between x and y, with proportion a

    :param x: quaternion tensor (N, S, J, 4)        (S, J, 4)
    :param y: quaternion tensor (N, S, J, 4)        (S, J, 4)
    :param a: interpolation weight (S, )
    :return: tensor of interpolation results
    """
    # flag = False
    # if len(x.shape) == 3:
    flag = True
    x = torch.unsqueeze(x, 0)
    y = torch.unsqueeze(y, 0)
    a = torch.unsqueeze(a, 0)

    len = torch.sum(x * y, axis=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    print("1", torch.zeros_like(x[..., 0]).shape)
    print("a", a.shape)

    a = torch.zeros_like(x[..., 0]) + a

    amount0 = torch.zeros_like(a)
    amount1 = torch.zeros_like(a)

    linear = (1.0 - len) < 0.01
    omegas = torch.arccos(len[~linear])
    sinoms = torch.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = torch.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = torch.sin(a[~linear] * omegas) / sinoms

    # reshape
    amount0 = amount0[..., None]
    amount1 = amount1[..., None]

    res = amount0 * x + amount1 * y
    
    if flag:
        res = torch.squeeze(res, 0)
    return res




def linear_interpolation_trajectory(coordinates):
    T, _ = coordinates.shape
    interpolated_coordinates = np.zeros((2*T - 1, 3))

    for i in range(T - 1):
        t0 = coordinates[i]
        t1 = coordinates[i + 1]

        # 线性插值
        interpolated_coordinates[2 * i] = t0
        interpolated_coordinates[2 * i + 1] = 0.5 * (t0 + t1)

    # 处理最后一个时间步
    interpolated_coordinates[2 * (T - 1)] = coordinates[T - 1]

    return interpolated_coordinates



# coordinates = np.random.rand(3,3)
# result_coordinates = linear_interpolation_trajectory(coordinates)
# print(coordinates)
# print(result_coordinates)
# print("result_coordinates", result_coordinates.shape)

# result_coordinates 的形状为（2T, 3）

# 示例使用
# 假设你的四元数tensor为 quaternion_tensor，形状为（T, 22, 4）
# 进行球面线性插值
# quaternion_tensor = np.random.rand(3,22,4)

def twice_qua(quaternion_tensor):
    # quaternion_tensor = torch.from_numpy(quaternion_tensor)
    left = quaternion_tensor[:-1]
    right = quaternion_tensor[1:]
    weight = (torch.ones_like(left) * 0.5)[:,:,0]
    print(weight)
    mid_tensor = quat_slerp(left, right, weight)
    print("mid_tensor", mid_tensor.shape)

    T = quaternion_tensor.shape[0]
    result_tensor = torch.zeros([2*T - 1, 22, 4])
    for i in range(T-1):
    
        result_tensor[2 * i] = quaternion_tensor[i]
        result_tensor[2 * i + 1] = mid_tensor[i]
    result_tensor[2 * (T - 1)] = quaternion_tensor[T - 1]

    print("quaternion_tensor", quaternion_tensor[:,0])
    print("result_tensor", result_tensor[:,0])
    print("result_tensor", result_tensor.shape)

    return result_tensor



if __name__ == '__main__':
    modir = '/data2/lrh/project/dance/Lodge/lodge302/experiments/Local_Module/AISTPP_Fine_Norm_128len_139/samples_dod_2499_2099_inpaint_soft_ddim_notranscontrol_2024-03-14-23-48-11/concat/npy' 
    outdir = '/data2/lrh/project/dance/Lodge/lodge302/experiments/Local_Module/AISTPP_Fine_Norm_128len_139/samples_dod_2499_2099_inpaint_soft_ddim_notranscontrol_2024-03-14-23-48-11/concat/twice' 

    if not os.path.exists(outdir):
        os.makedirs(outdir)


    for file in os.listdir(modir):
        if file[-3:] not in  ['npy', 'pkl']:
            continue
        mofile = os.path.join(modir, file)
        if file[-3:] == 'npy':
            data = np.load(mofile)
            if data.shape[1] == 139:
                data = torch.from_numpy(data[:,:139])
                trans = data[:,4:7].clone()
                poses6d = data[:,7:].reshape(data.shape[0], 22, 6).clone()
                # axis = poses6d.reshape(data.shape[0], -1, 6)
                # axis = ax_from_6v(torch.from_numpy(axis))       #.detach().cpu().numpy()
                qua = quat_from_6v(poses6d)
                print("qua", qua.shape)
            elif data.shape[1] == 135:
                data = torch.from_numpy(data[:,:135])
                trans = data[:,:3].clone()
                poses6d = data[:,3:].reshape(data.shape[0], 22, 6).clone()
                # axis = poses6d.reshape(data.shape[0], -1, 6)
                # axis = ax_from_6v(torch.from_numpy(axis))       #.detach().cpu().numpy()
                qua = quat_from_6v(poses6d)
                print("qua", qua.shape)
        elif file[-3:] == 'pkl':

            data = pickle.load(open((mofile), "rb"))
            print(data.keys())
            axis = torch.from_numpy(data['smpl_poses'] ).reshape(-1,22, 3)
            print("axis", axis.shape)
            trans = torch.from_numpy(data['smpl_trans'] ).clone()

            qua = quat_from_ax(axis)
            # print("poses6d", poses6d.shape)

        

        trans = trans.detach().cpu().numpy()
        result_trans = linear_interpolation_trajectory(trans)
        print("result_trans", result_trans.shape)

        result_qua = twice_qua(qua)
        print("result_qua", result_qua.shape)
        result_6v = quat_to_6v(result_qua)
        print("result_6v", result_6v.shape)

        result_6v = result_6v.reshape(result_6v.shape[0], 22*6).detach().cpu().numpy()
        data139 = np.concatenate([np.zeros([result_6v.shape[0],4]), result_trans, result_6v ], axis = 1)
        print("data139", data139.shape)

        savepath = os.path.join(outdir, file)
        np.save(savepath, data139)

