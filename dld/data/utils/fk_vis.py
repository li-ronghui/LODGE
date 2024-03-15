import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import librosa as lr
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import numpy as np
import soundfile as sf
import torch
from matplotlib import cm
from matplotlib.colors import ListedColormap
from pytorch3d.transforms import (axis_angle_to_quaternion, quaternion_apply,
                                  quaternion_multiply)
from tqdm import tqdm
from typing import NewType
Tensor = NewType('Tensor', torch.Tensor)
import torch.nn.functional as F
import pickle5 as pickle


smpl_joints = [
    "root",  # 0
    "lhip",  # 1
    "rhip",  # 2
    "belly", # 3
    "lknee", # 4
    "rknee", # 5
    "spine", # 6
    "lankle",# 7
    "rankle",# 8
    "chest", # 9
    "ltoes", # 10
    "rtoes", # 11
    "neck",  # 12
    "linshoulder", # 13
    "rinshoulder", # 14
    "head", # 15
    "lshoulder", # 16
    "rshoulder",  # 17
    "lelbow", # 18
    "relbow",  # 19
    "lwrist", # 20
    "rwrist", # 21
    "lhand", # 22
    "rhand", # 23
]

smplh_joints = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3'
]


smplx_joints = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw','left_eye_smplhf','right_eye_smplhf',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3'
]


smpl_parents = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]

smplh_parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
        16, 17, 18, 19, 20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
        35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50]

smplx_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53]


smpl_offsets = [
    [0.0, 0.0, 0.0],
    [0.05858135, -0.08228004, -0.01766408],
    [-0.06030973, -0.09051332, -0.01354254],
    [0.00443945, 0.12440352, -0.03838522],
    [0.04345142, -0.38646945, 0.008037],
    [-0.04325663, -0.38368791, -0.00484304],
    [0.00448844, 0.1379564, 0.02682033],
    [-0.01479032, -0.42687458, -0.037428],
    [0.01905555, -0.4200455, -0.03456167],
    [-0.00226458, 0.05603239, 0.00285505],
    [0.04105436, -0.06028581, 0.12204243],
    [-0.03483987, -0.06210566, 0.13032329],
    [-0.0133902, 0.21163553, -0.03346758],
    [0.07170245, 0.11399969, -0.01889817],
    [-0.08295366, 0.11247234, -0.02370739],
    [0.01011321, 0.08893734, 0.05040987],
    [0.12292141, 0.04520509, -0.019046],
    [-0.11322832, 0.04685326, -0.00847207],
    [0.2553319, -0.01564902, -0.02294649],
    [-0.26012748, -0.01436928, -0.03126873],
    [0.26570925, 0.01269811, -0.00737473],
    [-0.26910836, 0.00679372, -0.00602676],
    [0.08669055, -0.01063603, -0.01559429],
    [-0.0887537, -0.00865157, -0.01010708],
]


def set_line_data_3d(line, x):
    line.set_data(x[:, :2].T)
    line.set_3d_properties(x[:, 2])


def set_scatter_data_3d(scat, x, c):
    scat.set_offsets(x[:, :2])
    scat.set_3d_properties(x[:, 2], "z")
    scat.set_facecolors([c])


def get_axrange(poses):
    pose = poses[0]
    x_min = pose[:, 0].min()
    x_max = pose[:, 0].max()

    y_min = pose[:, 1].min()
    y_max = pose[:, 1].max()

    z_min = pose[:, 2].min()
    z_max = pose[:, 2].max()

    xdiff = x_max - x_min
    ydiff = y_max - y_min
    zdiff = z_max - z_min

    biggestdiff = max([xdiff, ydiff, zdiff])
    return biggestdiff


def plot_single_pose(num, poses, lines, ax, axrange, scat, contact, ske_parents):
    pose = poses[num]
    static = contact[num]
    indices = [7, 8, 10, 11]

    for i, (point, idx) in enumerate(zip(scat, indices)):
        position = pose[idx : idx + 1]
        color = "r" if static[i] else "g"
        set_scatter_data_3d(point, position, color)

    for i, (p, line) in enumerate(zip(ske_parents, lines)):
        # don't plot root
        if i == 0:
            continue
        # stack to create a line
        data = np.stack((pose[i], pose[p]), axis=0)
        set_line_data_3d(line, data)

    if num == 0:
        if isinstance(axrange, int):
            axrange = (axrange, axrange, axrange)
        xcenter, ycenter, zcenter = 0, 0, 2.5
        stepx, stepy, stepz = axrange[0] / 2, axrange[1] / 2, axrange[2] / 2

        x_min, x_max = xcenter - stepx, xcenter + stepx
        y_min, y_max = ycenter - stepy, ycenter + stepy
        z_min, z_max = zcenter - stepz, zcenter + stepz

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)


def skeleton_render(
    poses,
    epoch=0,
    out="renders",
    name="",
    sound=True,
    stitch=False,
    sound_folder="ood_sliced",
    contact=None,
    render=True,
    smpl_mode="smpl",       # 是否渲染双手
):
    if render:
        if smpl_mode=="smpl":
            poses = np.concatenate((poses[:, :23, :], np.expand_dims(poses[:, 37, :], axis=1)), axis=1)
            ske_parents = smpl_parents
        elif smpl_mode == "smplx":
            ske_parents = smplx_parents
        
        # generate the pose with FK
        Path(out).mkdir(parents=True, exist_ok=True)
        num_steps = poses.shape[0]      # 
        
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        
        point = np.array([0, 0, 1])
        normal = np.array([0, 0, 1])
        d = -point.dot(normal)
        xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 2), np.linspace(-1.5, 1.5, 2))
        z = (-normal[0] * xx - normal[1] * yy - d) * 1.0 / normal[2]
        # plot the plane
        ax.plot_surface(xx, yy, z, zorder=-11, cmap=cm.twilight)
        # Create lines initially without data
        lines = [
            ax.plot([], [], [], zorder=10, linewidth=1.5)[0]
            for _ in ske_parents
        ]
        scat = [
            ax.scatter([], [], [], zorder=10, s=0, cmap=ListedColormap(["r", "g", "b"]))
            for _ in range(4)
        ]
        axrange = 3

        # create contact labels
        feet = poses[:, (7, 8, 10, 11)]
        feetv = np.zeros(feet.shape[:2])
        feetv[:-1] = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)
        if contact is None:
            contact = feetv < 0.01
        else:
            contact = contact > 0.95

        # Creating the Animation object
        anim = animation.FuncAnimation(
            fig,
            plot_single_pose,
            num_steps,
            fargs=(poses, lines, ax, axrange, scat, contact, ske_parents),
            interval=1000 // 30,
        )
    if sound:
        # make a temporary directory to save the intermediate gif in
        if render:
            temp_dir = TemporaryDirectory()
            gifname = os.path.join(temp_dir.name, f"{epoch}.gif")
            anim.save(gifname)

        # stitch wavs
        if stitch:
            assert type(name) == list  # must be a list of names to do stitching
            name_ = [os.path.splitext(x)[0] + ".wav" for x in name]
            audio, sr = lr.load(name_[0], sr=None)
            ll, half = len(audio), len(audio) // 2
            total_wav = np.zeros(ll + half * (len(name_) - 1))
            total_wav[:ll] = audio
            idx = ll
            for n_ in name_[1:]:
                audio, sr = lr.load(n_, sr=None)
                total_wav[idx : idx + half] = audio[half:]
                idx += half
            # save a dummy spliced audio
            audioname = f"{temp_dir.name}/tempsound.wav" if render else os.path.join(out, f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.wav')
            sf.write(audioname, total_wav, sr)
            outname = os.path.join(
                out,
                f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.mp4',
            )
        else:
            assert type(name) == str
            assert name != "", "Must provide an audio filename"
            audioname = name
            outname = os.path.join(
                out, f"{epoch}_{os.path.splitext(os.path.basename(name))[0]}.mp4"
            )
        if render:
            print(f"ffmpeg -loglevel error -stream_loop 0 -y -i {gifname} -i {audioname} -shortest -c:v libx264 -crf 26 -c:a aac -q:a 4 {outname}")
            out = os.system(
                f"/home/lrh/Documents/ffmpeg-6.0-amd64-static/ffmpeg -loglevel error -stream_loop 0 -y -i {gifname} -i {audioname} -shortest -c:v libx264 -crf 26 -c:a aac -q:a 4 {outname}"
            )
    else:
        if render:
            # actually save the gif
            path = os.path.normpath(name)
            pathparts = path.split(os.sep)
            gifname = os.path.join(out, f"{pathparts[-1][:-4]}.gif")
            anim.save(gifname, savefig_kwargs={"transparent": True, "facecolor": "none"},)
    plt.close()


class SMPLSkeleton:
    def __init__(
        self, device=None,
    ):
        offsets = smpl_offsets
        parents = smpl_parents
        assert len(offsets) == len(parents)

        # self._offsets = torch.Tensor(offsets).to(device)
        self._offsets = torch.Tensor(offsets)
        self._parents = np.array(parents)
        self._compute_metadata()

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

    def forward(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 3) tensor of axis-angle rotations describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        # print("rotations shape", rotations.shape)
        # print("root_positions shape", root_positions.shape)
        assert len(rotations.shape) == 4
        assert len(root_positions.shape) == 3
        self._offsets = self._offsets.to(rotations.device)
        # transform from axis angle to quaternion
        N, L, J, _ = rotations.shape
        rotations = axis_angle_to_quaternion(rotations)
        # print("rotations shape", rotations.shape)
        if rotations.shape[2] == 22:
            rotations = torch.cat([rotations, torch.zeros([N, L ,2, 4],  device=rotations.device, dtype=torch.float32)], dim=2)


        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(
            rotations.shape[0],
            rotations.shape[1],
            self._offsets.shape[0],
            self._offsets.shape[1],
        )

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(
                    quaternion_apply(
                        rotations_world[self._parents[i]], expanded_offsets[:, :, i]
                    )
                    + positions_world[self._parents[i]]
                )
                if self._has_children[i]:
                    rotations_world.append(
                        quaternion_multiply(
                            rotations_world[self._parents[i]], rotations[:, :, i]
                        )
                    )
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)



class SMPLX_Skeleton:
    def __init__(
        self, device=None,
    ):
        # offsets = smpl_offsets
        self.device = device
        self.parents = smplx_parents
        # assert len(offsets) == len(parents)

        # self._offsets = torch.Tensor(offsets).to(device)
        
        
    def batch_rodrigues(self, rot_vecs: Tensor, epsilon: float = 1e-8,) -> Tensor:
        ''' Calculates the rotation matrices for a batch of rotation vectors
            Parameters
            ----------
            rot_vecs: torch.tensor Nx3
                array of N axis-angle vectors
            Returns
            -------
            R: torch.tensor Nx3x3
                The rotation matrices for the given axis-angle parameters
        '''
        batch_size = rot_vecs.shape[0]
        device, dtype = rot_vecs.device, rot_vecs.dtype

        angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
        rot_dir = rot_vecs / angle

        cos = torch.unsqueeze(torch.cos(angle), dim=1)
        sin = torch.unsqueeze(torch.sin(angle), dim=1)

        # Bx1 arrays
        rx, ry, rz = torch.split(rot_dir, 1, dim=1)
        K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

        zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
            .view((batch_size, 3, 3))

        ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
        rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
        return rot_mat

    def batch_rigid_transform(self,
        rot_mats: Tensor,
        joints: Tensor,
        parents: Tensor,
        dtype=torch.float32
    ) -> Tensor:
        """
        Applies a batch of rigid transformations to the joints

        Parameters
        ----------
        rot_mats : torch.tensor BxNx3x3
            Tensor of rotation matrices
        joints : torch.tensor BxNx3
            Locations of joints
        parents : torch.tensor BxN
            The kinematic tree of each object
        dtype : torch.dtype, optional:
            The data type of the created tensors, the default is torch.float32

        Returns
        -------
        posed_joints : torch.tensor BxNx3
            The locations of the joints after applying the pose rotations
        rel_transforms : torch.tensor BxNx4x4
            The relative (with respect to the root joint) rigid transformations
            for all the joints
        """

        joints = torch.unsqueeze(joints, dim=-1)
        # joints_check = joints.detach().cpu().numpy()

        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]

        transforms_mat = self.transform_mat(
            rot_mats.reshape(-1, 3, 3),
            rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, parents.shape[0]):
            # Subtract the joint location at the rest pose
            # No need for rotation, since it's identity when at rest
            curr_res = torch.matmul(transform_chain[parents[i]],
                                    transforms_mat[:, i])
            transform_chain.append(curr_res)

        transforms = torch.stack(transform_chain, dim=1)

        # The last column of the transformations contains the posed joints
        posed_joints = transforms[:, :, :3, 3]

        # joints_homogen = F.pad(joints, [0, 0, 0, 1])

        # rel_transforms = transforms - F.pad(
        #     torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

        return posed_joints #, rel_transforms

    def transform_mat(self, R: Tensor, t: Tensor) -> Tensor:
        ''' Creates a batch of transformation matrices
            Args:
                - R: Bx3x3 array of a batch of rotation matrices
                - t: Bx3x1 array of a batch of translation vectors
            Returns:
                - T: Bx4x4 Transformation matrix
        '''
        # No padding left or right, only add an extra row
        return torch.cat([F.pad(R, [0, 0, 0, 1]),
                        F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

    def motion_data_load_process(self, motionfile):
        if motionfile.split(".")[-1] == "pkl":
            pkl_data = pickle.load(open(motionfile, "rb"))
            if "pos" in pkl_data.keys():
                local_q_165 = torch.from_numpy(pkl_data["q"]).to(self.device).float()
                root_pos = torch.from_numpy(pkl_data["pos"]).to(self.device).float()
                root_pos = root_pos[:, :]  - root_pos[0, :]
                return local_q_165, root_pos  
            else:
                smpl_poses = pkl_data["smpl_poses"]         
                if smpl_poses.shape[0] != 150 and smpl_poses.shape[0] != 300:
                    smpl_poses = smpl_poses.reshape(150, -1)
                # modata = np.concatenate((pkl_data["smpl_trans"], smpl_poses), axis=1)
                # assert modata.shape[1] == 159
                # modata = torch.from_numpy(modata).to(f'cuda:{args.gpu}')
                root_pos = pkl_data["smpl_trans"]
                
                local_q = torch.from_numpy(smpl_poses).to(self.device).float()
                root_pos = torch.from_numpy(root_pos).to(self.device).float()
                local_q_165 = torch.cat([local_q[:, :66], torch.zeros([local_q.shape[0], 9], device=local_q.device, dtype=torch.float32), local_q[:, 66:]], dim=1).to(self.device).float()
                root_pos = root_pos[:, :]  - root_pos[0, :]
                return local_q_165, root_pos
        

    def forward(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (B, S, 55, 3)  或 (B, S, 52, 3) 或 (B, S, 55*3)  (B, S, 52*3)
         -- root_positions: (N, 3) 
         输出: N, 55, 3 关节点全局坐标
        """
        # assert len(rotations.shape) == 4
        # assert len(root_positions.shape) == 3
        # print(fk_device)
        if len(root_positions.shape) == 3:
            b, s, c = root_positions.shape
            root_positions = root_positions.view(b*s, -1)
        if len(rotations.shape) == 4:               # (B, S, 55, 3)  或 (B, S, 52, 3)
            b, s, c, _ = rotations.shape
            rotations = rotations.view(b*s, -1)     # (B, S, 55*3)  (B, S, 52*3)
        elif len(rotations.shape) == 3:
            b, s, _ = rotations.shape
            rotations = rotations.view(b*s, -1)
        assert len(rotations.shape) == 2
        assert len(root_positions.shape) == 2
        
        fk_device = rotations.device
        if rotations.shape[1] == 156:
            local_q_165 = torch.cat([rotations[:, :66], torch.zeros([rotations.shape[0], 9], device=fk_device, dtype=torch.float32), rotations[:, 66:]], dim=1).to(fk_device).float()
        elif rotations.shape[1] == 165:
            local_q_165 = rotations.to(fk_device).float()
        elif rotations.shape[1] == 66:
            local_q_165 = torch.cat([rotations[:, :66], torch.zeros([rotations.shape[0], 99], device=fk_device, dtype=torch.float32)], dim=1).to(fk_device).float()
        else:
            raise("rotations shape error!  rotations.shape is :", rotations.shape)
            # sys.exit(0)
        
        root_pos = root_positions.to(fk_device).float()
        assert local_q_165.shape[1] == 165
        
        
        B, C = local_q_165.shape
        # print("local_q shape is:", local_q_165.shape)
        rot_mats = self.batch_rodrigues(local_q_165.view(-1, 3)).view(
                [B, -1, 3, 3])
        J = np.load("dld/data/utils/smplx_neu_J_1.npy")
        J = torch.from_numpy(J).to(fk_device).unsqueeze(dim=0).repeat(B, 1, 1)

        self.parents = torch.Tensor(self.parents).long()
        J_transformed = self.batch_rigid_transform(rot_mats, J, self.parents, dtype=torch.float32)
        J_transformed += root_pos.unsqueeze(dim=1)
        # J_transformed = J_transformed.detach().cpu().numpy()

        return J_transformed
    
    
if __name__ == "__main__":
    print("1")
    device = f'cuda:{0}'
    
    smplx_fk = SMPLX_Skeleton(device = device)
    
    motion_file = "/home/data/lrh/datasets/fine_dance/magicsmpl/sliced/test/dances/012_slice0.pkl"
    music_file = "/home/data/lrh/datasets/fine_dance/magicsmpl/sliced/test/wavs/012_slice0.wav"
    local_q_165, root_pos = smplx_fk.motion_data_load_process(motion_file)
    
    joints = smplx_fk.forward(local_q_165, root_pos).detach().cpu().numpy()            # 150, 165     150, 3
    skeleton_render(
                joints,
                epoch=f"e{1}_b{1}",
                out="./output/temp",
                name=music_file,
                render=True,
                stitch=False,
                sound=True,
                smpl_mode="smplx"
            )
    
