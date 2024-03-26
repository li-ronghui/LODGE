# from gc import is_finalized
import pickle
import numpy as np
import torch
import cv2
import os

os.environ["PYOPENGL_PLATFORM"] = "osmesa" 
from tqdm import tqdm
from smplx import SMPL, SMPLX, SMPLH
import pyrender
import trimesh
import subprocess
import pickle
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_matrix, rotation_6d_to_matrix)

import sys
import argparse
from typing import NewType
Tensor = NewType('Tensor', torch.Tensor)

# from utils.smplfk import SMPLX_Skeleton
# from dataset.preprocess.foot_process.set_on_groud import set_on_ground_139

def quat_to_6v(q):
    assert q.shape[-1] == 4
    mat = quaternion_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat


def quat_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    quat = matrix_to_quaternion(mat)
    return quat


def ax_to_6v(q):
    assert q.shape[-1] == 3
    mat = axis_angle_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat


def ax_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    ax = matrix_to_axis_angle(mat)
    return ax

class MovieMaker():
    def __init__(self, save_path) -> None:
        
        self.mag = 2
        self.eyes = np.array([[3,-3,2], [0,0,-2], [0,0,4], [-8,-8,1], [0,-2,4], [0,2,4]])
        self.centers = np.array([[0,0,0],[0,0,0],[0,0.5,0],[0,0,-1], [0,0.5,0], [0,0.5,0]])
        self.ups = np.array([[0,0,1],[0,1,0],[0,1,0],[0,0,-1], [0,1,0], [0,1,0]])
        self.save_path = save_path
        
        self.fps = args.fps
        self.img_size = (1200,1200)

        ipadd = "10.103.11.45"
        if ipadd == "10.103.11.45":
            SMPLH_path = "/data/human/datasets/smpl_model/smplh/SMPLH_MALE.pkl"
            SMPL_path = "/data/human/datasets/smpl_model/smpl/SMPL_MALE.pkl"
            SMPLX_path = "/data/human/datasets/smpl_model/smplx/SMPLX_NEUTRAL.npz"
            trimesh_path = '/data2/lrh/floor/NORMAL_new.obj'
        elif ipadd == "10.103.11.40":
            SMPLH_path = "/data/lrh/smpl_model/smplh/SMPLH_MALE.pkl"
            SMPL_path =  "/data/lrh/smpl_model/smpl/SMPL_MALE.pkl"
            trimesh_path = '/data/lrh/floor/NORMAL_new.obj'
        else:
            raise("error of machine ip")


        self.smplh = SMPLH(SMPLH_path, use_pca=False, flat_hand_mean=True)
        self.smplh.to(f'cuda:{args.device}').eval()
        
        self.smpl = SMPL(SMPL_path)
        self.smpl.to(f'cuda:{args.device}').eval()

        self.smplx = SMPLX(SMPLX_path, use_pca=False, flat_hand_mean=True).eval()
        self.smplx.to(f'cuda:{args.device}').eval()

        self.scene = pyrender.Scene()
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = look_at(self.eyes[5], self.centers[5], self.ups[5])       # 2
        self.scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        self.scene.add(light, pose=camera_pose)
        self.r = pyrender.OffscreenRenderer(self.img_size[0], self.img_size[1])
        
        self.mesh = trimesh.load(trimesh_path)
        floor_mesh  = pyrender.Mesh.from_trimesh(self.mesh)   
        floor_node = self.scene.add(floor_mesh)


    def save_video(self, save_path, color_list):
        # save_path = os.path.join(save_path,'move.mp4')
        f = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videowriter = cv2.VideoWriter(save_path,f,self.fps,self.img_size)
        for i in range(len(color_list)):
            videowriter.write(color_list[i][:,:,::-1])
        videowriter.release()

    def get_imgs(self, motion):
        meshes = self.motion2mesh(motion)
        imgs = self.render_imgs(meshes)
        return np.concatenate(imgs, axis=1)

    def motion2mesh(self, motion):
        if args.mode == "smpl":
            output = self.smpl.forward(
                betas = torch.zeros([motion.shape[0], 10]).to(motion.device),
                transl = motion[:,:3],
                global_orient = motion[:,3:6],
                body_pose = torch.cat([motion[:,6:69], motion[:,69:72], motion[:,114:117]], dim=1)
                )
        elif args.mode == "smplh":
            output = self.smplh.forward(
                betas = torch.zeros([motion.shape[0], 10]).to(motion.device),
                transl = motion[:,:3],
                # transl = torch.tensor([[0,0,-1]]).expand(motion.shape[0],-1).to(motion.device) ,
                global_orient = motion[:,3:6],
                body_pose = motion[:,6:69],
                left_hand_pose = motion[:,69:114],
                right_hand_pose = motion[:,114:159],
                )
        elif args.mode == "smplx":
            output = self.smplx.forward(
                betas = torch.zeros([motion.shape[0], 10]).to(motion.device),
                # transl = motion[:,:3],
                transl = motion[:,:3],
                global_orient = motion[:,3:6],
                body_pose = motion[:,6:69],
                jaw_pose = torch.zeros([motion.shape[0], 3]).to(motion),
                leye_pose = torch.zeros([motion.shape[0], 3]).to(motion),
                reye_pose = torch.zeros([motion.shape[0], 3]).to(motion),
                left_hand_pose = torch.zeros([motion.shape[0], 45]).to(motion),
                right_hand_pose = torch.zeros([motion.shape[0], 45]).to(motion),
                expression= torch.zeros([motion.shape[0], 10]).to(motion),
                )

        
        meshes = []
        for i in range(output.vertices.shape[0]):
            if args.mode == 'smplh':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplh.faces)
            elif args.mode == 'smplx':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplx.faces)
            elif args.mode == 'smpl':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smpl.faces)
            # mesh.export(os.path.join(self.save_path, f'{i}.obj'))
            meshes.append(mesh)
        
        return meshes


    def render_multi_view(self, meshes, music_file, tab='', eyes=None, centers=None, ups=None, views=1):
        if eyes and centers and ups:
            assert eyes.shape == centers.shape == ups.shape
        else:
            eyes = self.eyes
            centers = self.centers
            ups = self.ups
        
        for i in range(views):
            color_list = self.render_single_view(meshes, eyes[1], centers[1], ups[1])
            movie_file = os.path.join(self.save_path, tab + '-' + str(i) + '.mp4')
            output_file = os.path.join(self.save_path, tab + '-' + str(i) + '-music.mp4')
            self.save_video(movie_file, color_list)
            if music_file is not None:
                # subprocess.run(['/home/lrh/Documents/ffmpeg-6.0-amd64-static/ffmpeg','-i',movie_file,'-i',music_file,'-shortest',output_file])
                subprocess.run(['ffmpeg','-i',movie_file,'-i',music_file,'-shortest',output_file])
            else:
                # subprocess.run(['/home/lrh/Documents/ffmpeg-6.0-amd64-static/ffmpeg','-i',movie_file,output_file])
                subprocess.run(['ffmpeg','-i',movie_file,output_file])
                os.remove(movie_file)

            
            

    def render_single_view(self, meshes):
        num = len(meshes)
        color_list = []
        for i in tqdm(range(num)):
            mesh_nodes = []
            for mesh in meshes[i]:
                render_mesh = pyrender.Mesh.from_trimesh(mesh)   
                mesh_node = self.scene.add(render_mesh)
                mesh_nodes.append(mesh_node)
            color, _ = self.r.render(self.scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            color = color.copy()
            color_list.append(color)
            for mesh_node in mesh_nodes:
                self.scene.remove_node(mesh_node)
        return color_list
    
    def render_imgs(self, meshes):
        colors = []
        for mesh in meshes:
            render_mesh = pyrender.Mesh.from_trimesh(mesh)   
            mesh_node = self.scene.add(render_mesh)
            color, _ = self.r.render(self.scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            colors.append(color)
            self.scene.remove_node(mesh_node)


        return colors
        # cv2.imwrite(os.path.join(self.save_path, 'test.jpg'), color[:,:,::-1])
    
    def run(self, seq_rot, music_file=None, tab='', save_pt=False):
        if isinstance(seq_rot, np.ndarray):
            seq_rot = torch.tensor(seq_rot, dtype=torch.float32, device=f'cuda:{args.device}')

        if save_pt:
            torch.save(seq_rot.detach().cpu(), os.path.join(self.save_path, tab +'_pose.pt'))

        B, D = seq_rot.shape
        if args.mode == "smpl":
            print("using smpl!!!")
            output = self.smpl.forward(
                betas = torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
                transl = seq_rot[:,:3],
                global_orient = seq_rot[:,3:6],
                body_pose = torch.cat([seq_rot[:,6:69], seq_rot[:,69:72], seq_rot[:,114:117]], dim=1)
                )
        
        elif args.mode == "smplh":
            print("using smplh!!!")
            output = self.smplh.forward(
                betas = torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
                transl = seq_rot[:,:3],
                global_orient = seq_rot[:,3:6],
                body_pose = seq_rot[:,6:69],
                left_hand_pose =  seq_rot[:,69:114],  # torch.zeros([seq_rot.shape[0], 45]).to(seq_rot.device),      # seq_rot[:,69:114],
                right_hand_pose = seq_rot[:,114:],    # torch.zeros([seq_rot.shape[0], 45]).to(seq_rot.device),      # 
                expression = torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
                )
            
        elif args.mode == "smplx":
            output = self.smplx.forward(
                betas = torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
                # transl = motion[:,:3],
                transl = seq_rot[:,:3],
                global_orient = seq_rot[:,3:6],
                body_pose = seq_rot[:,6:69],
                jaw_pose = torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
                leye_pose = torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
                reye_pose = torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
                left_hand_pose = torch.zeros([seq_rot.shape[0], 45]).to(seq_rot),
                right_hand_pose = torch.zeros([seq_rot.shape[0], 45]).to(seq_rot),
                expression= torch.zeros([seq_rot.shape[0], 10]).to(seq_rot),
                )
        
        N, V, DD = output.vertices.shape                # 150, 6890, 3
        vertices = output.vertices.reshape((B, -1, V, DD))  #  # 150, 1, 6890, 3
        
        meshes = []
        for i in range(B):
            # debug!
            # if int(i) > 10 and int(i) < 500:
            #     continue
            # if not int(i) % 4 == 0:
            #     continue
            # if int(i) > 1200:
            #     break
            view = []
            for v in vertices[i]:
                # vertices[:,2] *= -1
                if args.mode == 'smplh':
                    mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplh.faces)
                elif args.mode == 'smplx':
                    mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplx.faces)
                elif args.mode == 'smpl':
                    mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smpl.faces)
                # mesh.export(os.path.join(self.save_path, 'test.obj'))
                view.append(mesh)
            meshes.append(view)

        color_list = self.render_single_view(meshes)
        movie_file = os.path.join(self.save_path, tab + 'tmp.mp4')
        output_file = os.path.join(self.save_path, tab + 'z.mp4')
        self.save_video(movie_file, color_list)
        if music_file is not None:
            subprocess.run(['/home/lrh/Documents/ffmpeg-6.0-amd64-static/ffmpeg','-i',movie_file,'-i',music_file,'-shortest',output_file])
        else:

            subprocess.run(['/home/lrh/Documents/ffmpeg-6.0-amd64-static/ffmpeg','-i',movie_file,output_file])
        os.remove(movie_file)


def look_at(eye, center, up):
    front = eye - center
    front = front / np.linalg.norm(front)
    right = np.cross(up, front)
    right = right/ np.linalg.norm(right)
    up_new = np.cross(front, right)
    camera_pose = np.eye(4)
    camera_pose[:3,:3] = np.stack([right, up_new, front]).transpose()
    camera_pose[:3,3] = eye
    return camera_pose


def motion_data_load_process(motionfile):
    if motionfile.split(".")[-1] == "pkl":
        pkl_data = pickle.load(open(motionfile, "rb"))
        smpl_poses = pkl_data["smpl_poses"]
        modata = np.concatenate((pkl_data["smpl_trans"], smpl_poses), axis=1)
        if modata.shape[1] == 69:
            hand_zeros = np.zeros([modata.shape[0], 90], dtype=np.float32)
            modata = np.concatenate((modata, hand_zeros), axis=1)
        assert modata.shape[1] == 159
        modata[:, 1] = modata[:, 1] + 0 # + 1.25
        return modata
    elif motionfile.split(".")[-1] == "npy":
        modata = np.load(motionfile)
        if len(modata.shape) == 3 and modata.shape[1]%8==0:
            print("modata has 3 dim , reshape the batch to time!!!")
            modata = modata.reshape(-1, modata.shape[-1])
        if modata.shape[-1] == 315:
            print("modata.shape is:", modata.shape)
            rot6d = torch.from_numpy(modata[:,3:])
            T,C = rot6d.shape
            rot6d = rot6d.reshape(-1,6)
            axis = ax_from_6v(rot6d).view(T,-1).detach().cpu().numpy()
            modata = np.concatenate((modata[:,:3], axis), axis=1)
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 319:
            print("modata.shape is:", modata.shape)
            modata = modata[:,4:]
            rot6d = torch.from_numpy(modata[:,3:])
            T,C = rot6d.shape
            rot6d = rot6d.reshape(-1,6)
            axis = ax_from_6v(rot6d).view(T,-1).detach().cpu().numpy()
            modata = np.concatenate((modata[:,:3], axis), axis=1)
            # print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 159:
            print("modata.shape is:", modata.shape)
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 135:
            print("modata.shape is:", modata.shape)
            if len(modata.shape) == 3 and modata.shape[0] ==1:
                modata = modata.squeeze(0)
            rot6d = torch.from_numpy(modata[:,3:])
            T,C = rot6d.shape
            rot6d = rot6d.reshape(-1,6)
            axis = ax_from_6v(rot6d).view(T,-1).detach().cpu().numpy()
            hand_zeros = torch.zeros([T, 90]).to(rot6d).detach().cpu().numpy()
            modata = np.concatenate((modata[:,:3], axis, hand_zeros), axis=1)
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 139:
            print("modata.shape is:", modata.shape)
            modata = modata[:,4:]
            rot6d = torch.from_numpy(modata[:,3:])
            T,C = rot6d.shape
            rot6d = rot6d.reshape(-1,6)
            axis = ax_from_6v(rot6d).view(T,-1).detach().cpu().numpy()
            hand_zeros = torch.zeros([T, 90]).to(rot6d).detach().cpu().numpy()
            modata = np.concatenate((modata[:,:3], axis, hand_zeros), axis=1)
            print("modata.shape is:", modata.shape)
        else:
            raise("shape error!")
            
        modata[:, 1] = modata[:, 1] + 0 #+ 1.25
        return modata
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--modir", type=str, default="")
    parser.add_argument("--mode", type=str, default="smplx", choices=['smpl','smplh','smplx'])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--song", type=str, default=None)
    args = parser.parse_args()
    print(args.device)


    motion_dir = args.modir
    if args.save_path is not None:
        save_path = args.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = os.path.join(motion_dir, 'video')
        os.makedirs(save_path, exist_ok=True)


    music_dir = "experiments/DanceDiffuse_module/debug--0517_Norm_512len_315_transloss/val1640/samples_2023-05-17-20-54-05"
    for file in os.listdir(motion_dir):
        if file[-3:] in ["npy", "pkl"]:

            # if there have exist rendered video, continue
            flag = False
            for exists_file in os.listdir(save_path):
                if file[:-4] in exists_file:
                    flag = True
                    break
                else:
                    flag = False
            if flag:
                print("exist", file)
                continue

            print(file)
            motion_file = os.path.join(motion_dir, file)
            visualizer = MovieMaker(save_path=save_path)
            modata = motion_data_load_process(motion_file)
            visualizer.run(modata, tab=os.path.basename(motion_file).split(".")[0], music_file=None)
    print('done')
