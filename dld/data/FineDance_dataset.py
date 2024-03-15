# from sympy import O
import torch
from torch.utils import data
import numpy as np
import os
from tqdm import tqdm
import json
# import torchgeometry as tgy
from dld.data.render_joints.smplfk import set_on_ground_139, SMPLX_Skeleton

import sys
sys.path.insert(0,'.')
# from utils.parser_util import args

SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]

SMPLX_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13,
                        15, 17, 16, 19, 18, 21, 20, 22, 24, 23,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
SMPLX_POSE_FLIP_PERM = []
for i in SMPLX_JOINTS_FLIP_PERM:
    SMPLX_POSE_FLIP_PERM.append(3*i)
    SMPLX_POSE_FLIP_PERM.append(3*i+1)
    SMPLX_POSE_FLIP_PERM.append(3*i+2)

def flip_pose(pose):
    #Flip pose.The flipping is based on SMPLX parameters.
    pose = pose[:,SMPLX_POSE_FLIP_PERM]
    # we also negate the second and the third dimension of the axis-angle
    pose[:,1::3] = -pose[:,1::3]
    pose[:,2::3] = -pose[:,2::3]
    return pose

Genres_aist = {
    'gBR': 0,
    'gPO': 1,
    'gLO': 2,
    'gMH': 3,
    'gLH': 4,
    'gHO': 5,
    'gWA': 6,
    'gKR': 7,
    'gJS': 8,
    'gJB': 9,
}

Genres_fd = {            # Breaking
          'Breaking': 0,
          'Popping': 1,
          'Locking': 2,
          'Hiphop':3,
          'Urban':4,
          'Jazz':5,
          'jazz':5,

          'Tai':6,
          'Uighur':7,
          'Hmong':8,
          'Dai':6,
          'Wei':7,
          'Miao':8,

          'HanTang':9,
          'ShenYun':10,
          'Kun':11,
          'DunHuang':12,

          'Korean':13,
          'Choreography':14,
          'Chinese':15,
}

def music2genre(label_dir):
    music_genre = {}
    for file in os.listdir(label_dir):
        name = file.split(".")[0]
        jsonfile = os.path.join(label_dir, file)
        with open(jsonfile,"r") as f:
            genredict = json.load(f)
        genre = genredict['style2']

        music_genre[name] = genre

    return music_genre

class FineDance_Smpl(data.Dataset):
    def __init__(self, args, istrain, dataname=None):
        self.motion_dir =  eval(f"args.DATASET.{dataname.upper()}.MOTION")     #'/data/lrh/datasets/fine_dance/origin/motion_feature319'
        self.music_dir = eval(f"args.DATASET.{dataname.upper()}.MUSIC")        #'/data/lrh/datasets/fine_dance/origin/music_feature35'
        if 'FINEDANCE' in dataname:
            self.music2genre = music2genre(eval(f"args.DATASET.{dataname.upper()}.LABEL"))

        self.istrain = istrain
        self.args = args
        self.seq_len = args.FINEDANCE.full_seq_len
        slide = args.FINEDANCE.full_seq_len // args.FINEDANCE.windows

        self.motion_index = []
        self.music_index = []
        motion_all = []
        music_all = []
        genre_list = []

        
        ignor_list, train_list, test_list = self.get_train_test_list(dataset = dataname)
        if self.istrain:
            self.datalist= train_list
        else:
            self.datalist = test_list

        total_length = 0            # 将数据集中的所有motion用同一个index索引

        # debug_num = 0
        for name in tqdm(self.datalist):
            # debug_num += 1
            # if debug_num>30:
            #     break

            name = name + ".npy"
            
            if name[:-4] in ignor_list:
                continue
            
            motion = np.load(os.path.join(self.motion_dir, name))

            # wheather change the ground height?  if changed, please be careful of normalizer and foot loss!
            # motion = torch.from_numpy(motion)[:, :139]
            # motion = set_on_ground_139(motion, smplx_model, -1.2)
            # motion = motion.detach().cpu().numpy()

            if dataname == "AISTPP":
                motion = motion[::2]
            music = np.load(os.path.join(self.music_dir, name))

            min_all_len = min(motion.shape[0], music.shape[0])
            motion = motion[:min_all_len]
          
            if motion.shape[-1] == 168:
                motion = np.concatenate([motion[:,:69], motion[:,78:]], axis=1)     # 22,  25
            elif motion.shape[-1] == 319 and args.FINEDANCE.nfeats ==139:
                motion = motion[:,:139]
            elif motion.shape[-1] == 315 or motion.shape[-1] == 139:
                if motion.shape[-1] == 139:
                    motion = motion[:,:139]
            elif motion.shape[-1] == 263 or motion.shape[-1] == 266:
                # motion = (motion - self.mean) / self.std
                pass
            else:
                print("motion.shape", motion.shape)
                raise("input motion shape error! not 168 or 319!")
            music = music[:min_all_len]        
            nums = (min_all_len-self.seq_len) // slide + 1          # 舍弃了最后一段不满seq_len的motion


            if 'FINEDANCE' in dataname:
                genre = self.music2genre[name.split(".")[0]]
                # print("genre1 ", genre)
                genre = np.array(Genres_fd[genre])
                genre = torch.from_numpy(genre).unsqueeze(0)
            elif 'AISTPP' in dataname:
                genre = name.split('_')[0]
                genre = np.array(Genres_aist[genre])
                genre = torch.from_numpy(genre).unsqueeze(0)
            
            if self.istrain:
                clip_index = []
                for i in range(nums):
                    motion_clip = motion[i * slide: i * slide + self.seq_len]
                    if motion_clip.std(axis=0).mean() > 0.07:           # judge wheather the motion clip is effective
                        clip_index.append(i)
                index = np.array(clip_index) * slide + total_length     # clip_index is local index 
                genre_list = genre_list + len(clip_index)*[genre]
            else:
                index = np.arange(nums) * slide + total_length
                genre_list = genre_list + nums*[genre]

            if args.FINEDANCE.mix:
                motion_index = []
                music_index = []
                num = (len(index) - 1) // 8 + 1
                for i in range(num):
                    motion_index_tmp, music_index_tmp = np.meshgrid(index[i*8:(i+1)*8], index[i*8:(i+1)*8])         
                    motion_index += motion_index_tmp.reshape((-1)).tolist()
                    music_index += music_index_tmp.reshape((-1)).tolist()
            else:
                motion_index = index.tolist()
                music_index = index.tolist()

            
            self.motion_index += motion_index
            self.music_index += music_index
            total_length += min_all_len

            assert len(self.motion_index) == len(genre_list)
            motion_all.append(motion)
            music_all.append(music)

        self.motion = np.concatenate(motion_all, axis=0).astype(np.float32)
        self.music = np.concatenate(music_all, axis=0).astype(np.float32)
        self.genre_list = genre_list

        self.len = len(self.motion_index)
        print(f'FineDance has {self.len} samples..')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        motion_index = self.motion_index[index]
        music_index = self.music_index[index]
        motion = self.motion[motion_index:motion_index+self.seq_len]
        music = self.music[music_index:music_index+self.seq_len]
        genre = self.genre_list[index]

        return motion, music, genre
    
    def get_train_test_list(self, dataset="FineDance"):
        if dataset in ["AISTPP", "AISTPP_60FPS"]:
            train = []
            test = []
            ignore = []

            train_file = open('/data2/lrh/dataset/aist/data/origin/aist_plusplus_final/splits/crossmodal_train.txt', 'r')
            for fname in train_file.readlines():
                train.append(fname.strip())
            train_file.close()

            test_file = open('/data2/lrh/dataset/aist/data/origin/aist_plusplus_final/splits/crossmodal_test.txt', 'r')
            for fname in test_file.readlines():
                test.append(fname.strip())
            test_file.close()
                              
            test_file = open('/data2/lrh/dataset/aist/data/origin/aist_plusplus_final/splits/crossmodal_val.txt', 'r')
            for fname in test_file.readlines():
                test.append(fname.strip())
            test_file.close()

            ignore_file = open('/data2/lrh/dataset/aist/data/origin/aist_plusplus_final/ignore_list.txt', 'r')
            for fname in ignore_file.readlines():
                ignore.append(fname.strip())
            ignore_file.close()

            return ignore, train, test

        elif dataset == "AISTPP_LONG263":
            train = []
            test = []
            ignore = []
            print("modir", self.motion_dir)
            for file in os.listdir(self.motion_dir):
                if file[-4:] != '.npy':
                    continue
                file = file.split('.')[0]
                if file.split('_')[-1] in ['mLH5', 'mJS4', 'mBR3', 'mMH2', 'mPO1', 'mWA0']:
                    test.append(file)
                else:
                    train.append(file)

            return  ignore, train, test


        else:
            all_list = []
            train_list = []
            for i in range(1,212):
                all_list.append(str(i).zfill(3))
    
            test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
            ignor_list = ["116", "117", "118", "119", "120", "121", "122", "123", "202"]
            # flist = ['106','008','011', '012','017','018','019','021','023','024','025','026','027','028','030','031','034','035','038','039','040','043','044','045','048','105','104','108','109','110','111','112','115','148','155','170'] + ['190'] + ['208','200','185','180','173','152'] + ['154'] + ['046','113', '153']
            tradition_list = ['005', '007', '008', '015', '017', '018', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '032', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '126', '127', '132', '133', '134',  '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '151', '152', '153', '154', '155', '170']
            morden_list = []
            for one in all_list:
                if one not in tradition_list:
                    morden_list.append(one)

            ignor_list = ignor_list
            for one in all_list:
                if one not in test_list:
                    train_list.append(one)
            
            if self.args.FINEDANCE.partial == 'full':
                return ignor_list, train_list, test_list
            elif self.args.FINEDANCE.partial == 'morden':
                for one in train_list:
                    if one in tradition_list:
                        train_list.remove(one)
                for one in test_list:
                    if one in tradition_list:
                        test_list.remove(one)
                return ignor_list, train_list, test_list
            elif self.args.FINEDANCE.partial == 'tradition':
                for one in train_list:
                    if one in morden_list:
                        train_list.remove(one)
                for one in test_list:
                    if one in morden_list:
                        test_list.remove(one)
                return ignor_list, train_list, test_list



class DoubleDance(data.Dataset):
    def __init__(self, args, istrain, dataname=None):
        self.motion_dir =  eval(f"args.DATASET.{dataname.upper()}.MOTION")     #'/data/lrh/datasets/fine_dance/origin/motion_feature319'
        self.music_dir = eval(f"args.DATASET.{dataname.upper()}.MUSIC")        #'/data/lrh/datasets/fine_dance/origin/music_feature35'
        self.texinfo = eval(f"args.DATASET.{dataname.upper()}.TXTINFO")
        self.istrain = istrain
        self.args = args
        self.seq_len = args.FINEDANCE.full_seq_len
        slide = args.FINEDANCE.full_seq_len // args.FINEDANCE.windows

        self.motion_index = []
        self.music_index = []
        motion_all_a = []
        motion_all_b = []
        music_all = []
        
        test_list = ['008','062','072','033','057']
        with open(self.texinfo, 'r') as file:
            dance_pairs_data = file.readlines()

        total_length = 0            # 将数据集中的所有motion用同一个index索引
        # debug_num = 0
        for i in range(0, len(dance_pairs_data), 2):
            # debug_num += 1
            # if debug_num>2:
            #     break

            # 解析第一行数据
            first_line = dance_pairs_data[i].strip().split('_')
            pair_number = first_line[0]

            if self.istrain:
                if pair_number in test_list:
                    continue
            else:
                if not pair_number in test_list:
                    continue

            file_number1 = first_line[1]
            npy_file_path_a = os.path.join(self.motion_dir, f'{file_number1}.npy' )
            second_line = dance_pairs_data[i + 1].strip().split('_')
            file_number2 = second_line[1]
            npy_file_path_b = os.path.join(self.motion_dir, f'{file_number2}.npy' )
            music_file_path = os.path.join(self.music_dir, f'{file_number1}.npy' )

            motion_a = np.load(npy_file_path_a)
            motion_b = np.load(npy_file_path_b)
            music = np.load(music_file_path)

            min_all_len = min(motion_a.shape[0],  motion_b.shape[0], music.shape[0])
            motion_a = motion_a[:min_all_len]
            motion_b = motion_b[:min_all_len]
            music = music[:min_all_len]
          
            if motion_a.shape[-1] == 168:
                motion_a = np.concatenate([motion_a[:,:69], motion_a[:,78:]], axis=1)     # 22,  25
            elif motion_a.shape[-1] == 319:
                pass
            elif motion_a.shape[-1] == 315 or motion_a.shape[-1] == 139:
                if motion_a.shape[-1] == 139:
                    motion_a = motion_a[:,:139]
            elif motion_a.shape[-1] == 263 or motion_a.shape[-1] == 266:
                # motion = (motion - self.mean) / self.std
                pass
            else:
                print("motion_a.shape", motion_a.shape)
                raise("input motion_a shape error! not 168 or 319!")
            nums = (min_all_len-self.seq_len) // slide + 1         


            if self.istrain:
                clip_index = []
                for i in range(nums):
                    motion_clip_a = motion_a[i * slide: i * slide + self.seq_len]
                    if motion_clip_a.std(axis=0).mean() > 0.07:          
                        clip_index.append(i)
                index = np.array(clip_index) * slide + total_length     # clip_index is local index
            else:
                index = np.arange(nums) * slide + total_length

            motion_all_a.append(motion_a)
            motion_all_b.append(motion_b)
            music_all.append(music)

            if args.FINEDANCE.mix:
                motion_index = []
                music_index = []
                num = (len(index) - 1) // 8 + 1
                for i in range(num):
                    motion_index_tmp, music_index_tmp = np.meshgrid(index[i*8:(i+1)*8], index[i*8:(i+1)*8])       
                    motion_index += motion_index_tmp.reshape((-1)).tolist()
                    music_index += music_index_tmp.reshape((-1)).tolist()
            else:
                motion_index = index.tolist()
                music_index = index.tolist()

            
            self.motion_index += motion_index
            self.music_index += music_index
            total_length += min_all_len


        self.motion_a = np.concatenate(motion_all_a, axis=0).astype(np.float32)
        self.motion_b = np.concatenate(motion_all_b, axis=0).astype(np.float32)
        self.music = np.concatenate(music_all, axis=0).astype(np.float32)

        self.len = len(self.motion_index)
        print(f'FineDance has {self.len} samples..')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        motion_index = self.motion_index[index]
        music_index = self.music_index[index]
        motion_a = self.motion_a[motion_index:motion_index+self.seq_len]
        motion_b = self.motion_b[motion_index:motion_index+self.seq_len]
        # if motion.shape[-1] != 263 and motion.shape[-1] != 266:
        #     if motion.shape[-1] == 319 or motion.shape[-1] == 139:
        #         # motion[:, 4:7]  = motion[:, 4:7] - motion[:1, 4:7]           # The first 4 dimension are foot contact
        #         motion[:, 4]  = motion[:, 4] - motion[:1, 4] 
        #         motion[:, 6]  = motion[:, 6] - motion[:1, 6] 
        #     else:
        #         motion[:, 0] = motion[:, 0] - motion[:1, 0]
        #         motion[:, 2] = motion[:, 2] - motion[:1, 2]

        music = self.music[music_index:music_index+self.seq_len]
        # if np.random.rand(1) > 0.5:
        #     motion = motion[:,self.mirror_idx]
        return motion_a, motion_b, music
    
    def get_train_test_list(self):
        all_list = []
        train_list = []
        for i in range(1,212):
            all_list.append(str(i).zfill(3))
  
        test_list = ['008','062','072','033','057']
     
        ignor_list = []
        for one in all_list:
            if one not in test_list:
                train_list.append(one)
                
        return ignor_list, train_list, test_list
                



if __name__ == '__main__':

    # dataset = Magic()
    # a = dataset[10]
    
    a, b, c = get_train_test_list("/home/data/lrh/datasets/fine_dance/origin/motion_npy_smpl")

    # motion_dir = '/data/human/datasets/data/smpl168'
    # ouput_dir = '/data/human/datasets/data/smpl159'
    # for i in tqdm(range(1,212)):
    #     file = '{:0>3}.npy'.format(i)
    #     motion = np.load(os.path.join(motion_dir, file))
    #     transl = motion[1:,:3] - motion[:-1,:3]
    #     transl = np.concatenate([np.zeros((1,3)), transl], axis=0)
    #     motion = torch.tensor(np.concatenate([motion[:,:69], motion[:,78:]], axis=1), dtype=torch.float32)
    #     rot = torch.tensor([[[1,0,0,0],[0,0,-1,0],[0,1,0,0]]], dtype=torch.float32).expand(1000,-1,-1)
    #     mat = tgy.angle_axis_to_rotation_matrix(motion[:,3:6])
    #     mat = torch.bmm(rot, mat)
    #     aa = tgy.rotation_matrix_to_angle_axis(mat)
    #     motion[:,3:6] = aa
    #     rot = torch.tensor([[[1,0,0],[0,0,-1],[0,1,0]]], dtype=torch.float32).expand(1000,-1,-1)
    #     motion[:,:3] = torch.bmm(rot, motion[:,:3].unsqueeze(2))[:,:,0]
    #     motion = np.concatenate([transl, motion[:,3:69], motion[:,78:]], axis=1)
    #     np.save(os.path.join(ouput_dir, file), motion)
    print('done')