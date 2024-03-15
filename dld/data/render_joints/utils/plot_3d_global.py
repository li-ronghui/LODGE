import torch 
import matplotlib.pyplot as plt
import numpy as np
import io
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
import imageio

def plot_3d_motion(args, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')
    
    
    joints, out_name, title = args
    
    data = joints.copy().reshape(len(joints), -1, 3)
    
    nb_joints = joints.shape[1]
    smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    limits = 1000 if nb_joints == 21 else 2
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):

        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(b=False)
        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            ## Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)
        fig = plt.figure(figsize=(480/96., 320/96.), dpi=96) if nb_joints == 21 else plt.figure(figsize=(10, 10), dpi=96)
        if title is not None :
            wraped_title = '\n'.join(wrap(title, 40))
            fig.suptitle(wraped_title, fontsize=16)
        ax = p3.Axes3D(fig)
        
        init()
        
        # ax.lines = []
        # ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        #         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
        if out_name is not None : 
            plt.savefig(out_name, dpi=96)
            plt.close()
            
        else : 
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96)
            io_buf.seek(0)
            # print(fig.bbox.bounds)
            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return arr

    out = []
    for i in range(frame_number) : 
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)



def plot_3d_motion_55(args, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')
    
    
    joints, out_name, title = args
    
    data = joints.copy().reshape(len(joints), -1, 3)
    
    nb_joints = joints.shape[1]
    smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15, 22], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20], [22, 23], [22, 24], [20, 25, 26, 27], [20, 28, 29, 30], [20, 31, 32, 33], [20, 34, 35, 36], [20, 37, 38, 39], [21, 40, 41, 42], [21, 43, 44, 45], [21, 46, 47, 48], [21, 49, 50, 51], [21, 52, 53, 54]]
    limits = 1000 if nb_joints == 21 else 2
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred'
              ]
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):

        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(b=False)
        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            ## Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)
        fig = plt.figure(figsize=(480/96., 320/96.), dpi=96) if nb_joints == 21 else plt.figure(figsize=(20, 10), dpi=96)
        if title is not None :
            wraped_title = '\n'.join(wrap(title, 40))
            fig.suptitle(wraped_title, fontsize=16)
        # fig.suptitle("why", fontsize=16)
        ax = fig.add_subplot(111, projection='3d')
        # ax = p3.Axes3D(fig)
        
        init()
        
        # ax.lines = []
        # ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        # ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        # ax.scatter3D(5,5,5, c='r', marker='o')
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
        if out_name is not None : 
            plt.savefig(out_name, dpi=96)
            plt.close()
            
        else : 
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96)
            io_buf.seek(0)
            # print(fig.bbox.bounds)
            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return arr

    out = []
    for i in range(frame_number) :
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)

def plot_3d_inter_motion(args, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')
    
    
    joints0,joints1, out_name, title = args

    data0 = joints0.copy().reshape(len(joints0), -1, 3)
    data1 = joints1.copy().reshape(len(joints1), -1, 3)

    nb_joints = joints0.shape[1]
    smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    limits = 1000 if nb_joints == 21 else 2
    MINS = np.minimum(data0.min(axis=0).min(axis=0),data1.min(axis=0).min(axis=0))
    MAXS = np.maximum(data0.max(axis=0).max(axis=0),data1.max(axis=0).max(axis=0))
    MINS0 = data0.min(axis=0).min(axis=0)
    MINS1 = data1.min(axis=0).min(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data0.shape[0]
    #     print(data.shape)

    data0[:, :, 1] -= MINS0[1]
    trajec0 = data0[:, 0, [0, 2]]
    # data0[..., 0] -= data0[:, 0:1, 0]
    # data0[..., 2] -= data0[:, 0:1, 2]

    data1[:, :, 1] -= MINS1[1]
    trajec1 = data1[:, 0, [0, 2]]

    def update(index):

        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(b=False)
        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            ## Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)
        fig = plt.figure(figsize=(480/96., 320/96.), dpi=96) if nb_joints == 21 else plt.figure(figsize=(20, 20), dpi=96)
        if title is not None :
            wraped_title = '\n'.join(wrap(title, 40))
            fig.suptitle(wraped_title, fontsize=16)
        # ax = p3.Axes3D(fig)
        ax = fig.add_subplot(111, projection='3d')
        
        init()
        
        # ax.lines = []
        # ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec0[index, 0], MAXS[0] - trajec0[index, 0], 0, MINS[2] - trajec0[index, 1],
                     MAXS[2] - trajec0[index, 1])
        #         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

        if index > 1:
            ax.plot3D(trajec0[:index, 0] - trajec0[index, 0], np.zeros_like(trajec0[:index, 0]),
                      trajec0[:index, 1] - trajec0[index, 1], linewidth=1.0,
                      color='blue')
            ax.plot3D(trajec1[:index, 0] - trajec1[index, 0], np.zeros_like(trajec1[:index, 0]),
                      trajec1[:index, 1] - trajec1[index, 1], linewidth=1.0,
                      color='red')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data0[index, chain, 0], data0[index, chain, 1], data0[index, chain, 2], linewidth=linewidth,
                      color=color)
            ax.plot3D(data1[index, chain, 0], data1[index, chain, 1], data1[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
        if out_name is not None : 
            plt.savefig(out_name, dpi=96)
            plt.close()
            
        else : 
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96)
            io_buf.seek(0)
            # print(fig.bbox.bounds)
            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return arr

    out = []
    for i in range(frame_number) : 
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)


def plot_3d_inter_motion_55(args, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    joints0,joints1,motion3d0,motion3d1, out_name, title = args

    data0 = joints0.copy().reshape(len(joints0), -1, 3)
    data1 = joints1.copy().reshape(len(joints1), -1, 3)
    motion0 = motion3d0.copy().reshape(len(motion3d0), -1)
    motion1 = motion3d1.copy().reshape(len(motion3d1), -1)
    x0=motion0[0,0]
    z0=motion0[0,2]
    x1=motion1[0,0]
    z1=motion1[0,2]

    nb_joints = joints0.shape[1]
    smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7],
                          [3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                                                                  [0, 3, 6, 9, 12, 15, 22], [9, 14, 17, 19, 21],
                                                                  [9, 13, 16, 18, 20], [22, 23], [22, 24],
                                                                  [20, 25, 26, 27], [20, 28, 29, 30], [20, 31, 32, 33],
                                                                  [20, 34, 35, 36], [20, 37, 38, 39], [21, 40, 41, 42],
                                                                  [21, 43, 44, 45], [21, 46, 47, 48], [21, 49, 50, 51],
                                                                  [21, 52, 53, 54]]
    limits = 1000 if nb_joints == 21 else 2
    MINS = np.minimum(data0.min(axis=0).min(axis=0),data1.min(axis=0).min(axis=0))
    MAXS = np.maximum(data0.max(axis=0).max(axis=0),data1.max(axis=0).max(axis=0))
    MINS0 = data0.min(axis=0).min(axis=0)
    MINS1 = data1.min(axis=0).min(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred'
              ]
    frame_number = data0.shape[0]
    #     print(data.shape)
    # height_offset = MINS[1]
    data0[:, :, 1] -= MINS0[1]
    data0[:,:,0] += x0
    data0[:,:,2] += z0
    trajec0 = data0[:, 0, [0, 2]]
    # data0[..., 0] -= data0[:, 0:1, 0]
    # data0[..., 2] -= data0[:, 0:1, 2]

    data1[:, :, 1] -= MINS1[1]
    data1[:,:,0] += x1
    data1[:,:,2] += z1
    trajec1 = data1[:, 0, [0, 2]]

    # data1[..., 0] -= data1[:, 0:1, 0]
    # data1[..., 2] -= data1[:, 0:1, 2]
    def update(index):

        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(b=False)

        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            ## Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)

        fig = plt.figure(figsize=(480 / 96., 320 / 96.), dpi=96) if nb_joints == 21 else plt.figure(figsize=(20, 20),
                                                                                                    dpi=96)
        if title is not None:
            wraped_title = '\n'.join(wrap(title, 40))
            fig.suptitle(wraped_title, fontsize=16)
        # fig.suptitle("why", fontsize=16)
        ax = fig.add_subplot(111, projection='3d')
        # ax = p3.Axes3D(fig)

        init()

        # ax.lines = []
        # ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec0[index, 0], MAXS[0] - trajec0[index, 0], 0, MINS[2] - trajec0[index, 1],
                     MAXS[2] - trajec0[index, 1])
        # ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

        if index > 1:
            ax.plot3D(trajec0[:index, 0] - trajec0[index, 0], np.zeros_like(trajec0[:index, 0]),
                      trajec0[:index, 1] - trajec0[index, 1], linewidth=1.0,
                      color='blue')
            ax.plot3D(trajec1[:index, 0] - trajec1[index, 0], np.zeros_like(trajec1[:index, 0]),
                      trajec1[:index, 1] - trajec1[index, 1], linewidth=1.0,
                      color='red')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data0[index, chain, 0], data0[index, chain, 1], data0[index, chain, 2], linewidth=linewidth,
                      color=color)
            ax.plot3D(data1[index, chain, 0], data1[index, chain, 1], data1[index, chain, 2], linewidth=linewidth,
                      color=color)

        # ax.scatter3D(5,5,5, c='r', marker='o')
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        if out_name is not None:
            plt.savefig(out_name, dpi=96)
            plt.close()

        else:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96)
            io_buf.seek(0)
            # print(fig.bbox.bounds)
            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return arr

    out = []
    for i in range(frame_number):
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)

def plot_3d_inter_motion_55_rela(args, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')
    print("plot_3d_inter_motion_55_rela")
    joints0,joints1, out_name, title = args
    joints0 = joints0.copy().reshape(len(joints0), -1, 3)
    joints1 = joints1.copy().reshape(len(joints1), -1, 3)

    data0 = joints0[:,:55,:]
    data1 = joints1[:,:55,:]
    rela_dis0 = joints0[:,55:,:].reshape(len(joints0), 3)
    rela_dis1 = joints1[:, 55:, :].reshape(len(joints1), 3)

    import pickle
    rela_dis_dir = '/home/nfs/zyl/dataset/fine_dance/total/relative_dis.pkl'
    with open(rela_dis_dir, 'rb') as file:
        rela_dis_dict = pickle.load(file)
    rela_dis_mean = rela_dis_dict['Mean']
    rela_dis_std = rela_dis_dict['std']

    nb_joints = data0.shape[1]
    smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7],
                          [3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                                                                  [0, 3, 6, 9, 12, 15, 22], [9, 14, 17, 19, 21],
                                                                  [9, 13, 16, 18, 20], [22, 23], [22, 24],
                                                                  [20, 25, 26, 27], [20, 28, 29, 30], [20, 31, 32, 33],
                                                                  [20, 34, 35, 36], [20, 37, 38, 39], [21, 40, 41, 42],
                                                                  [21, 43, 44, 45], [21, 46, 47, 48], [21, 49, 50, 51],
                                                                  [21, 52, 53, 54]]
    limits = 1000 if nb_joints == 21 else 2
    MINS = np.minimum(data0.min(axis=0).min(axis=0),data1.min(axis=0).min(axis=0))
    MAXS = np.maximum(data0.max(axis=0).max(axis=0),data1.max(axis=0).max(axis=0))
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred'
              ]
    frame_number = data0.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data0[:, :, 1] -= height_offset
    trajec0 = data0[:, 0, [0, 2]]
    # data0[..., 0] -= data0[:, 0:1, 0]
    # data0[..., 2] -= data0[:, 0:1, 2]
    for t in range(len(data1)):
        data1[t][:,0]+=rela_dis0[t][0]
        data1[t][:, 2] += rela_dis0[t][1]
    data1[:, :, 1] -= height_offset
    trajec1 = data1[:, 0, [0, 2]]
    data1[..., 0] -= data1[:, 0:1, 0]
    data1[..., 2] -= data1[:, 0:1, 2]
    def update(index):

        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(b=False)

        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            ## Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)

        fig = plt.figure(figsize=(480 / 96., 320 / 96.), dpi=96) if nb_joints == 21 else plt.figure(figsize=(20, 20),
                                                                                                    dpi=96)
        if title is not None:
            wraped_title = '\n'.join(wrap(title, 40))
            fig.suptitle(wraped_title, fontsize=16)
        # fig.suptitle("why", fontsize=16)
        ax = fig.add_subplot(111, projection='3d')
        # ax = p3.Axes3D(fig)

        init()

        # ax.lines = []
        # ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec0[index, 0], MAXS[0] - trajec0[index, 0], 0, MINS[2] - trajec0[index, 1],
                     MAXS[2] - trajec0[index, 1])
        # ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

        if index > 1:
            ax.plot3D(trajec0[:index, 0] - trajec0[index, 0], np.zeros_like(trajec0[:index, 0]),
                      trajec0[:index, 1] - trajec0[index, 1], linewidth=1.0,
                      color='blue')
            ax.plot3D(trajec1[:index, 0] - trajec1[index, 0], np.zeros_like(trajec1[:index, 0]),
                      trajec1[:index, 1] - trajec1[index, 1], linewidth=1.0,
                      color='red')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data0[index, chain, 0], data0[index, chain, 1], data0[index, chain, 2], linewidth=linewidth,
                      color=color)
            ax.plot3D(data1[index, chain, 0], data1[index, chain, 1], data1[index, chain, 2], linewidth=linewidth,
                      color=color)

        # ax.scatter3D(5,5,5, c='r', marker='o')
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        if out_name is not None:
            plt.savefig(out_name, dpi=96)
            plt.close()

        else:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96)
            io_buf.seek(0)
            # print(fig.bbox.bounds)
            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return arr

    out = []
    for i in range(frame_number):
        out.append(update(i))
    print("stacking")
    out = np.stack(out, axis=0)
    print("stacked")
    return torch.from_numpy(out)

def draw_to_batch(smpl_joints_batch, title_batch=None, outname=None, joints_num=22) : 
    
    batch_size = len(smpl_joints_batch)
    out = []
    for i in range(batch_size) : 
        print("smpl_joints_batch[i]", smpl_joints_batch[i].shape)
        if joints_num == 22:
            out.append(plot_3d_motion([smpl_joints_batch[i], None, title_batch[i] if title_batch is not None else None]))
        elif joints_num == 55:
            print('drawing 55 joints')
            out.append(plot_3d_motion_55([smpl_joints_batch[i], None, title_batch[i] if title_batch is not None else None]))
        if outname is not None:
            # imageio.mimsave(outname[i], np.array(out[-1]), fps=30)
            imageio.mimsave(outname[i], np.array(out[-1]), duration=(1000*1/30))
    out = torch.stack(out, axis=0)
    return out


def draw_inter_to_batch(smpl_joints_batch0, smpl_joints_batch1, motion3d0, motion3d1, title_batch=None, outname=None, joints_num=22,use_rela=False):
    batch_size = len(smpl_joints_batch0)
    out = []
    for i in range(batch_size):
        print("smpl_joints_batch[i]", smpl_joints_batch0[i].shape)
        if joints_num == 22:
            out.append(
                plot_3d_inter_motion([smpl_joints_batch0[i], smpl_joints_batch1[i], None, title_batch[i] if title_batch is not None else None]))
        elif joints_num == 55:
            print('drawing 55 joints')
            if use_rela:
                out.append(
                    plot_3d_inter_motion_55_rela([smpl_joints_batch0[i], smpl_joints_batch1[i], None,
                                             title_batch[i] if title_batch is not None else None]))
            else:
                out.append(
                    plot_3d_inter_motion_55([smpl_joints_batch0[i],smpl_joints_batch1[i],motion3d0,motion3d1, None, title_batch[i] if title_batch is not None else None]))
        if outname is not None:
            # imageio.mimsave(outname[i], np.array(out[-1]), fps=30)
            print("saving",outname[i])
            imageio.mimsave(outname[i], np.array(out[-1]), duration=(1000 * 1 / 30))
            print("saved")
    out = torch.stack(out, axis=0)
    return out
    




