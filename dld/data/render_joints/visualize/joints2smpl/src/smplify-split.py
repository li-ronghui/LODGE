import torch
import os, sys
import pickle
import smplx
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from customloss import (camera_fitting_loss, 
                        body_fitting_loss, 
                        camera_fitting_loss_3d,
                        body_fitting_loss_smplx, 
                        )
from prior import MaxMixturePrior, L2Prior, create_prior
from visualize.joints2smpl.src import config



@torch.no_grad()
def guess_init_3d(model_joints, 
                  j3d, 
                  joints_category="orig"):
    """Initialize the camera translation via triangle similarity, by using the torso joints        .
    :param model_joints: SMPL model with pre joints
    :param j3d: 25x3 array of Kinect Joints
    :returns: 3D vector corresponding to the estimated camera translation
    """
    # get the indexed four
    gt_joints = ['RHip', 'LHip', 'RShoulder', 'LShoulder']
    gt_joints_ind = [config.AMASS_JOINT_MAP[joint] for joint in gt_joints]
    
    if joints_category=="orig":
        joints_ind_category = [config.JOINT_MAP[joint] for joint in gt_joints]
    elif joints_category=="AMASS":
        joints_ind_category = [config.AMASS_JOINT_MAP[joint] for joint in gt_joints] 
    elif joints_category=="SMPLX55":
        joints_ind_category = [config.AMASS_JOINT_MAP[joint] for joint in gt_joints] 
    else:
        print("NO SUCH JOINTS CATEGORY!") 

    sum_init_t = (j3d[:, joints_ind_category] - model_joints[:, gt_joints_ind]).sum(dim=1)
    init_t = sum_init_t / 4.0
    return init_t


# SMPLIfy 3D
class SMPLify3D():
    """Implementation of SMPLify, use 3D joints."""

    def __init__(self,
                 smplxmodel,
                 step_size=1e-2,
                 batch_size=1,
                 num_iters=100,
                 use_collision=False,
                 use_lbfgs=True,
                 joints_category="orig",
                 device=torch.device('cuda:0'),
                 ):

        # Store options
        self.batch_size = batch_size
        self.device = device
        self.step_size = step_size

        self.num_iters = num_iters
        # --- choose optimizer
        self.use_lbfgs = use_lbfgs
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder=config.GMM_MODEL_DIR,
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
   
    
        # self.left_hand_prior = L2Prior(dtype=torch.float32)
        # self.right_hand_prior = L2Prior(dtype=torch.float32)


        # collision part
        self.use_collision = use_collision
        if self.use_collision:
            self.part_segm_fn = config.Part_Seg_DIR
        
        # reLoad SMPL-X model
        self.smplx = smplxmodel
        self.model_faces = smplxmodel.faces_tensor.view(-1)

        # select joint joint_category
        self.joints_category = joints_category
        
        if joints_category=="orig":
            self.smpl_index = config.full_smpl_idx
            self.corr_index = config.full_smpl_idx 
        elif joints_category=="AMASS":
            self.smpl_index = config.amass_smpl_idx
            self.corr_index = config.amass_idx
        elif joints_category=="SMPLX55":
            self.smpl_index = config.smplx55_idx
            self.corr_index = config.smplx55_idx
        else:
            self.smpl_index = None 
            self.corr_index = None
            print("NO SUCH JOINTS CATEGORY!")

    # ---- get the man function here ------
    def __call__(self, init_pose, init_betas, init_cam_t, init_jaw_pose, init_leye_pose, init_reye_pose, init_left_hand, init_right_hand, j3d, conf_3d=1.0, seq_ind=0):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            j3d: joints 3d aka keypoints
            conf_3d: confidence for 3d joints
			seq_ind: index of the sequence
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
        """

        # # # add the mesh inter-section to avoid
        search_tree = None
        pen_distance = None
        filter_faces = None
        
        if self.use_collision:
            from mesh_intersection.bvh_search_tree import BVH
            import mesh_intersection.loss as collisions_loss
            from mesh_intersection.filter_faces import FilterFaces

            search_tree = BVH(max_collisions=8)

            pen_distance = collisions_loss.DistanceFieldPenetrationLoss(
                           sigma=0.5, point2plane=False, vectorized=True, penalize_outside=True)

            if self.part_segm_fn:
                # Read the part segmentation
                part_segm_fn = os.path.expandvars(self.part_segm_fn)
                with open(part_segm_fn, 'rb') as faces_parents_file:
                    face_segm_data = pickle.load(faces_parents_file,  encoding='latin1')
                faces_segm = face_segm_data['segm']
                faces_parents = face_segm_data['parents']
                # Create the module used to filter invalid collision pairs
                filter_faces = FilterFaces(
                    faces_segm=faces_segm, faces_parents=faces_parents,
                    ign_part_pairs=None).to(device=self.device)
                    
                    
        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # use guess 3d to get the initial
        smpl_output = self.smplx(global_orient=global_orient,
                                body_pose=body_pose,
                                betas=betas)
        model_joints = smpl_output.joints

        init_cam_t = guess_init_3d(model_joints, j3d, self.joints_category).unsqueeze(1).detach()
        camera_translation = init_cam_t.clone()
        
        preserve_pose = init_pose[:, 3:].detach().clone()
       # -------------Step 1: Optimize camera translation and body orientation--------
        # Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]

        if self.use_lbfgs:
            camera_optimizer = torch.optim.LBFGS(camera_opt_params, max_iter=self.num_iters,
                                                 lr=self.step_size, line_search_fn='strong_wolfe')
            for i in range(10):
                def closure():
                    camera_optimizer.zero_grad()
                    smpl_output = self.smplx(global_orient=global_orient,
                                            body_pose=body_pose,
                                            jaw_pose = init_jaw_pose,
                                            leye_pose = init_leye_pose,
                                            reye_pose = init_reye_pose,
                                            left_hand_pose = init_left_hand,
                                            right_hand_pose = init_right_hand,
                                            betas=betas)
                    model_joints = smpl_output.joints
                    # print('model_joints', model_joints.shape)
                    # print('camera_translation', camera_translation.shape)
                    # print('init_cam_t', init_cam_t.shape)
                    # print('j3d', j3d.shape)
                    loss = camera_fitting_loss_3d(model_joints, camera_translation,
                                                  init_cam_t, j3d, self.joints_category)
                    loss.backward()
                    return loss

                camera_optimizer.step(closure)
        else:
            camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

            for i in range(20):
                smpl_output = self.smplx(global_orient=global_orient,
                                        body_pose=body_pose,
                                        jaw_pose = init_jaw_pose,
                                        leye_pose = init_leye_pose,
                                        reye_pose = init_reye_pose,
                                        left_hand_pose = init_left_hand,
                                        right_hand_pose = init_right_hand,
                                        betas=betas)
                model_joints = smpl_output.joints

                loss = camera_fitting_loss_3d(model_joints[:, self.smpl_index], camera_translation,
                                              init_cam_t,  j3d[:, self.corr_index], self.joints_category)
                camera_optimizer.zero_grad()
                loss.backward()
                camera_optimizer.step()

        # Fix camera translation after optimizing camera
        # --------Step 2: Optimize body joints --------------------------
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = True
        
        init_jaw_pose.requires_grad = True
        init_leye_pose.requires_grad = True
        init_reye_pose.requires_grad = True
        init_left_hand.requires_grad = False
        init_right_hand.requires_grad = False

        # --- if we use the sequence, fix the shape
        if seq_ind == 0:
            betas.requires_grad = True
            body_opt_params = [body_pose, betas, global_orient, camera_translation, init_jaw_pose, init_leye_pose, init_reye_pose]
        else:
            betas.requires_grad = False
            body_opt_params = [body_pose, global_orient, camera_translation, init_jaw_pose, init_leye_pose, init_reye_pose]

        
        if self.use_lbfgs:
            body_optimizer = torch.optim.LBFGS(body_opt_params, max_iter=self.num_iters,
                                               lr=self.step_size, line_search_fn='strong_wolfe')
            for i in tqdm(range(10)):
                def closure():
                    body_optimizer.zero_grad()
                    smpl_output = self.smplx(global_orient=global_orient,
                                            body_pose=body_pose,
                                            jaw_pose = init_jaw_pose,
                                            leye_pose = init_leye_pose,
                                            reye_pose = init_reye_pose,
                                            left_hand_pose = init_left_hand,
                                            right_hand_pose = init_right_hand,
                                            betas=betas)
                    model_joints = smpl_output.joints
                    model_vertices = smpl_output.vertices

                    loss = body_fitting_loss_smplx(body_pose, preserve_pose, model_joints, camera_translation,
                                                j3d,
                                                joints3d_conf=conf_3d,
                                                joint_loss_weight=600.0,
                                                hand_joint_loss_weight=0.0,
                                                use_collision=self.use_collision, 
                                                model_vertices=model_vertices, model_faces=self.model_faces,
                                                search_tree=search_tree, pen_distance=pen_distance, filter_faces=filter_faces)
                    loss.backward()
                    return loss

                body_optimizer.step(closure)
        else:
            body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

            for i in tqdm(range(10)):
                smpl_output = self.smplx(global_orient=global_orient,
                                        body_pose=body_pose,
                                        jaw_pose = init_jaw_pose,
                                        leye_pose = init_leye_pose,
                                        reye_pose = init_reye_pose,
                                        left_hand_pose = init_left_hand,
                                        right_hand_pose = init_right_hand,
                                        betas=betas)
                model_joints = smpl_output.joints
                model_vertices = smpl_output.vertices

                loss = body_fitting_loss_smplx(body_pose, preserve_pose, model_joints, camera_translation,
                                                j3d,
                                                joints3d_conf=conf_3d,
                                                joint_loss_weight=600.0,
                                                hand_joint_loss_weight=0.0,
                                                use_collision=self.use_collision, 
                                                model_vertices=model_vertices, model_faces=self.model_faces,
                                                search_tree=search_tree, pen_distance=pen_distance, filter_faces=filter_faces)
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()



        # Fix body after optimizing body pose
        # --------Step 3: Optimize hand joints --------------------------
        # Optimize only the hand pose
        body_pose.requires_grad = False
        global_orient.requires_grad = False
        camera_translation.requires_grad = False
        init_jaw_pose.requires_grad = False
        init_leye_pose.requires_grad = False
        init_reye_pose.requires_grad = False
        init_left_hand.requires_grad = True
        init_right_hand.requires_grad = True

        # --- if we use the sequence, fix the shape
        if seq_ind == 0:
            betas.requires_grad = True
            hand_opt_params = [init_left_hand, betas, init_right_hand]
        else:
            betas.requires_grad = False
            hand_opt_params = [init_left_hand, init_right_hand]

            

        if self.use_lbfgs:
            print("use_lbfgs")
            hand_optimizer = torch.optim.LBFGS(hand_opt_params, max_iter=self.num_iters,
                                               lr=self.step_size, line_search_fn='strong_wolfe')
            for i in tqdm(range(30)):
                def closure():
                    hand_optimizer.zero_grad()
                    smpl_output = self.smplx(global_orient=global_orient,
                                            body_pose=body_pose,
                                            jaw_pose = init_jaw_pose,
                                            leye_pose = init_leye_pose,
                                            reye_pose = init_reye_pose,
                                            left_hand_pose = init_left_hand,
                                            right_hand_pose = init_right_hand,
                                            betas=betas)
                    model_joints = smpl_output.joints
                    model_vertices = smpl_output.vertices

                    loss = body_fitting_loss_smplx(body_pose, preserve_pose, model_joints, camera_translation,
                                                j3d,
                                                joints3d_conf=conf_3d,
                                                joint_loss_weight=0.0,
                                                hand_joint_loss_weight=600.0,
                                                use_collision=self.use_collision, 
                                                model_vertices=model_vertices, model_faces=self.model_faces,
                                                search_tree=search_tree, pen_distance=pen_distance, filter_faces=filter_faces)
                    loss.backward()
                    return loss

                hand_optimizer.step(closure)
        else:
            hand_optimizer = torch.optim.Adam(hand_opt_params, lr=self.step_size, betas=(0.9, 0.999))

            for i in tqdm(range(30)):
                smpl_output = self.smplx(global_orient=global_orient,
                                        body_pose=body_pose,
                                        jaw_pose = init_jaw_pose,
                                        leye_pose = init_leye_pose,
                                        reye_pose = init_reye_pose,
                                        left_hand_pose = init_left_hand,
                                        right_hand_pose = init_right_hand,
                                        betas=betas)
                model_joints = smpl_output.joints
                model_vertices = smpl_output.vertices

                loss = body_fitting_loss_smplx(body_pose, preserve_pose, model_joints, camera_translation,
                                                j3d,
                                                joints3d_conf=conf_3d,
                                                joint_loss_weight=0.0,
                                                hand_joint_loss_weight=600.0,
                                                use_collision=self.use_collision, 
                                                model_vertices=model_vertices, model_faces=self.model_faces,
                                                search_tree=search_tree, pen_distance=pen_distance, filter_faces=filter_faces)
                hand_optimizer.zero_grad()
                loss.backward()
                hand_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smplx(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            model_joints = smpl_output.joints
            model_vertices = smpl_output.vertices

            final_loss = body_fitting_loss_smplx(body_pose, preserve_pose, model_joints, camera_translation,
                                                j3d,
                                                joints3d_conf=conf_3d,
                                                joint_loss_weight=600.0,
                                                hand_joint_loss_weight=600.0,
                                                use_collision=self.use_collision, 
                                                model_vertices=model_vertices, model_faces=self.model_faces,
                                                search_tree=search_tree, pen_distance=pen_distance, filter_faces=filter_faces)

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose, init_jaw_pose, init_leye_pose, init_reye_pose, init_left_hand, init_right_hand], dim=-1).detach()
        betas = betas.detach()

        return vertices, joints, pose, betas, camera_translation, final_loss
