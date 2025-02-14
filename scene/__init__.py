#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import glob
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
import torch.nn as nn
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], init_train_args=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians


        # Light params
        # values: [intensity, attenuation k, attenuation power, rot_angle_0=0, rot_angle_1=0, rot_angle_2=2]

        try:
            initial_values = torch.tensor([init_train_args.light_intensity_init, 
                                           init_train_args.attenuation_k_init, 
                                           init_train_args.attenuation_power_init, 
                                           1.0, 0.0, 0.0, 
                                           0.0, 1.0,0.0,
                                           0.0,0.0,1.0], dtype=torch.float, device="cuda")  
            self.light = nn.Parameter(initial_values.requires_grad_(True))
            self.optimizer_light = torch.optim.Adam([self.light], lr=init_train_args.light_lr, eps=1e-15)
        except:
            initial_values = torch.tensor([1.0, 
                                           0.8, 
                                           2.0, 
                                           1.0, 0.0, 0.0, 
                                           0.0, 1.0, 0.0,
                                           0.0, 0.0, 1.0], dtype=torch.float, device="cuda")
            
            self.light = nn.Parameter(initial_values.requires_grad_(True))
            self.optimizer_light = torch.optim.Adam([self.light], lr=0.003, eps=1e-15)

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        
        if "C3VD" in args.source_path:
            scene_info=sceneLoadTypeCallbacks["colon"](args.source_path)
        elif "Rotate" in args.source_path:
            scene_info=sceneLoadTypeCallbacks["colonRotate"](args.source_path)
        else:
            print(f"your source path is: {args.source_path}")
            raise NotImplementedError

        if shuffle:
            random.seed(42)
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # CAM object directly returned from colon reader
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            print("Loading point cloud from iteration {}".format(self.loaded_iter))
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.spatial_lr_scale = self.cameras_extent
            try:
                self.gaussians.mlp.load_state_dict(torch.load(self.model_path + "/chkpnt_mlp" + str(self.loaded_iter) + ".pth"))
                self.gaussians.positional_encoding_gauss.load_state_dict(torch.load(self.model_path + "/chkpnt_grid_gauss" + str(self.loaded_iter) + ".pth"))
                self.gaussians.positional_encoding_camera.load_state_dict(torch.load(self.model_path + "/chkpnt_grid_camera" + str(self.loaded_iter) + ".pth"))

                self.light = torch.load(self.model_path + "/chkpnt_light" + str(self.loaded_iter) + ".pth")

            except:
                print("&&&&&&&&&&&& &&&&&&&&&&& NO CHECKPOINTS LOADED FOR GRID AND MLP AND LIGHT.")
                raise FileNotFoundError
        else:
            print("Creating point cloud from scene")
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, args, init_args=init_train_args)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]