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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, cdist, build_rotation, rot_to_quat_batch
import time
from tqdm import tqdm 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from gaussian_norms import compute_gauss_norm
import faiss
import faiss.contrib.torch_utils
from scene.mlp import MLP, PosEmbedding, HashGrid
from utils.sh_utils import eval_sh
import torch.nn.functional as F

class GaussianModel:


    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        def return_surface_normal(scaling, scaling_modifier, rotation):
            input_rotation_normalized = rotation / rotation.norm(p=2, dim=1, keepdim=True)
            cuda_normal = compute_gauss_norm(scaling, input_rotation_normalized, scaling_modifier)
            return cuda_normal
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.surface_normal = return_surface_normal
    
    def setup_mlp(self):
        self.mlp_W = 128
        self.mlp_D = 4
        self.positional_encoding_camera = HashGrid(input_dim=4).cuda()

        encoding_dims = self.positional_encoding_camera.encoding.n_output_dims
        self.mlp = MLP(self.max_sh_degree, self.mlp_W, self.mlp_D, self.use_hg, encoding_dims=encoding_dims).cuda()


    def __init__(self, sh_degree : int):

        torch.manual_seed(42)

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.use_hg = None
        self.setup_functions()

        self.closest_point_indices = torch.empty(0, dtype=torch.long)
        self.original_normals = torch.empty(0)
        self.faiss_index = None
        self.lambda_norm = None
        self.triangles = torch.empty(0)
        self.eps_s0 = 1e-8

        self.start_mlp_iter =1000 #defined only here
        self.random_noise_val = 0.005 # will be overwritten in training_setup()
        self.end_diffuse_loss_iter = 3000 # will be overwritten in training_setup()
        self.max_scale = 0.1 # defined only here, TODO: parametrize
    


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.use_hg,
            self.roughness,
            self.F_0,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.use_hg,
        self.roughness,
        self.F_0) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.lambda_norm = training_args.lambda_norm

    @property
    def get_scaling(self):
        self.s0 = torch.ones(self._scaling.shape[0], 1).cuda() * self.eps_s0
        return torch.cat([self.scaling_activation(self._scaling[:, [0, 1]]),self.s0], dim=1).clamp(max=self.max_scale*self.spatial_lr_scale)

    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_roughness(self):
        # based on human testines textures
        return torch.clamp(self.roughness, 0.0, 0.7)
    
    @property
    def get_F_0(self):
        # based on human testines textures
        return torch.clamp(self.F_0, 0, 0.035)
    
    def compute_gaussian_rgb(self, camera_center):
        shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
        dir_pp = (self.get_xyz - camera_center.repeat(self.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        return sh2rgb +0.5 #torch.clamp_min(sh2rgb + 0.5, 0.0)
    
    
    def compute_positional_encoding_camera(self, L, dist):
        """ We can think of camera as individual gaussian light """
        
        cat_encodings = self.positional_encoding_camera(torch.cat([L, dist], dim=1)) 

        return cat_encodings

    def compute_mlp_outputs(self, base_color, light_gauss_dist, N, L, randomize_input = False):

        random_noise_value = self.random_noise_val
        if randomize_input:
            n, mult = N.shape
            noise = torch.cuda.FloatTensor(n, mult).normal_() * random_noise_value
            N = N+noise
            N = N/(N.norm(dim=1, keepdim=True)+self.eps_s0)

            n, mult = L.shape
            noise = torch.cuda.FloatTensor(n, mult).normal_() * random_noise_value
            L = L+noise
            L = L/(L.norm(dim=1, keepdim=True)+self.eps_s0)

            n, mult = light_gauss_dist.shape
            noise = torch.cuda.FloatTensor(n, mult).normal_() * random_noise_value
            light_gauss_dist = light_gauss_dist+noise

        inputs = [base_color, light_gauss_dist, N, L, torch.sum(N * L, dim=1, keepdim=True)]

        if self.use_hg:
            inputs.insert(0, self.compute_positional_encoding_camera(L, light_gauss_dist))

        input = torch.cat(inputs, dim=1)
        output = self.mlp(input)
        return output
    

    def compute_lighted_rgb(self, camera_center, light, ret_loss=False, iter = 0, 
                            randomize_input=False, disable_reflections = False, 
                            light_center_sep=None, light_rot = None, spotlight_attenuation_power = 0.0):
        
        if torch.is_tensor(light_center_sep):
            light_center_raw = light_center_sep
        else:
            light_center_raw = camera_center

        base_color = self.compute_gaussian_rgb(camera_center)

        #Light color is [1,1,1,]
        light_intensity, attenuation_k, attenuation_power, light_adjustment = light[0], light[1], light[2], light[3:12]
        
        # ------ HOW TO SET LIGHT DIRECTION - either basic version, or with d optimized
        # light dir == view dir - the basic option
        light_center = light_center_raw
        dir_pp_light = (self.get_xyz - light_center.repeat(self.get_features.shape[0], 1))
        dir_pp_light = torch.nn.functional.normalize(dir_pp_light, dim=1)

        # # light source is offset by d 
        # offset = light_adjustment[:3] #only couple of entries for light adjustment are finally needed
        # light_center = light_center_raw+offset
        # dir_pp_light = (self.get_xyz - light_center.repeat(self.get_features.shape[0], 1))
        # dir_pp_light = torch.nn.functional.normalize(dir_pp_light, dim=1)
        
        #--------------
        
        
        dir_gauss_lightcenter = (self.get_xyz - light_center.repeat(self.get_features.shape[0], 1))
        light_gauss_dist = dir_gauss_lightcenter.norm(dim=1, keepdim=True) / self.spatial_lr_scale

        dir_pp_camera = (self.get_xyz - camera_center.repeat(self.get_features.shape[0], 1))
        camera_gauss_dist = dir_pp_camera.norm(dim=1, keepdim=True) / self.spatial_lr_scale 
        # self.spatial_lr_scale  - its scene radius

        normal = self.get_gaussian_normals()[:, 3:]

        # Normalize the input vectors
        N = torch.nn.functional.normalize(normal, dim=1)
        L = -torch.nn.functional.normalize(dir_pp_light, dim=1) 
        V = -torch.nn.functional.normalize(dir_pp_camera, dim=1)
        
        # normals always towards camera
        N_dot_V = torch.sum(N * V, dim=1, keepdim=True)  # [N, 1]
        N = torch.where(N_dot_V < 0, -N, N)  # Flip N if N_dot_V < 0

        # cosine
        N_dot_L = torch.clamp(torch.sum(N * L, dim=1, keepdim=True), min=0.0)
        
        # Compute distance attenuation (light intensity fades with distance)
        denom_A = (1.0 + attenuation_k * (light_gauss_dist) ** attenuation_power)
        attenuation_raw_coeffs = 1.0 / (denom_A + (denom_A == 0).float() * 1e-6)
        attenuation_coeffs = torch.clamp(attenuation_raw_coeffs, 0, 1)
        
        # ======== DIFFUSE 
        diffuse_component_mlp = self.compute_mlp_outputs(
            base_color, light_gauss_dist, N, L, randomize_input=randomize_input)

        # Diffuse component - from MLP, light transfer eq
        I_diffuse_color_mlp = light_intensity  *(base_color * diffuse_component_mlp) * attenuation_coeffs * N_dot_L

        #Diffuse component - from coefficients, light transfer eq
        I_diffuse_color_coeffs = light_intensity  * N_dot_L * attenuation_coeffs * base_color
        

        # ======== PBR reflections

        # Compute halfway vector for specular component
        H = F.normalize(L + V, dim=1)
        
        # Fresnel Effect (Schlick's approximation) for non-metals (F0 = 0.04) -value from gpt
        fresnel = self.get_F_0 + (1 - self.get_F_0) * (1 - torch.sum(H * V, dim=1, keepdim=True)) ** 5
        
        # Specular component using Cook-Torrance model
        N_dot_H = torch.clamp(torch.sum(N * H, dim=1, keepdim=True), min=0.0)
        alpha = (self.get_roughness) ** 2
        

        # Microfacet distribution function (Trowbridge-Reitz GGX)
        denom_D = (torch.pi * ((N_dot_H ** 2) * (alpha ** 2 - 1) + 1) ** 2)        
        D = (alpha ** 2) / (denom_D + (denom_D == 0).float() * 1e-6)

        # Geometric attenuation (Smith GGX)
        k = (self.roughness + 1) ** 2 / 8
        N_dot_V_clamped = torch.clamp(torch.sum(N * V, dim=1, keepdim=True), min=0.0)
        N_dot_L_clamped = torch.clamp(N_dot_L, min=0.0)
        
        denom_V = N_dot_V_clamped * (1 - k) + k
        denom_L = N_dot_L_clamped * (1 - k) + k
        G1_V = N_dot_V_clamped / (denom_V + (denom_V == 0).float() * 1e-6)
        G1_L = N_dot_L_clamped / (denom_L + (denom_L == 0).float() * 1e-6)
        
        G = G1_V * G1_L

        # Final Cook-Torrance specular term
        spec_denom = (4 * N_dot_V_clamped * N_dot_L_clamped)
        specular_component_coeffs =  (fresnel * D * G) / (spec_denom + (spec_denom == 0).float() * 1e-6)
        I_specular_coeffs = light_intensity * attenuation_coeffs * specular_component_coeffs*N_dot_L

        # ======== TOTAL 
      
        if iter>self.start_mlp_iter: #warmup for Light params + diffuse MLP
            I_diffuse_rough, I_specular_rough = I_diffuse_color_mlp*(1-fresnel),  I_specular_coeffs
        else:
            I_diffuse_rough, I_specular_rough  = I_diffuse_color_coeffs*(1-fresnel), I_specular_coeffs

        
        I_diffuse_final, I_specular_final = torch.clamp_min(I_diffuse_rough, 0), torch.clamp_min(I_specular_rough, 0)
            

        if ret_loss:
            if iter<=self.start_mlp_iter:
                # at the begining we "initialize"  mlp with coeff values. 
                linear_loss_factor = 1.0
            elif self.end_diffuse_loss_iter <= self.start_mlp_iter:
                linear_loss_factor = 0.0
            else: 
                #we add more freedom to diffuse mlp to be able to model more complicated relations
                linear_loss_factor = torch.clamp(\
                    torch.tensor(1- ((iter- self.start_mlp_iter) / (self.end_diffuse_loss_iter - self.start_mlp_iter + self.eps_s0))), 0, 1)
            
            diffuse_loss = linear_loss_factor * \
                    (((1 - diffuse_component_mlp)**2)
                    )
            
            albedo_loss = ((base_color - base_color.mean(dim=0)) ** 2).mean()
            roughness_loss = ((self.roughness - self.roughness.mean(dim=0)) ** 2).mean()
            f0_loss = ((self.F_0 - self.F_0.mean(dim=0)) ** 2).mean()
                        
        # to render light rotations (separate light from camera):
        if light_rot is not None:

            # Normalize the camera normal
            light_normal = torch.nn.functional.normalize(light_rot[:, 2], dim=0)

            # Repeat camera normal for all features
            light_normal = light_normal.repeat(self.get_features.shape[0], 1)

            # Compute dot product (cosine of angle for spotlight angle)
            cos_theta = torch.sum(L * -light_normal, dim=1).unsqueeze(1)

            # Define scaling factors based on the cosine value
            scaling_factor = torch.where(
                cos_theta > 0, 
                torch.abs(cos_theta**spotlight_attenuation_power),
                0
            )

            # Apply scaling to I_diffuse_final and I_specular_final
            I_diffuse_final = scaling_factor* I_diffuse_final
            I_specular_final = scaling_factor * I_specular_final    
        

        if disable_reflections:
            reflected_rgb= I_diffuse_final
        else:
            reflected_rgb= I_diffuse_final+I_specular_final

        if ret_loss:
            return reflected_rgb, (diffuse_loss, albedo_loss, roughness_loss, f0_loss)
        else:
            return reflected_rgb
    

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_closest_point_indices(self):
        return self.closest_point_indices

    @property
    def get_original_normals(self):
        return self.original_normals
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def get_surface_normal(self, scaling_modifier = 1):
        return self.surface_normal(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def find_closest_indices(self, points, n_closest:int=1): 
        if self.faiss_index is None:
            raise ValueError("faiss index not initialized")
        _, closest_indices = self.faiss_index.search(points.contiguous(), n_closest)
        return closest_indices.squeeze()

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, args, init_args=None):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        roughness = (init_args.roughness_init*torch.ones((fused_color.shape[0], 1))).float().cuda()
        F_0 = (init_args.f0_init*torch.ones((fused_color.shape[0], 1))).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001) # find mean dist to nearest 3 neighbors
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.roughness = nn.Parameter(roughness.requires_grad_(True))
        self.F_0 = nn.Parameter(F_0.requires_grad_(True))

        self.original_normals = self.compute_point_cloud_normals(k=init_args.K_normals, plotting=False, create_faiss_index=True)
        self.closest_point_indices = self.find_closest_indices(self.get_xyz)

        print("Finished creating Gaussian model from point cloud.")

    def training_setup(self, training_args, tuning=False):
        self.setup_mlp()
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        if tuning == False:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': [self.roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
                {'params': [self.F_0], 'lr': training_args.f0_lr, "name": "F_0"},
            ]

            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)

            self.lambda_norm = training_args.lambda_norm
            self.end_diffuse_loss_iter=training_args.end_diffuse_loss_iter
            self.random_noise_val = training_args.random_noise_val

            
        l_mlp = [
            {'params': [*self.positional_encoding_camera.parameters()], 'lr': training_args.grid_lr, "name": "hash_grid_camera"},
            {'params': [*self.mlp.parameters()], 'lr': training_args.mlp_lr, "name": "mlp"},]
        self.optimizer_mlp = torch.optim.Adam(l_mlp, lr=0.0, eps=1e-15)
        print('MLP optimizer parameters: ', [p["name"] for p in self.optimizer_mlp.param_groups])
        
       
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]+1): #3rd scale
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self.roughness.shape[1]):
            l.append('roughness_{}'.format(i))
        for i in range(self.F_0.shape[1]):
            l.append('F_0_{}'.format(i))
        return l

    def save_ply(self, path, normals=None):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        if normals is None:
            normals = np.zeros_like(xyz)
        else:
            normals = normals.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        # add 3rd scale
        log_value = torch.log(torch.tensor(self.eps_s0)).item()
        new_column = log_value * np.ones((scale.shape[0], 1))
        scale = np.concatenate((scale, new_column), axis=1)

        ###
        rotation = self._rotation.detach().cpu().numpy()
        roughness = self.roughness.detach().cpu().numpy()
        F_0 = self.F_0.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, roughness, F_0), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        # cut 2nd scale  needed for modifications or supersplat 
        scale_names = scale_names[:2]

        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        roughness_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("roughness")]
        roughness_names = sorted(roughness_names, key = lambda x: int(x.split('_')[-1]))
        roughness = np.zeros((xyz.shape[0], len(roughness_names)))
        for idx, attr_name in enumerate(roughness_names):
            roughness[:, idx] = np.asarray(plydata.elements[0][attr_name])

        F_0_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("F_0")]
        F_0_names = sorted(F_0_names, key = lambda x: int(x.split('_')[-1]))
        F_0 = np.zeros((xyz.shape[0], len(F_0_names)))
        for idx, attr_name in enumerate(F_0_names):
            F_0[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
        self.F_0 = nn.Parameter(torch.tensor(F_0, dtype=torch.float, device="cuda").requires_grad_(True))


        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.roughness = optimizable_tensors["roughness"]
        self.F_0 = optimizable_tensors["F_0"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self.closest_point_indices = self.closest_point_indices[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_roughness, new_F_0):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "roughness": new_roughness,
        "F_0": new_F_0}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.roughness = optimizable_tensors["roughness"]
        self.F_0 = optimizable_tensors["F_0"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        new_closest_point_indices = self.find_closest_indices(new_xyz)
        

        if new_closest_point_indices.ndim == 0:
            if self.faiss_index is None:
                raise ValueError("faiss index not initialized")
            else:
                self.closest_point_indices = torch.cat((self.closest_point_indices, new_closest_point_indices.unsqueeze(dim=0)), dim=0)
        else:
            self.closest_point_indices = torch.cat((self.closest_point_indices, new_closest_point_indices), dim=0)
        


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        if self.lambda_norm != 0:
            stds = self.get_scaling[selected_pts_mask].repeat(N,1)*0.1 # limit movement so normals are valid 
        else:
            stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda") 
        samples = torch.normal(mean=means, std=stds) 
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))[:, [0, 1]]
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_F_0 = self.F_0[selected_pts_mask].repeat(N,1)
        new_roughness = self.roughness[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_roughness, new_F_0)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_roughness = self.roughness[selected_pts_mask]
        new_F_0 = self.F_0[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_roughness, new_F_0)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        #if there are more valid points then 200k, you can prune 
        if (~prune_mask).sum()>150000:
            self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def get_gaussian_normals(self):
        """
        Compute the normal of the 3D Gaussian and concatenate with xyz.
        """
        normal_mat_normalized = torch.nn.functional.normalize(self.get_surface_normal(), dim=1) 
        normals_normalized = torch.cat((self.get_xyz, normal_mat_normalized), dim=1) 
        return normals_normalized

    def compute_point_cloud_normals(self, k=10, plotting=False, create_faiss_index=True):
        """
        Compute normal vectors for each point in the point cloud using PCA on the neighborhood.

        param:
            k (int): Number of nearest neighbors to consider for each point.
        """
        start = time.time()
        print("Computing point cloud normals... This will only be done once.\n")

        xyz = self.get_xyz

        # break up xyz into chunks to avoid BREAKING THINGS dammit
        size_of_xyz = xyz.shape[0]
        chunk_size = int(size_of_xyz / 1000)

        all_normals = None

        num_chunks = (xyz.shape[0] + chunk_size - 1) // chunk_size
        pbar = tqdm(total=num_chunks, desc="Computing normals")

        for i in range(0, xyz.shape[0], chunk_size):
            chunk = xyz[i:i+chunk_size]

            distances = cdist(chunk, xyz)

            k_neighbors_indices = torch.topk(distances, k=k+1, largest=False, sorted=False)[1][:, 1:]

            flat_indices = k_neighbors_indices.reshape(-1)

            actual_chunk_size = chunk.shape[0]

            neighbors = torch.index_select(xyz, 0, flat_indices).reshape(actual_chunk_size, k, xyz.shape[1])

            neighbors_centered = neighbors - neighbors.mean(dim=1, keepdim=True)

            covariance_matrices = torch.matmul(neighbors_centered.transpose(-2, -1), neighbors_centered) / (k - 1)

            _, eigenvectors = torch.linalg.eigh(covariance_matrices)

            normals = eigenvectors[..., 0]

            eps  = 1e-8
            normals_normalized = torch.nn.functional.normalize(normals, dim=1, eps=eps) 

            if all_normals is None:
                all_normals = normals_normalized
            else:
                all_normals = torch.cat((all_normals, normals_normalized), dim=0)

            pbar.update(1)
        pbar.close()

        all_normals = torch.cat((xyz, all_normals), dim=1) # size is (N, 6)

        end = time.time()
        print("Finished computing normals in {} seconds.".format(end-start))

        if create_faiss_index:
            # initialize faiss index
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.GpuIndexFlatL2(res, 3)
            normals = all_normals[:, :3].detach().contiguous()
            self.faiss_index.add(normals)
        
        return all_normals
    
    # games reparametrization
    def prepare_vertices(self):
        """
        Prepare psudo-mesh face based on Gaussian.
        """ 

        self.s0 = torch.ones(self._scaling.shape[0], 1).cuda() * self.eps_s0
        scales = torch.cat([self.scaling_activation(self._scaling[:, [0,1]]), self.s0], dim=1)

        rotation = self._rotation
        R = build_rotation(rotation)
        R = R.transpose(-2, -1)

        v1 = self._xyz
    
        s_2 = scales[:, 0] 
        s_3 = scales[:, 1] 
        _v2 = v1 + s_2.reshape(-1, 1) * R[:, 0]
        _v3 = v1 + s_3.reshape(-1, 1) * R[:, 1]        

        mask = s_2 > s_3

        v2 = torch.zeros_like(_v2)
        v3 = torch.zeros_like(_v3)

        v2[mask] = _v2[mask]
        v3[mask] = _v3[mask]

        # v2[~mask] = _v3[~mask]
        # v3[~mask] = _v2[~mask]

        v2[~mask] = _v2[~mask]
        v3[~mask] = _v3[~mask]

        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

        self.triangles = torch.stack([v1, v2, v3], dim = 1)

    # back from games reparametrization
    def prepare_scaling_rot(self, eps=1e-8):
        """
        Approximate covariance matrix and calculate scaling/rotation tensors.
        Prepare parametrized Gaussian.
        """

        def dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)

        def proj(v, u):
            """
            projection of vector v onto subspace spanned by u

            vector u is assumed to be already normalized
            """
            coef = dot(v, u)
            return coef * u

        v1 = self.triangles[:, 0].clone()
        v2 = self.triangles[:, 1].clone()
        v3 = self.triangles[:, 2].clone()

        _s2 = v2 - v1
        _s3 = v3 - v1

        r1 = torch.cross(_s2, _s3)
        s2 = torch.linalg.vector_norm(_s2, dim=-1, keepdim=True) + eps
        _s3_norm = torch.linalg.vector_norm(_s3, dim=-1, keepdim=True) + eps

        r1 = r1 / (torch.linalg.vector_norm(r1, dim=-1, keepdim=True) + eps)
        r2 = _s2 / s2
        r3 = _s3 - proj(_s3, r1) - proj(_s3, r2)
        r3 = r3 / (torch.linalg.vector_norm(r3, dim=-1, keepdim=True) + eps)
        s3 = dot(_s3, r3)

        scales = torch.cat([s2, s3], dim=1)
        self._scaling = self.scaling_inverse_activation(scales)


        rotation = torch.stack([r2, r3, r1], dim=1)
        rotation = rotation.transpose(-2, -1)

        self._rotation = rot_to_quat_batch(rotation)
        

