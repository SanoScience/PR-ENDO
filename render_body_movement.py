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
import copy
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
import random
from PIL import ImageDraw


def transform_sinus(triangles, t):
    # Linearly scales amplitude
    triangles_new = triangles.clone()
    amplitude = t  
    period = torch.pi / 15 
    triangles_new[:, :, 1] += amplitude * torch.sin(triangles[:, :, 2] * period)
    
    return triangles_new, amplitude, period


def transform_save_gaussians(gaussians, scene):
    if hasattr(gaussians, 'prepare_vertices'):
            gaussians.prepare_vertices()
    if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()

    #t = torch.linspace(0, 4*torch.pi, 100 )
    ampl = 1.5
    t = torch.linspace(0, ampl, 30 )


    v1, v2, v3 = gaussians.v1, gaussians.v2, gaussians.v3
    triangles = torch.stack([v1, v2, v3], dim=1)

    for i, t_i in enumerate(tqdm(t, desc="body movement progress")):

        old_xyz = gaussians._xyz.clone()
        new_triangles, _, per = transform_sinus(triangles, t_i)
        
        gaussians.triangles = new_triangles
        _xyz = new_triangles[:, 0]
        gaussians.v2 = new_triangles[:, 1]
        gaussians.v3 = new_triangles[:, 2]
        gaussians.prepare_scaling_rot()
        gaussians._xyz=_xyz

        ply_dir = os.path.join(scene.model_path, f"sinus_animation_ampl{ampl}_per{round(per, 2)}")
        if not os.path.exists(ply_dir):
            os.makedirs(ply_dir)
        ply_path = os.path.join(ply_dir, f"gaussians_{str(i).zfill(3)}.ply")

        gaussians.save_ply(ply_path)

        #restore old xyz
        gaussians._xyz = old_xyz

    return ply_dir

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians.mlp.eval()
    gaussians.positional_encoding_camera.eval()

    with torch.no_grad():

        if (not args.skip_test) and (len(scene.getTrainCameras()) > 0):

            modification_folder = transform_save_gaussians(gaussians, scene) # transform_save_gaussians returns folder path
            modification_paths = sorted([os.path.join(modification_folder, f) for f in os.listdir(modification_folder) if "ply" in f])
    
            viewpoint_stack = scene.getTrainCameras()
            
            # select only n camera poses for body movement
            random.seed(42)
            random.shuffle(viewpoint_stack)
            viewpoint_stack = viewpoint_stack[:5]

            for i, viewpoint_cam in tqdm(enumerate(viewpoint_stack), desc="rendering progress"):
                gaussians.load_ply(os.path.join(dataset.model_path,
                                                "point_cloud",
                                                "iteration_" + str(scene.loaded_iter),
                                                "point_cloud.ply"))
                gt_image = torch.clamp(viewpoint_cam.original_image, 0.0, 1.0)
                torchvision.utils.save_image(gt_image, os.path.join(modification_folder, f"camera{i}_original_gt.png"))

                override_color = gaussians.compute_lighted_rgb(camera_center = viewpoint_cam.camera_center, light = scene.light, iter = scene.loaded_iter)
                render_pkg = render(viewpoint_cam, gaussians, pipe=pipe, bg_color=background, override_color = override_color)
                rgb = render_pkg['render']
                torchvision.utils.save_image(torch.clamp(rgb, 0.0, 1.0), os.path.join(modification_folder, f"camera{i}_original_render.png"))
                
                rgbmaps = []
                for idx, modification_path in enumerate(modification_paths):
                    gaussians.load_ply(modification_path)
                    override_color = gaussians.compute_lighted_rgb(camera_center = viewpoint_cam.camera_center, light = scene.light, iter = scene.loaded_iter)

                    render_pkg = render(viewpoint_cam, gaussians, pipe=pipe, bg_color=background, override_color = override_color)
                    rgb = render_pkg['render']
                    torchvision.utils.save_image(torch.clamp(rgb, 0.0, 1.0), os.path.join(modification_folder, f"camera{i}_timestep{idx}.png"))
                    pil_rendering = torchvision.transforms.functional.to_pil_image(torch.clamp(rgb, 0.0, 1.0))
                    
                    # Draw text on the image
                    draw = ImageDraw.Draw(pil_rendering)
                    draw.text(( pil_rendering.width - 10 * len("sinus_games"), 20), "sinus_games", fill="green" )  # Adjust position
    
                    rgbmaps.append(pil_rendering)

                rgbmaps = rgbmaps + rgbmaps[::-1]
                rgbmaps = rgbmaps + rgbmaps
                writer = imageio.get_writer(os.path.join(modification_folder, f"camera{i}.mp4"), fps=24)
                for k in range(len(rgbmaps)):
                    frame = np.array(rgbmaps[k])
                    writer.append_data(frame)
                writer.close()


        
