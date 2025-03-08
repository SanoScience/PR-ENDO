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
from scene import Scene
import os
import time
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.spiral_utils import spiral_cam_info
import re
import imageio
import numpy as np
from PIL import Image, ImageDraw
import copy
import random

def generate_video(imgs_path, text_to_add = ''):

    image_files = [f for f in os.listdir(imgs_path) if re.match(r'\d+\.png$', f) and '_depth' not in f]
    image_files_sorted = sorted(image_files, key=lambda x: int(re.findall(r'\d+', x)[0]))
    writer = imageio.get_writer(f"{imgs_path}/0000_video.mp4", fps=20)
    for img_name in image_files_sorted:
        img_path = os.path.join(imgs_path, img_name)
        img = Image.open(img_path)
        # Draw text on the image
        draw = ImageDraw.Draw(img)
        draw.text(( img.width - 10 * len(text_to_add), 20), text_to_add, fill="green" )  # Adjust position
    
        writer.append_data(np.array(img))
    writer.close()


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, light, loaded_iter):
    main_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    render_path = os.path.join(main_path, "renders")
    gts_path = os.path.join(main_path, "gt")
    seaprate_path_fixed_cam = os.path.join(main_path, "separate_light_camera_fixed_cam")
    seaprate_path_fixed_light = os.path.join(main_path, "separate_light_camera_fixed_light")
    disable_reflections_path = os.path.join(main_path, "disable_reflections")
    normal_path = os.path.join(main_path, "normals")
    albedo_path = os.path.join(main_path, "albedo")
    views = sorted(views, key=lambda obj: obj.colmap_id)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(disable_reflections_path, exist_ok=True)
    makedirs(albedo_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    
    middle_cam_idx = int(len(views)//2)
    middle_cam = views[middle_cam_idx]
    first_cam = views[0]

    random.seed(42)
    random_ids = random.sample(range(len(views)), min(20,len(views)))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        gt = view.original_image.cuda()
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        override_color = gaussians.compute_lighted_rgb(camera_center = view.camera_center, light = light, iter = loaded_iter)
        renderpkg = render(view, gaussians, pipeline, background, override_color = override_color)
        rendering = renderpkg["render"].clamp(0,1)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        
        if True: #idx in random_ids:
            override_color = gaussians.compute_lighted_rgb(camera_center = view.camera_center, light = light, iter = loaded_iter, disable_reflections=True)
            renderpkg = render(view, gaussians, pipeline, background, override_color = override_color)
            rendering = renderpkg["render"].clamp(0,1)
            torchvision.utils.save_image(rendering, os.path.join(disable_reflections_path, '{0:05d}'.format(idx) + ".png"))

            override_color = gaussians.compute_gaussian_rgb(view.camera_center)
            renderpkg = render(view, gaussians, pipeline, background, override_color = override_color)
            rendering = renderpkg["render"].clamp(0,1)
            torchvision.utils.save_image(rendering, os.path.join(albedo_path, '{0:05d}'.format(idx) + ".png"))

            # render normals
            dir_pp_camera = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_features.shape[0], 1))
            normal = gaussians.get_gaussian_normals()[:, 3:]

            # # Normalize the input vectors
            N = torch.nn.functional.normalize(normal, dim=1)
            V = torch.nn.functional.normalize(dir_pp_camera, dim=1)
            
            # # normals always towards camera
            N_dot_V = torch.sum(N * -V, dim=1, keepdim=True)  # [N, 1]
            N = torch.where(N_dot_V < 0, -N, N)  # Flip N if N_dot_V < 0
            override_color = (N+1)/2
            renderpkg = render(view, gaussians, pipeline, background, override_color = override_color)
            rendering = renderpkg["render"].clamp(0,1)
            torchvision.utils.save_image(rendering, os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"))
        
        
    
    # Make videos
    generate_video(render_path, "renders")
    generate_video(albedo_path, "albedo")
    generate_video(disable_reflections_path, "disable_refl")
    generate_video(normal_path, "normals")


    # Performence check
    render_time_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()
        start = time.time()
        override_color = gaussians.compute_lighted_rgb(camera_center = view.camera_center, light = light, iter = loaded_iter)
        renderpkg = render(view, gaussians, pipeline, background, override_color = override_color)

        torch.cuda.synchronize()
        end = time.time()
        render_time_list.append((end-start)*1000)
    with open(os.path.join(model_path, 'render_time.txt'), 'w') as f:
        for t in render_time_list:
            f.write("%.2fms\n"%t)
        f.write("Mean time: %.2fms\n"%(np.mean(render_time_list[5:])))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_spiral : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        gaussians.mlp.eval()
        gaussians.positional_encoding_camera.eval()

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, scene.light, scene.loaded_iter)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, scene.light, scene.loaded_iter)
            
        # if not skip_spiral: 
        #     cams = spiral_cam_info(dataset.model_path, focal=0.7)
        #     render_set(dataset.model_path, "spiral", scene.loaded_iter,cams, gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_spiral", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)


    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_spiral)
