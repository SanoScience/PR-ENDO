# Required imports
import torch
import os
from tqdm import tqdm
from gaussian_renderer import render
import torchvision
import imageio
from utils.train_utils import rotate_camera, interpolate_cameras
import numpy as np
import random

if __name__ == "__main__":
    # Set up command line argument parser
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import GaussianModel
    from scene import Scene

    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--spotlight_attenuation_power", default=0, type=int)
    parser.add_argument("--max_rot", default=12, type=int)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Extract model and pipeline parameters
    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    iteration = scene.loaded_iter
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Set models to evaluation mode
    gaussians.mlp.eval()
    gaussians.positional_encoding_camera.eval()

    max_rot = args.max_rot
    spotlight_attenuation_power = args.spotlight_attenuation_power

    # Create output directory for videos
    video_dir = os.path.join(scene.model_path, f"rotate_light_video_maxrot{max_rot}_cospower{spotlight_attenuation_power}")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    steps=15
    max_rotations = [
        (max_rot, max_rot, max_rot),   
        (-max_rot, -max_rot, -max_rot),
        (max_rot, -max_rot, max_rot),
        (-max_rot, max_rot, -max_rot)
    ]
    

    with torch.no_grad():
        # If test cameras exist
        if len(scene.getTestCameras()) > 0:

            # Randomly sample 1-2 test cameras
            viewpoint_stack = scene.getTestCameras()
            random.seed(42)
            random.shuffle(viewpoint_stack)
            viewpoint_stack = viewpoint_stack[:1]

            for i, viewpoint_cam in tqdm(enumerate(viewpoint_stack), desc="render video progress"):

                # Process each maximum rotation
                for idx, max_rotation in enumerate(max_rotations):
                    
                    # Generate rotated matrix
                    rotated_camera = rotate_camera(viewpoint_cam, 
                                                   torch.clamp(viewpoint_cam.original_image.to("cuda"), 0.0, 1.0),
                                                   angle=max_rotation)
                    interpolated_cameras = interpolate_cameras(viewpoint_cam, rotated_camera, steps=20)
                    gif_imgs = []
    
                    # Render interpolated frames
                    for ic in interpolated_cameras:
                        
                        # Compute override color with randomized settings
                        override_color = gaussians.compute_lighted_rgb(
                            camera_center=ic.camera_center,
                            light=scene.light,
                            iter=iteration,
                            light_center_sep=ic.camera_center,
                            light_rot=torch.tensor(ic.R, device=gaussians.get_xyz.device, dtype=torch.float32),
                            spotlight_attenuation_power = spotlight_attenuation_power
                        )

                        # Use the original settings for rendering
                        image_rendered = render(
                            viewpoint_cam,
                            gaussians,
                            pipe=pipe,
                            bg_color=background,
                            override_color=override_color
                        )["render"]

                        # Clamp and save image
                        image_rendered = torch.clamp(image_rendered, 0.0, 1.0)
                        pil_rendering = torchvision.transforms.functional.to_pil_image(image_rendered)
                        gif_imgs.append(pil_rendering)

                    # Save the entire sequence as a video
                    video_path = os.path.join(video_dir, f"{viewpoint_cam.image_name}_rotid{idx}.mp4")
                    writer = imageio.get_writer(video_path, fps=24)
                    for img in gif_imgs:
                        writer.append_data(np.array(img))
                    writer.close()
