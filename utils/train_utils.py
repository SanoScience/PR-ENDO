import os
import torch
import uuid
from argparse import Namespace
from .image_utils import psnr
from pytorch_msssim import ms_ssim
from .loss_utils import ssim
import lpips
from scene import Scene
import numpy as np
from PIL import Image
import copy
import imageio
import torch.nn.functional as F
import torchvision
from utils.augmented_rotation_utils import rotate_matrix 


from scene.cameras import Camera
from scipy.spatial.transform import Rotation

def rotate_camera(camera: Camera, gt_image, angle=[-5, 5, -5]) -> Camera:
    rotation = Rotation.from_euler('xyz', angle, degrees=True)
    r_rotated = rotation.apply(camera.R)
    cam = Camera(colmap_id=camera.uid, R=r_rotated, T=camera.T, 
        FoVx=camera.FoVx, FoVy=camera.FoVy, 
        image=gt_image, gt_alpha_mask=None,depth=camera.original_depth,
        image_name="", uid=camera.uid, cx=camera.cx, cy=camera.cy, data_device=camera.original_image.device)
    return cam

# Function to interpolate matrices for smooth rotations
def interpolate_matrices(mat1, mat2, steps=20):
    """Interpolate between two matrices in a linear fashion"""
    return [(1 - alpha) * mat1 + alpha * mat2 for alpha in torch.linspace(0, 1, steps)]

def interpolate_cameras(camera1: Camera, camera2: Camera, steps=20):
    """Interpolate between two camera objects in a linear fashion."""
    interpolated_cameras = []
    for alpha in torch.linspace(0, 1, steps):
        R_interp = (1 - alpha) * camera1.R + alpha * camera2.R
        R_interp = R_interp.numpy()
        
        T_interp = (1 - alpha) * camera1.T + alpha * camera2.T
        T_interp = T_interp.numpy()
        interpolated_camera = Camera(
            colmap_id=camera1.uid,
            R=R_interp,
            T=T_interp,
            FoVx=camera1.FoVx,
            FoVy=camera1.FoVy,
            image=camera1.original_image,
            gt_alpha_mask=None,
            depth=camera1.original_depth,
            image_name=camera1.image_name,
            uid=camera1.uid,
            cx=camera1.cx,
            cy=camera1.cy,
            data_device=camera1.original_image.device
        )
        interpolated_cameras.append(interpolated_camera)
    return interpolated_cameras

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("Tensorboard found")
except ImportError:
    TENSORBOARD_FOUND = False

def prepare_output_and_logger(args):
    """
    Prepares the output folder and Tensorboard logger.

    Parameters:
        args (Namespace): Arguments from the command line.
    """ 
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10]) 
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, verbose, log_file_path = "results.txt"):
    """
    Logs training progress and evaluates the model at specific iterations.

    Parameters:
        tb_writer (SummaryWriter): TensorBoard writer for logging.
        iteration (int): Current iteration number.
        Ll1 (Tensor): L1 loss for logging.
        loss (Tensor): Total loss for logging.
        l1_loss (function): L1 loss function.
        elapsed (float): Elapsed time for the iteration.
        testing_iterations (list): Iterations at which to perform evaluation.
        scene (Scene): Scene object containing training and test data.
        renderFunc (function): Function to render images.
        renderArgs (tuple): Additional arguments for the render function.
        verbose (bool): If True, perform detailed evaluation and logging.
    """
    
    current_test_psnr = torch.tensor(0)
    gif_dir = f"{scene.model_path}/rotate_gifs"
    os.makedirs(gif_dir, exist_ok=True) 
    
    if tb_writer:
        # log basic metrics
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # train_cameras_filtered = [c for c in scene.getTrainCameras() if "aux" not in str(c.image_name)]
        validation_configs = (
            {'name': 'test', 'cameras' : scene.getTestCameras()}, 
            )
        
        lpips_model = lpips.LPIPS(net='vgg').cuda() if verbose else None

        for config in validation_configs:
            if not config['cameras']:
                continue

            l1_test, psnr_test = [], []
            lpips_test, ssim_test, mssim_test = ([] for _ in range(3)) if verbose else (None, None, None)

            for idx, viewpoint in enumerate(config['cameras']):

                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                override_color = scene.gaussians.compute_lighted_rgb(camera_center = viewpoint.camera_center, light = scene.light, iter=iteration)
                image_mlp = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=override_color)["render"], 0.0, 1.0)


                if tb_writer and idx < 5:
                    tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render_base", image[None], global_step=iteration)
                    tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render_lit", image_mlp[None], global_step=iteration)
                    torchvision.utils.save_image(image_mlp, os.path.join(gif_dir, f'{viewpoint.image_name}_lit_original_pose'+ ".png"))
                    
                    if iteration == testing_iterations[0] or True:
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)
                    
                    # render normals we use for training
                    dir_pp_camera = (scene.gaussians.get_xyz - viewpoint.camera_center.repeat(scene.gaussians.get_features.shape[0], 1))
                    normal = scene.gaussians.get_gaussian_normals()[:, 3:]

                    # # Normalize the input vectors
                    N = torch.nn.functional.normalize(normal, dim=1)
                    V = torch.nn.functional.normalize(dir_pp_camera, dim=1)
                    
                    # # normals always towards camera
                    N_dot_V = torch.sum(N * -V, dim=1, keepdim=True)  # [N, 1]
                    N = torch.where(N_dot_V < 0, -N, N)  # Flip N if N_dot_V < 0
                    override_color = (N+1)/2
                    
                    render_normals = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=override_color)["render"], 0.0, 1.0)
                    tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/normals_render", render_normals[None], global_step=iteration)

                    # disable reflections
                    override_color = scene.gaussians.compute_lighted_rgb(camera_center = viewpoint.camera_center, light = scene.light, iter=iteration, disable_reflections=True)
                    render_dis_ref = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=override_color)["render"], 0.0, 1.0)
                    tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/disable_reflections", render_dis_ref[None], global_step=iteration)

                    # Create custom camera and rotate
                    rotated_camera = rotate_camera(viewpoint, gt_image)
        
                    # #---------
                    image_random = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render_random_base", image_random[None], global_step=iteration)

                    override_color = scene.gaussians.compute_lighted_rgb(camera_center = rotated_camera.camera_center, light = scene.light, iter=iteration)
                    image_random_mlp = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=override_color)["render"], 0.0, 1.0)

                    tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render_random_lit", image_random_mlp[None], global_step=iteration)
                    torchvision.utils.save_image(image_random_mlp, os.path.join(gif_dir, f'{viewpoint.image_name}_lit_rotated_pose' + ".png"))

                    # ----------------------------
                    # Function to interpolate matrices for smooth rotations.
                    # ----------------------------

                    interpolated_cameras = interpolate_cameras(viewpoint, rotated_camera, steps=20)
                    gif_imgs = []

                    for camera_rotated in interpolated_cameras:
                        cam_center = camera_rotated.camera_center

                        override_color = scene.gaussians.compute_lighted_rgb(
                            camera_center=cam_center, 
                            light=scene.light, 
                            iter=iteration
                        )

                        image_random_interp = torch.clamp(
                            renderFunc(
                                camera_rotated, 
                                scene.gaussians, 
                                *renderArgs, 
                                override_color=override_color
                            )["render"], 
                            0.0, 1.0
                        )
                        pil_rendering = torchvision.transforms.functional.to_pil_image(image_random_interp)
                        gif_imgs.append(pil_rendering)

                    writer = imageio.get_writer(f"{gif_dir}/{viewpoint.image_name}.mp4", fps=20)
                    for img in gif_imgs:
                        writer.append_data(np.array(img))
                    writer.close()
                    # ----------------------------

                l1_test.append(l1_loss(image_mlp, gt_image).mean().double())
                psnr_test.append(psnr(image_mlp, gt_image).mean().double())

                if verbose:
                    lpips_test.append(lpips_model(image_mlp, gt_image).mean().double())
                    ssim_test.append(ssim(image_mlp, gt_image).mean().double())
                    mssim_test.append(ms_ssim(image_mlp.unsqueeze(0), gt_image.unsqueeze(0), data_range=1.0).mean().double())
            
            l1_test_avg, psnr_test_avg = sum(l1_test) / len(l1_test), sum(psnr_test) / len(psnr_test)

            if verbose or True:
                # Calculate verbose metrics averages and stds
                lpips_test_avg, ssim_test_avg, mssim_test_avg = sum(lpips_test) / len(lpips_test), sum(ssim_test) / len(ssim_test), sum(mssim_test) / len(mssim_test)
                lpips_test_std, ssim_test_std, mssim_test_std = torch.std(torch.tensor(lpips_test)), torch.std(torch.tensor(ssim_test)), torch.std(torch.tensor(mssim_test))
                
                # Print verbose metrics
                avg_to_print = (
                    f"\n[ITER {iteration}] Evaluating {config['name']} avg: "
                    f"L1 {round(l1_test_avg.item(), 2)} PSNR {round(psnr_test_avg.item(), 2)} "
                    f"LPIPS {round(lpips_test_avg.item(), 2)} SSIM {round(ssim_test_avg.item(), 2)} "
                    f"MSSIM {round(mssim_test_avg.item(), 2)}"
                )

                std_to_print = (
                    f"[ITER {iteration}] Evaluating {config['name']} std: "
                    f"L1 {round(torch.std(torch.tensor(l1_test)).item(), 2)} PSNR {round(torch.std(torch.tensor(psnr_test)).item(), 2)} "
                    f"LPIPS {round(lpips_test_std.item(), 2)} SSIM {round(ssim_test_std.item(), 2)} "
                    f"MSSIM {round(mssim_test_std.item(), 2)}"
                )

                print(avg_to_print)
                print(std_to_print)
                with open(log_file_path, "a") as log_file:
                    log_file.write(avg_to_print + "\n")
                    log_file.write(std_to_print + "\n")

            # Log basic and verbose metrics
            if tb_writer:
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test_avg, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test_avg, iteration)
                if verbose:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test_avg, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test_avg, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - mssim', mssim_test_avg, iteration)

            if config['name']=='test':
                current_test_psnr = psnr_test_avg

        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return current_test_psnr.item()

def save_image(tensor, filename, source_path):
    """
    Saves a tensor as an image.

    Parameters:
        tensor (Tensor): Tensor to save as an image.
        filename (str): Name of the file to save the image to.
        source_path (str): Path to the folder where the image should be saved.
    """
    array = tensor.detach().cpu().numpy()
    if array.shape[0] == 1: 
        array = np.squeeze(array, axis=0)
    else:
        array = array.transpose(1, 2, 0)  # Convert from CHW to HWC for RGB images.
    array = (array * 255).astype(np.uint8)
    Image.fromarray(array).save(os.path.join(source_path, filename))

def save_example_images(image, gt_image, depth, gt_depth, iteration, source_path):
    """
    Saves example images for debugging purposes.
    
    Parameters:
        image (Tensor): Rendered image.
        gt_image (Tensor): Ground truth image.
        depth (Tensor): Rendered depth map.
        gt_depth (Tensor): Ground truth depth map.
        iteration (int): Current iteration number.
        source_path (str): Path to the folder where the images should be saved.
    """
    save_image(image, "render_" + str(iteration) + ".png", source_path)
    save_image(gt_image, "gt_" + str(iteration) + ".png", source_path)
    save_image(depth, "depth_" + str(iteration) + ".png", source_path)
    save_image(gt_depth, "gt_depth_" + str(iteration) + ".png", source_path)
