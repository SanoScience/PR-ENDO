import os
import sys
import uuid
import warnings
from random import randint
from argparse import ArgumentParser

# Third-party imports
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import lpips
from tqdm import tqdm
import copy
# Local application imports
from utils.loss_utils import l1_loss, ssim, compute_geometric_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.train_utils import prepare_output_and_logger, training_report, save_example_images
import torch

import json
import sys


# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

def training(dataset, opt, pipe, args):
    """
    Executes the training loop for the specified dataset and model parameters.

    Parameters:
        dataset (object): The dataset to be used for training.
        opt (object): Optimization parameters.
        pipe (object): Pipeline parameters.
        args (Namespace): Command-line arguments containing various training options.
    """

    # -----------------------------------------------------------
    # Initialize training parameters and setup
    # -----------------------------------------------------------

    params = {
        'test_iterations': args.test_iterations,
        'save_iterations': args.save_iterations,
        'checkpoint_iterations': args.checkpoint_iterations,
        'checkpoint': args.start_checkpoint,
        'debug_from': args.debug_from,
        'verbose': args.verbose,
        'save_img_from_itr': args.save_img_from_itr
    }

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, init_train_args=opt)
    gaussians.training_setup(opt)

    # load checkpoint if specified
    if params['checkpoint']:
        (model_params, first_iter) = torch.load(params['checkpoint'])
        gaussians.restore(model_params, opt)

    # set background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    
    # initialize normals for geometric loss if weight is not 0
    if opt.lambda_norm != 0:
        original_normals = gaussians.get_original_normals.detach()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # prepare stuff for logging best model

    log_file_path = os.path.join(scene.model_path, "results.txt")
    if os.path.exists(log_file_path):
        os.remove(log_file_path) 
    with open(log_file_path, 'w') as file:
        file.write(f"results for {scene.model_path}\n")

    best_model ={'iter':0, 'test_avg_psnr':0}

    # -----------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------

    print("\n[ITER {}] Saving Checkpoint before training".format(0))
    scene.save(0)

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        randinteger = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(randinteger)

        # -----------------------------------------------------------
        # Rendering
        # -----------------------------------------------------------

        if (iteration - 1) == params['debug_from']:
            pipe.debug = True


        override_color, (diffuse_loss, albedo_loss, roughness_loss, f0_loss) = gaussians.compute_lighted_rgb(
            camera_center = viewpoint_cam.camera_center, light = scene.light, ret_loss=True, iter=iteration, randomize_input=True)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color = override_color)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # -----------------------------------------------------------
        # Loss computation
        # -----------------------------------------------------------
        gt_image = viewpoint_cam.original_image.cuda() 
        L1_images = l1_loss(image, gt_image)

        
        loss = (1.0 - opt.lambda_dssim) * L1_images
        psnr_ = psnr(image, gt_image).mean().double()

        if opt.lambda_depth !=0 and viewpoint_cam.original_depth is not None:
            
            gt_depth = viewpoint_cam.original_depth.cuda()
            depth = render_pkg["depth"]
            
            L_depths = F.huber_loss(depth, gt_depth, delta=0.2) 
            loss += opt.lambda_depth * L_depths 

        if opt.lambda_dssim != 0:
            L_dssim = 1.0 - ssim(image, gt_image)
            loss += opt.lambda_dssim * L_dssim

        if opt.lambda_norm != 0 and iteration > opt.lambda_norm_start and iteration % opt.lambda_norm_skip == 0:
            gaussian_normals = gaussians.get_gaussian_normals()
            closest_point_indices = gaussians.get_closest_point_indices
            L_normals = compute_geometric_loss(gaussian_normals, original_normals, closest_point_indices)
            loss += opt.lambda_norm * L_normals 
        
        sys_exit = False
        if loss.isnan():
            print('nan')
            with open(log_file_path, "a") as file:
                file.write("\n" + "LOSS NAN" + "\n")
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            print("\n[ITER {}] Saving MLP & GRID Checkpoint".format(iteration))
            torch.save(gaussians.mlp.state_dict(), scene.model_path + "/chkpnt_mlp" + str(iteration) + ".pth")
            torch.save(gaussians.positional_encoding_camera.state_dict(), scene.model_path + "/chkpnt_grid_camera" + str(iteration) + ".pth")
            
            print("\n[ITER {}] Saving Light Checkpoint".format(iteration))
            torch.save(scene.light, scene.model_path + "/chkpnt_light" + str(iteration) + ".pth")
            params['test_iterations'].append(iteration)
            sys_exit = True

        
        # Add diffuse and tissue consistency losses
    
        #exception! diffuse loss can be only applied to visible gaussians. Otherwise we  get incorrectly assumed values
        loss += opt.diffuse_loss_weight * diffuse_loss[visibility_filter].mean()
        loss += opt.albedo_loss_weight * albedo_loss
        loss += opt.roughness_loss_weight * roughness_loss
        loss += opt.f0_loss_weight * f0_loss
        loss.backward()

        iter_end.record()

        # -----------------------------------------------------------
        # Training report
        # -----------------------------------------------------------

        with torch.no_grad():

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            total_points = gaussians.get_xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "PSNR": f"{psnr_:.{3}f}", "Points": f"{total_points}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Save images
            if params['save_img_from_itr'] and iteration in params['save_img_from_itr']:
                save_example_images(image, gt_image, depth, gt_depth, iteration, dataset.source_path)

            # Log and save
            report_params = {
                'tb_writer': tb_writer,
                'iteration': iteration,
                'Ll1': L1_images,
                'loss': loss,
                'l1_loss': l1_loss,
                'elapsed': iter_start.elapsed_time(iter_end),
                'testing_iterations': params['test_iterations'],
                'scene': scene,
                'renderFunc': render,
                'renderArgs': (pipe, background),
                'verbose': True, #params['verbose']
                'log_file_path':log_file_path
            }
            
            current_test_psnr = training_report(**report_params)
            if current_test_psnr>=best_model['test_avg_psnr']:
                best_model['test_avg_psnr']=current_test_psnr
                best_model['iter']=iteration

            if (iteration in params['test_iterations']):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            if (iteration in params['save_iterations']):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                scene.save(iteration)
            
            if sys_exit:
                sys.exit()

            # -----------------------------------------------------------
            # Adding Gaussian points 
            # -----------------------------------------------------------

            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # -----------------------------------------------------------
            # Optimization step
            # -----------------------------------------------------------

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                gaussians.optimizer_mlp.step()
                gaussians.optimizer_mlp.zero_grad(set_to_none = True)

                scene.optimizer_light.step()
                scene.optimizer_light.zero_grad(set_to_none = True)

            if (iteration in params['checkpoint_iterations']):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                print("\n[ITER {}] Saving MLP & GRID Checkpoint".format(iteration))
                torch.save(gaussians.mlp.state_dict(), scene.model_path + "/chkpnt_mlp" + str(iteration) + ".pth")
                torch.save(gaussians.positional_encoding_camera.state_dict(), scene.model_path + "/chkpnt_grid_camera" + str(iteration) + ".pth")
                
                print("\n[ITER {}] Saving Light Checkpoint".format(iteration))
                torch.save(scene.light, scene.model_path + "/chkpnt_light" + str(iteration) + ".pth")
    
    with open(log_file_path, "a") as file:
        file.write("\n" + "BEST MODEL" + "\n")
        file.write(json.dumps(best_model) + "\n")
        file.write(json.dumps({"NUM GAUSSIANS ":len(gaussians.get_xyz)}))
            


if __name__ == "__main__":

    # set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 10000, 20000, 30000, 40000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 10000, 20000, 30000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1000, 10000, 20000, 30000, 40000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--save_img_from_itr", nargs="+", type=int, default=None) 
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    assert args.save_iterations == args.checkpoint_iterations #TODO clean it
    assert args.save_iterations == args.test_iterations #TODO clean it
    
    print("Optimizing " + args.model_path)

    # initialize system state (RNG)
    safe_state(args.quiet)


    # start network gui
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # train
    train_params = {
        'dataset': lp.extract(args),
        'opt': op.extract(args),
        'pipe': pp.extract(args),
        'args': args,
    }
    training(**train_params)

    # finished
    print("\nTraining complete.")
