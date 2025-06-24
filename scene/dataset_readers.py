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
import sys
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # for "OSError: unrecognized data stream contents when reading image file"
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB, RGB2SH
from scene.gaussian_model import BasicPointCloud
from scene.rnnslam_loader import readRNN
from utils.obj_utils import read_obj 
from natsort import natsorted
import glob
import random
import torch.nn.functional as F
import torch
import os
from utils.system_utils import searchForMaxIteration
from enum import Enum
from utils.general_utils import build_rotation

MAX_N_POINTS = 700_000

class PinkColor(float, Enum):
    R = 188/255.0
    G = 99/255.0
    B = 76/255.0


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    depth_path: str
    depth_name: str
    width: int
    height: int
    cx: float
    cy: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, depths_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        depth_path = os.path.join(depths_folder, os.path.basename(extr.name))
        depth_name = os.path.basename(depth_path).split(".")[0]
        depth = Image.open(depth_path)
        depth = np.array(depth)
        depth = depth / np.max(depth)
        depth = (depth * 255).astype(np.float32)
        # if depth image has rgb values so (1, 1, 3) just take one channel of last dimension
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        depth = Image.fromarray(depth) 

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, 
                              depth=depth, depth_name=depth_name, depth_path=depth_path,
                              width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normals=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, depths=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    depth_dir = "depths" if depths == None else depths
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                           images_folder=os.path.join(path, reading_dir), depths_folder=os.path.join(path, depth_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readRNNSceneInfo(path, images, depths, eval, llffhold=8):

    extrinsics_file = [f for f in os.listdir(path) if f.endswith("_poses.txt")][0]
    extr_file_path = os.path.join(path, extrinsics_file)
    intrinsics_file = "cameras.txt"
    intr_file_path = os.path.join(path, intrinsics_file)

    reading_dir = "images" if images == None else images
    depths_dir = "depths" if depths == None else depths

    cam_infos_unsorted = readRNN(extrinsics_file=extr_file_path, intrinsics_file=intr_file_path, 
                                    images_folder=os.path.join(path, reading_dir), depths_folder=os.path.join(path, depths_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.uid)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)


    obj_file = [f for f in os.listdir(path) if f.endswith(".obj")][0]
    obj_path = os.path.join(path, obj_file)
    ply_path = os.path.join(path, "points3D.ply")

    # convert obj to ply
    if not os.path.exists(ply_path):
        print("Converting mesh.obj to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb = read_obj(obj_path)
        except Exception as e:
            print('Error reading obj file: {}'.format(e))
            return None
        storePly(ply_path, xyz, rgb*255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def getNerfppNorm2colon(w2c):
    cam_centers = np.linalg.inv(w2c)[:, :3, 3]  # Get scene radius
    center = np.mean(cam_centers, 0)[None]
    translate = -center
    radius = 1.1 * np.max(np.linalg.norm(cam_centers - center, axis=-1))
    return {"translate": translate, "radius": radius}

def readColonSceneInfo(path):
    
    images_folder = os.path.join(path)
    campose_folder = os.path.join(path.replace("C3VD", "C3VD_endogslam_optimized")) 

    params_path = os.path.join(campose_folder, "params.npz") 

    found_iter = searchForMaxIteration(os.path.join(campose_folder, "point_cloud"))
    point_cloud_path = os.path.join(campose_folder, "point_cloud", f"iteration_{found_iter}", "point_cloud.ply")

    # path to save auxiliary ply file (slightly converted point_cloud.ply required by original gaussians)
    ply_path = os.path.join(campose_folder, "point_cloud", f"iteration_{found_iter}", "point_cloud_auxiliary.ply")

    
    # read camera parameters
    params = np.load(params_path) 
    color_paths = natsorted(glob.glob(f"{images_folder}/color/*.png"))
    depth_paths = natsorted(glob.glob(f"{images_folder}/depth/*.tiff"))

    
    train_cameras = []

    for time_idx in range(len(params["gt_w2c_all_frames"])):
        # Get the estimated rotation & translation
        curr_cam_rot = F.normalize(torch.tensor(params['cam_unnorm_rots'][..., time_idx]))
        curr_cam_tran = params['cam_trans'][..., time_idx]
        R = build_rotation(curr_cam_rot).transpose(1, 2).squeeze().cpu().numpy()
        T = curr_cam_tran

        k = params["intrinsics"]
        fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]

        image = Image.open(color_paths[time_idx])
        orig_w, orig_h = image.size
        depth = np.array(Image.open(depth_paths[time_idx]), dtype=np.float64)
        depth = depth / 2.55 # depth scale for C3VD
        depth =(depth * 255).astype(np.float32)
        depth = Image.fromarray(depth) 
    
        FovY = focal2fov(fy, orig_h)
        FovX = focal2fov(fx, orig_w)

        train_cameras.append(CameraInfo(
            uid=time_idx,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            depth=depth,
            width=orig_w,
            height=orig_h,
            image_path=color_paths[time_idx],
            image_name=f"{time_idx}",
            depth_path=depth_paths[time_idx],
            depth_name="",
            cx=cx,
            cy=cy,
        ))
    
    num_cameras = len(train_cameras)
    all_idx = set(range(num_cameras))
    eval_idx = set(range(7, num_cameras, 8)) # we don't expect eval in early frames. We dont do eval now, params npz only for train
        
    train_idx = all_idx - eval_idx
    eval_idx = sorted(list(eval_idx))
    train_idx = sorted(list(train_idx))
    train_cameras_final = [train_cameras[i] for i in train_idx]
    test_cameras_final = [train_cameras[i] for i in eval_idx]
    
    random.seed(42)
    random.shuffle(train_cameras_final)
    random.shuffle(test_cameras_final)

    # normalization
    nerf_normalization = getNerfppNorm(train_cameras_final)
    
    # point cloud
    init_pt_cld = PlyData.read(point_cloud_path)
    f_dc_0 = np.array(init_pt_cld['vertex'].data['f_dc_0'])
    f_dc_1 = np.array(init_pt_cld['vertex'].data['f_dc_1'])
    f_dc_2 = np.array(init_pt_cld['vertex'].data['f_dc_2'])
    shs = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=1)
    rgb_colors = SH2RGB(shs)
    rgb_colors = np.clip(rgb_colors, 0.0, 1.0)

    fixed_ratio = 0.5 #we can even set 1 and start from orange values. Should be ok
    rgb_colors[:,0] = rgb_colors[:,0]*(1-fixed_ratio) + PinkColor.R * fixed_ratio
    rgb_colors[:,1] = rgb_colors[:,1]*(1-fixed_ratio) + PinkColor.G * fixed_ratio
    rgb_colors[:,2] = rgb_colors[:,2]*(1-fixed_ratio) + PinkColor.B * fixed_ratio
    rgb_colors = np.clip(rgb_colors, 0.0, 1.0)

    x = np.array(init_pt_cld['vertex'].data['x'])
    y = np.array(init_pt_cld['vertex'].data['y'])
    z = np.array(init_pt_cld['vertex'].data['z'])
    xyz = np.stack([x, y, z], axis=1)

    # Limit to max N init points
    total_init_points = min(xyz.shape[0], MAX_N_POINTS)  
    indices = np.random.choice(xyz.shape[0], total_init_points, replace=False)
    sampled_xyz = xyz[indices]
    sampled_rgb_colors = rgb_colors[indices]

    pcd = BasicPointCloud(points=sampled_xyz, colors=sampled_rgb_colors, normals=np.zeros((len(sampled_xyz), 3)))
    storePly(ply_path, sampled_xyz, sampled_rgb_colors * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cameras_final,
                           test_cameras=test_cameras_final,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

########################################################################
#### RotateColon datareader - TODO clean duplicated code
from tqdm import tqdm
import OpenEXR
import copy
    

def readRotateSceneInfo(path):

    extension = ".png"

    train_cameras = []
    with open(os.path.join(path, "transforms.json")) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in tqdm(enumerate(frames), desc="Loading train cameras"):
            
            frame_zfill = frame['file_path'].zfill(4)
            cam_name = os.path.join(path, "train_views", f"{frame_zfill}{extension}")
            depth_name = os.path.join(path, "depth_train", f"{frame_zfill}.exr")

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Read IMG - lots of redundant code
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path).convert("RGBA")
            w, h = image.size[0], image.size[1]

            cx = w / 2
            cy = h / 2

        
            # Get FoVs
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            # Read depth
            with OpenEXR.File(depth_name) as infile:
                depth = infile.channels()["ViewLayer.Depth.Z"].pixels
                
            depth = depth / 1.4 #depth scale for ColonRotate
            depth =(depth * 255).astype(np.float32)
            depth = Image.fromarray(depth) 

            #create camera object - completely multiplied code, redundant in so many places...
            camera_train = CameraInfo(uid=idx, 
                                R=R, T=T, 
                                FovX=FovX, 
                                FovY=FovY, 
                                image=image, 
                                depth=depth,
                                image_path=image_path,
                                image_name=image_name,
                                depth_name=depth_name,
                                depth_path=depth_name,
                                width=w, height=h,
                                cx=cx, cy=cy)
            
            
            train_cameras.append(camera_train)

        

    test_cameras = []
    with open(os.path.join(path, "transforms_test.json")) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        # Load testing cameras
        for idx, frame in tqdm(enumerate(frames), desc="Loading test cameras"):
            frame_zfill = frame['file_path'].zfill(4)
            cam_name = os.path.join(path, "test_views_1", f"{frame_zfill}{extension}")
            
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]


            # Read IMG - lots of redundant code
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path).convert("RGBA")
            w, h = image.size[0]//1, image.size[1]//1
            cx = w / 2
            cy = h / 2

            
            # Get FoVs
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            #create camera object - completely multiplied code, redundant in so many places...
            placeholder_depth = copy.copy(depth)
            camera_test = CameraInfo(uid=idx, 
                                R=R, T=T, 
                                FovX=FovX, 
                                FovY=FovY, 
                                image=image, 
                                depth=placeholder_depth,
                                image_path=image_path,
                                image_name=image_name,
                                depth_name=None,
                                depth_path=None,
                                width=w, height=h,
                                cx=cx, cy=cy)
            
            
            
            test_cameras.append(camera_test)

    #normalize scene
    nerf_normalization = getNerfppNorm(train_cameras)
    
    # prepare init point cloud
    original_ply_path = os.path.join(path, "init_point_cloud.ply")
    auxiliary_ply_path = os.path.join(path, "init_point_cloud_auxiliary.ply")
    init_pt_cld = PlyData.read(original_ply_path)

    x = init_pt_cld['vertex'].data['x']
    y = init_pt_cld['vertex'].data['y']
    z = init_pt_cld['vertex'].data['z']
    x_array = np.array(x)
    y_array = np.array(y)
    z_array = np.array(z)
    xyz = np.stack([x_array, y_array, z_array], axis=1)
    rgb_colors = xyz.copy()


    r_fixed = 188/255.
    g_fixed = 99/255.
    b_fixed = 76/255.

    rgb_colors[:,0] = rgb_colors[:,0]*0 +r_fixed
    rgb_colors[:,1] = rgb_colors[:,1]*0 +g_fixed
    rgb_colors[:,2] = rgb_colors[:,2]*0 +b_fixed
    rgb_colors = np.clip(rgb_colors, 0.0, 1.0)

    
    pcd = BasicPointCloud(points=xyz, colors=rgb_colors, normals=np.zeros((len(xyz), 3)))
    storePly(auxiliary_ply_path, xyz, rgb_colors * 255)
    try:
        pcd = fetchPly(auxiliary_ply_path)
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                        train_cameras=train_cameras,
                        test_cameras=test_cameras,
                        nerf_normalization=nerf_normalization,
                        ply_path=auxiliary_ply_path)
            
    return scene_info
###########################################################

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo, 
    "RNN" : readRNNSceneInfo,
    "colon": readColonSceneInfo, #Dataset from endoGslam
    "rotate": readRotateSceneInfo, #Dataset RotateColon
}
