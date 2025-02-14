import torch
from scipy.spatial.transform import Rotation as R

def rotate_matrix(matrix, angle = [-5,5,-5]):
    # Convert matrix to numpy for processing
    matrix_np = matrix.cpu().numpy()

    # Create rotation object (30 degrees = pi/6 radians)
    rotation = R.from_euler('xyz', angle, degrees=True)

    # Apply the rotation to the matrix
    rotated_matrix_np = rotation.apply(matrix_np)

    # Convert back to torch tensor
    rotated_matrix = torch.tensor(rotated_matrix_np).to(matrix.device)
    
    return rotated_matrix

import torch

def rotate_matrix_torch(matrix, angle = [-5,5,-5]):
    raise NotImplemented
    #actually this function breaks grad flow
    # angles should be a tuple (theta_x, theta_y, theta_z) in radians
    theta_x, theta_y, theta_z = angle

    # Rotation matrix around x-axis (pitch)
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(theta_x), -torch.sin(theta_x)],
        [0, torch.sin(theta_x), torch.cos(theta_x)]
    ], device = matrix.device)

    # Rotation matrix around y-axis (yaw)
    Ry = torch.tensor([
        [torch.cos(theta_y), 0, torch.sin(theta_y)],
        [0, 1, 0],
        [-torch.sin(theta_y), 0, torch.cos(theta_y)]
    ], device = matrix.device)

    # Rotation matrix around z-axis (roll)
    Rz = torch.tensor([
        [torch.cos(theta_z), -torch.sin(theta_z), 0],
        [torch.sin(theta_z), torch.cos(theta_z), 0],
        [0, 0, 1]
    ], device = matrix.device)

    # Combined rotation matrix: Rz * Ry * Rx
    rotation_matrix = Rz @ Ry @ Rx
    return rotation_matrix #@ matrix

