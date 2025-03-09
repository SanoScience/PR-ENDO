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

