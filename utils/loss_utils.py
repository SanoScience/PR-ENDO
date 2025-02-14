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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def compute_geometric_loss(gaussian_normals, original_normals, closest_point_indices):
    """    
    Compute the geometric loss between gaussian normals and original normals.

    :param 
        gaussian_normals: Tensor of shape (N, 6) representing gaussian normals.
        original_normals: Tensor of shape (M, 6) representing original normals.
        closest_point_indices: Tensor of shape (N, 1) representing the indices of the closest points in the original point cloud.

    :return: The computed L1 loss.
    """
    closest_original_normals = original_normals[closest_point_indices, 3:]
    cosine_sim = (gaussian_normals[:, 3:] * closest_original_normals).sum(dim=1)
    loss = 1 - cosine_sim.abs()
    return loss.mean()


def attenuation_loss(light_gauss_distance, attenuation, num_pairs=10000):
    """
    Penalize a set of random pairs based on the condition:
    If light_gauss_distance[i] > light_gauss_distance[j] but attenuation[i] >= attenuation[j],
    apply a squared error penalty.

    Args:
        light_gauss_distance (torch.Tensor): Tensor of shape [N, 1] containing Gaussian distances.
        attenuation (torch.Tensor): Tensor of shape [N, 1] containing attenuation values.
        num_pairs (int): The number of random pairs to sample and penalize.

    Returns:
        penalty_sum (torch.Tensor): The total penalty calculated as squared error for the violating pairs.
    """
    N = light_gauss_distance.size(0)

    # Randomly select num_pairs distinct pairs of indices
    idx1 = torch.randint(0, N, (num_pairs,))  # Random indices for the first element of each pair
    idx2 = torch.randint(0, N, (num_pairs,))  # Random indices for the second element of each pair

    # Extract the corresponding camera distances and attenuation values
    distance1 = light_gauss_distance[idx1]
    distance2 = light_gauss_distance[idx2]
    attenuation1 = attenuation[idx1]
    attenuation2 = attenuation[idx2]

    # Check the condition: light_gauss_distance[i] > light_gauss_distance[j]
    # and attenuation[i] <= attenuation[j]
    penalty_mask = (distance1 > distance2) & (attenuation1 >= attenuation2)

    # Calculate the squared error for these pairs
    squared_error = (attenuation1 - attenuation2) ** 2

    # Apply the penalty only for pairs where the condition is met
    penalties = penalty_mask.float() * squared_error

    # Sum all penalties to get a total penalty score
    penalty_sum = penalties.sum()

    return penalty_sum