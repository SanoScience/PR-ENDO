import torch
import torch.nn.functional as F

# Convert RGB images to grayscale using standard weights for R, G, B
def rgb_to_grayscale(img):
        weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=img.dtype, device=img.device)
        grayscale_img = (img * weights.view(3, 1, 1)).sum(dim=0)
        return grayscale_img


def compare_reflection_area(im1: torch.Tensor, im2: torch.Tensor) -> float:
    """
    Compare the white areas of two RGB images represented as torch tensors and return the ratio of the areas.

    Parameters
    ----------
    im1 : torch.Tensor
        First image tensor (shape: [3, H, W]).
    im2 : torch.Tensor
        Second image tensor (shape: [3, H, W]).

    Returns
    -------
    float
        The ratio of the white area in im2 to the white area in im1.
    """
    
    
    im1_gray = rgb_to_grayscale(im1)
    im2_gray = rgb_to_grayscale(im2)
    
    # Binarize the grayscale images (assuming white areas are > 200)
    threshold = 220 / 255.0  # Normalize threshold to range [0, 1] for grayscale images
    binarized_im1 = (im1_gray > threshold).float()  # Binary tensor: 1 for white, 0 for black
    binarized_im2 = (im2_gray > threshold).float()  # Binary tensor: 1 for white, 0 for black

    # Perform morphological closing (dilation followed by erosion)
    kernel = torch.ones((2,2), dtype=binarized_im1.dtype, device=binarized_im1.device)
    dilated_im1 = F.conv2d(binarized_im1.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=5)
    closed_im1 = (dilated_im1 > 0).float()  # Threshold to create a binary mask

    dilated_im2 = F.conv2d(binarized_im2.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=5)
    closed_im2 = (dilated_im2 > 0).float()  # Threshold to create a binary mask

    # Perform morphological opening (erosion followed by dilation)
    eroded_im1 = F.conv2d(closed_im1, kernel.unsqueeze(0).unsqueeze(0), padding=5)
    opened_im1 = (eroded_im1 > 0).float()  # Threshold to create a binary mask

    eroded_im2 = F.conv2d(closed_im2, kernel.unsqueeze(0).unsqueeze(0), padding=5)
    opened_im2 = (eroded_im2 > 0).float()  # Threshold to create a binary mask

    # Calculate the area (number of white pixels)
    area_im1 = opened_im1.sum()
    area_im2 = opened_im2.sum()

    # Calculate the ratio of the areas
    area_ratio = area_im2 / (area_im1 + 1e-10)

    return area_ratio, opened_im1, opened_im2


def zero_value_penalty(value, max_penalty=10.0):
    
    # Compute a penalty that increases as A and B get closer
    penalty = torch.exp(-value) * max_penalty
    # Return the mean penalty
    return penalty.sum()
