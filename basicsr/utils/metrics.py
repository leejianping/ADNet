"""
Utility functions for calculating metrics and saving images.
"""
import math
import numpy as np
import torch
import cv2
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
    Args:
        img1 (torch.Tensor): First image [batch_size, channels, H, W].
        img2 (torch.Tensor): Second image [batch_size, channels, H, W].
        max_val (float): Maximum pixel value (default: 1.0 for normalized images).
    Returns:
        float: PSNR value.
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse.item()))

def calculate_ssim(img1, img2):
    """
    Calculate SSIM (Structural Similarity Index) between two images.
    Args:
        img1 (torch.Tensor): First image [batch_size, channels, H, W].
        img2 (torch.Tensor): Second image [batch_size, channels, H, W].
    Returns:
        float: SSIM value.
    """
    img1_np = img1.squeeze().cpu().numpy()
    img2_np = img2.squeeze().cpu().numpy()
    return ssim(img1_np, img2_np, data_range=1.0)

def imwrite(img, path):
    """
    Save image to disk.
    Args:
        img (numpy.ndarray): Image array [H, W] or [H, W, C].
        path (str): Output file path.
    """
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(-1)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(path, img)

def to_numpy(tensor):
    """
    Convert torch tensor to numpy array.
    Args:
        tensor (torch.Tensor): Input tensor.
    Returns:
        numpy.ndarray: Converted array.
    """
    return tensor.detach().cpu().numpy()