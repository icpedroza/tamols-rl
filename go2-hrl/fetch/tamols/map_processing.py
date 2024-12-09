from scipy import ndimage
import numpy as np

def get_derivative_kernel():
    """Get 1D 5-point central finite difference kernel for first derivative"""
    # 1st order derivative kernel from equation (24)
    return np.array([1, -8, 0, 8, -1]) / (12 * 0.04)

def compute_gradients(height_map):
    """
    Compute gradients using 1D 5-point central finite difference kernel
    Args:
        height_map: 2D height map array
    Returns:
        gradient_x: Gradient in x direction
        gradient_y: Gradient in y direction
    """
    kernel = get_derivative_kernel()
    
    # Compute first derivatives
    gradient_x = ndimage.convolve1d(height_map, kernel, axis=1)
    gradient_y = ndimage.convolve1d(height_map, kernel, axis=0)
    
    return gradient_x, gradient_y

def process_height_maps(h_raw, sigma1=1.0, sigma2=2.0):
    """
    Process raw height map to generate h_s1 and h_s2 layers as described.
    
    Args:
        h_raw: Raw height map array
        sigma1: Standard deviation for first Gaussian filter
        sigma2: Standard deviation for final Gaussian filter (sigma2 > sigma1)
    
    Returns:
        h_s1: Gaussian filtered map for gradient computation
        h_s2: Virtual floor map
    """
    h_s1 = ndimage.gaussian_filter(h_raw, sigma=sigma1)
    
    h_median = ndimage.median_filter(h_raw, size=3)
    
    delta_h = h_raw - h_median
    
    mask = np.ones_like(h_raw)
    mask[delta_h > 0] = 1  # stepping stone
    mask[delta_h < 0] = 1  # gap
    mask[delta_h == 0] = np.inf
    
    binary_mask = (mask == 1)
    dilated_binary = ndimage.binary_dilation(binary_mask)
    
    h_dilated = np.copy(h_raw)
    for i in range(h_raw.shape[0]):
        for j in range(h_raw.shape[1]):
            if dilated_binary[i, j]:
                i_start = max(0, i-1)
                i_end = min(h_raw.shape[0], i+2)
                j_start = max(0, j-1)
                j_end = min(h_raw.shape[1], j+2)
                neighborhood = h_raw[i_start:i_end, j_start:j_end]
                h_dilated[i, j] = np.max(neighborhood)
    
    h_s2 = ndimage.gaussian_filter(h_dilated, sigma=sigma2)

    gradients = {
        'h': compute_gradients(h_raw),
        'h_s1': compute_gradients(h_s1),
        'h_s2': compute_gradients(h_s2)
    }
    
    return h_s1, h_s2, gradients