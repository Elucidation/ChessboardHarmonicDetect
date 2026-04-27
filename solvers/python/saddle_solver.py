import cv2
import numpy as np

def _get_saddle(gray_img):
    img = gray_img
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    gxx = cv2.Sobel(gx, cv2.CV_32F, 1, 0)
    gyy = cv2.Sobel(gy, cv2.CV_32F, 0, 1)
    gxy = cv2.Sobel(gx, cv2.CV_32F, 0, 1)
    
    # Inverse everything so positive equals more likely.
    S = -gxx*gyy + gxy**2

    # Calculate subpixel offsets
    denom = (gxx*gyy - gxy*gxy)
    sub_s = np.divide(gy*gxy - gx*gyy, denom, out=np.zeros_like(denom), where=denom!=0)
    sub_t = np.divide(gx*gxy - gy*gxx, denom, out=np.zeros_like(denom), where=denom!=0)
    return S, sub_s, sub_t, gx, gy

def _fast_nonmax_sup(img, win=11):
    element = np.ones([win, win], np.uint8)
    img_dilate = cv2.dilate(img, element)
    peaks = cv2.compare(img, img_dilate, cv2.CMP_EQ)
    img[peaks == 0] = 0


def find_saddle_points(image: np.ndarray, max_pts: int = 0, filter_t_corners: bool = True) -> np.ndarray:
    """
    Finds saddle points (X-corners) in an image.
    
    Args:
        image (np.ndarray): The input image array.
        max_pts (int): Maximum number of points to return. If 0, return all points.
        filter_t_corners (bool): If True, filters out T-corners on the edge of the board.
        
    Returns:
        np.ndarray: A 2D array of lattice points of shape (N, 2).
    """
    # Convert to grayscale if the image has 3 channels
    if len(image.shape) == 3:
        # Check if the image was loaded with RGB format and convert it to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    winsize = 10
    
    gray = cv2.blur(gray, (3, 3)) # Blur it
    saddle, sub_s, sub_t, gx, gy = _get_saddle(gray)
    _fast_nonmax_sup(saddle)

    # Strip off low points
    saddle[saddle < 10000] = 0
    sub_idxs = np.nonzero(saddle)
    spts = np.argwhere(saddle).astype(np.float64)[:, [1, 0]] # Return in x,y order instead of row-col

    # Add on sub-pixel offsets
    subpixel_offset = np.array([sub_s[sub_idxs], sub_t[sub_idxs]]).transpose()
    spts = spts + subpixel_offset

    # Sort points by saddle strength before clipping
    saddle_strengths = saddle[sub_idxs]
    sorted_indices = np.argsort(saddle_strengths)[::-1]
    spts = spts[sorted_indices]
    
    # Filter points
    if len(spts) > 0 and filter_t_corners:
        # 1. Edge Clipping
        near_edge = np.logical_or(
            np.any(spts <= winsize, axis=1),
            np.any(spts[:, [1, 0]] >= np.array(gray.shape) - winsize - 1, axis=1)
        )
        
        # 2. Vectorized Symmetry Filter (Rose Plot Symmetry)
        h, w_img = gray.shape
        mag = np.sqrt(gx**2 + gy**2)
        ixs = np.round(spts[:, 0]).astype(int)
        iys = np.round(spts[:, 1]).astype(int)
        
        # Ensure window is within image bounds for safe indexing
        ixs_safe = np.clip(ixs, 5, w_img - 6)
        iys_safe = np.clip(iys, 5, h - 6)
        
        # Precompute relative offsets for 40 square boundary pixels (clockwise)
        dx = np.concatenate([np.arange(-5, 6), np.ones(9, dtype=int)*5, np.arange(5, -6, -1), np.ones(9, dtype=int)*-5])
        dy = np.concatenate([np.ones(11, dtype=int)*-5, np.arange(-4, 5), np.ones(11, dtype=int)*5, np.arange(4, -5, -1)])
        
        # Extract all boundary rings at once: Shape (N, 40)
        ring_mags = mag[iys_safe[:, None] + dy, ixs_safe[:, None] + dx]
        
        # Calculate magnitude symmetry score using 180-degree periodic correlation
        row_sums = np.sum(ring_mags, axis=1, keepdims=True)
        norm_mags = ring_mags / (row_sums + 1e-6)
        scores = np.sum(norm_mags * np.roll(norm_mags, 20, axis=1), axis=1)
        
        # 3. Extract Intensity Ring for Point Symmetry
        ring_intensities = gray[iys_safe[:, None] + dy, ixs_safe[:, None] + dx].astype(np.float32)
        
        # Calculate Intensity Symmetry (NCC on the ring)
        # Subtract mean of each ring
        ring_means = np.mean(ring_intensities, axis=1, keepdims=True)
        ring_centered = ring_intensities - ring_means
        
        # 180-degree correlation (periodic shift by 20 in a 40-pixel ring)
        ring_rot = np.roll(ring_centered, 20, axis=1)
        num = np.sum(ring_centered * ring_rot, axis=1)
        den = np.sqrt(np.sum(ring_centered**2, axis=1) * np.sum(ring_rot**2, axis=1))
        # ncc_scores will be near 1.0 for symmetric corners, near -1.0 for anti-symmetric
        ncc_scores = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
        
        # Combined filter: high magnitude symmetry AND high intensity symmetry
        # X-corners have 4 clear peaks (scores) and point symmetry (ncc_scores)
        # NOTE : Hardcoded values here with some assumptions about input image scale etc.
        valid_mask = ~near_edge & (scores >= 0.02) & (ncc_scores >= 0.2)
            
        spts = spts[valid_mask]

    # Take only the top max_pts
    if max_pts > 0 and len(spts) > max_pts:
        spts = spts[:max_pts]

    return spts
