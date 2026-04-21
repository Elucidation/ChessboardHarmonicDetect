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

def _clip_bounding_points(pts, img_shape, winsize=10):
    # Points are given in x,y coords, not r,c of the image shape
    a = ~np.any(np.logical_or(pts <= winsize, pts[:, [1, 0]] >= np.array(img_shape) - winsize - 1), axis=1)
    return pts[a, :]

def find_saddle_points(image: np.ndarray, max_pts: int = 0) -> np.ndarray:
    """
    Function to load an image and generate saddle/lattice points.
    
    Args:
        image (np.ndarray): The input image array.
        max_pts (int): Maximum number of points to return. If 0, return all points.
        
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
    
    # Remove those points near winsize edges
    spts = _clip_bounding_points(spts, gray.shape, winsize)

    # Take only the top max_pts
    if max_pts > 0 and len(spts) > max_pts:
        spts = spts[:max_pts]

    return spts
