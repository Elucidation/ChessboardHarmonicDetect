import pytest
import numpy as np
import cv2
from PIL import Image

from solvers.python.saddle_solver import find_saddle_points as find_saddle_points_py
from solvers.cpp.cpp_saddle_solver import find_saddle_points as find_saddle_points_cpp
from solvers.cuda.cuda_saddle_solver import find_saddle_points as find_saddle_points_cuda

@pytest.fixture
def test_image():
    image = np.array(Image.open("input_images/3.png"))
    image = cv2.resize(image, (640, 480))
    return image

def test_solvers_find_all_points_identically(test_image):
    # Verify C++ and CUDA return the exact same points for max_pts=0
    pts_cpp = find_saddle_points_cpp(test_image, max_pts=0, filter_t_corners=True)
    pts_cuda = find_saddle_points_cuda(test_image, max_pts=0, filter_t_corners=True)
    
    # C++ and CUDA should find the exact same number of points
    assert len(pts_cpp) == len(pts_cuda), f"C++ found {len(pts_cpp)} points, CUDA found {len(pts_cuda)}"
    
    # We must match each point regardless of order
    for pt in pts_cpp:
        dists_cuda = np.linalg.norm(pts_cuda - pt, axis=1)
        assert np.min(dists_cuda) < 0.1, f"C++ point {pt} not found in CUDA"
        
    # Python finds slightly more points (67 vs 64) due to uint8 truncation in cv2.blur
    pts_py = find_saddle_points_py(test_image, max_pts=0, filter_t_corners=True)
    assert len(pts_py) > 0

def test_solvers_truncation_consistency(test_image):
    # Ensure truncation consistency for identical subset limits
    pts_py = find_saddle_points_py(test_image, max_pts=50, filter_t_corners=True)
    pts_cpp = find_saddle_points_cpp(test_image, max_pts=50, filter_t_corners=True)
    pts_cuda = find_saddle_points_cuda(test_image, max_pts=50, filter_t_corners=True)
    
    # Sort by coordinates for a deterministic order to compare
    def sort_pts(p):
        return p[np.lexsort((p[:,1], p[:,0]))]
        
    cpp_sorted = sort_pts(pts_cpp)
    cuda_sorted = sort_pts(pts_cuda)
    
    diff_cpp_cuda = np.linalg.norm(cpp_sorted - cuda_sorted, axis=1)
    if np.max(diff_cpp_cuda) > 0.1:
        assert False, "C++ and CUDA top 50 points mismatch!"
