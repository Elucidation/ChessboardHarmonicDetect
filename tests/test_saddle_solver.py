import sys
import os
import pytest
import numpy as np
import cv2

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saddle_solver import find_saddle_points as python_find_saddle_points
from cuda_saddle_solver import find_saddle_points as cuda_find_saddle_points

def test_saddle_solvers_match():
    # Load test image
    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'input_images', '3.png')
    if not os.path.exists(img_path):
        pytest.skip(f"Test image not found at {img_path}")
        
    image = cv2.imread(img_path)
    if image is None:
        pytest.skip(f"Failed to load image at {img_path}")
        
    image = cv2.resize(image, (640, 480))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    max_pts = 50
    
    # Run python solver
    py_pts = python_find_saddle_points(image, max_pts=max_pts, filter_t_corners=True)
    
    # Run cuda solver
    cu_pts = cuda_find_saddle_points(image, max_pts=max_pts, filter_t_corners=True)
    
    print(f"Python returned {len(py_pts)} points")
    print(f"CUDA returned {len(cu_pts)} points")
    
    # Assert counts match or are very close
    assert abs(len(py_pts) - len(cu_pts)) <= 5, f"Point count mismatch: Py={len(py_pts)}, Cu={len(cu_pts)}"
    
    if len(py_pts) == 0:
        return
        
    # Match points via nearest neighbor (order might slightly differ for tie-breaks)
    py_matched = np.zeros(len(py_pts), dtype=bool)
    cu_matched = np.zeros(len(cu_pts), dtype=bool)
    
    tolerance = 2.0 # Allow 2 pixels tolerance for sub-pixel precision and float differences
    
    for i, p_cu in enumerate(cu_pts):
        dists = np.linalg.norm(py_pts - p_cu, axis=1)
        min_idx = np.argmin(dists)
        if dists[min_idx] <= tolerance and not py_matched[min_idx]:
            py_matched[min_idx] = True
            cu_matched[i] = True
            
    # Calculate match rate
    match_rate_cu = np.sum(cu_matched) / len(cu_pts) if len(cu_pts) > 0 else 0
    match_rate_py = np.sum(py_matched) / len(py_pts) if len(py_pts) > 0 else 0
    
    print(f"CUDA match rate: {match_rate_cu*100:.1f}%")
    print(f"Python match rate: {match_rate_py*100:.1f}%")
    
    assert match_rate_cu >= 0.85, f"Only {match_rate_cu*100:.1f}% of CUDA points matched Python points"
    assert match_rate_py >= 0.85, f"Only {match_rate_py*100:.1f}% of Python points matched CUDA points"
