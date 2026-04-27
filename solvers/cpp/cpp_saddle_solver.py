import ctypes
import os
import numpy as np
import cv2

class SaddlePoint(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("S", ctypes.c_float)
    ]

class CPPSaddleSolver:
    def __init__(self):
        dll_name = "cpp_saddle_solver.dll" if os.name == "nt" else "cpp_saddle_solver.so"
        lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dll_name)
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Could not find C++ library at: {lib_path}. Please build it first using build_cpp_saddle_solver.bat.")
            
        self.lib = ctypes.CDLL(lib_path)
        
        self.lib.find_saddle_points_cpp.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), # h_img
            ctypes.POINTER(SaddlePoint),    # h_out_pts
            ctypes.c_int,                   # w
            ctypes.c_int,                   # h
            ctypes.c_bool                   # filter_t_corners
        ]
        self.lib.find_saddle_points_cpp.restype = ctypes.c_int
        self.lib.free_saddle_resources_cpp.argtypes = []

    def __del__(self):
        if hasattr(self, 'lib'):
            self.lib.free_saddle_resources_cpp()

_cpp_solver = None

def find_saddle_points(image: np.ndarray, max_pts: int = 0, filter_t_corners: bool = True) -> np.ndarray:
    """
    Finds saddle points (X-corners) in an image using a C++ CPU kernel.
    
    Args:
        image (np.ndarray): The input image array.
        max_pts (int): Maximum number of points to return. If 0, return all points.
        filter_t_corners (bool): If True, filters out T-corners on the edge of the board.
        
    Returns:
        np.ndarray: A 2D array of lattice points of shape (N, 2).
    """
    global _cpp_solver
    if _cpp_solver is None:
        _cpp_solver = CPPSaddleSolver()
        
    # Convert to grayscale if the image has 3 channels
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    # Ensure C-contiguous memory layout
    if not gray.flags['C_CONTIGUOUS']:
        gray = np.ascontiguousarray(gray)
        
    h, w = gray.shape
    
    # Allocate output buffer (maximum 10000 points expected)
    out_pts = (SaddlePoint * 10000)()
    img_ptr = gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    
    # Run C++ Kernel
    count = _cpp_solver.lib.find_saddle_points_cpp(img_ptr, out_pts, w, h, filter_t_corners)
    
    if count == 0:
        return np.empty((0, 2), dtype=np.float64)
        
    # Convert points back to numpy array
    # Preallocate an array directly from memory to be faster
    # but a loop is also fine for max ~200 points.
    pts_array = np.zeros((count, 3), dtype=np.float64)
    for i in range(count):
        pts_array[i, 0] = out_pts[i].x
        pts_array[i, 1] = out_pts[i].y
        pts_array[i, 2] = out_pts[i].S
        
    # Sort points by saddle strength for consistency with python/cuda versions
    sorted_indices = np.argsort(pts_array[:, 2])[::-1]
    sorted_pts = pts_array[sorted_indices]
    
    # Take only the top max_pts
    if max_pts > 0 and len(sorted_pts) > max_pts:
        sorted_pts = sorted_pts[:max_pts]

    # Return only x and y coordinates
    return sorted_pts[:, :2]
