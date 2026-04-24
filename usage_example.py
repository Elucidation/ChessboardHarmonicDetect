import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time

# Import our custom solvers
from solvers.cuda.cuda_saddle_solver import find_saddle_points as find_saddle_points_cuda
from solvers.cpp.cpp_saddle_solver import find_saddle_points as find_saddle_points_cpp
from solvers.python.saddle_solver import find_saddle_points as find_saddle_points_py
from harmonic_solver import estimate_chess_grid, estimate_homography
from utils_visualize import visualize_reconstruction

def load_and_plot_saddles(image_path: str, img_size: tuple = (640, 480), output_path: str = None, solver_choice: str = 'all'):
    """
    Loads an image, gets saddle points, and plots the original and overlaid images side by side.
    
    Args:
        image_path (str): Path to the image file.
        img_size (tuple): Size of the image to resize to, saddle solver is tuned for default img_size.
    """
    # 1. Load the image
    image = np.array(Image.open(image_path))
    image = cv2.resize(image, img_size)
    
    solvers_to_run = []
    if solver_choice in ['all', 'cuda']: solvers_to_run.append(('CUDA', find_saddle_points_cuda)) # ~1ms
    if solver_choice in ['all', 'cpp']: solvers_to_run.append(('C++', find_saddle_points_cpp)) # ~8ms
    if solver_choice in ['all', 'python']: solvers_to_run.append(('Python', find_saddle_points_py)) # ~20ms
    
    if 'CUDA' in [s[0] for s in solvers_to_run]:
        # Warm up call to initialize CUDA context and allocate memory (~90ms One time cost)
        t_start = time.time()
        _ = find_saddle_points_cuda(np.zeros_like(image), max_pts=50, filter_t_corners=True)
        t_warmup = time.time() - t_start
        print(f"CUDA init took {t_warmup*1000:.1f} ms. NOTE: One-time initialization cost.")

    import os
    
    for name, solver_func in solvers_to_run:
        print(f"--- Running {name} Solver ---")
        
        t_start = time.time()
        points = solver_func(image, max_pts=50, filter_t_corners=True)
        t_saddle_ms = (time.time() - t_start) * 1000
        print(f"{name} solver took {t_saddle_ms:.1f} ms.")
        
        # 3. Estimate chess grid points (1-2ms)
        t_start = time.time()
        chess_grid_points, basis_vectors, debug_info = estimate_chess_grid(points)
        t_grid_ms = (time.time() - t_start) * 1000
        print(f"Chess grid estimated in {t_grid_ms:.1f} ms.")
        
        # 4. Estimate homography
        H = estimate_homography(points, chess_grid_points)
        
        # Setup output path
        current_out = output_path
        if current_out is None:
            # Create a folder for each solver's output if not provided
            folder_name = name.replace('+', 'p').lower()
            out_dir = os.path.join("outputs", folder_name)
            os.makedirs(out_dir, exist_ok=True)
            current_out = os.path.join(out_dir, "out.png")
            
        timing_str = f"Timing: {name} saddle took ({t_saddle_ms:.1f} ms) + harmonic ({t_grid_ms:.1f} ms) = {t_saddle_ms + t_grid_ms:.1f} ms"
        
        # 5. Visualize reconstruction
        visualize_reconstruction(image, points, chess_grid_points, H, basis_vectors, debug_info, current_out, timing_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run chessboard harmonic detection and visualization.")
    parser.add_argument("-i", "--input", type=str, default="input_images/3.png", help="Path to the input image.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to save the output visualization. If not provided, saves in outputs/<solver>/.")
    parser.add_argument("--solver", type=str, choices=['python', 'cpp', 'cuda', 'all'], default='all', help="Which solver to run.")
    args = parser.parse_args()

    load_and_plot_saddles(args.input, output_path=args.output, solver_choice=args.solver)
