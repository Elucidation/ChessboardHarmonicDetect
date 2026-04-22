import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time

# Import our custom solvers
from saddle_solver import find_saddle_points
from harmonic_solver import estimate_chess_grid, estimate_homography
from utils_visualize import visualize_reconstruction

def load_and_plot_saddles(image_path: str, img_size: tuple = (640, 480), output_path: str = None):
    """
    Loads an image, gets saddle points, and plots the original and overlaid images side by side.
    
    Args:
        image_path (str): Path to the image file.
        img_size (tuple): Size of the image to resize to, saddle solver is tuned for default img_size.
    """
    # 1. Load the image
    image = np.array(Image.open(image_path))
    image = cv2.resize(image, img_size)
    
    # 2. Get saddle points (~15ms)
    # NOTE : This is much slower than it could be, ML-based would be faster, or C++/GPU implementation etc.
    t_start = time.time()
    points = find_saddle_points(image, max_pts=50, filter_t_corners=True)
    t_saddle = time.time() - t_start
    print(f"Saddle points found in {t_saddle*1000:.1f} ms.")
    
    # 3. Estimate chess grid points (1-2ms)
    t_start = time.time()
    chess_grid_points, basis_vectors, debug_info = estimate_chess_grid(points, return_debug=True)
    t_grid = time.time() - t_start
    print(f"Chess grid estimated in {t_grid*1000:.1f} ms.")
    
    # 4. Estimate homography
    H = estimate_homography(points, chess_grid_points)
    
    # 5. Visualize reconstruction
    visualize_reconstruction(image, points, chess_grid_points, H, basis_vectors, debug_info, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run chessboard harmonic detection and visualization.")
    parser.add_argument("-i", "--input", type=str, default="input_images/3.png", help="Path to the input image.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to save the output visualization. If not provided, displays on screen.")
    args = parser.parse_args()

    load_and_plot_saddles(args.input, output_path=args.output)
