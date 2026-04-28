import numpy as np
import cv2
from PIL import Image
import os

# Import the core components
from solvers.python.saddle_solver import find_saddle_points
from solvers.harmonic_solver import estimate_chess_grid, estimate_homography
from utils_visualize import visualize_reconstruction

def run_simple_detection(image_path: str, output_path: str = "output_simple.png"):
    """
    A simple example of the chessboard detection pipeline using the Python solver.
    """
    print(f"Loading image: {image_path}")
    # 1. Load and resize image
    # The saddle solver is tuned for images around 640x480
    image = np.array(Image.open(image_path))
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
    
    print("Finding saddle points...")
    # 2. Detect saddle points (X-corners)
    # max_pts=50 limits to the strongest candidates
    points = find_saddle_points(image, max_pts=50, filter_t_corners=True)
    
    print(f"Found {len(points)} candidate points. Estimating chess grid...")
    # 3. Use harmonic analysis to find the underlying grid structure
    chess_grid_points, basis_vectors, debug_info = estimate_chess_grid(points)
    
    # 4. Estimate the homography mapping the image to the ideal grid
    H = estimate_homography(points, chess_grid_points)
    
    print(f"Saving visualization to: {output_path}")
    # 5. Generate and save the final visualization
    visualize_reconstruction(
        image, 
        points, 
        chess_grid_points, 
        H, 
        basis_vectors, 
        debug_info, 
        output_path, 
        timing_str="Simple Python Solver Example"
    )

if __name__ == "__main__":
    # Ensure the input image exists
    input_img = "input_images/3.png"
    if not os.path.exists(input_img):
        print(f"Error: {input_img} not found. Please run this from the project root.")
    else:
        run_simple_detection(input_img)
