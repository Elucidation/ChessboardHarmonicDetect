import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Import our custom solvers
from saddle_solver import find_saddle_points
from harmonic_solver import estimate_chess_grid, estimate_homography

def visualize_reconstruction(image: np.ndarray, lattice_points: np.ndarray, 
                             chess_grid_points: np.ndarray, homography_matrix: np.ndarray):
    """
    Function to visualize the reconstructed grid on the original image.
    
    Args:
        image (np.ndarray): Original input image.
        lattice_points (np.ndarray): 2D array of lattice points of shape (N, 2).
        chess_grid_points (np.ndarray): 2D array of chess grid points of shape (N_grid, 2).
        homography_matrix (np.ndarray): Estimated homography matrix of shape (3, 3).
    """
    # TODO: Implement visualization logic
    pass

def load_and_plot_saddles(image_path: str, img_size: tuple = (320, 240)):
    """
    Loads an image, gets saddle points, and plots the original and overlaid images side by side.
    
    Args:
        image_path (str): Path to the image file.
        img_size (tuple): Size of the image to resize to, saddle solver is tuned for 320x240.
    """
    # 1. Load the image
    image = np.array(Image.open(image_path))
    image = cv2.resize(image, img_size)
    
    # 2. Get saddle points
    points = find_saddle_points(image, max_pts=100)
    
    # 3. Create matplotlib plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original on the left
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Plot original with points on the right
    ax2.imshow(image)
    if points is not None and len(points) > 0:
        ax2.scatter(points[:, 0], points[:, 1], c='red', marker='x', s=40, label='Saddle Points')
        ax2.legend()
    ax2.set_title("Image with Saddle Points")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test the function with an example image from the repo
    # You can comment this out or change it to test other images
    example_image_path = "input_images/0.jpg"
    # example_image_path = "input_images/1.png"
    load_and_plot_saddles(example_image_path)
