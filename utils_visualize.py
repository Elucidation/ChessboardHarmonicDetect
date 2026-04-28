# Some functions for matplotlib visualization for the usage_example overlay.
import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_checkerboard_on_image(image, homography, alpha_blend=0.4) -> np.ndarray:
    """
    Draws a warped checkerboard pattern on an image using a homography.
    The checkerboard is defined as a centered 8x8 grid in "lattice space".
    """
    if homography is None:
        return image
    
    rows, cols = 8, 8
    # Notice the order: y, x for meshgrid to match the vertex ordering
    y, x = np.meshgrid(np.arange(-4, -4+rows+1), np.arange(-4, -4+cols+1))
    object_points = np.stack((x.ravel(), y.ravel()), axis=-1).astype(np.float32)
    
    warped_points = cv2.perspectiveTransform(object_points.reshape(-1, 1, 2), homography)

    overlay = image.copy()
    
    if warped_points is not None:
        warped_points = warped_points.reshape(-1, 2).astype(np.int32)
        verts_per_row = cols + 1
        for r in range(rows):
            for c in range(cols):
                p1_idx = r * verts_per_row + c
                p2_idx = p1_idx + 1
                p3_idx = (r + 1) * verts_per_row + c
                p4_idx = p3_idx + 1
                corners = np.array([warped_points[p1_idx], warped_points[p2_idx], warped_points[p4_idx], warped_points[p3_idx]])
                
                # Image is in RGB, so (0, 255, 0) is green, (255, 0, 0) is red
                color = (0, 255, 0) if (r + c) % 2 == 0 else (255, 0, 0)
                cv2.fillPoly(overlay, [corners], color)

    final_img = cv2.addWeighted(overlay, alpha_blend, image, 1 - alpha_blend, 0)
    return final_img

def visualize_reconstruction(image: np.ndarray, lattice_points: np.ndarray, 
                             chess_grid_points: np.ndarray, homography_matrix: np.ndarray,
                             basis_vectors: np.ndarray, debug_info: dict = None,
                             output_path: str = None, timing_str: str = ""):
    """
    Visualizes the reconstructed grid on the original image.
    
    Args:
        image (np.ndarray): Original input image.
        lattice_points (np.ndarray): 2D array of lattice points of shape (N, 2).
        chess_grid_points (np.ndarray): 2D array of chess grid points of shape (N_grid, 2).
        homography_matrix (np.ndarray): Estimated homography matrix of shape (3, 3).
        basis_vectors (np.ndarray): Estimated basis vectors of the lattice of shape (2, 2).
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    if timing_str:
        fig.suptitle(timing_str, fontsize=16, fontweight='bold')
    axes = axes.ravel()
    
    # 1. Original Image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 2. Saddle Points
    axes[1].imshow(image)
    if lattice_points is not None and len(lattice_points) > 0:
        axes[1].scatter(lattice_points[:, 0], lattice_points[:, 1], c='red', marker='x', s=40, label='Saddle Points')
        
        if chess_grid_points is not None:
            # Find which points successfully map to the 8x8 board inliers
            grid_mask = (chess_grid_points[:, 0] >= -4) & (chess_grid_points[:, 0] <= 4) & \
                        (chess_grid_points[:, 1] >= -4) & (chess_grid_points[:, 1] <= 4)
            
            if homography_matrix is not None:
                warped_pts = cv2.perspectiveTransform(chess_grid_points.reshape(-1, 1, 2).astype(np.float32), homography_matrix).reshape(-1, 2)
                dists = np.linalg.norm(warped_pts - lattice_points, axis=1)
                inliers = dists < 10.0 # Pixel distance threshold
                valid_mask = grid_mask & inliers
            else:
                valid_mask = grid_mask
                
            valid_pts = lattice_points[valid_mask]
            valid_grid = chess_grid_points[valid_mask]
            
            axes[1].scatter(valid_pts[:, 0], valid_pts[:, 1], facecolors='none', edgecolors='green', s=80, label='Grid Match')
            
            for pt, grid_pt in zip(valid_pts, valid_grid):
                axes[1].text(pt[0] + 5, pt[1] + 5, f"{int(grid_pt[0])},{int(grid_pt[1])}", color='green', fontsize=8, weight='bold')
                
        axes[1].legend(loc='upper right')
    
    axes[1].set_title("Saddle Points")
    axes[1].axis('off')
    
    # 3. Pairwise Displacements
    axes[2].set_title("Pairwise Displacements")
    if lattice_points is not None and len(lattice_points) > 0:
        displacements = lattice_points[:, np.newaxis, :] - lattice_points[np.newaxis, :, :]
        flat_displacements = displacements.reshape(-1, 2)
        axes[2].scatter(flat_displacements[:, 0], flat_displacements[:, 1], c='red', marker='.', s=2)
        
        if basis_vectors is not None and len(basis_vectors) >= 2:
            axes[2].quiver(
                [0, 0], [0, 0],
                basis_vectors[:, 0], basis_vectors[:, 1],
                angles='xy', scale_units='xy', scale=1,
                color=['cyan', 'magenta'], width=0.01
            )
            
    # Based on the image size, setting scale properly.
    displacement_scale = max(image.shape[0], image.shape[1]) // 2
    axes[2].set_xlim(-displacement_scale, displacement_scale)
    axes[2].set_ylim(-displacement_scale, displacement_scale)
    axes[2].set_aspect('equal')
    axes[2].set_xlabel('Dx (px)')
    axes[2].set_ylabel('Dy (px)')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].invert_yaxis()
    
    # 4. 2D Histogram & Peaks
    axes[3].set_title("2D Histogram & Peaks")
    if debug_info is not None and 'density_map' in debug_info:
        density_map = debug_info['density_map']
        extent = debug_info['extent']
        peak_vecs = debug_info['peak_vectors']
        
        axes[3].imshow(density_map, origin='lower', extent=extent, cmap='viridis')
        if len(peak_vecs) > 0:
            axes[3].scatter(peak_vecs[:, 0], peak_vecs[:, 1], c='red', marker='x', s=40)
            
    axes[3].set_aspect('equal')
    axes[3].set_xlabel('Dx (px)')
    axes[3].set_ylabel('Dy (px)')
    axes[3].invert_yaxis()

    # 5. Chessboard Overlay
    img_overlay = draw_checkerboard_on_image(image, homography_matrix, alpha_blend=0.4)
    axes[4].imshow(img_overlay)
    axes[4].set_title("Reconstructed Chessboard")
    axes[4].axis('off')
    
    # 6. Overhead Warped View
    axes[5].set_title("Overhead Warped View")
    if homography_matrix is not None:
        try:
            H_inv = np.linalg.inv(homography_matrix)
            
            # Map grid space [-4, 4] to a 500x500 image with 50px margins
            dest_size = 500
            margin = 50
            scale = (dest_size - 2 * margin) / 8.0
            center = dest_size / 2.0
            
            grid_to_dest = np.array([
                [scale, 0.0, center],
                [0.0, scale, center],
                [0.0, 0.0, 1.0]
            ])
            
            H_img_to_dest = grid_to_dest @ H_inv
            
            warped_img = cv2.warpPerspective(image, H_img_to_dest, (dest_size, dest_size))
            axes[5].imshow(warped_img)
            
            # Draw grid lines for the ideal 8x8 tiling
            for i in range(-4, 5):
                pos = scale * i + center
                axes[5].axvline(x=pos, color='cyan', linestyle='--', alpha=0.7)
                axes[5].axhline(y=pos, color='cyan', linestyle='--', alpha=0.7)
                
            axes[5].set_xlim(0, dest_size)
            axes[5].set_ylim(dest_size, 0) # Invert y to match image coords
            axes[5].axis('off')
        except np.linalg.LinAlgError:
            axes[5].axis('off')
    else:
        axes[5].axis('off')
        
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
