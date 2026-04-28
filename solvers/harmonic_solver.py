import numpy as np
import cv2
from typing import Tuple, Optional, Dict, List, Any

def find_lattice_basis_vectors(
    points: np.ndarray,
    num_vectors: int = 8,
    displacements: Optional[np.ndarray] = None,
    bins: int = 100,
) -> Tuple[np.ndarray, dict]:
    """
    Finds the most prominent displacement vectors between points using a 2D histogram
    and peak detection, serving as candidates for lattice basis vectors.

    Args:
        points (np.ndarray): A 2D array of lattice points of shape (N, 2).
        num_vectors (int): The number of basis vectors to find (default: 8).
        displacements (Optional[np.ndarray]): Pre-computed displacements of shape (N, N, 2).
                                             If None, computed from points.
        bins (int): The number of bins to use for the 2D histogram (default: 100).
        
    Returns:
        Tuple[np.ndarray, dict]:
            - basis_vectors (np.ndarray): The estimated lattice basis vectors of shape (num_vectors, 2).
            - debug_info (dict): A dictionary containing debug information.
    """
    # Calculate all pairwise displacements between points
    if displacements is None:
        displacements = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    flat_dists = displacements.reshape((-1, 2))

    # Determine histogram range focusing on the core 60% of displacements to avoid outliers
    dx_sorted = np.sort(flat_dists[:, 0])
    dy_sorted = np.sort(flat_dists[:, 1])
    n = len(dx_sorted)
    i_lo, i_hi = int(0.20 * (n - 1)), int(0.80 * (n - 1))
    x_min, x_max = dx_sorted[i_lo], dx_sorted[i_hi]
    y_min, y_max = dy_sorted[i_lo], dy_sorted[i_hi]

    # Build 2D histogram using np.bincount
    x_scale = bins / (x_max - x_min)
    y_scale = bins / (y_max - y_min)
    xi = ((flat_dists[:, 0] - x_min) * x_scale).astype(np.intp)
    yi = ((flat_dists[:, 1] - y_min) * y_scale).astype(np.intp)
    mask = (xi >= 0) & (xi < bins) & (yi >= 0) & (yi < bins)
    flat_idx = xi[mask] * bins + yi[mask]
    hist = np.bincount(flat_idx, minlength=bins * bins).reshape(bins, bins)
    x_bin_width = (x_max - x_min) / bins
    y_bin_width = (y_max - y_min) / bins

    # Smooth the histogram to create a continuous density map
    sigma = 3
    ksize = 2 * int(4 * sigma + 0.5) + 1
    density_map = cv2.GaussianBlur(
        hist.T.astype(np.float32), (ksize, ksize), sigma
    )

    # Find local maxima
    min_distance = 5
    kernel_size = 2 * min_distance + 1
    dilated = cv2.dilate(
        density_map, np.ones((kernel_size, kernel_size), dtype=np.float32)
    )
    threshold = 0.01 * density_map.max()
    peak_mask = (density_map == dilated) & (density_map > threshold)
    coordinates = np.argwhere(peak_mask)

    # Convert peak bin coordinates back to continuous displacement vectors
    if len(coordinates) > 0:
        peak_vectors = np.column_stack([
            x_min + (coordinates[:, 1] + 0.5) * x_bin_width,
            y_min + (coordinates[:, 0] + 0.5) * y_bin_width,
        ])
    else:
        peak_vectors = np.empty((0, 2))
    peak_scores = np.array([density_map[c[0], c[1]] for c in coordinates])

    # NOTE : Returning debug info for visualization purposes, 
    # adding ~0.20ms overhead to this function, not needed for the actual solver.
    debug_info = {
        'density_map': density_map,
        'extent': [x_min, x_max, y_min, y_max],
        'peak_vectors': peak_vectors.copy(),
    }

    # Remove the peak at the origin (zero displacement) as it's not a valid basis vector
    if len(peak_vectors) > 0:
        origin_idx = np.argmin(np.linalg.norm(peak_vectors, axis=1))
        peak_vectors = np.delete(peak_vectors, origin_idx, axis=0)
        peak_scores = np.delete(peak_scores, origin_idx, axis=0)

    # Score vectors based on harmonic relationships and select the best ones
    harmonic_scores = _calculate_harmonic_scores(peak_vectors, peak_scores)
    final_vectors = _select_best_vectors(peak_vectors, harmonic_scores, num_vectors)
    
    return _pad_vectors(final_vectors, num_vectors), debug_info

def _calculate_harmonic_scores(
    peak_vectors: np.ndarray, peak_scores: np.ndarray
) -> np.ndarray:
    """
    Scores basis vector candidates by checking for harmonic relationships 
    (integer multiples of smaller vectors in the same direction).

    Args:
        peak_vectors (np.ndarray): Candidate basis vectors of shape (N, 2).
        peak_scores (np.ndarray): Scores for each candidate vector of shape (N,).
        
    Returns:
        np.ndarray: Harmonic scores for each candidate vector of shape (N,).
    """
    harmonic_scores = peak_scores.copy()
    num_peaks = len(peak_vectors)
    if num_peaks == 0:
        return harmonic_scores

    COS_SIM_THRESH = 0.985
    MAG_RATIO_TOL = 0.15

    # Compute pairwise magnitudes to check for integer multiples
    mags = np.linalg.norm(peak_vectors, axis=1)
    mags_i = mags[:, np.newaxis]
    mags_j = mags[np.newaxis, :]

    # Condition 1: Candidate harmonic vector must be significantly larger
    condition1 = mags_j > mags_i * 1.5

    # Condition 2: Vectors must point in almost the exact same direction (high cosine similarity)
    dot_products = peak_vectors @ peak_vectors.T
    mag_products = mags_i * mags_j
    cos_sim_matrix = dot_products / (mag_products + 1e-9)
    condition2 = cos_sim_matrix > COS_SIM_THRESH

    # Condition 3: The ratio of their magnitudes should be very close to an integer
    mag_ratio_matrix = mags_j / (mags_i + 1e-9)
    k = np.round(mag_ratio_matrix)
    condition3 = np.abs(mag_ratio_matrix - k) < MAG_RATIO_TOL

    # Combine masks to find valid harmonic pairs, ignoring self-comparisons
    identity = np.eye(num_peaks, dtype=bool)
    combined_mask = condition1 & condition2 & condition3 & ~identity

    # Distribute the score of the larger harmonic back to the fundamental vector
    peak_scores_j_broadcast = np.tile(peak_scores[np.newaxis, :], (num_peaks, 1))
    k_safe = np.where(k == 0, 1, k)
    contributions = np.where(combined_mask, peak_scores_j_broadcast / k_safe, 0)

    harmonic_scores += np.sum(contributions, axis=1)

    return harmonic_scores

def _select_best_vectors(
    peak_vectors: np.ndarray, harmonic_scores: np.ndarray, num_vectors: int
) -> List[np.ndarray]:
    """
    Selects the best non-redundant basis vectors based on harmonic scores,
    ensuring they are not collinear or overlapping.

    Args:
        peak_vectors (np.ndarray): Candidate basis vectors of shape (N, 2).
        harmonic_scores (np.ndarray): Harmonic scores for each candidate vector of shape (N,).
        num_vectors (int): The number of basis vectors to select (default: 8).
        
    Returns:
        List[np.ndarray]: Selected basis vectors of shape (num_vectors, 2).
    """
    magnitudes = np.linalg.norm(peak_vectors, axis=1)
    final_metric = harmonic_scores / (magnitudes + 1e-6)
    top_indices = np.argsort(final_metric)[::-1]

    final_vectors: List[np.ndarray] = []
    COS_SIM_THRESH = 0.985

    for idx in top_indices:
        if len(final_vectors) >= num_vectors:
            break

        candidate_vec = peak_vectors[idx]

        if candidate_vec[0] < -1e-6:
            candidate_vec = -candidate_vec
        elif abs(candidate_vec[0]) < 1e-6 and candidate_vec[1] < -1e-6:
            candidate_vec = -candidate_vec

        is_redundant = False
        for existing_vec in final_vectors:
            norm_product = np.linalg.norm(candidate_vec) * np.linalg.norm(existing_vec)
            if norm_product > 1e-6:
                cos_sim = np.dot(candidate_vec, existing_vec) / norm_product
                if abs(cos_sim) > COS_SIM_THRESH:
                    is_redundant = True
                    break

        if not is_redundant:
            final_vectors.append(candidate_vec)

    return final_vectors

def _pad_vectors(vectors: List[np.ndarray], target_count: int) -> np.ndarray:
    """Pads the list of vectors with zeros to reach the target count."""
    output_array = np.array(vectors)
    num_found = len(output_array)
    if num_found < target_count:
        padding = np.zeros((target_count - num_found, 2))
        if num_found == 0:
            return padding
        output_array = np.vstack([output_array, padding])
    return output_array

def reproject_points(
    points: np.ndarray, basis_vectors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects points onto the ideal integer lattice defined by the basis vectors
    and estimates the sub-pixel center offset.

    Args:
        points (np.ndarray): Input lattice points of shape (N, 2).
        basis_vectors (np.ndarray): Basis vectors of shape (2, 2).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - points_lattice (np.ndarray): Reprojected lattice points of shape (N, 2).
            - center (np.ndarray): Estimated center offset of shape (2,).
    """
    if np.linalg.det(basis_vectors) == 0:
        return points, np.mean(points, axis=0)

    # Transform points into the coordinate space defined by the basis vectors
    B_inv = np.linalg.inv(basis_vectors)
    points_lattice = points @ B_inv

    # Remove the mean offset to center the points around the origin temporarily
    points_lattice_offset = points_lattice.mean(axis=0)
    points_lattice -= points_lattice_offset

    # Use a circular mean to robustly estimate the fractional sub-grid offset
    fractional_parts = np.mod(points_lattice, 1)
    angles = 2 * np.pi * fractional_parts
    mean_angle = np.angle(np.mean(np.exp(1j * angles), axis=0))
    points_lattice_frac_offset = np.mod(mean_angle / (2 * np.pi), 1)
    points_lattice -= points_lattice_frac_offset

    # Snap points to nearest integers and remove any remaining global integer offset
    integer_indices = np.round(points_lattice).astype(int)
    integer_offset = np.round(np.mean(integer_indices, axis=0)).astype(int)
    points_lattice -= integer_offset
    integer_indices -= integer_offset

    # Calculate the ideal lattice points and use median difference to find the robust center
    lattice_part = integer_indices @ basis_vectors
    origin_estimates = points - lattice_part
    best_center = np.median(origin_estimates, axis=0)
    
    # Map the best center back into lattice space to finalize the zero-centered indices
    best_center_lattice = (
        (best_center @ B_inv)
        - points_lattice_offset
        - points_lattice_frac_offset
        - integer_offset
    )
    points_lattice -= best_center_lattice

    return points_lattice, best_center

def get_lattice_and_reproject(
    points: np.ndarray, num_vectors: int = 2, displacements: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Finds the 2D lattice basis vectors for the given points and reprojects 
    the points onto this ideal lattice.

    Args:
        points (np.ndarray): Input lattice points of shape (N, 2).
        num_vectors (int): The number of basis vectors to find (default: 8).
        displacements (Optional[np.ndarray]): Pre-computed displacements of shape (N, N, 2).
                                             If None, computed from points.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
            - basis_vectors (np.ndarray): The estimated lattice basis vectors of shape (2, 2).
            - projected_points (np.ndarray): Reprojected lattice points of shape (N, 2).
            - center (np.ndarray): Estimated center offset of shape (2,).
            - debug_info (dict): A dictionary containing debug information.
    """
    assert(len(points.shape) == 2)
    
    basis_vectors, debug_info = find_lattice_basis_vectors(
        points, num_vectors=num_vectors, displacements=displacements
    )

    angles = np.arctan2(basis_vectors[:, 1], basis_vectors[:, 0])
    basis_vectors = basis_vectors[np.argsort(angles)]

    projected_points, center = reproject_points(points, basis_vectors)

    return basis_vectors, projected_points, center, debug_info

def estimate_chess_grid(lattice_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Estimates the mapping of lattice points to the chess grid.
    
    Args:
        lattice_points (np.ndarray): A 2D array of lattice points of shape (N, 2).
        
    Returns:
        Tuple[np.ndarray, np.ndarray, dict]:
            - Estimated chess grid points of shape (N_grid, 2).
            - Basis vectors of shape (2, 2).
            - Debug dictionary containing density map and peaks.
    """
    if len(lattice_points) < 4:
        return np.zeros_like(lattice_points), np.zeros((2, 2)), {}
        
    basis_vectors, projected_points, _, debug_info = get_lattice_and_reproject(
        lattice_points, num_vectors=2
    )
    return np.round(projected_points), basis_vectors, debug_info

def estimate_homography(lattice_points: np.ndarray, chess_grid_points: np.ndarray) -> np.ndarray:
    """
    Estimates the homography matrix to warp the image to the chess grid.
    
    Args:
        lattice_points (np.ndarray): 2D array of lattice points of shape (N, 2).
        chess_grid_points (np.ndarray): 2D array of chess grid points of shape (N_grid, 2).
        
    Returns:
        np.ndarray: Homography matrix of shape (3, 3).
    """
    if len(lattice_points) < 4:
        return np.eye(3)
    H, _ = cv2.findHomography(chess_grid_points, lattice_points, cv2.RANSAC)
    if H is None:
        return np.eye(3)
    return H