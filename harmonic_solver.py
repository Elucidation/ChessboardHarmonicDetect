import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from typing import Tuple, Optional, Dict, List, Any

def find_lattice_basis_vectors(
    points: np.ndarray,
    num_vectors: int = 8,
    displacements: Optional[np.ndarray] = None,
    bins: int = 100,
    return_debug: bool = False,
) -> Any:
    if displacements is None:
        displacements = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    flat_dists = displacements.reshape((-1, 2))

    x_min, x_max = np.percentile(flat_dists[:, 0], [20, 80])
    y_min, y_max = np.percentile(flat_dists[:, 1], [20, 80])
    hist, x_edges, y_edges = np.histogram2d(
        flat_dists[:, 0],
        flat_dists[:, 1],
        bins=bins,
        range=[[x_min, x_max], [y_min, y_max]],
    )

    density_map = gaussian_filter(hist.T, sigma=3)

    coordinates = peak_local_max(density_map, min_distance=5, threshold_rel=0.01)
    x_bin_width = x_edges[1] - x_edges[0]
    y_bin_width = y_edges[1] - y_edges[0]
    peak_vectors = np.array(
        [
            [x_edges[c[1]] + x_bin_width / 2, y_edges[c[0]] + y_bin_width / 2]
            for c in coordinates
        ]
    )
    peak_scores = np.array([density_map[c[0], c[1]] for c in coordinates])

    debug_info = {}
    if return_debug:
        debug_info = {
            'density_map': density_map,
            'extent': [x_min, x_max, y_min, y_max],
            'peak_vectors': peak_vectors.copy(),
        }

    if len(peak_vectors) > 0:
        origin_idx = np.argmin(np.linalg.norm(peak_vectors, axis=1))
        peak_vectors = np.delete(peak_vectors, origin_idx, axis=0)
        peak_scores = np.delete(peak_scores, origin_idx, axis=0)

    harmonic_scores = _calculate_harmonic_scores(peak_vectors, peak_scores)
    final_vectors = _select_best_vectors(peak_vectors, harmonic_scores, num_vectors)
    
    if return_debug:
        return _pad_vectors(final_vectors, num_vectors), debug_info
    return _pad_vectors(final_vectors, num_vectors)

def _calculate_harmonic_scores(
    peak_vectors: np.ndarray, peak_scores: np.ndarray
) -> np.ndarray:
    harmonic_scores = peak_scores.copy()
    num_peaks = len(peak_vectors)
    if num_peaks == 0:
        return harmonic_scores

    COS_SIM_THRESH = 0.985
    MAG_RATIO_TOL = 0.15

    mags = np.linalg.norm(peak_vectors, axis=1)
    mags_i = mags[:, np.newaxis]
    mags_j = mags[np.newaxis, :]

    condition1 = mags_j > mags_i * 1.5

    dot_products = peak_vectors @ peak_vectors.T
    mag_products = mags_i * mags_j
    cos_sim_matrix = dot_products / (mag_products + 1e-9)
    condition2 = cos_sim_matrix > COS_SIM_THRESH

    mag_ratio_matrix = mags_j / (mags_i + 1e-9)
    k = np.round(mag_ratio_matrix)
    condition3 = np.abs(mag_ratio_matrix - k) < MAG_RATIO_TOL

    identity = np.eye(num_peaks, dtype=bool)
    combined_mask = condition1 & condition2 & condition3 & ~identity

    peak_scores_j_broadcast = np.tile(peak_scores[np.newaxis, :], (num_peaks, 1))
    
    k_safe = np.where(k == 0, 1, k)
    contributions = np.where(combined_mask, peak_scores_j_broadcast / k_safe, 0)

    harmonic_scores += np.sum(contributions, axis=1)

    return harmonic_scores

def _select_best_vectors(
    peak_vectors: np.ndarray, harmonic_scores: np.ndarray, num_vectors: int
) -> List[np.ndarray]:
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
    if np.linalg.det(basis_vectors) == 0:
        return points, np.mean(points, axis=0)

    B_inv = np.linalg.inv(basis_vectors)
    points_lattice = points @ B_inv

    points_lattice_offset = points_lattice.mean(axis=0)
    points_lattice -= points_lattice_offset

    fractional_parts = np.mod(points_lattice, 1)
    angles = 2 * np.pi * fractional_parts
    mean_angle = np.angle(np.mean(np.exp(1j * angles), axis=0))
    points_lattice_frac_offset = np.mod(mean_angle / (2 * np.pi), 1)
    points_lattice -= points_lattice_frac_offset

    integer_indices = np.round(points_lattice).astype(int)
    integer_offset = np.round(np.mean(integer_indices, axis=0)).astype(int)
    points_lattice -= integer_offset
    integer_indices -= integer_offset

    lattice_part = integer_indices @ basis_vectors
    origin_estimates = points - lattice_part
    best_center = np.median(origin_estimates, axis=0)
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
    return_debug: bool = False
) -> Any:
    assert(len(points.shape) == 2)
    
    if return_debug:
        basis_vectors, debug_info = find_lattice_basis_vectors(
            points, num_vectors=num_vectors, displacements=displacements, return_debug=True
        )
    else:
        basis_vectors = find_lattice_basis_vectors(
            points, num_vectors=num_vectors, displacements=displacements
        )

    angles = np.arctan2(basis_vectors[:, 1], basis_vectors[:, 0])
    basis_vectors = basis_vectors[np.argsort(angles)]

    projected_points, center = reproject_points(points, basis_vectors)

    if return_debug:
        return basis_vectors, projected_points, center, debug_info
    return basis_vectors, projected_points, center

def estimate_chess_grid(lattice_points: np.ndarray, return_debug: bool = False) -> Any:
    """
    Estimates the mapping of lattice points to the chess grid.
    
    Args:
        lattice_points (np.ndarray): A 2D array of lattice points of shape (N, 2).
        return_debug (bool): If True, returns additional debug information.
        
    Returns:
        If return_debug is False:
            Tuple[np.ndarray, np.ndarray]: 
                - Estimated chess grid points of shape (N_grid, 2).
                - Basis vectors of shape (2, 2).
        If return_debug is True:
            Tuple[np.ndarray, np.ndarray, dict]:
                - Estimated chess grid points of shape (N_grid, 2).
                - Basis vectors of shape (2, 2).
                - Debug dictionary containing density map and peaks.
    """
    if len(lattice_points) < 4:
        if return_debug:
            return np.zeros_like(lattice_points), np.zeros((2, 2)), {}
        return np.zeros_like(lattice_points), np.zeros((2, 2))
        
    if return_debug:
        basis_vectors, projected_points, _, debug_info = get_lattice_and_reproject(
            lattice_points, num_vectors=2, return_debug=True
        )
        return np.round(projected_points), basis_vectors, debug_info
    else:
        basis_vectors, projected_points, _ = get_lattice_and_reproject(
            lattice_points, num_vectors=2
        )
        return np.round(projected_points), basis_vectors

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