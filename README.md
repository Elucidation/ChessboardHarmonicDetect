# ChessboardHarmonicDetect
Detects chessboard poses from images by locating saddle points and using harmonic analysis. Here is the output from the example script showing it working.

![example output](outputs/output_plot_1.png)


I also made a [youtube video](https://youtu.be/ikdNyfMvQsA?si=wtFThdHmDZqxIK-M) explaining how the algorithm works:

[![ChessboardPoseEstimation_YouTube](https://img.youtube.com/vi/ikdNyfMvQsA/mqdefault.jpg)](https://youtu.be/ikdNyfMvQsA?si=wtFThdHmDZqxIK-M)

If you use this code, please cite it as below:

    Ansari, S. (2026). ChessboardHarmonicDetect (Version 1.0.0) [Computer software]. https://github.com/Elucidation/ChessboardHarmonicDetect


This currently uses only computer vision algorithms, no machine learning. It can be greatly improved with ML to refine the points passed in.

## Usage

Run the usage example to visualize the detection pipeline.

Interactive display:
```bash
python usage_example.py --input input_images/3.png
```

Save to file:
```bash
python usage_example.py --input input_images/3.png --output output_plot.png
```

## Files

- **`saddle_solver.py`**: Detects sub-pixel saddle points (X-corners).
- **`harmonic_solver.py`**: Estimates the 2D chessboard lattice and homography matrix from saddle points.
- **`usage_example.py`**: End-to-end usage example with matplotlib visualization.
- **`utils_visualize.py`**: Some functions to do the example matplotlib overlay.

## Important Functions

### `find_saddle_points(image: np.ndarray, max_pts: int = 0) -> np.ndarray`
- **Inputs**: 
  - `image` (np.ndarray): The input image array (RGB or Grayscale).
  - `max_pts` (int): Maximum number of top-scoring saddle points to return. Set to `0` to return all detected points.
- **Outputs**: 
  - `np.ndarray`: A 2D array of sub-pixel accurate `(x, y)` saddle points of shape `(N, 2)`.

### `estimate_chess_grid(lattice_points: np.ndarray) -> Tuple`
- **Inputs**: 
  - `lattice_points` (np.ndarray): The 2D array of saddle points returned by `find_saddle_points()`.
- **Outputs**: 
  - `chess_grid_points` (np.ndarray): Estimated ideal integer chess grid coordinates for each saddle point.
  - `basis_vectors` (np.ndarray): The `2x2` matrix of the estimated lattice basis vectors.
  - `debug_info` (dict): Density map and peak vectors for plotting displacements.

### `estimate_homography(lattice_points: np.ndarray, chess_grid_points: np.ndarray) -> np.ndarray`
- **Inputs**: 
  - `lattice_points` (np.ndarray): Actual saddle points in the image coordinate space.
  - `chess_grid_points` (np.ndarray): Idealized integer grid coordinates corresponding to the saddle points.
- **Outputs**: 
  - `np.ndarray`: A `3x3` homography matrix for warping between the image plane and the ideal chessboard grid, estimated via RANSAC.

## All Outputs

![output 0](outputs/output_plot_0.png)

![output 1](outputs/output_plot_1.png)

![output 2](outputs/output_plot_2.png)

![output 3](outputs/output_plot_3.png)

