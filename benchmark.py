import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import random
from tqdm import tqdm

# Import our custom solvers
from solvers.cuda.cuda_saddle_solver import find_saddle_points as find_saddle_points_cuda
from solvers.cpp.cpp_saddle_solver import find_saddle_points as find_saddle_points_cpp
from solvers.python.saddle_solver import find_saddle_points as find_saddle_points_py
from solvers.harmonic_solver import estimate_chess_grid, estimate_homography
from utils_visualize import visualize_reconstruction

def load_and_plot_saddles(image_path: str, img_size: tuple = (640, 480), output_path: str = None, 
                          solver_choice: str = 'all', num_trials: int = 1):
    """
    Loads an image, gets saddle points, and plots the original and overlaid images side by side.
    
    Args:
        image_path (str): Path to the image file.
        img_size (tuple): Size of the image to resize to, saddle solver is tuned for default img_size.
        output_path (str): Path to save the output visualization. If not provided, saves in outputs/<solver>/.
        solver_choice (str): Which solver to run.
        num_trials (int): Number of trials to run for each solver.
    """
    # 1. Load the image
    image = np.array(Image.open(image_path))
    image = cv2.resize(image, img_size)
    
    solvers_to_run = []
    if solver_choice in ['all', 'cuda']: solvers_to_run.append(('CUDA', find_saddle_points_cuda)) # ~1ms
    if solver_choice in ['all', 'cpp']: solvers_to_run.append(('C++', find_saddle_points_cpp)) # ~8ms
    if solver_choice in ['all', 'python']: solvers_to_run.append(('Python', find_saddle_points_py)) # ~20ms

    # solvers_to_run.reverse() # Test ordering

    if 'CUDA' in [s[0] for s in solvers_to_run]:
        # Warm up call to initialize CUDA context and allocate memory (~90ms One time cost)
        t_start = time.time()
        _ = find_saddle_points_cuda(np.random.randint(0, 255, size=(img_size[1], img_size[0])).astype(np.uint8), max_pts=50, filter_t_corners=True)
        t_warmup = time.time() - t_start
        print(f"CUDA init took {t_warmup*1000:.1f} ms. NOTE: One-time initialization cost.")

    if 'C++' in [s[0] for s in solvers_to_run]:
        # Warm up call to initialize OpenMP and allocate buffers (~10-20ms One time cost)
        t_start = time.time()
        _ = find_saddle_points_cpp(np.random.randint(0, 255, size=(img_size[1], img_size[0])).astype(np.uint8), max_pts=50, filter_t_corners=True)
        t_warmup = time.time() - t_start
        print(f"C++ init took {t_warmup*1000:.1f} ms. NOTE: One-time initialization cost.")

    # Warm up the harmonic solver (SciPy/NumPy overhead)
    t_start = time.time()
    dummy_points = np.random.rand(50, 2).astype(np.float64)
    _ = estimate_chess_grid(dummy_points)
    t_warmup = time.time() - t_start
    print(f"Harmonic init took {t_warmup*1000:.1f} ms. NOTE: One-time initialization cost.")

    
    stats = {name: {'saddle': [], 'harmonic': []} for name, _ in solvers_to_run}
    # Store one set of results per solver for final visualization
    visualization_data = {}
    
    trial_iterator = range(num_trials)
    if num_trials > 1:
        trial_iterator = tqdm(trial_iterator, desc="Benchmarking", unit="trial")

    for trial in trial_iterator:
        # Shuffle the order per trial to avoid bias from CPU caching/frequency scaling
        if num_trials > 1:
            random.shuffle(solvers_to_run)
            
        for name, solver_func in solvers_to_run:
            if num_trials == 1:
                print(f"--- Running {name} Solver ---")
            
            t_start = time.time()
            points = solver_func(image, max_pts=50, filter_t_corners=True)
            t_saddle_ms = (time.time() - t_start) * 1000
            
            # 3. Estimate chess grid points (1-2ms)
            t_start = time.time()
            chess_grid_points, basis_vectors, debug_info = estimate_chess_grid(points)
            t_grid_ms = (time.time() - t_start) * 1000
            
            stats[name]['saddle'].append(t_saddle_ms)
            stats[name]['harmonic'].append(t_grid_ms)
            
            if num_trials == 1:
                print(f"{name} solver took {t_saddle_ms:.1f} ms + harmonic ({t_grid_ms:.1f} ms) = {t_saddle_ms + t_grid_ms:.1f} ms")
            
            # 4. Estimate homography
            H = estimate_homography(points, chess_grid_points)
            
            # Store data for the final visualization (from the first time this solver runs)
            if name not in visualization_data:
                visualization_data[name] = {
                    'points': points,
                    'chess_grid_points': chess_grid_points,
                    'H': H,
                    'basis_vectors': basis_vectors,
                    'debug_info': debug_info
                }

    # 6. Save visualizations with average timings
    print("\n--- Saving Visualizations ---")
    for name, data in visualization_data.items():
        avg_saddle = np.mean(stats[name]['saddle'])
        avg_harmonic = np.mean(stats[name]['harmonic'])
        
        current_out = output_path
        if current_out is None:
            folder_name = name.replace('+', 'p').lower()
            out_dir = os.path.join("outputs", folder_name)
            os.makedirs(out_dir, exist_ok=True)
            input_stem = os.path.splitext(os.path.basename(image_path))[0]
            current_out = os.path.join(out_dir, f"out_{input_stem}.png")
            
        timing_str = f"Avg Timing ({num_trials} trials): {name} saddle ({avg_saddle:.2f} ms) + harmonic ({avg_harmonic:.2f} ms) = {avg_saddle + avg_harmonic:.2f} ms"
        visualize_reconstruction(image, data['points'], data['chess_grid_points'], data['H'], data['basis_vectors'], data['debug_info'], current_out, timing_str)

    # 7. Print summary stats
    if num_trials > 1:
        print("\n" + "="*40)
        print(f"BENCHMARK SUMMARY ({num_trials} trials, randomized order)")
        print("="*40)
        for name in stats:
            avg_saddle = np.mean(stats[name]['saddle'])
            avg_harmonic = np.mean(stats[name]['harmonic'])
            std_saddle = np.std(stats[name]['saddle'])
            print(f"{name:8}: Saddle {avg_saddle:6.2f} ms (±{std_saddle:4.2f}) | Harmonic {avg_harmonic:5.2f} ms | Total {avg_saddle + avg_harmonic:6.2f} ms")
        print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run chessboard harmonic detection and visualization.")
    parser.add_argument("-i", "--input", type=str, default="input_images/3.png", help="Path to the input image.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to save the output visualization. If not provided, saves in outputs/<solver>/.")
    parser.add_argument("--solver", type=str, choices=['python', 'cpp', 'cuda', 'all'], default='all', help="Which solver to run.")
    parser.add_argument("-n", "--trials", type=int, default=1, help="Number of trials for benchmarking.")
    args = parser.parse_args()

    load_and_plot_saddles(args.input, output_path=args.output, solver_choice=args.solver, num_trials=args.trials)
