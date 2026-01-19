import sys
import os

# Add the project root to python path to find 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# benchmark.py
import argparse
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
from src.engine.rbergomi import RBergomiEngine

def run_benchmark(custom_range_n=None):
    """
    Benchmarks the execution time of Cholesky vs Hybrid (FFT) methods
    for the rough Bergomi model simulation.

    Args:
        custom_n_values (int, optional): Numbers of N to test.
    """
    with open('../config.yaml', 'r') as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)
    T = data["model_parameters"]["T"]
    H = data["model_parameters"]["H"]
    eta = data["model_parameters"]["eta"]
    rho = data["model_parameters"]["rho"]
    xi_0 = data["model_parameters"]["xi"]
    S_0 = data["model_parameters"]["S0"]
    n_paths = data["simulation_settings"]["n_paths"]

    if custom_range_n and custom_range_n > 0:
        N_values = [500*k for k in range(1,custom_range_n+1)]
        print(f"Benchmarking N values: {N_values}")
    else:
        # Default values if no argument is provided
        N_values = [100, 250, 500, 1000, 1500, 2000, 2500]
        print(f"Benchmarking default N values: {N_values}")

    times_cholesky = []
    times_hybrid = []

    print(f"{'N':<10} | {'Cholesky (s)':<15} | {'Hybrid (s)':<15} | {'Speedup':<10}")
    print("-" * 60)

    for N in N_values:
        engine = RBergomiEngine(T, N, H, eta, rho, xi_0, S_0)
        
        start = time.time()
        engine.simulate_cholesky(n_paths)
        end = time.time()
        times_cholesky.append(end - start)

        start = time.time()
        engine.simulate_hybrid(n_paths)
        end = time.time()
        times_hybrid.append(end - start)
        
        t_chol = times_cholesky[-1] if times_cholesky[-1] is not None else float('nan')
        t_hyb = times_hybrid[-1]
        
        ratio = t_chol / t_hyb if (t_chol and t_hyb > 0) else 0
        
        print(f"{N:<10} | {t_chol:<15.4f} | {t_hyb:<15.4f} | x{ratio:.1f}")

    plot_results(N_values, times_cholesky, times_hybrid, n_paths)

def plot_results(N_values, times_cholesky, times_hybrid, n_paths):
    """
    Generates and SAVES comparison plots for the benchmark results.
    Saves to: ./out/benchmark/
    """
    output_dir = "../out/benchmark"
    os.makedirs(output_dir, exist_ok=True)

    valid_indices = [i for i, t in enumerate(times_cholesky) if t is not None]
    valid_N = [N_values[i] for i in valid_indices]
    valid_chol = [times_cholesky[i] for i in valid_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(valid_N, valid_chol, 'o-', label='Cholesky (Exact) - O(N^3)', color='red', linewidth=2)
    plt.plot(N_values, times_hybrid, 'o-', label='Hybrid (FFT) - O(N log N)', color='green', linewidth=2)
    
    plt.title(f"rBergomi Performance (n_paths={n_paths})")
    plt.xlabel("Number of time steps (N)")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.grid(True)
    
    save_path_linear = os.path.join(output_dir, "benchmark_linear.png")
    plt.savefig(save_path_linear)
    print(f"Graph saved to: {save_path_linear}")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.loglog(valid_N, valid_chol, 'o-', label='Cholesky (Slope ~3)', color='red')
    plt.loglog(N_values, times_hybrid, 'o-', label='Hybrid (Slope ~1)', color='green')
    
    plt.title("Complexity Analysis (Log-Log Scale)")
    plt.xlabel("log(N)")
    plt.ylabel("log(Time)")
    plt.legend()
    plt.grid(True, which="both", ls="-")
    
    save_path_log = os.path.join(output_dir, "benchmark_loglog.png")
    plt.savefig(save_path_log)
    print(f"Graph saved to: {save_path_log}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark rBergomi simulation methods.")
    parser.add_argument("custom_n_values", type=int, help="Numbers of N to test.")
    args = parser.parse_args()

    run_benchmark(args.custom_n_values)