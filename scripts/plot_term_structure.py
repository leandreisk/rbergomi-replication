import sys
import os
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.stats import norm

plt.style.use('seaborn-v0_8-whitegrid')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.engine.rbergomi import RBergomiEngine
from src.pricing.pricer import MonteCarloPricer

def delta_to_strike(delta_call, S, T, sigma):
    delta_call = np.clip(delta_call, 1e-6, 1.0 - 1e-6)
    
    d1 = norm.ppf(delta_call)
    
    sigma_sqrt_t = sigma * np.sqrt(T)
    log_moneyness = d1 * sigma_sqrt_t - 0.5 * sigma**2 * T
    
    K = S * np.exp(-log_moneyness)
    return K

def plot_volatility_term_structure():
    """
    Reproduces the Volatility Term Structure graph (McCrickerd & Pakkanen).
    Plots Smiles for maturities: 1 Week, 1 Month, 3 Months, 6 Months, 1 Year.
    """
    with open('./config.yaml', 'r') as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)

    T = data["model_parameters"]["T"]
    H = data["model_parameters"]["H"]
    eta = data["model_parameters"]["eta"]
    rho = data["model_parameters"]["rho"]
    xi_0 = data["model_parameters"]["xi"]
    S_0 = data["model_parameters"]["S0"]
    
    N = data["model_parameters"]["n_steps"]      
    n_paths = data["simulation_settings"]["n_paths"]
    seed = data["simulation_settings"]["seed"]

    np.random.seed(seed)

    cmap = plt.get_cmap('viridis') # 'plasma' 'coolwarm' 'magma' 'viridis'
    colors = [cmap(i) for i in [0.1, 0.25, 0.4, 0.6, 0.8, 0.95][::-1]]

    scenarios = [
        (1/365, colors[0], '1D (1 Day)',    (-0.05, 0.05)),
        (1/52,  colors[1], '1W (1 Week)',   (-0.1, 0.1)),
        (1/12,  colors[2], '1M (1 Month)',  (-0.2, 0.2)),
        (3/12,  colors[3], '3M (3 Months)', (-0.25, 0.25)),
        (6/12,  colors[4], '6M (6 Months)', (-0.3, 0.3)),
        (1.0,   colors[5], '1Y (1 Year)',   (-0.4, 0.4))
    ]

    plt.figure(figsize=(10, 7))
    print(f"Starting Term Structure Simulation (H={H}, rho={rho})...")

    for T, color, label, ran in scenarios:
        k_range = np.linspace(ran[0], ran[1], 50)
        deltas = np.linspace(0.95, 0.05, 19)
        t0 = time.time()

        strikes = []
        vol_approx = np.sqrt(xi_0)
        for d in deltas:
            k = delta_to_strike(d, S_0, T, vol_approx)
            strikes.append(k)
        strikes = np.array(strikes)
        k_range = np.log(strikes / S_0)
        
        engine = RBergomiEngine(T, N, H, eta, rho, xi_0, S_0)
        pricer = MonteCarloPricer(engine)
        k_out = S_0 * np.exp(k_range)
        
        try:
            prices, iv_out = pricer.compute_smile(k_range, n_paths)
            
            valid_mask = ~np.isnan(iv_out) & (iv_out > 1e-6)

            x = k_range[valid_mask]
            y = iv_out[valid_mask]

            if len(x) > 3:
                x_smooth = np.linspace(x.min(), x.max(), 200)
                
                spline = make_interp_spline(x, y, k=3)
                y_smooth = spline(x_smooth)
                
                plt.plot(x_smooth, y_smooth, color=color, linewidth=2, label=label)
                
                plt.plot(x, y, 'o', color=color, markersize=4, alpha=0.4)
            
            else:
                plt.plot(x, y, 'o-', color=color, label=label)
            
            dt = time.time() - t0
            print(f"  -> Finished {label} in {dt:.2f}s")
            
        except Exception as e:
            print(f"  -> Error on {label}: {e}")

    plt.title(f"rBergomi Volatility Term Structure\n"
              f"$\\rho={rho}, H={H}, \\xi_0={xi_0}, \\eta={eta}$", fontsize=14)
    plt.xlabel(r"Log-Moneyness $k = log(K/S_0)$", fontsize=12)
    plt.ylabel(r"Implied Volatility $\sigma_{BS}(k, T)$", fontsize=12)
    plt.legend(title="Maturity")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    
    output_dir = os.path.join(project_root, "out", "smiles")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"term_structure_H{H}_rho{rho}_eta{eta}_N{N}.png"
    save_path = os.path.join(output_dir, filename)
    
    plt.savefig(save_path)
    print(f"\nGraph saved to: {save_path}")

if __name__ == "__main__":
    plot_volatility_term_structure()