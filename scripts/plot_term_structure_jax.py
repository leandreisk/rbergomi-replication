import sys
import os
from scipy.interpolate import make_interp_spline
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

# Setup JAX
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" 
jax.config.update("jax_enable_x64", True)

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.engine.rbergomi_jax import RBergomiJAXEngine
from src.pricing.pricer_jax import MonteCarloPricerJAX

plt.style.use('seaborn-v0_8-whitegrid')

def delta_to_strike(delta_call, S, T, sigma):
    """
    Convertit un Delta de Call en Strike K.
    Delta_Call = N(d1)
    Si delta -> 1, K -> 0 (Deep ITM)
    Si delta -> 0, K -> inf (Deep OTM)
    """
    delta_call = jnp.clip(delta_call, 1e-6, 1.0 - 1e-6)
    
    d1 = norm.ppf(delta_call)
    
    sigma_sqrt_t = sigma * jnp.sqrt(T)
    log_moneyness = d1 * sigma_sqrt_t - 0.5 * sigma**2 * T
    
    K = S * jnp.exp(-log_moneyness)
    return K

def plot_volatility_term_structure():
    config_path = os.path.join(project_root, 'config.yaml')
    try:
        with open(config_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.SafeLoader)
        
        H = data["model_parameters"]["H"]
        eta = data["model_parameters"]["eta"]
        rho = data["model_parameters"]["rho"]
        xi_0 = data["model_parameters"]["xi"]
        S_0 = data["model_parameters"]["S0"]
        
        N = data["model_parameters"]["n_steps"]
        n_paths_total = data["simulation_settings"]["n_paths"]
        seed = data["simulation_settings"]["seed"]
        
    except FileNotFoundError:
        print("Warning: config.yaml not found. Using default parameters.")
        xi_0 = 0.235**2 
        eta = 1.9
        H = 0.07
        rho = -0.9
        S_0 = 100.0

        N = 312 
        n_paths_total = 400_000 
    
    BATCH_SIZE = 50_000
    
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in [0.1, 0.25, 0.4, 0.6, 0.8, 0.95][::-1]]
    
    scenarios = [
        (1/365, colors[0], '1D (1 Day)',    (-0.05, 0.05)),
        (1/52,  colors[1], '1W (1 Week)',   (-0.1, 0.1)),
        (1/12,  colors[2], '1M (1 Month)',  (-0.2, 0.2)),
        (3/12,  colors[3], '3M (3 Months)', (-0.25, 0.25)),
        (6/12,  colors[4], '6M (6 Months)', (-0.3, 0.3)),
        (1.0,   colors[5], '1Y (1 Year)',   (-0.4, 0.4))
    ]

    deltas = jnp.linspace(0.95, 0.05, 19)
    
    key = jax.random.PRNGKey(seed)
    
    plt.figure(figsize=(12, 7))
    
    plt.figure(figsize=(10, 7))
    print(f"--- Term Structure Simulation ---")
    print(f"Device: {jax.devices()[0]}")
    print(f"Paths: {n_paths_total} (Batch Size limit: 50k)")

    key = jax.random.PRNGKey(42)

    for T, color, label, ran in scenarios:
        key, subkey = jax.random.split(key)
        
        strikes = []
        vol_approx = np.sqrt(xi_0)
        for d in deltas:
            k = delta_to_strike(d, S_0, T, vol_approx)
            strikes.append(k)
            
        strikes_jax = jnp.array(strikes)
        k_range_jax = jnp.log(strikes_jax / S_0)
        
        t0 = time.time()
        
        engine = RBergomiJAXEngine(T, N, H, eta, rho, xi_0, S_0)
        
        pricer = MonteCarloPricerJAX(engine, batch_size=BATCH_SIZE)
        
        try:
            strikes_out, iv_out = pricer.compute_smile(subkey, k_range_jax, n_paths_total)
            
            iv_out.block_until_ready()
            
            iv_np = np.array(iv_out)
            k_np = np.array(k_range_jax)
            
            valid_mask = ~np.isnan(iv_np) & (iv_np > 1e-6)
            valid_mask = ~np.isnan(iv_np) & (iv_np > 1e-6)
            x = k_np[valid_mask]
            y = iv_np[valid_mask]

            if len(x) > 3:
                x_smooth = np.linspace(x.min(), x.max(), 200)
                
                spline = make_interp_spline(x, y, k=3)
                y_smooth = spline(x_smooth)
                
                plt.plot(x_smooth, y_smooth, color=color, linewidth=2, label=label)
                
                plt.plot(x, y, 'o', color=color, markersize=4, alpha=0.4)
            
            else:
                plt.plot(x, y, 'o-', color=color, label=label)
            
            dt = time.time() - t0
            print(f"  -> {label}: Done in {dt:.2f}s")
            
        except Exception as e:
            print(f"  -> Error on {label}: {e}")

    plt.title(f"rBergomi Volatility Term Structure \n"
              f"$\\rho={rho}, H={H}, \\xi_0={xi_0}, \\eta={eta}$", fontsize=14)
    plt.xlabel(r"Log-Moneyness $k = \log(K/S_0)$", fontsize=12)
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