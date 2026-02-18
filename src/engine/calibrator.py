import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.engine.rbergomi import RBergomiEngine
from src.pricing.pricer import MonteCarloPricer

class RBergomiCalibrator:
    def __init__(self, engine: RBergomiEngine, pricer: MonteCarloPricer, data, n_paths=1000):
        """
        Args:
            engine: rBergomiEngine Instance.
            pricer: MonteCarloPricer Instance.
            data (pd.DataFrame): Dataset with 'market_strike', 'ivs', 'option_type'.
            n_paths (int): Number of path for the pricing (turbo: 1000 is enough).
        """
        self.engine = engine
        self.pricer = pricer
        self.n_paths = n_paths
        
        self.market_strikes = np.array(data['market_strike'].values, dtype=float)
        self.market_ivs = np.array(data['ivs'].values, dtype=float)
        
        raw_types = data['option_type'].astype(str).str.upper().values
        self.is_call = (raw_types == 'C') | (raw_types == 'CALL')
        self.is_put = (raw_types == 'P') | (raw_types == 'PUT')
        
        if len(self.market_strikes) != len(self.market_ivs):
            raise ValueError("Numbers of strikes and vol inconsistent.")
            
        print(f"Calibrator initialized with {len(self.market_strikes)} options "
              f"({np.sum(self.is_call)} Calls, {np.sum(self.is_put)} Puts).")

    def objective_function(self, params, seed=42):
        """
        Loss function to optimize (RMSE).
        """
        rho, eta = params
        
        self.engine.rho = rho
        self.engine.eta = eta
        model_ivs = np.zeros_like(self.market_ivs)
        
        try:
            if np.any(self.is_call):
                strikes_calls = self.market_strikes[self.is_call]
                np.random.seed(seed) 
                _, _, ivs_c = self.pricer.price_european_option_turbo(
                    K=strikes_calls,
                    n_paths=self.n_paths,
                    option_type='call',
                    return_iv=True
                )
                
                if ivs_c is None or np.isnan(ivs_c).any():
                    return 1e6
                    
                model_ivs[self.is_call] = ivs_c

            if np.any(self.is_put):
                strikes_puts = self.market_strikes[self.is_put]
                np.random.seed(seed) 
                _, _, ivs_p = self.pricer.price_european_option_turbo(
                    K=strikes_puts,
                    n_paths=self.n_paths,
                    option_type='put',
                    return_iv=True
                )
                
                if ivs_p is None or np.isnan(ivs_p).any():
                    return 1e6
                
                model_ivs[self.is_put] = ivs_p

            diff = model_ivs - self.market_ivs
            rmse = np.sqrt(np.mean(diff**2))

            print(f" >> Step: rho={rho:+.4f} | eta={eta:.4f} | RMSE={rmse:.5f}")
            
            return rmse

        except Exception as e:
            print(f"Warning: Pricing failed for params {params}: {e}")
            return 1e6

    def calibrate(self, initial_guess=[-0.7, 1.9]):
        """
        Starting optimization with L-BFGS-B.
        """
        bounds = [(-0.99, 0.99), (0.1, 4.0)]
        
        print(f"Starting calibration (n_paths={self.n_paths})...")
        
        result = minimize(
            self.objective_function,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-5, 'disp': True}
        )
        
        if result.success:
            print(f"Calibration done : rho={result.x[0]:.4f}, eta={result.x[1]:.4f}, RMSE={result.fun:.5f}")
            self.engine.rho = result.x[0]
            self.engine.eta = result.x[1]
            return result.x
        else:
            print(f"Calibration failed: {result.message}")
            return None