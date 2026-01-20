import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from src.engine.rbergomi import RBergomiEngine

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Computes the Black-Scholes European option price.
    Vectorized for Strikes (K) and Maturities (T).

    Args:
        S (float): Spot price.
        K (np.array): Strike prices.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate.
        sigma (float): Volatility (decimal, e.g., 0.2 for 20%).
        option_type (str): 'call' or 'put'.

    Returns:
        np.array: Option prices.
    """
    K = np.atleast_1d(K)

    if T <= 1e-6:
        if option_type == "call":
            price =  np.maximum(0.0, S - K)
        else:
            price =  np.maximum(0.0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price =  S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price =  K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    if len(price) == 1:
        return price[0]
    return price

def implied_volatility(price, S, K, T, r, option_type='call'):
    """
    Computes Implied Volatility from a market price using numerical inversion.
    Uses Brent's method for stability.

    Args:
        price (float or np.array): The option price (e.g., from Monte Carlo).
        S (float): Spot price.
        K (float or np.array): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        option_type (str): 'call' or 'put'.

    Returns:
        float or np.array: The implied volatility (sigma).
    """
    prices = np.atleast_1d(price)
    strikes = np.atleast_1d(K)
    
    if len(prices) != len(strikes) and len(strikes) > 1:
        raise ValueError("Price and Strike arrays must have the same length.")

    n = len(prices)
    ivs = np.zeros(n)

    def _solve_single_iv(p, k):
        intrinsic = max(0.0, S - k) if option_type == 'call' else max(0.0, k - S)
        if p < intrinsic or p < 1e-6:
            return np.nan

        def objective_fn(sigma):
            return black_scholes_price(S, k, T, r, sigma, option_type) - p
        
        try:
            return brentq(objective_fn, 1e-4, 5.0)
        except ValueError:
            return np.nan 

    for i in range(n):
        k_val = strikes[i] if len(strikes) > 1 else strikes[0]
        p_val = prices[i]  if len(prices) > 1  else prices[0]
        ivs[i] = _solve_single_iv(p_val, k_val)

    if np.ndim(price) == 0 and np.ndim(K) == 0:
        return ivs[0]
    return ivs


class MonteCarloPricer:
    """
    Wrapper class to price European options using a simulation engine.
    """

    def __init__(self, engine: RBergomiEngine):
        """
        Args:
            engine (RBergomiEngine): An initialized rBergomi simulation engine.
        """
        self.engine = engine

    def price_european_option(self, K, n_paths, option_type='call', return_iv=False):
        """
        Prices European options for a range of strikes.

        Args:
            K (np.array): Array of strike prices.
            n_paths (int): Number of Monte Carlo paths to generate.
            option_type (str): 'call' or 'put'.
            return_iv (bool): If True, computes and returns Implied Volatilities.

        Returns:
            tuple: (price, std_error, implied_vol or None) Contains 'price', 'std_error', and optionally 'implied_vol'.
        """
        K = np.atleast_1d(K)
        o_type = 1 if option_type=='call' else -1
        v, S = self.engine.simulate_hybrid(n_paths)
        S_T = S[-1]
        
        payoffs = np.maximum(o_type*(S_T[:, np.newaxis] - K), 0)
        
        price = np.mean(payoffs, axis=0)
        
        std_err = np.std(payoffs, axis=0) / np.sqrt(n_paths)
        
        if return_iv :
            iv = implied_volatility(price, self.engine.S_0, K, self.engine.T, 0, option_type=option_type)
            return price, std_err, iv
        else :
            return price, std_err, None 


    def compute_smile(self, k_log_moneyness, n_paths):
        """
        Generates the Volatility Smile for specific log-moneyness levels.
        This is useful to reproduce the graphs from the McCrickerd paper.

        Args:
            k_log_moneyness (np.array): Log-moneyness levels k = log(K/S0).
            n_paths (int): Number of paths.

        Returns:
            tuple: (prices, Implied_Vols)
        """
        K_values = self.engine.S_0 * np.exp(k_log_moneyness)
        
        mask_puts = K_values < self.engine.S_0
        mask_calls = ~mask_puts
        
        ivs = np.zeros_like(K_values)
        prices = np.zeros_like(K_values)
        
        #  OTM PUTS
        if np.any(mask_puts):
            K_puts = K_values[mask_puts]
            prices_put, _, iv_puts = self.price_european_option(
                K_puts, 
                n_paths, 
                option_type='put',
                return_iv=True
            )
            ivs[mask_puts] = iv_puts
            prices[mask_puts] = prices_put

        #  OTM CALLS
        if np.any(mask_calls):
            K_calls = K_values[mask_calls]
            prices_call, _, iv_calls = self.price_european_option(
                K_calls, 
                n_paths, 
                option_type='call',
                return_iv=True
            )
            ivs[mask_calls] = iv_calls
            prices[mask_calls] = prices_call
            
        return prices, ivs