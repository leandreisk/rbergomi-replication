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

def bs_price_vectorized(vol_squared, spot, strike, option_type='call'):
    """
    Formule de Black-Scholes vectorisée.
    vol_squared: Variance totale (sigma^2 * T) ou variance intégrée résiduelle.
    """
    # Protection contre division par zéro et valeurs négatives
    vol_squared = np.maximum(vol_squared, 1e-12)
    sqrt_v = np.sqrt(vol_squared)
    
    # Diffusion correcte des dimensions pour numpy (strikes en colonnes, chemins en lignes)
    # Si spot est (N,), il devient (N, 1) pour matcher strike (1, K)
    if spot.ndim == 1:
        spot = spot[:, np.newaxis]
    if np.ndim(strike) == 1:
        strike = strike[np.newaxis, :]
    if vol_squared.ndim == 1:
        vol_squared = vol_squared[:, np.newaxis]
        sqrt_v = sqrt_v[:, np.newaxis]

    d1 = (np.log(spot / strike) + 0.5 * vol_squared) / sqrt_v
    d2 = d1 - sqrt_v
    
    if option_type == 'call':
        price = spot * norm.cdf(d1) - strike * norm.cdf(d2)
    else:
        price = strike * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
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

    def price_european_option(self, K, n_paths, option_type='call', return_iv=False, S=None):
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
        if S is None :
            _, S = self.engine.simulate_hybrid(n_paths)
        S_T = S[-1]
        
        payoffs = np.maximum(o_type*(S_T[:, np.newaxis] - K), 0)
        
        price = np.exp(-self.engine.r*self.engine.T)*np.mean(payoffs, axis=0)
        
        std_err = np.std(payoffs, axis=0) / np.sqrt(n_paths)
        
        if return_iv :
            iv = implied_volatility(price, self.engine.S_0, K, self.engine.T, self.engine.r, option_type=option_type)
            return price, std_err, iv
        else :
            return price, std_err, None 

    def price_european_option_turbo(self, K, n_paths, option_type='call', return_iv=False, increments=None):
        K = np.atleast_1d(K)
        T = self.engine.T
        dt = self.engine.dt
        rho = self.engine.rho

        if increments is None :
            dW_vol, _ = self.engine._generate_hybrid_increments(n_paths)
            W_tilde = self.engine._fft_convolution(dW_vol)
        else : 
            dW_vol, W_tilde = increments[0], increments[1]
        t_vec = self.engine.times[:, None] # Shape (N+1, 1)
        
        V = self.engine.xi_0 * np.exp(self.engine.eta * W_tilde - 0.5 * self.engine.eta**2 * t_vec**(2 * self.engine.H))
        
        V_start = V[:-1, :] 
        
        IV = np.sum(V_start, axis=0) * dt
        
        stoch_int_W1 = np.sum(np.sqrt(V_start) * dW_vol, axis=0)
        
        log_S1 = self.engine.r * T -0.5 * (rho**2) * IV + rho * stoch_int_W1
        S1_T = self.engine.S_0 * np.exp(log_S1)
        
        residual_var = (1 - rho**2) * IV
        X = bs_price_vectorized(residual_var, S1_T, K, option_type)
        
        Q_hat = np.max(IV)
        
        control_var = rho**2 * (Q_hat - IV)
        Y = bs_price_vectorized(control_var, S1_T, K, option_type)
        
        E_Y = bs_price_vectorized(np.full_like(IV, rho**2 * Q_hat), self.engine.S_0 * np.exp(self.engine.r * T), K, option_type)
        
        mean_X = np.mean(X, axis=0)
        mean_Y = np.mean(Y, axis=0)
        
        cov_XY = np.mean((X - mean_X) * (Y - mean_Y), axis=0)
        var_Y = np.var(Y, axis=0)
        
        alpha = np.zeros_like(mean_X)
        mask = var_Y > 1e-12
        alpha[mask] = -cov_XY[mask] / var_Y[mask]
        
        controlled_samples = X + alpha * (Y - E_Y)
        price = np.exp(-self.engine.r*self.engine.T)*np.mean(controlled_samples, axis=0)
        
        std_err = np.std(controlled_samples, axis=0) / np.sqrt(n_paths)
        
        if return_iv:
            iv = implied_volatility(price, self.engine.S_0, K, T, self.engine.r, option_type)
            return price, std_err, iv
        
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
        _, S = self.engine.simulate_hybrid(n_paths)
        
        #  OTM PUTS
        if np.any(mask_puts):
            K_puts = K_values[mask_puts]
            prices_put, _, iv_puts = self.price_european_option(
                K_puts, 
                n_paths, 
                option_type='put',
                return_iv=True,
                S=S
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
                return_iv=True,
                S=S
            )
            ivs[mask_calls] = iv_calls
            prices[mask_calls] = prices_call
            
        return prices, ivs
    
    def compute_smile_turbo(self, k_log_moneyness, n_paths):
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
        # _, S = self.engine.simulate_hybrid(n_paths)
        dW_vol, _ = self.engine._generate_hybrid_increments(n_paths)
        W_tilde = self.engine._fft_convolution(dW_vol)
        increments = (dW_vol, W_tilde)
        
        
        #  OTM PUTS
        if np.any(mask_puts):
            K_puts = K_values[mask_puts]
            prices_put, _, iv_puts = self.price_european_option_turbo(
                K_puts, 
                n_paths, 
                option_type='put',
                return_iv=True,
                increments=increments
            )
            ivs[mask_puts] = iv_puts
            prices[mask_puts] = prices_put

        #  OTM CALLS
        if np.any(mask_calls):
            K_calls = K_values[mask_calls]
            prices_call, _, iv_calls = self.price_european_option_turbo(
                K_calls, 
                n_paths, 
                option_type='call',
                return_iv=True,
                increments=increments
            )
            ivs[mask_calls] = iv_calls
            prices[mask_calls] = prices_call
            
        return prices, ivs