import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from jax.scipy.stats import norm

from src.engine.rbergomi_jax import RBergomiJAXEngine

jax.config.update("jax_enable_x64", True)

@jit
def black_scholes_price(S, K, T, r, sigma, is_call):
    """
    Computes the Black-Scholes European option price.
    Vectorized for Strikes (K) and Maturities (T).
    """
    T = jnp.maximum(T, 1e-6)
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    call_price = S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)
    put_price = K * jnp.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return jnp.where(is_call, call_price, put_price)

@jit
def implied_vol_solver(target_price, S, K, T, r, is_call):
    """
    Computes Implied Volatility from a market price using numerical inversion (Bissection).
    """
    low, high = 1e-4, 5.0
    tol = 1e-6
    
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, is_call) - target_price

    def body_fun(state):
        low, high, _ = state
        mid = (low + high) / 2.0
        val = objective(mid)
        new_high = jnp.where(val > 0, mid, high)
        new_low = jnp.where(val <= 0, mid, low)
        return (new_low, new_high, mid)

    def cond_fun(state):
        low, high, mid = state
        return (high - low) > tol

    intrinsic = jnp.where(is_call, jnp.maximum(0.0, S - K), jnp.maximum(0.0, K - S))
    is_valid = target_price > (intrinsic + 1e-9)

    init_val = (low, high, 0.0)
    _, _, final_mid = jax.lax.while_loop(cond_fun, body_fun, init_val)
    
    return jnp.where(is_valid, final_mid, jnp.nan)

implied_vol_vmap = vmap(implied_vol_solver, in_axes=(0, None, 0, None, None, 0))

class MonteCarloPricerJAX:
    def __init__(self, engine: RBergomiJAXEngine, batch_size: int = 50_000):
        """
        Args:
            engine: rBergomi Engine JAX.
            batch_size: Size of batch send to gpu
        """
        self.engine = engine
        self.batch_size = batch_size

    def _process_batches(self, key, n_paths, payoff_fn):
        n_full_batches = n_paths // self.batch_size
        remainder = n_paths % self.batch_size
        
        total_payoff_sum = None 
        
        for i in range(n_full_batches):
            key, subkey = jax.random.split(key)
            
            _, S = self.engine.simulate_hybrid(subkey, self.batch_size)
            S_T = S[-1, :] # (batch_size,)
            
            batch_payoffs = payoff_fn(S_T) # Shape (n_strikes,) or (1,)
            
            if total_payoff_sum is None:
                total_payoff_sum = batch_payoffs
            else:
                total_payoff_sum += batch_payoffs
            
            del S, S_T

        if remainder > 0:
            key, subkey = jax.random.split(key)
            _, S = self.engine.simulate_hybrid(subkey, remainder)
            S_T = S[-1, :]
            batch_payoffs = payoff_fn(S_T)
            
            if total_payoff_sum is None:
                total_payoff_sum = batch_payoffs
            else:
                total_payoff_sum += batch_payoffs

        return total_payoff_sum

    def compute_smile(self, key, k_log_moneyness, n_paths):
        K_values = self.engine.S_0 * jnp.exp(k_log_moneyness)
        
        is_put_mask = K_values < self.engine.S_0
        signs = jnp.where(is_put_mask, -1.0, 1.0)
        
        def smile_payoff_aggregator(S_T_batch):
            diff = S_T_batch[:, None] - K_values[None, :]
            payoffs = jnp.maximum(signs * diff, 0.0)
            return jnp.sum(payoffs, axis=0)

        total_payoff_sum = self._process_batches(key, n_paths, smile_payoff_aggregator)
        
        prices_otm = total_payoff_sum / n_paths
        
        is_call_mask = ~is_put_mask
        
        ivs = implied_vol_vmap(
            prices_otm, 
            self.engine.S_0, 
            K_values, 
            self.engine.T, 
            0.0, 
            is_call_mask
        )
        
        return K_values, ivs

    def price_option_batch(self, key, K, n_paths, is_call=True):
        K = jnp.atleast_1d(K)
        payoff_sign = 1.0 if is_call else -1.0
        
        def option_payoff_aggregator(S_T_batch):
            payoffs = jnp.maximum(payoff_sign * (S_T_batch[:, None] - K[None, :]), 0.0)
            return jnp.sum(payoffs, axis=0)
            
        total_payoff_sum = self._process_batches(key, n_paths, option_payoff_aggregator)
        
        prices = total_payoff_sum / n_paths
        
        r = 0.0
        is_call_mask = jnp.full_like(K, is_call, dtype=bool)
        
        ivs = implied_vol_vmap(prices, self.engine.S_0, K, self.engine.T, r, is_call_mask)

        return prices, ivs