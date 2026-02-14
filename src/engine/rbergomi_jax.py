import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

jax.config.update("jax_enable_x64", True)

class RBergomiJAXEngine:
    def __init__(self, T, N, H, eta, rho, xi_0, S_0):
        """
        JAX-optimized RBergomi Engine.
        This class is stateless to comply with JAX functional paradigm.
        """
        self.T = T
        self.N = N
        self.dt = T / N
        self.H = H
        self.eta = eta
        self.rho = rho
        self.xi_0 = xi_0
        self.S_0 = S_0
        
        self.times = jnp.linspace(0, T, N + 1)
        
        k = jnp.arange(N)
        t_mid = (k + 0.5) * self.dt
        b = t_mid ** (H - 0.5)
        b = b.at[0].set(0.0)
        
        b_padded = jnp.pad(b, (0, N), mode='constant')
        self.b_freq = jnp.fft.rfft(b_padded)
        
        self.w_sing = (self.dt ** (self.H - 0.5)) / jnp.sqrt(2 * self.H)
        self.sqrt_2H = jnp.sqrt(2 * self.H)
        self.sqrt_dt = jnp.sqrt(self.dt)
        self.sqrt_one_minus_rho2 = jnp.sqrt(1 - self.rho**2)

    @partial(jit, static_argnames=['self', 'n_paths'])
    def simulate_hybrid(self, key, n_paths):
        """
        Run the hybrid simulation on GPU/TPU.
        
        Args:
            key (jax.random.PRNGKey): Random seed for JAX.
            n_paths (int): Number of paths to simulate.
            
        Returns:
            tuple: (v, S)
                v: Variance paths (N+1, n_paths)
                S: Price paths (N+1, n_paths)
        """
        if n_paths % 2 != 0:
            raise ValueError(f"n_paths ({n_paths}) must be even for antithetic variates.")

        n_half = n_paths // 2

        key1, key2 = jax.random.split(key)
        
        dW1_half = self.sqrt_dt * jax.random.normal(key1, shape=(self.N, n_half))
        dW2_half = self.sqrt_dt * jax.random.normal(key2, shape=(self.N, n_half))

        dW1 = jnp.concatenate([dW1_half, -dW1_half], axis=1)
        dW2 = jnp.concatenate([dW2_half, -dW2_half], axis=1)
        dW_vol = dW1
        dZ_price = self.rho * dW1 + self.sqrt_one_minus_rho2 * dW2

        dW_padded = jnp.pad(dW_vol, ((0, self.N), (0, 0)), mode='constant')
        
        dW_freq = jnp.fft.rfft(dW_padded, axis=0)
        
        fft_product = dW_freq * self.b_freq[:, None]
        
        W_tilde_fft = jnp.fft.irfft(fft_product, n=2*self.N, axis=0)[:self.N]

        W_corr = self.w_sing * dW_vol
        W_tilde_partial = self.sqrt_2H * (W_tilde_fft + W_corr)
        
        zeros_row = jnp.zeros((1, n_paths))
        W_tilde = jnp.vstack([zeros_row, W_tilde_partial])

        t_col = self.times[:, None]
        drift_correction = 0.5 * self.eta**2 * (t_col ** (2 * self.H))
        v = self.xi_0 * jnp.exp(self.eta * W_tilde - drift_correction)

        V_start = v[:-1, :]
        
        log_returns = jnp.sqrt(V_start) * dZ_price - 0.5 * V_start * self.dt
        cum_log_returns = jnp.cumsum(log_returns, axis=0)
        
        cum_log_returns_padded = jnp.vstack([zeros_row, cum_log_returns])
        S = self.S_0 * jnp.exp(cum_log_returns_padded)

        return v, S