import numpy as np
import pandas as pd 
from scipy.linalg import toeplitz
import time

import numpy as np

class RBergomiEngine:
    def __init__(self, T, N, H, eta, rho, xi_0, S_0, r):
        self.T = T
        self.N = N
        self.dt = T / N
        self.H = H
        self.eta = eta
        self.rho = rho
        self.xi_0 = xi_0
        self.S_0 = S_0
        self.r = r
        
        self.times = np.linspace(0, T, N+1)
        
        self.L = None

    def _cov_volterra_kernel(self):
        """
        Compute E[W_tilde(t1) * W_tilde(t2)] via a Riemann sum approximation.
    
        Args:
            None
        Returns:
            ndarray: Covariance matrix of shape (M, M), where M = len(self.times).
        """
        kernel_vector = np.zeros_like(self.times)
        kernel_vector[1:] = self.times[1:]**(self.H - 0.5)
        singular_weight = (self.dt**(self.H - 0.5)) / np.sqrt(2 * self.H)
        kernel_vector[1] = singular_weight
        kernel_vector[0] = 0.0
        K = toeplitz(kernel_vector, r=np.zeros_like(kernel_vector))
        cov_matrix = self.dt * (K @ K.T)
        return 2 * self.H * cov_matrix
    
    def _cov_brownian(self):
        """
        Compute the standard Brownian motion covariance matrix.
    
        Args:
            None
        Returns:
            ndarray: Covariance matrix of shape (N+1, N+1).
        """
        return np.minimum(self.times[:, None], self.times[None, :])

    def _cov_cross_term(self):
        """
        Compute the cross-covariance matrix between the Volterra process and Brownian motion from 
        the closed-form covariance formula from Bayer et al. (2016).
    
        Args:
            None.
        Returns:
            ndarray: Cross-covariance matrix of shape (N+1, N+1).
        """
        const = self.rho * np.sqrt(2*self.H) / (self.H+0.5)
        dt_matrix = self.times[:, None] - np.minimum(self.times[:, None], self.times[None, :])
        cov_matrix = const * (self.times[:, None]**(self.H+0.5) - dt_matrix**(self.H+0.5))
        return cov_matrix

    def _build_cholesky_matrix(self):
        """
        Construct the joint 2M x 2M covariance matrix and compute its Cholesky factor.
    
        Args:
            None.
        Returns:
            None: Stores the lower triangular matrix in self.L.
        """
        cov_vol = self._cov_volterra_kernel() 
        cov_brown = self._cov_brownian()
        cov_cross = self._cov_cross_term()
        cov = np.block([
            [cov_vol, cov_cross],
            [cov_cross.T, cov_brown]
        ])
        self.L = np.linalg.cholesky(cov + np.eye(cov.shape[0]) * 1e-9)

    def simulate_cholesky(self, n_paths):
        """
        Simulate joint paths for variance (v) and price (S) using a Cholesky decomposition.
    
        Args:
            n_paths (int): Number of paths (trajectories) to simulate.
        Returns:
            tuple: (v, S) as ndarrays of shape (N + 1, n_paths), where M is the number of time steps.
        """
        if self.L is None:
            self._build_cholesky_matrix()
    
        M = len(self.times)
        Z = np.random.randn(2*M, n_paths)
    
        noise = self.L @ Z
        W_tilde, Z_price = noise[:M,:], noise[M:,:]
    
        t = self.times[:, None]
        v = self.xi_0 * np.exp(self.eta * W_tilde - 0.5 * self.eta**2 * t**(2 * self.H))
    
        dZ = np.diff(Z_price, axis=0)
        V_start = v[:-1, :]
        log_returns = np.sqrt(V_start) * dZ + (self.r - 0.5 * V_start) * self.dt
        cum_log_returns = np.cumsum(log_returns, axis=0)
    
        zeros_row = np.zeros((1, n_paths))
        cum_log_returns_padded = np.vstack([zeros_row, cum_log_returns])
        S = self.S_0 * np.exp(cum_log_returns_padded)

        return v, S
    
    def _generate_hybrid_increments(self, n_paths):
        """
        Generating dW_vol and dZ_price for increments of the process
    
        Args:
            n_paths (int): Number of paths (trajectories) to simulate.
        Returns:
            tuple: Two ndarrays (dW_vol, dZ_price) of shape (N, n_paths).
        """
        half_paths = n_paths // 2
        
        z1 = np.random.randn(self.N, half_paths)
        z2 = np.random.randn(self.N, half_paths)

        z1_full = np.concatenate([z1, -z1], axis=1)
        z2_full = np.concatenate([z2, -z2], axis=1)

        dW1 = np.sqrt(self.dt) * z1_full
        dW2 = np.sqrt(self.dt) * z2_full
    
        dW_vol = dW1
        dZ_price = self.rho * dW1 + np.sqrt(1-self.rho**2) * dW2
    
        return dW_vol, dZ_price

    def _fft_convolution(self, dW_vol):
        """
        Compute the Volterra integral via FFT-based linear convolution.
        Includes a singularity correction for the fractional kernel at the origin.

        Args:
            dW_vol (ndarray): Brownian increments of shape (N, n_paths).
        Returns:
            ndarray: The discretized Volterra process of shape (N + 1, n_paths).
    """
        k = np.arange(self.N)
        t_mid = (k + 0.5) * self.dt
    
        b = t_mid ** (self.H - 0.5)
        b[0] = 0
        b_padded = np.pad(b, (0, self.N), mode='constant')
        dW_padded = np.pad(dW_vol, ((0, self.N), (0, 0)), mode='constant')
    
        b_freq = np.fft.rfft(b_padded)
        dW_vol_freq = np.fft.rfft(dW_padded, axis=0)
    
        fft_product = dW_vol_freq * b_freq[:, None]
        W_tilde_fft = np.fft.irfft(fft_product, n=2*self.N, axis=0)[:self.N]
    
        w_sing = (self.dt ** (self.H - 0.5)) / np.sqrt(2 * self.H)
        W_corr = w_sing * dW_vol
        W_tilde = np.sqrt(2 * self.H) * W_tilde_fft + np.sqrt(2 * self.H) * W_corr
    
        n_paths = dW_vol.shape[1]
        zeros_row = np.zeros((1, n_paths))
        W_tilde = np.vstack([zeros_row, W_tilde])
    
        return W_tilde

    def simulate_hybrid(self, n_paths):
        """
        Simulate variance (v) and price (S) paths using the hybrid FFT scheme.

        Args:
            n_paths (int): Number of paths (trajectories) to simulate.
        Returns:
            tuple: (v, S) as ndarrays of shape (N + 1, n_paths).
        """
        dW_vol, dZ_price = self._generate_hybrid_increments(n_paths)
    
        W_tilde = self._fft_convolution(dW_vol)
    
        t = self.times[:, None]
        v = self.xi_0 * np.exp(self.eta * W_tilde - 0.5 * self.eta**2 * t**(2 * self.H))
    
        V_start = v[:-1, :]
        log_returns = np.sqrt(V_start) * dZ_price + (self.r - 0.5 * V_start) * self.dt
        cum_log_returns = np.cumsum(log_returns, axis=0)
    
        zeros_row = np.zeros((1, n_paths))
        cum_log_returns_padded = np.vstack([zeros_row, cum_log_returns])
        S = self.S_0 * np.exp(cum_log_returns_padded)
    
        return v, S