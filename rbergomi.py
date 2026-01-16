import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from scipy.linalg import toeplitz
import time

import numpy as np

class RBergomiEngine:
    def __init__(self, T, N, H, eta, rho, xi_0, S_0):
        self.T = T
        self.N = N
        self.dt = T / N
        self.H = H
        self.eta = eta
        self.rho = rho
        self.xi_0 = xi_0
        self.S_0 = S_0
        
        self.times = np.linspace(0, T, N+1)
        
        self.L = None

    def _cov_volterra_kernel(self):
        """
        Compute E[W_tilde(t1) * W_tilde(t2)]
        With a simple Riemann sum to approximate volterra covariance
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
        Bloc Bas-Droite (N x N).
        Cov(Z_ti, Z_tj) = min(ti, tj)
        """
        return np.minimum(self.times[:, None], self.times[None, :])

    def _cov_cross_term(self):
        """
        Compute E[W_tilde(t_vol) * Z(t_price)]
        Using Bayer et al. (2016) formulas  (Section 4).
        """
        const = self.rho * np.sqrt(2*self.H) / (self.H+0.5)
        dt_matrix = self.times[:, None] - np.minimum(self.times[:, None], self.times[None, :])
        cov_matrix = const * (self.times[:, None]**(self.H+0.5) - dt_matrix**(self.H+0.5))
        return cov_matrix

    def _build_cholesky_matrix(self):
        """
        Build the 2N x 2N covariance matrix and use Cholesky decomposition.
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
        Generate v_t and S_T

        Args:
            n_paths (int): Number of paths (trajectories) to simulate.
        """
        if self.L is None:
            print("Compute Cholesky...")
            start = time.time()
            self._build_cholesky_matrix()
            print(f"Done in {(time.time() - start):.3f} seconds")
        start = time.time()
        print("Computing L*Z...")
        M = len(self.times)
        Z = np.random.randn(2*M, n_paths)
        noise = self.L @ Z
        print(f"Done in {(time.time() - start):.3f} seconds")
        start = time.time()
        print("Computing paths...")
        W_tilde, Z_price = noise[:M,:], noise[M:,:]
        t = self.times[:, None]
        v = self.xi_0 * np.exp(self.eta * W_tilde - 0.5 * self.eta**2 * t**(2 * self.H))
        dZ = np.diff(Z_price, axis=0)
        V_start = v[:-1, :]
        log_returns = np.sqrt(V_start) * dZ - 0.5 * V_start * self.dt
        cum_log_returns = np.cumsum(log_returns, axis=0)
        zeros_row = np.zeros((1, n_paths))
        cum_log_returns_padded = np.vstack([zeros_row, cum_log_returns])
        print(f"Done in {(time.time() - start):.3f} seconds")
        S = self.S_0 * np.exp(cum_log_returns_padded)

        return v, S
    
    def _generate_hybrid_increments(self, n_paths):
        """
        Generating dW_vol and dZ_price for increments of the process

        Args:
            n_paths (int): Number of paths (trajectories) to simulate.
        """
        dW1, dW2 = np.sqrt(self.dt)*np.random.randn(self.N, n_paths), np.sqrt(self.dt)*np.random.randn(self.N, n_paths)
        dW_vol = dW1
        dZ_price = self.rho * dW1 + np.sqrt(1-self.rho**2) * dW2
        return dW_vol, dZ_price

    def _fft_convolution(self, dW_vol):
        """
        Fonction interne pour calculer l'intégrale de Volterra via FFT.
        """
        # 1. Créer noyau b
        # 2. Padding (double taille)
        # 3. FFT -> Produit -> IFFT
        # 4. Troncature
        # Retourne W_tilde
        pass

    def simulate_hybrid(self, n_paths):
        """
        Simulation via FFT.
        Le code est 100% autonome ici aussi.
        """
        # 1. Génération Incréments
        dW_vol, dZ_price = self._generate_hybrid_increments(n_paths)
        
        # 2. Calcul W_tilde (Mémoire longue)
        W_tilde = self._fft_convolution(dW_vol)
        
        # 3. Calcul Variance (V_t)
        # v = xi * exp(...) (Même formule que exact)
        
        # 4. Calcul Prix (S_t) - Méthode Euler Exponentiel
        # ATTENTION : Ici dZ_price sont DÉJÀ des sauts (incréments).
        # On n'utilise PAS np.diff ici.
        
        # ... Logique Euler ...
        # log_ret = sqrt(v) * dZ_price - ...
        # S = S0 * exp(cumsum(log_ret))
        
        pass
        # return S, v