import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from scipy.linalg import toeplitz

import numpy as np

class RBergomiEngine:
    def __init__(self, T, N, H, eta, rho, xi_0):
        self.T = T
        self.N = N
        self.dt = T / N
        self.H = H
        self.eta = eta
        self.rho = rho
        self.xi_0 = xi_0
        
        self.times = np.linspace(0, T, N+1) # Shape (N,)
        
        self.L = self.construct_covariance_matrix()

    def cov_volterra_kernel(self):
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
    
    def cov_brownian(self):
        """
        Bloc Bas-Droite (N x N).
        Cov(Z_ti, Z_tj) = min(ti, tj)
        """
        return np.minimum(self.times[:, None], self.times[None, :])

    def cov_cross_term(self):
        """
        Compute E[W_tilde(t_vol) * Z(t_price)]
        Using Bayer et al. (2016) formulas  (Section 4).
        """
        const = self.rho * np.sqrt(2*self.H) / (self.H+0.5)
        dt_matrix = self.times[:, None] - np.minimum(self.times[:, None], self.times[None, :])
        cov_matrix = const * (self.times[:, None]**(self.H+0.5) - dt_matrix**(self.H+0.5))
        return cov_matrix

    def construct_covariance_matrix(self):
        """
        Build the 2N x 2N covariance matrix and use Cholesky decomposition.
        """
        cov_vol = self.cov_volterra_kernel() 
        cov_brown = self.cov_brownian()
        cov_cross = self.cov_cross_term()
        cov = np.block([
            [cov_vol, cov_cross],
            [cov_cross.T, cov_brown]
        ])
        L = np.linalg.cholesky(cov)
        return L

    def simulate_paths(self, n_paths):
        """
        Generate v_t and S_T
        """
        # 1. Générer Gaussiennes Z (2N, n_paths)
        # 2. Corréler : correlated_bruit = self.L @ Z
        # 3. Extraire W_tilde (partie 1) et Z_price (partie 2)
        # 4. Construire v_t = xi_0 * exp(...)
        # 5. Construire S_t = S_0 * exp(...)
        pass