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
        
        self.times = np.linspace(0, T, N) # Shape (N,)
        
        self.L = self.construct_covariance_matrix()

    def cov_volterra_kernel(self, t_vec):
        """
        Compute E[W_tilde(t1) * W_tilde(t2)]
        With a simple Riemann sum
        """
        # A CORRIGER
        kernel_vector = np.zeros_like(t_vec)
        kernel_vector[1:] = t_vec[1:]**(self.H - 0.5)
        kernel_vector[0] = (self.dt)**(self.H - 0.5)/np.sqrt(2*self.H)
        kernel_matrix = toeplitz(kernel_vector, np.zeros_like(kernel_vector))
        kernel_matrix = np.tril(kernel_matrix)
        kernel_matrix[0, 0] = 0.0
        return self.dt*kernel_matrix@kernel_matrix.T

    def cov_brownian(self, t_vec):
        """
        Bloc Bas-Droite (N x N).
        Cov(Z_ti, Z_tj) = min(ti, tj)
        """
        return np.minimum(t_vec[:, None], t_vec[None, :])

    def cov_cross_term(self, t_vol_vec, t_price_vec):
        """
        Compute E[W_tilde(t_vol) * Z(t_price)]
        Using Bayer et al. (2016) formulas  (Section 4).
        """
        const = self.rho * np.sqrt(2*self.H) / (self.H+0.5)
        dt_matrix = t_vol_vec[:, None] - np.minimum(t_vol_vec[:, None], t_price_vec[None, :])
        cov_matrix = const * (t_vol_vec[:, None]**(self.H+0.5) - dt_matrix**(self.H+0.5))
        return cov_matrix

    def construct_covariance_matrix(self):
        """
        Build the 2N x 2N covariance matrix and use Cholesky decomposition.
        """
        # Initialiser matrice vide (2N, 2N)
        # Remplir le bloc haut-gauche (Volterra)
        # Remplir le bloc bas-droite (Brownien standard: min(ti, tj))
        # Remplir les blocs croisés (Haut-Droit et Bas-Gauche)
        
        # Faire Cholesky
        # return L
        pass

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