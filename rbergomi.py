import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from scipy.linalg import toeplitz

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
        L = np.linalg.cholesky(cov + np.eye(cov.shape[0]) * 1e-9)
        return L

    def simulate_paths(self, n_paths):
        """
        Generate v_t and S_T
        """
        M = len(self.times)
        Z = np.random.randn(2*M, n_paths)
        noise = self.L @ Z
        W_tilde, Z_price = noise[:M,:], noise[M:,:]
        t = self.times[:, None]
        v = self.xi_0 * np.exp(self.eta * W_tilde - 0.5 * self.eta**2 * t**(2 * self.H))
        dZ = np.diff(Z_price, axis=0)
        V_start = v[:-1, :]
        log_returns = np.sqrt(V_start) * dZ - 0.5 * V_start * self.dt
        cum_log_returns = np.cumsum(log_returns, axis=0)
        zeros_row = np.zeros((1, n_paths))
        cum_log_returns_padded = np.vstack([zeros_row, cum_log_returns])
        S = self.S_0 * np.exp(cum_log_returns_padded)
        return v, S
    
    def simulate_hybrid_paths(self, n_paths, kappa=1):
        """
        Simulation via Schéma Hybride (Bennedsen et al. 2017).
        Complexité: O(N log N) via FFT.
        
        Args:
            kappa (int): Nombre de pas de temps traités de manière exacte (partie singulière).
                        kappa=1 est souvent suffisant pour une bonne précision.
        """
        # 1. Génération des Bruits de base (Incréments)
        # On a besoin de dW (pour la vol) et dZ (pour le prix)
        # Ils sont corrélés avec rho.
        
        # a. Générer deux bruits indépendants dW1 et dW2 de taille (N, n_paths)
        # Ce sont des incréments Gaussiens ~ N(0, dt)
        
        # b. Construire les bruits corrélés
        # dW_vol = dW1
        # dZ_price = rho * dW1 + sqrt(1 - rho^2) * dW2
        
        # 2. Construction du Processus de Volterra (W_tilde) en 2 parties
        
        # --- PARTIE A : CONVOLUTION FFT (Le "Long-Terme") ---
        # On veut calculer la convolution de dW_vol avec le noyau b(t) = t^(H-0.5)
        # Astuce FFT: Pour éviter la "convolution circulaire" (effets de bord), 
        # on doit doubler la taille des vecteurs (Padding avec des zéros).
        
        # i. Définir le vecteur noyau 'b' sur la grille
        # b[k] = (t_k)^(H-0.5) pour k allant de 0 à N-1
        # (Attention à b[0] qui est infini, on le gère dans la partie singulière, 
        # pour la FFT on peut le mettre à 0 ou l'interpoler)
        
        # ii. Padding
        # Étendre dW_vol et b à la taille 2*N (ajouter des zéros à la fin)
        
        # iii. Appliquer la FFT (scipy.fft.fft)
        # fft_dW = fft(dW_pad)
        # fft_b = fft(b_pad)
        
        # iv. Multiplication dans le domaine fréquentiel
        # fft_product = fft_dW * fft_b
        
        # v. Inverse FFT
        # convolution_result = ifft(fft_product)
        
        # vi. Tronquer
        # On ne garde que les N premiers points (la partie réelle)
        # C'est notre intégrale "grossière" I_fft
        
        
        # --- PARTIE B : CORRECTION SINGULIÈRE (Le "Short-Terme") ---
        # La FFT lisse trop le point t=0. On doit réinjecter l'énergie de la singularité
        # qui vient des 'kappa' derniers pas de temps.
        
        # On calcule explicitement la covariance sur les 'kappa' derniers lags
        # et on ajuste le résultat de la FFT (soustraire la partie mal intégrée 
        # et ajouter la partie exacte).
        # (Pour une implémentation simple "Turbocharging", on peut juste faire :
        # W_tilde = I_fft (sur les lags > kappa) + Somme_Exacte (sur lags <= kappa))
        
        
        # 3. Reconstitution (Comme avant)
        # W_tilde = Resultat_Hybrid
        # V = xi_0 * exp(...)
        # S = Euler Exponentiel avec dZ_price
        
        pass