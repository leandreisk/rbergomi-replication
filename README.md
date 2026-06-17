# Rough Bergomi (rBergomi) Model: Empirical Calibration & Pricing

This repository implements the **Rough Bergomi (rBergomi)** stochastic volatility model. The project features real-world empirical validation, variance reduction techniques, and full calibration to the S&P 500 options surface.

*This project was developed as part of the PhD course **FINA 80222A: Contingent Claims in Incomplete Markets** at HEC Montréal.*

The codebase focuses on reproducing the characteristic **At-The-Money (ATM) volatility skew** for short maturities and simulating Volterra processes using efficient discretization schemes.

## 🎯 Project Objectives

* **Empirical Roughness Measurement**: Extracting the Hurst exponent ($H$) directly from historical S&P 500 realized volatility using scaling moments.
* **Volterra Process Simulation**: Implementation of fractional kernels to capture the non-Markovian dynamics of fractional Brownian motion.
* **IV Skew Recovery**: Recovering the power-law explosion of the skew as time-to-maturity $T \to 0$.
* **Variance Reduction**: Utilizing a Mixed Estimator (Conditional Monte Carlo + Control Variate) to drastically reduce computational Monte Carlo error.

## 📂 Project Structure

* `notebooks/`: Contains `rBergomi.ipynb`, an interactive notebook featuring detailed step-by-step mathematical explanations, empirical findings, and exploratory scripts.
* `src/`: Core modular engine divided into `engine/` (simulation logic), `pricing/` (Monte Carlo and Mixed estimators), and `utils/`.
* `scripts/`: Executable Python scripts for automated tasks, including `benchmark.py` (performance testing) and `plot_term_structure.py` (IV surface generation).
* `out/`: Destination directory for generated artifacts.
* `config.yaml`: Centralized configuration file for model parameters ($H, \eta, \rho, \text{steps}, \text{paths}$).
* `requirements.txt`: Project dependencies.
* `Rough Volatility From Empirical Evidence to Pricing.pdf`: Presentation slides detailing the mathematical framework and empirical results.

## 🛠 Features & Roadmap

- [x] Initial engine implementation via exact Cholesky decomposition.
- [x] Replace Cholesky ($O(N^3)$) with the hybrid scheme ($O(N \log N)$) via FFT to handle large time steps
- [x] Compare the speed and accuracy of the hybrid and Cholesky schemes.
- [x] Implied Volatility (IV) surface generation for European options.
- [x] Variance reduction via Mixed Estimator combining Black-Scholes analytical integration and a synthetic Timer option control variate.
- [x] Real market calibration on the S&P 500 (Sept 16, 2022) optimizing $\eta, \rho, H,$ and $\xi_0$.

---

## 📚 References

[1] **Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018).** *Volatility is rough*. Quantitative Finance, 18(6), 933-949.

[2] **Bayer, C., Friz, P., & Gatheral, J. (2016).** *Pricing under rough volatility*. Quantitative Finance, 16(6), 887-904.

[3] **McCrickerd, R., & Pakkanen, M. S. (2018).** *Turbocharging Monte Carlo pricing for the rough Bergomi model*. Quantitative Finance, 18(11), 1877-1886.

[4] **Bergomi, L. (2005).** *Smile Dynamics II*. Société Générale, Equity Derivatives Research.

[5] **Bennedsen, M., Lunde, A., & Pakkanen, M. S. (2017).** *Hybrid scheme for Brownian semistationary processes.* Finance and Stochastics, 21(4), 931–965.