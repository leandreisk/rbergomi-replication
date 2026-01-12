# rBergomi Model Replication

This repository implements the **Rough Bergomi (rBergomi)** stochastic volatility model. The project focuses on reproducing the characteristic **At-The-Money (ATM) volatility skew** for short maturities and simulating Volterra processes using efficient discretization schemes.



## 🎯 Project Objectives

* **Volterra Process Simulation**: Implementation of kernels to capture the "rough" (non-Markovian) dynamics of fractional Brownian motion.
* **IV Skew Recovery**: Recovering the power-law explosion of the skew as time-to-maturity $T \to 0$.
* **Scheme Comparison**: Evaluating simulation efficiency (e.g., hybrid schemes) for rough paths.

## 📂 Project Structure

* `rbergomi.py`: Core engine containing the `rBergomi` class and simulation logic.
* `main.py`: Execution script for Monte Carlo simulations and visualization.
* `config.yaml`: Centralized parameters ($H, \eta, \rho, \text{steps}, \text{paths}$).
* `out/`: Directory for generated plots, skews, and reports.

## 🧮 Theoretical Background

The variance process is modeled as:
$$v_t = v_0 \exp \left( \eta \sqrt{2H} \int_0^t (t-s)^{H-1/2} dW_s - \frac{1}{2} \eta^2 t^{2H} \right)$$

Key focus is placed on the **Hurst exponent ($H$)**. When $H < 0.5$, the model generates the "rough" trajectories necessary to match market-observed volatility surfaces.

## 🛠 Features & Roadmap
- [ ] Efficient Monte Carlo pricing for European options.
- [ ] Implied Volatility (IV) surface generation.
- [ ] Integration of the Bennedsen et al. (2017) hybrid scheme.
- [ ] Parallelization with Numba/Multiprocessing.

## 📚 References

This implementation is based on the following seminal work:

* **Bayer, C., Friz, P., & Gatheral, J. (2016).** *Pricing under rough volatility.* Quantitative Finance, 16(6), 887-904.

* **Bennedsen, M., Lunde, A., & Pakkanen, M. S. (2017).** *Hybrid scheme for Brownian semistationary processes.* Finance and Stochastics, 21(4), 1107-1165.