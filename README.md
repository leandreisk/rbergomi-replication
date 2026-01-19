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
- [x] Initial engine implementation via Cholesky decomposition.
- [x] Replace Cholesky $(O(N^3))$ with the hybrid scheme $(O(N \log N))$ via FFT to handle large time steps (Bennedsen et al., 2017).
- [x] Compare the speed of the hybrid and Cholesky schemes.
- [ ] Implied Volatility (IV) surface generation for European option.
- [ ] Training an MLP (Multi-Layer Perceptron) for instantaneous inversion of market parameters.
- [ ] SDE Simulation via Neural SDEs

## 📚 References

This implementation is based on the following seminal work:

* **Bayer, C., Friz, P., & Gatheral, J. (2016).** *Pricing under rough volatility.* Quantitative Finance, 16(6), 887-904.

* **McCrickerd, R., & Pakkanen, M. S. (2018).** *Turbocharging Monte Carlo pricing for the rough Bergomi model.* Quantitative Finance, 18(11), 1877-1886.

* **Bennedsen, M., Lunde, A., & Pakkanen, M. S. (2017).** *Hybrid scheme for Brownian semistationary processes.* Finance and Stochastics, 21(4), 1107-1165.