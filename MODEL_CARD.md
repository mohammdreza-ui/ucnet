# Model Card: UCNet (Uncertainty-Calibrated Nets)

## Model Details
- **Model Name:** UCNet (Uncertainty-Calibrated Nets from Scratch)
- **Architecture Type:** Reverse-mode automatic differentiation engine and probabilistic neural network framework implemented in pure NumPy. Supports Linear, Conv1D, LayerNorm, activations, Heteroscedastic Gaussian NLL, Deep Ensembles ($M$ members), and Bayesian Linear Regression.
- **Developer:** Mohammadreza Salehi
- **Release Date:** October 2026
- **License:** MIT
- **Contact:** mohammadrezasalehi88@gmail.com

---

## Intended Use
- **Primary Use Cases:**
  - Educational and research demonstration of first-principles automatic differentiation, backpropagation, and exact gradient checking without high-level deep learning frameworks.
  - Quantitative predictive uncertainty estimation in regression tasks subject to distribution shifts (extrapolation shifts, input noise corruption, instrument domain variations).
  - Gas absorption spectroscopy calibration (TDLAS integration).
- **Out-of-Scope / Misuse:**
  - Direct deployment to high-throughput production GPU inference pipelines (UCNet is implemented strictly in CPU NumPy for transparency).
  - Safety-critical autonomous decision-making without domain-specific human verification.

---

## Training and Evaluation Data
1. **Synthetic Heteroscedastic Benchmark:** $y = \sin(3x) + \epsilon(x)$ with $\sigma(x) = 0.05 + 0.2|x|$ evaluated on $x \in [-1, 1]$ (in-distribution) and $[1, 2] \cup [-2, -1]$ (extrapolation).
2. **UCI / Scikit-Learn Tabular Benchmarks:**
   - Diabetes (442 samples, 10 features; scikit-learn)
   - Energy Efficiency (768 samples, 8 features; UCI)
   - Concrete Compressive Strength (1030 samples, 8 features; UCI)
   All three are the real datasets: the loaders raise an error rather than substituting simulated data.
3. **Simulated TDLAS Spectroscopic Scans:** 500 scans of length 250 across 4 distinct optical instruments with varying path lengths, baseline polynomial drift, and detector noise.

---

## Evaluation Metrics & Calibration Methodology
- **Negative Log-Likelihood (NLL):** Proper scoring rule penalizing miscalibrated probabilities.
- **Continuous Ranked Probability Score (CRPS):** Strictly proper scoring rule for continuous predictive distributions.
- **Empirical Coverage (Cov 50%, 80%, 90%, 95%):** Fraction of true targets falling within the central prediction intervals.
- **Probability Integral Transform (PIT) & Kolmogorov-Smirnov Test:** Uniformity test of $u = F(y)$ to detect overconfidence (U-shape) vs underconfidence ($\cap$-shape).
- **Regression Calibration Error (CalErr):** Mean squared error between nominal and empirical cumulative proportions (Kuleshov et al., 2018).
- **Sharpness:** Mean predictive standard deviation ($\bar{\sigma}$), evaluated alongside coverage to ensure intervals are informative.
- **Non-Parametric Bootstrap:** 1000 resamples to construct 95% confidence intervals on metric point estimates and paired model differences.

---

## Key Performance Summary
All results are mean ± std over 5 seeds; see [docs/results.md](docs/results.md) for tables and the decision rule.
- **Parity with PyTorch:** Unit tests confirm analytical gradients and optimizer steps match PyTorch in float64 within $10^{-10}$ relative tolerance.
- **Single network vs ensemble under shift:** the ensemble (M=5) has lower NLL than a single heteroscedastic network in all 6 shifted tabular scenarios (e.g. Energy extrapolation 2.65 vs 142.6), and 90% coverage closer to nominal in 4 of 6 (Energy and Concrete; not Diabetes). It remains under-covering on some shifts (Energy noise shift 0.65, Concrete noise shift 0.47 against 0.90).
- **Not uniformly best:** a linear Bayesian baseline matches or beats the ensemble on NLL for all Diabetes scenarios and on Concrete noise shift.
- **Simulated spectroscopy:** with a held-out device, the classical-style linear baseline is sharp and accurate in-distribution but its 95% coverage falls to 0.63 ± 0.26 (varying from 0.19 to 0.96 across seeds); the ensemble keeps ~0.98 coverage with intervals about four times wider. The pre-registered H3 is not confirmed under its decision rule.

---

## Limitations
1. **Hardware Constraints:** UCNet is designed strictly for single-threaded or multi-threaded CPU execution via NumPy/BLAS; it does not utilize GPU CUDA acceleration.
2. **Convolutional Scalability:** Conv1D (sliding-window views + einsum) measured about $8.8\times$ slower than PyTorch on a small benchmark (single thread). Conv1D and LayerNorm are verified by gradient and PyTorch-parity tests but are not used in the reported experiments.
3. **Dataset Scale:** Evaluated primarily on small-to-medium sized regression datasets ($N \le 20,000$).
4. **Spectroscopic Simulation:** The spectral data is generated from Beer-Lambert equations with simulated instrument variation, not measured laboratory data. Only one shift type (held-out device) is simulated, the neural baselines are small MLPs, and the classical reference is Bayesian linear regression rather than a Levenberg-Marquardt / Cramér-Rao fit.
5. **Statistical power:** five seeds; bootstrap intervals over five per-seed values are coarse. NLL, CRPS and coverage for ensembles use the moment-matched Gaussian; only PIT/KS/CalErr use the exact mixture CDF.
6. **Log-variance clamp:** predicted log-variance is clamped to $[-10, 5]$ in training and prediction, which has zero gradient outside the range.

---

## Ethical Considerations
Predictive models used in critical diagnostic or physical monitoring settings can produce overconfident failures when environment conditions drift. UCNet explicitly measures and quantifies model disagreement and aleatoric noise to alert operators when predictions enter untrusted regimes.
