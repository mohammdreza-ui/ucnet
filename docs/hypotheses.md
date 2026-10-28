# Pre-Registered Scientific Hypotheses

**Project:** UCNet (Uncertainty-Calibrated Nets)  
**Date Pre-Registered:** 2026-10-27  
**Commit Status:** Registered prior to running benchmark experiments.

---

## 1. Context and Research Questions
Modern deep learning models tend to produce overconfident predictions when evaluated outside their training distribution. In high-stakes applications such as quantitative absorption spectroscopy and critical regression tasks, faithful calibration and truthful uncertainty estimation are paramount.

This study systematically compares:
- **M0**: Homoscedastic baseline (MSE loss with constant variance estimated from validation residuals)
- **M1**: Single Heteroscedastic neural network (predicting mean $\mu(x)$ and log-variance $s(x) = \ln \sigma^2(x)$)
- **M2**: Deep Ensemble of heteroscedastic networks ($M=5$ members) combined via the Law of Total Variance
- **M4**: Classical analytical Bayesian Linear Regression (BLR) / physical least-squares reference

---

## 2. Pre-Registered Hypotheses

### Hypothesis H1 (Extrapolation Coverage)
> **Under feature extrapolation distribution shift, the empirical 90% coverage of Deep Ensemble (M2) is closer to the nominal 90% level than that of a single heteroscedastic network (M1):**
> $$|\text{Cov}_{90}(\text{M2}) - 0.90| < |\text{Cov}_{90}(\text{M1}) - 0.90|$$
> **Rationale:** In out-of-distribution regions, epistemic uncertainty (member disagreement) expands significantly, inflating the predictive interval and compensating for extrapolation errors that would cause single overconfident models to fail.

### Hypothesis H2 (Negative Log-Likelihood)
> **Deep Ensemble (M2) achieves strictly lower (superior) average Negative Log-Likelihood (NLL) on shifted test distributions than a single network (M1):**
> $$\text{NLL}(\text{M2}) < \text{NLL}(\text{M1})$$
> **Rationale:** By averaging predictions over multiple local minima, ensembles regularize predictions and avoid catastrophically high log-loss penalties on outliers.

### Hypothesis H3 (Spectroscopic Domain Shift)
> **Under instrument domain shifts (inter-device variation, optical baseline drift, path length fluctuations), the classical analytical baseline (M4) provides calibrated coverage (~95%) in-distribution, but exhibits calibration degradation under physical shift. Deep Ensemble (M2) maintains higher retention of nominal interval coverage under instrument shift.**

---

## 3. Decision Rules & Statistical Verification Protocol
1. **5 Independent Random Seeds:** Seeds `[42, 123, 456, 789, 1024]` evaluated across all datasets and shift scenarios.
2. **Paired Difference Testing:** For each seed $k$, compute $\Delta_k = \text{Metric}(\text{M1})_k - \text{Metric}(\text{M2})_k$.
3. **Statistical Significance Criteria:** A hypothesis is formally **CONFIRMED** if and only if:
   - The paired difference $\Delta_k$ maintains consistent sign across all 5 seeds, AND
   - The 95% non-parametric bootstrap confidence interval ($B=1000$) on the mean difference does NOT contain zero.
   Otherwise, the hypothesis is reported as **NOT CONFIRMED** without modification or post-hoc reframing.
4. **Sharpness Requirement:** Calibration metrics must always be reported alongside sharpness (mean predictive standard deviation) to prevent rewarding trivially uninformative inflated intervals.

---

## 4. Post-registration notes (added after the experiments; the registered text above is unchanged)

- **Scope of H1/H2:** evaluated for M1 vs M2 in each of the six shift scenarios (3 datasets × extrapolation / 0.5-std noise
  shift). The in-distribution scenario is reported for context only. The registration named extrapolation shift for H1; the
  noise shift was added as a second shift type and is reported separately.
- **Decision rule implemented as registered:** M2 ahead in every seed AND bootstrap CI of the mean advantage strictly
  above 0 (`ucnet.metrics.hypothesis_verdict`).
- **H3 operationalisation (fixed after seeing the data):** "~95%" was turned into thresholds (M4 in-distribution coverage
  within 0.03 of 0.95; M4 coverage lower under shift in every seed; M2 shift coverage closer to 0.95 than M4 in every seed
  with a CI above 0). The registration did not specify these numbers. Result: not confirmed (docs/results.md §4).
- **Errata:** a first analysis reported H1-H3 as confirmed; it was produced by code defects and a re-defined H3 and has been
  withdrawn (docs/results.md §8).

