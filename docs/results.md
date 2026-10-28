# UCNet: Empirical Evaluation & Findings

All numbers below come from the experiment scripts in `experiments/`, run on the corrected code (see
[§8 Corrections](#8-corrections-to-the-earlier-version-of-this-report)). Every run writes a manifest
(`results/<experiment>/*_manifest.json`: seeds, configuration, library versions, git commit) and its console log
(`results/logs/*.log`). Per-seed raw metrics are in the `*_per_seed.csv` files. The `results/` directory is
git-ignored, so regenerate it with the commands in the README.

Protocol: 5 seeds `[42, 123, 456, 789, 1024]`; standardisation fitted on training data only; (lr, weight decay)
chosen on validation NLL only and shared by M0, M1 and every M2 member; test sets used once. Tables show mean ± std over
seeds. NLL, CRPS, coverage and sharpness use the moment-matched Gaussian `(mu*, sigma*^2)` of each predictor; the PIT,
KS and CalErr values use the exact predictive CDF (the mixture CDF for ensembles). CalErr is the *mean* squared gap
over 100 nominal levels (the plan's formula sums over levels, which only rescales it).

---

## 1. Summary of the pre-registered hypotheses

Decision rule (docs/hypotheses.md §3): *confirmed* only if M2 is ahead in **every** seed **and** the 95% bootstrap CI of
the mean advantage is strictly above zero. With 5 seeds the bootstrap is resampling only five numbers, so a CI is a
coarse summary.

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** M2's 90% coverage is closer to nominal than M1's under shift | **Partly confirmed: 4 of 6 shift scenarios** | Confirmed on Energy and Concrete (extrapolation and noise). **Not confirmed on either Diabetes shift:** the mean advantage is positive for noise shift (+0.094, CI [+0.045, +0.134]) but not in every seed, and the extrapolation CI includes 0. |
| **H2** M2 has lower NLL than M1 on shifted data | **Confirmed in 6 of 6 shift scenarios** | All seeds positive and CI above 0 everywhere. Most of the magnitude comes from M1's catastrophic overconfidence outliers (e.g. Energy extrapolation NLL 142.6 ± 136.3 for M1 vs 2.65 ± 0.47 for M2). On in-distribution data H2 is mixed (Diabetes +0.11 confirmed; Energy −0.10 and Concrete +0.03 not confirmed), which matches the hypothesis being about shifted data. |
| **H3** M4 is calibrated (~95%) in-distribution but degrades under instrument shift, while M2 keeps higher coverage | **Not confirmed** under the rule below; the direction of the effect is partly supported | See §4. |

Important context for H1 and H2: M2 is not uniformly the best model. The linear Bayesian baseline M4 matches or beats M2 on
NLL for all three Diabetes scenarios, ties it on Energy extrapolation (2.659 vs 2.645) and beats it clearly on Concrete
noise shift (4.06 vs 8.37), and the homoscedastic M0 beats M2 on Concrete noise shift (7.75 vs 8.37). The ensemble fixes
the single-network failure; it does not make the uncertainty correct (M2 coverage is 0.65 on Energy noise shift and
0.47 on Concrete noise shift against a nominal 0.90).

---

## 2. Experiment A: synthetic heteroscedastic benchmark

`y = sin(3x) + eps`, `sigma(x) = 0.05 + 0.2|x|`; train on `[-1, 1]`, extrapolate to `[-2,-1] ∪ [1,2]`; 5 seeds.
NN hyper-parameters fixed a priori (width 64, tanh, lr 0.01, weight decay 1e-4); BLR `(alpha, beta)` tuned on validation.

| Model | Domain | NLL | CRPS | Cov 90% | Sharpness | CalErr |
|---|---|---|---|---|---|---|
| M0 Homoscedastic | In-distribution | -0.344 ± 0.036 | 0.094 ± 0.003 | 0.875 ± 0.013 | 0.160 ± 0.011 | 0.001 ± 0.000 |
|  | Extrapolation | 4.965 ± 0.683 | 0.365 ± 0.014 | 0.353 ± 0.047 | 0.160 ± 0.011 | 0.035 ± 0.005 |
| M1 Single heteroscedastic | In-distribution | -0.478 ± 0.039 | 0.092 ± 0.003 | 0.895 ± 0.026 | 0.164 ± 0.004 | 0.000 ± 0.000 |
|  | Extrapolation | 1.334 ± 0.288 | 0.355 ± 0.030 | 0.653 ± 0.057 | 0.376 ± 0.018 | 0.011 ± 0.003 |
| M2 Deep ensemble (M=5) | In-distribution | -0.486 ± 0.034 | 0.092 ± 0.003 | 0.898 ± 0.020 | 0.165 ± 0.003 | 0.001 ± 0.001 |
|  | Extrapolation | 1.168 ± 0.106 | 0.348 ± 0.014 | 0.665 ± 0.050 | 0.377 ± 0.025 | 0.010 ± 0.003 |
| M4 Bayesian lin. reg. (RBF) | In-distribution | -0.335 ± 0.051 | 0.094 ± 0.003 | 0.851 ± 0.007 | 0.148 ± 0.003 | 0.001 ± 0.000 |
|  | Extrapolation | 7.926 ± 5.176 | 0.575 ± 0.065 | 0.301 ± 0.187 | 0.262 ± 0.149 | 0.056 ± 0.013 |

Findings:
1. **Noise recovery.** In-distribution, the learned aleatoric standard deviation matches the true `sigma(x)` with RMSE
   0.021 ± 0.002 (M1) and 0.018 ± 0.002 (M2), against a true range of 0.05–0.25.
2. **In-distribution calibration** is good for the three neural models (90% coverage 0.875–0.898); M4 is 0.851.
3. **Extrapolation fails for every model.** Coverage is 0.353 (M0), 0.653 (M1), 0.665 (M2) and 0.301 (M4), all well below
   0.90. M2 has the lowest NLL (1.168 ± 0.106 vs 1.334 ± 0.288 for M1) and a smaller seed-to-seed spread, but the
   differences in coverage and CRPS (0.348 vs 0.355) are small. No formal test was run on this experiment.
4. **Epistemic uncertainty grows, but not enough.** M2's epistemic standard deviation is 5.98 ± 2.19 times larger
   outside `[-1, 1]` than inside, yet coverage stays at 0.67: the members extrapolate the mean along similar smooth
   trends (the figure shows the mean missing the sine's turn at `x ≈ ±1.57`), so their agreement is not evidence of being
   right. The learned aleatoric σ also grows with |x| (as the true σ does), so part of the widening is not epistemic.
5. PIT histograms (`figures/synthetic_m1_ood_pit_histogram.png`, `..._m2_...`) and the reliability diagram are
   U-shaped (over-confident) for both, as expected from the coverage numbers.

---

## 3. Experiment B: tabular benchmarks (Diabetes, Energy, Concrete; real data)

Diabetes comes from scikit-learn, Energy Efficiency and Concrete Compressive Strength from the UCI repository
(downloaded once and cached as CSV; the loaders raise an error if the real data cannot be read). 70/15/15 random split
for in-distribution and noise shift. Extrapolation: the 70% lowest values of the feature most correlated with the target
form train/validation, the top 30% are the test set. M4 is linear Bayesian regression with `(alpha, beta)` tuned on
validation NLL.


#### diabetes (lr=0.01, weight decay=0.01)

| Scenario | Metric | M0 | M1 | M2 (M=5) | M4 (BLR) |
|---|---|---|---|---|---|
| In-distribution | NLL | 5.497 ± 0.085 | 5.567 ± 0.091 | 5.458 ± 0.034 | 5.440 ± 0.032 |
|  | Cov 90% | 0.887 ± 0.070 | 0.827 ± 0.035 | 0.869 ± 0.030 | 0.884 ± 0.039 |
|  | CRPS | 32.96 ± 2.13 | 33.65 ± 1.11 | 32.42 ± 0.91 | 31.74 ± 1.12 |
|  | Sharpness | 54.08 ± 4.51 | 51.02 ± 4.08 | 53.11 ± 2.56 | 55.19 ± 0.43 |
| Extrapolation | NLL | 5.844 ± 0.112 | 5.962 ± 0.183 | 5.637 ± 0.046 | 5.615 ± 0.049 |
|  | Cov 90% | 0.756 ± 0.044 | 0.901 ± 0.070 | 0.941 ± 0.011 | 0.795 ± 0.069 |
|  | CRPS | 40.96 ± 1.90 | 48.03 ± 13.05 | 36.90 ± 1.35 | 36.21 ± 1.01 |
|  | Sharpness | 50.45 ± 3.40 | 110.60 ± 55.69 | 77.13 ± 5.23 | 51.75 ± 6.89 |
| Noise shift (0.5 std) | NLL | 5.600 ± 0.109 | 5.831 ± 0.200 | 5.552 ± 0.078 | 5.527 ± 0.040 |
|  | Cov 90% | 0.836 ± 0.065 | 0.758 ± 0.035 | 0.857 ± 0.045 | 0.890 ± 0.028 |
|  | CRPS | 35.87 ± 2.82 | 37.06 ± 1.25 | 35.11 ± 2.05 | 34.30 ± 1.62 |
|  | Sharpness | 54.08 ± 4.51 | 50.34 ± 4.47 | 54.78 ± 3.06 | 56.97 ± 1.22 |

#### energy (lr=0.01, weight decay=0.0001)

| Scenario | Metric | M0 | M1 | M2 (M=5) | M4 (BLR) |
|---|---|---|---|---|---|
| In-distribution | NLL | 1.068 ± 0.249 | 1.086 ± 0.180 | 1.188 ± 0.160 | 2.460 ± 0.036 |
|  | Cov 90% | 0.898 ± 0.024 | 0.933 ± 0.045 | 0.966 ± 0.020 | 0.893 ± 0.015 |
|  | CRPS | 0.41 ± 0.11 | 0.62 ± 0.12 | 0.65 ± 0.09 | 1.53 ± 0.07 |
|  | Sharpness | 0.74 ± 0.20 | 1.11 ± 0.17 | 1.35 ± 0.16 | 3.21 ± 0.02 |
| Extrapolation | NLL | 18.062 ± 10.568 | 142.621 ± 136.293 | 2.645 ± 0.474 | 2.659 ± 0.152 |
|  | Cov 90% | 0.311 ± 0.116 | 0.261 ± 0.128 | 0.770 ± 0.158 | 0.896 ± 0.057 |
|  | CRPS | 2.99 ± 0.96 | 2.87 ± 2.18 | 1.76 ± 0.52 | 2.03 ± 0.33 |
|  | Sharpness | 0.75 ± 0.08 | 0.37 ± 0.08 | 2.51 ± 1.35 | 3.35 ± 0.10 |
| Noise shift (0.5 std) | NLL | 60.215 ± 11.905 | 167.381 ± 112.506 | 3.893 ± 0.521 | 3.978 ± 0.717 |
|  | Cov 90% | 0.129 ± 0.038 | 0.266 ± 0.079 | 0.652 ± 0.064 | 0.769 ± 0.125 |
|  | CRPS | 5.67 ± 0.70 | 3.60 ± 0.53 | 2.61 ± 0.16 | 8.98 ± 10.34 |
|  | Sharpness | 0.74 ± 0.20 | 1.30 ± 0.38 | 2.90 ± 0.45 | 28.68 ± 48.02 |

#### concrete (lr=0.01, weight decay=0.01)

| Scenario | Metric | M0 | M1 | M2 (M=5) | M4 (BLR) |
|---|---|---|---|---|---|
| In-distribution | NLL | 3.266 ± 0.091 | 3.099 ± 0.064 | 3.073 ± 0.068 | 3.797 ± 0.036 |
|  | Cov 90% | 0.885 ± 0.035 | 0.910 ± 0.017 | 0.941 ± 0.021 | 0.924 ± 0.015 |
|  | CRPS | 3.47 ± 0.26 | 3.25 ± 0.25 | 3.18 ± 0.29 | 5.99 ± 0.25 |
|  | Sharpness | 5.86 ± 0.28 | 5.69 ± 0.43 | 5.98 ± 0.33 | 12.01 ± 0.07 |
| Extrapolation | NLL | 4.165 ± 0.175 | 5.050 ± 0.856 | 3.602 ± 0.028 | 4.164 ± 0.042 |
|  | Cov 90% | 0.661 ± 0.037 | 0.651 ± 0.056 | 0.869 ± 0.023 | 0.786 ± 0.008 |
|  | CRPS | 5.37 ± 0.15 | 5.86 ± 0.20 | 4.93 ± 0.11 | 8.03 ± 0.18 |
|  | Sharpness | 5.04 ± 0.36 | 5.65 ± 0.85 | 8.28 ± 0.86 | 10.50 ± 0.12 |
| Noise shift (0.5 std) | NLL | 7.746 ± 0.426 | 23.671 ± 9.362 | 8.366 ± 0.574 | 4.062 ± 0.050 |
|  | Cov 90% | 0.459 ± 0.050 | 0.396 ± 0.060 | 0.467 ± 0.057 | 0.846 ± 0.019 |
|  | CRPS | 11.62 ± 0.79 | 12.96 ± 0.99 | 12.04 ± 0.77 | 7.84 ± 0.34 |
|  | Sharpness | 5.86 ± 0.28 | 5.53 ± 0.40 | 6.51 ± 0.23 | 12.10 ± 0.05 |


#### Hypothesis verdicts (M2 vs M1)

"Advantage" is positive when M2 is better: for H1 it is `|cov90 − 0.9|` of M1 minus that of M2; for H2 it is
`NLL(M1) − NLL(M2)`. "All seeds" means the advantage is positive in each of the 5 seeds.

| Dataset | Scenario | H1 advantage [95% CI] | H1 all seeds | H1 | H2 advantage [95% CI] | H2 all seeds | H2 |
|---|---|---|---|---|---|---|---|
| diabetes | In-distribution | +0.038 [+0.009, +0.059] | no | not confirmed | +0.109 [+0.037, +0.205] | yes | **confirmed** |
| diabetes | Extrapolation | +0.017 [-0.013, +0.045] | no | not confirmed | +0.325 [+0.212, +0.480] | yes | **confirmed** |
| diabetes | Noise shift (0.5 std) | +0.094 [+0.045, +0.134] | no | not confirmed | +0.279 [+0.151, +0.436] | yes | **confirmed** |
| energy | In-distribution | -0.011 [-0.026, +0.004] | no | not confirmed | -0.101 [-0.233, +0.024] | no | not confirmed |
| energy | Extrapolation | +0.460 [+0.333, +0.545] | yes | **confirmed** | +139.976 [+33.559, +267.387] | yes | **confirmed** |
| energy | Noise shift (0.5 std) | +0.386 [+0.290, +0.491] | yes | **confirmed** | +163.488 [+74.966, +270.656] | yes | **confirmed** |
| concrete | In-distribution | -0.022 [-0.044, -0.005] | no | not confirmed | +0.026 [-0.004, +0.057] | no | not confirmed |
| concrete | Extrapolation | +0.215 [+0.163, +0.267] | yes | **confirmed** | +1.448 [+0.741, +2.295] | yes | **confirmed** |
| concrete | Noise shift (0.5 std) | +0.071 [+0.055, +0.083] | yes | **confirmed** | +15.305 [+7.710, +22.789] | yes | **confirmed** |

Observations:
- Under shift, M1 is often badly over-confident on Energy and Concrete (90% coverage 0.26 and 0.27 on Energy, 0.65 and
  0.40 on Concrete for extrapolation and noise), which is what produces the very large M1 NLL values.
- M2 improves calibration relative to M1 but remains under-covering on Energy (0.770 extrapolation, 0.652 noise) and
  Concrete noise shift (0.467), so "closer to nominal" is not the same as "calibrated".
- M4 (a linear model) is competitive or better on Diabetes, Energy extrapolation and Concrete noise shift, and clearly
  worse in-distribution on Energy (NLL 2.46 vs about 1.1) and Concrete (3.80 vs 3.07), where the relationship is
  nonlinear.

---

## 4. Experiment C: simulated TDLAS spectra, held-out instrument

Data are simulated (Beer–Lambert, Lorentzian line, per-device path length, baseline offset and noise; 500 scans of
length 250, 4 devices, regenerated for each seed so the device parameters differ between seeds). Device 0 is the shifted
test instrument; the in-distribution test set is a separate 15% split of devices 1–3, disjoint from train and
validation. Only this one shift type is simulated (not the six scenarios of the original plan). The classical
reference M4 is **Bayesian linear regression** on the raw scan with validation-tuned `(alpha, beta)`; it is not the
Levenberg–Marquardt / Cramér–Rao reference of the plan. The neural models are small MLPs (one hidden layer of 32 tanh
units, no CNN) and are clearly weaker predictors here than M4 (held-out-device RMSE about 38 ppm for M2 and 49 ppm for M1,
against 26 ppm for M4).

| Model | Condition | NLL | CRPS (ppm) | Cov 95% | Sharpness (ppm) | CalErr |
|---|---|---|---|---|---|---|
| M0 | In-distribution test | 4.705 ± 0.302 | 15.16 ± 4.18 | 0.907 ± 0.036 | 21.90 ± 4.61 | 0.004 ± 0.002 |
|  | Held-out device | 5.094 ± 0.354 | 18.47 ± 2.84 | 0.808 ± 0.090 | 21.90 ± 4.61 | 0.052 ± 0.040 |
| M1 | In-distribution test | 5.446 ± 0.133 | 26.20 ± 2.84 | 0.871 ± 0.016 | 42.54 ± 6.37 | 0.002 ± 0.001 |
|  | Held-out device | 5.504 ± 0.160 | 28.64 ± 2.24 | 0.884 ± 0.044 | 45.86 ± 4.72 | 0.017 ± 0.008 |
| M2 | In-distribution test | 5.063 ± 0.107 | 20.12 ± 1.34 | 0.996 ± 0.007 | 51.92 ± 9.27 | 0.006 ± 0.004 |
|  | Held-out device | 5.208 ± 0.150 | 23.49 ± 3.15 | 0.983 ± 0.026 | 56.25 ± 15.43 | 0.031 ± 0.018 |
| M4 | In-distribution test | 4.020 ± 0.260 | 7.51 ± 1.91 | 0.953 ± 0.041 | 13.76 ± 3.29 | 0.003 ± 0.001 |
|  | Held-out device | 5.957 ± 1.987 | 17.47 ± 6.97 | 0.625 ± 0.264 | 13.76 ± 3.43 | 0.167 ± 0.092 |

Per-seed 95% coverage on the held-out device: M4 = 0.96, 0.55, 0.19, 0.82, 0.61 (seeds 42, 123, 456, 789, 1024); M2 =
1.00, 1.00, 0.93, 1.00, 0.98.

**H3 verdict: not confirmed.** The components, with thresholds fixed after seeing the data because the pre-registered
text only says "~95%" (`results/spectral/spectral_h3.csv`):

| Component | Result | Pass |
|---|---|---|
| (a) M4 in-distribution 95% coverage within 0.03 of nominal | mean absolute error 0.037 | no |
| (b) M4 coverage drops under shift in every seed | mean drop 0.328, but seed 42 rises (0.95 → 0.96) | no |
| (c) M2 shift coverage closer to nominal than M4 in every seed, CI above 0 | mean advantage +0.291, CI [+0.066, +0.534], but not every seed (seed 42) | no |

Reading: the effect predicted by H3 is present in the means and in 4 of 5 seeds. M4 is sharp (14 ppm) and accurate
in-distribution (NLL 4.02, CRPS 7.5 ppm), and its coverage then collapses on the held-out device (0.625 ± 0.264). M2
keeps coverage near 0.98, but only through intervals about four times wider (sharpness 56 ppm), and its PIT histogram
is concentrated in the centre (over-dispersed; `figures/spectral_m2_device_shift_pit.png`). Both M0 and M4 beat M2 on
CRPS under shift, and M0 has the lowest NLL under shift (5.094 vs 5.208 for M2 and 5.957 for M4). The M4 PIT
(`figures/spectral_m4_device_shift_pit.png`) is skewed towards 1: the held-out device is biased, not just noisier. With
only one simulated shift type, five seeds and an under-tuned neural baseline, this experiment does not support a general
claim about ML versus classical methods.

---

## 5. Ablations (Diabetes, extrapolation shift, 5 seeds)

Base configuration: width 32, lr 0.01, weight decay 0.01 (the validation choice for Diabetes).

| Ensemble size M | NLL | CRPS | Cov 90% | Sharpness | CalErr |
|---|---|---|---|---|---|
| 1 | 5.825 ± 0.081 | 42.64 ± 3.77 | 0.844 ± 0.030 | 68.8 ± 6.0 | 0.0168 ± 0.0128 |
| 3 | 5.654 ± 0.062 | 37.55 ± 1.67 | 0.916 ± 0.038 | 74.9 ± 11.2 | 0.0042 ± 0.0039 |
| 5 | 5.637 ± 0.046 | 36.90 ± 1.35 | 0.941 ± 0.011 | 77.1 ± 5.2 | 0.0030 ± 0.0020 |
| 10 | 5.631 ± 0.033 | 36.47 ± 0.71 | 0.955 ± 0.018 | 80.5 ± 6.5 | 0.0015 ± 0.0009 |

| Hidden width (M=5) | NLL | CRPS | Cov 90% | Sharpness | CalErr |
|---|---|---|---|---|---|
| 16 | 5.630 ± 0.044 | 36.66 ± 1.34 | 0.950 ± 0.010 | 77.7 ± 3.2 | 0.0023 ± 0.0017 |
| 32 | 5.637 ± 0.046 | 36.90 ± 1.35 | 0.941 ± 0.011 | 77.1 ± 5.2 | 0.0030 ± 0.0020 |
| 64 | 5.672 ± 0.072 | 37.74 ± 1.84 | 0.923 ± 0.014 | 79.8 ± 12.3 | 0.0030 ± 0.0013 |
| 128 | 5.693 ± 0.038 | 38.29 ± 1.15 | 0.923 ± 0.025 | 79.7 ± 10.1 | 0.0026 ± 0.0013 |

| Weight decay (M=5) | NLL | CRPS | Cov 90% | Sharpness | CalErr |
|---|---|---|---|---|---|
| 0.0 | 5.650 ± 0.046 | 37.40 ± 1.46 | 0.949 ± 0.007 | 80.4 ± 5.1 | 0.0036 ± 0.0025 |
| 0.0001 | 5.650 ± 0.046 | 37.39 ± 1.46 | 0.949 ± 0.007 | 80.3 ± 5.1 | 0.0036 ± 0.0025 |
| 0.01 | 5.637 ± 0.046 | 36.90 ± 1.35 | 0.941 ± 0.011 | 77.1 ± 5.2 | 0.0030 ± 0.0020 |
| 0.1 | 5.554 ± 0.019 | 35.43 ± 0.86 | 0.901 ± 0.021 | 63.0 ± 1.9 | 0.0047 ± 0.0023 |

- NLL and calibration error improve with ensemble size, with diminishing returns after M=3. Coverage rises monotonically
  and **overshoots** nominal at M=10 (0.955); M=3 (0.916) is the closest to 0.90.
- Wider networks do not help (NLL 5.630 at width 16, 5.693 at width 128).
- Weight decay 0.1 gives the best test NLL (5.554) and coverage closest to nominal (0.901). This is not used for
  selection: the validation-selected value stays 0.01.

---

## 6. Runtime (CPU, float64, 1 thread, median of 7 blocks after warm-up)

| Operation | UCNet (ms) | PyTorch (ms) | UCNet / PyTorch |
|---|---|---|---|
| 2-layer tanh MLP 128-128-128, batch 64, fwd+bwd | 0.267 | 0.316 | 0.84× |
| Conv1D (C8->16, L200, K7, batch 32) fwd+bwd | 9.740 | 1.105 | 8.82× |
| Adam step (10,000 parameters) | 0.058 | 0.061 | 0.94× |

These are micro-benchmarks on small shapes and mostly measure framework overhead. The pure-NumPy engine is on par
with PyTorch for a small dense MLP and an Adam step, and about 8.8× slower for `Conv1D`.

---

## 7. Limitations

- Five seeds; bootstrap intervals over five per-seed values are coarse.
- NLL, CRPS and coverage for ensembles use the moment-matched Gaussian, not the exact mixture; only PIT/KS/CalErr use the mixture CDF.
- The spectral data are simulated, with one shift type and a weak neural baseline; M4 is BLR, not LM + Cramér–Rao.
- Hyper-parameters were tuned once per dataset on a single in-distribution validation split, then reused for shifted scenarios.
- The log-variance output is clamped to `[-10, 5]` in both training and prediction; the clamp has zero gradient outside the range.
- Tabular and spectral models use MLPs only; `Conv1D` and `LayerNorm` are validated by gradient checks and PyTorch parity tests but not used in the experiments.

---

## 8. Corrections to the earlier version of this report

An earlier version of this report and of the README reported different numbers and stronger conclusions. A code review found
and the following changes fixed:

1. `DeepEnsemble.predict` did not apply the log-variance clamp used in training, giving variances up to 1e37 under shift
   (visible as Concrete noise-shift sharpness ≈ 9e8). It now clamps identically.
2. `mixture_cdf` broadcast a flat target vector into an N×N matrix, so the synthetic-experiment PIT/KS/CalErr values were
   computed from N² meaningless values. Fixed and guarded by a shape check; PIT now uses the exact mixture CDF everywhere.
3. The "Concrete" rows were produced by a silent synthetic fallback. Loaders now raise on failure; Concrete is the real
   UCI data and all earlier Concrete numbers were discarded.
4. H3 was reported as confirmed after being re-defined in terms of NLL/CRPS. The criterion is now the pre-registered
   coverage-based one (components in §4) and the verdict is "not confirmed".
5. The spectral in-distribution "test" set was the validation set; M4 had untuned hyper-parameters. Both fixed; results
   now use 5 seeds.
6. Early stopping restored the best weights only when patience ran out; `Tensor.backward` double-counted shared
   intermediate nodes on repeated calls; `clip_grad_norm` failed on read-only/aliased gradients; the runtime benchmark
   was not like-for-like. All fixed, with regression tests in `tests/test_regressions.py`.
