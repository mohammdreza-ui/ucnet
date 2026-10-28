"""
Experiment C: Simulated TDLAS spectra under inter-instrument (device-level) shift.
Tests pre-registered Hypothesis H3 over 5 seeds.

Protocol notes:
- Device 0 is held out as the shifted test instrument. The in-distribution test set is a separate
  split of devices 1-3 that is disjoint from train AND validation (the validation set is only used
  for early stopping, M0's sigma and hyper-parameter tuning).
- (lr, weight_decay) are tuned on the first seed's validation split and shared by M0, M1 and M2.
  M4 is Bayesian linear regression on the raw scan with (alpha, beta) tuned on validation NLL.
  It is NOT the Levenberg-Marquardt + Cramer-Rao classical reference of the original plan.
- The data are simulated here (Beer-Lambert + Lorentzian line); only one shift type (held-out device)
  is generated, not the six scenarios of the original plan.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    evaluate_predictor,
    train_homoscedastic,
    train_probabilistic_member,
    tune_blr,
    tune_mlp_hyperparams,
)

from ucnet.data import (  # noqa: E402
    StandardScaler,
    generate_simulated_spectral_dataset,
    save_spectral_contract_npz,
    split_by_device,
)
from ucnet.ensemble import DeepEnsemble  # noqa: E402
from ucnet.metrics import hypothesis_verdict, make_original_scale_cdf  # noqa: E402
from ucnet.runlog import write_manifest  # noqa: E402
from ucnet.viz import plot_calibration_curve, plot_pit_histogram  # noqa: E402

SEEDS = (42, 123, 456, 789, 1024)
MODELS = ("M0", "M1", "M2", "M4")
KEYS = ("nll", "crps", "cov95", "sharp", "calerr")


def run_spectral_experiment(
    results_dir: str = "results/spectral",
    data_path: str = "data/spectral_dataset.npz",
    n_samples: int = 500,
    scan_length: int = 250,
    n_devices: int = 4,
    epochs: int = 80,
    seeds: tuple[int, ...] = SEEDS,
) -> dict:
    out_dir = Path(results_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir, "spectral", {
        "seeds": list(seeds), "n_samples": n_samples, "scan_length": scan_length, "n_devices": n_devices,
        "epochs": epochs, "patience": 15, "test_device": 0, "ensemble_size": 5,
    })

    print("\n==================================================================")
    print("  Running Experiment C: TDLAS Spectral Benchmark (Device Shift, 5 seeds)")
    print("==================================================================")

    res = {m: {c: {k: [] for k in KEYS} for c in ("in", "shift")} for m in MODELS}
    per_seed_rows: list[dict] = []
    hp: dict = {}
    pit_cache: dict = {}

    for si, seed in enumerate(seeds):
        X_raw, y_raw, dev_raw, scen_raw, meta = generate_simulated_spectral_dataset(
            n_samples=n_samples, scan_length=scan_length, n_devices=n_devices, seed=seed)
        if si == 0:
            save_spectral_contract_npz(data_path, X_raw, y_raw, dev_raw, scen_raw, meta)
            print(f"Spectral contract dataset (seed {seed}) serialized to {data_path}")

        split = split_by_device(X_raw, y_raw, dev_raw, test_device=0, val_ratio=0.15, test_in_ratio=0.15, seed=seed)
        assert split.X_test_in is not None and split.y_test_in is not None
        scaler_x = StandardScaler().fit(split.X_train)
        scaler_y = StandardScaler().fit(split.y_train)
        X_tr, y_tr = scaler_x.transform(split.X_train), scaler_y.transform(split.y_train).reshape(-1, 1)
        X_val, y_val = scaler_x.transform(split.X_val), scaler_y.transform(split.y_val).reshape(-1, 1)
        eval_sets = {
            "in": (scaler_x.transform(split.X_test_in), split.y_test_in.ravel()),
            "shift": (scaler_x.transform(split.X_test), split.y_test.ravel()),
        }

        if si == 0:
            hp = tune_mlp_hyperparams(scan_length, X_tr, y_tr, X_val, y_val, seed=seed,
                                      activation="tanh", epochs=epochs)
            print(f"Selected on validation: lr={hp['lr']}, weight_decay={hp['weight_decay']} "
                  f"(val NLL {hp['val_nll']:.4f})")
        kw = {"activation": "tanh", "lr": hp["lr"], "weight_decay": hp["weight_decay"], "epochs": epochs}

        m0 = train_homoscedastic(scan_length, X_tr, y_tr, X_val, y_val, seed=seed, **kw)
        m1, _ = train_probabilistic_member(scan_length, X_tr, y_tr, X_val, y_val, seed=seed, **kw)
        members = [train_probabilistic_member(scan_length, X_tr, y_tr, X_val, y_val, seed=seed * 100 + m, **kw)[0]
                   for m in range(5)]
        models = {
            "M0": m0, "M1": DeepEnsemble([m1]), "M2": DeepEnsemble(members),
            "M4": tune_blr(X_tr, y_tr, X_val, y_val, basis="linear",
                           alphas=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)),
        }
        for name, model in models.items():
            for cond, (X_ev, y_ev) in eval_sets.items():
                rep, mu, std = evaluate_predictor(model, X_ev, y_ev, scaler_y)
                vals = {"nll": rep.nll, "crps": rep.crps, "cov95": rep.coverage_95,
                        "sharp": rep.sharpness, "calerr": rep.calibration_error}
                for k, v in vals.items():
                    res[name][cond][k].append(v)
                per_seed_rows.append({"seed": seed, "model": name, "condition": cond, **vals,
                                      "cov90": rep.coverage_90, "ks_stat": rep.ks_statistic, "ks_p": rep.ks_pvalue,
                                      "z_mean": rep.z_mean, "z_var": rep.z_var,
                                      "rmse": float(np.sqrt(np.mean((mu.ravel() - y_ev) ** 2)))})
                if si == 0 and cond == "shift" and name in ("M2", "M4"):
                    cdf = make_original_scale_cdf(model.mixture_cdf, X_ev, float(scaler_y.mean_[0, 0]),
                                                  float(scaler_y.scale_[0, 0]))
                    pit_cache[name] = np.clip(cdf(y_ev), 1e-9, 1 - 1e-9)
        print(f"  seed {seed}: " + " | ".join(
            f"{m} in/shift cov95={res[m]['in']['cov95'][-1]:.2f}/{res[m]['shift']['cov95'][-1]:.2f}"
            f" NLL={res[m]['in']['nll'][-1]:.2f}/{res[m]['shift']['nll'][-1]:.2f}" for m in MODELS))

    # ---- summary table ----
    def stat(a: list[float]) -> str:
        return f"{np.mean(a):.3f} +/- {np.std(a):.3f}"

    print("\n" + "-" * 118)
    print(f"{'Model':<6} | {'Cond':<6} | {'NLL':<16} | {'CRPS':<16} | {'Cov95':<16} | {'Sharp':<16} | {'CalErr':<16}")
    print("-" * 118)
    summary_rows = []
    for m in MODELS:
        for cond in ("in", "shift"):
            r = res[m][cond]
            print(f"{m:<6} | {cond:<6} | {stat(r['nll']):<16} | {stat(r['crps']):<16} | {stat(r['cov95']):<16} | "
                  f"{stat(r['sharp']):<16} | {stat(r['calerr']):<16}")
            row = {"model": m, "condition": cond}
            for k in KEYS:
                row[f"{k}_mean"], row[f"{k}_std"] = float(np.mean(r[k])), float(np.std(r[k]))
            summary_rows.append(row)
    print("-" * 118)

    # ---- H3 components (thresholds fixed ex post; hypotheses.md only says "~95%") ----
    m4_in, m4_sh = np.array(res["M4"]["in"]["cov95"]), np.array(res["M4"]["shift"]["cov95"])
    m2_sh = np.array(res["M2"]["shift"]["cov95"])
    a_close = float(np.mean(np.abs(m4_in - 0.95)))
    b_drop = m4_in - m4_sh
    adv_ret = (np.abs(m4_sh - 0.95) - np.abs(m2_sh - 0.95)).tolist()
    c_mean, c_lo, c_hi, c_ok = hypothesis_verdict(adv_ret)
    h3 = {
        "a_m4_in_cov95_mean_abs_err": a_close, "a_pass": a_close <= 0.03,
        "b_m4_cov_drop_mean": float(np.mean(b_drop)), "b_pass": bool(np.all(b_drop > 0)),
        "c_m2_closer_to_nominal_advantage": c_mean, "c_ci_low": c_lo, "c_ci_high": c_hi, "c_pass": c_ok,
    }
    h3["h3_confirmed"] = bool(h3["a_pass"] and h3["b_pass"] and h3["c_pass"])
    print("\nH3 components (a: M4 in-dist cov95 within 0.03 of 0.95; b: M4 coverage drops under shift in every seed;"
          " c: M2 shift coverage closer to 0.95 than M4, all seeds + CI>0)")
    for k, v in h3.items():
        print(f"  {k}: {v}")

    for name, rows in (("spectral_summary.csv", summary_rows), ("spectral_per_seed.csv", per_seed_rows)):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(out_dir / "spectral_h3.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(h3.keys()))
        w.writeheader()
        w.writerow(h3)

    # ---- figures (seed 42, exact predictive CDFs) ----
    for name, label in (("M4", "M4 (BLR)"), ("M2", "M2 (Deep Ensemble)")):
        fig = plot_pit_histogram(pit_cache[name], save_path=fig_dir / f"spectral_{name.lower()}_device_shift_pit.png",
                                 title=f"{label} PIT, held-out device (seed {seeds[0]})")
        plt.close(fig)
    fig = plot_calibration_curve(pit_cache["M2"], save_path=fig_dir / "spectral_m2_device_shift_calibration_curve.png",
                                 title="Deep Ensemble calibration curve, held-out device")
    plt.close(fig)
    print(f"Spectral summary saved to {out_dir / 'spectral_summary.csv'}")
    return {"summary": summary_rows, "h3": h3}


if __name__ == "__main__":
    run_spectral_experiment()
