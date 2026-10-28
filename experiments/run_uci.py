"""
Experiment B: Tabular Benchmarks (Diabetes, Energy, Concrete) across 5 Random Seeds.
Evaluates In-Distribution, Extrapolation Shift, and Input Noise Shift.
Tests Pre-registered Hypotheses H1 (Coverage) and H2 (NLL) with the decision rule of
docs/hypotheses.md: M2 ahead in every seed AND bootstrap CI of the mean advantage strictly above 0.

Protocol notes:
- (lr, weight_decay) are chosen once per dataset on the in-distribution validation split of the first
  seed, then shared by M0, M1 and all M2 members (equal tuning budget). The test sets are never used.
- M4 is Bayesian linear regression with (alpha, beta) tuned on validation NLL.
- PIT uses each model's exact predictive CDF (mixture CDF for ensembles).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

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
    load_concrete_dataset,
    load_diabetes_dataset,
    load_energy_dataset,
    make_extrapolation_shift_split,
    make_in_distribution_split,
    make_noise_shift_split,
)
from ucnet.ensemble import DeepEnsemble  # noqa: E402
from ucnet.metrics import hypothesis_verdict  # noqa: E402
from ucnet.runlog import write_manifest  # noqa: E402

SEEDS = (42, 123, 456, 789, 1024)
SHIFTS = ("in_distribution", "extrapolation", "noise_shift_0.5")
MODELS = ("M0", "M1", "M2", "M4")


def make_split(raw_X: np.ndarray, raw_y: np.ndarray, shift: str, seed: int):
    if shift == "in_distribution":
        return make_in_distribution_split(raw_X, raw_y, seed=seed)
    if shift == "extrapolation":
        return make_extrapolation_shift_split(raw_X, raw_y, seed=seed)
    if shift == "noise_shift_0.5":
        return make_noise_shift_split(make_in_distribution_split(raw_X, raw_y, seed=seed), noise_level=0.5, seed=seed)
    raise ValueError(f"Unknown shift: {shift}")


def evaluate_tabular_dataset(
    dataset_name: str,
    load_fn: Callable,
    seeds: Sequence[int] = SEEDS,
    epochs: int = 100,
    results_dir: str = "results/tabular",
) -> tuple[list[dict], list[dict]]:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n==================================================================")
    print(f"  Evaluating Tabular Dataset: {dataset_name.upper()} (Seeds: {list(seeds)})")
    print("==================================================================")

    raw_X, raw_y, _ = load_fn()
    in_dim = raw_X.shape[1]
    print(f"Loaded {dataset_name}: X={raw_X.shape}, y range=[{raw_y.min():.3f}, {raw_y.max():.3f}]")

    # Validation-only hyperparameter selection on the first seed's in-distribution split
    tune_split = make_in_distribution_split(raw_X, raw_y, seed=seeds[0])
    sx, sy = StandardScaler().fit(tune_split.X_train), StandardScaler().fit(tune_split.y_train)
    hp = tune_mlp_hyperparams(
        in_dim, sx.transform(tune_split.X_train), sy.transform(tune_split.y_train).reshape(-1, 1),
        sx.transform(tune_split.X_val), sy.transform(tune_split.y_val).reshape(-1, 1),
        seed=seeds[0], epochs=epochs,
    )
    print(f"Selected on validation: lr={hp['lr']}, weight_decay={hp['weight_decay']} (val NLL {hp['val_nll']:.4f})")
    kw = {"lr": hp["lr"], "weight_decay": hp["weight_decay"], "epochs": epochs}

    summary_rows: list[dict] = []
    per_seed_rows: list[dict] = []

    for shift in SHIFTS:
        print(f"\n--- Scenario: {shift.upper()} ---")
        res: dict[str, dict[str, list[float]]] = {
            m: {k: [] for k in ("nll", "cov90", "crps", "sharp", "calerr")} for m in MODELS
        }

        for seed in seeds:
            split = make_split(raw_X, raw_y, shift, seed)
            scaler_x = StandardScaler().fit(split.X_train)
            scaler_y = StandardScaler().fit(split.y_train)
            X_tr = scaler_x.transform(split.X_train)
            y_tr = scaler_y.transform(split.y_train).reshape(-1, 1)
            X_val = scaler_x.transform(split.X_val)
            y_val = scaler_y.transform(split.y_val).reshape(-1, 1)
            X_te = scaler_x.transform(split.X_test)
            y_te = split.y_test.ravel()

            m0 = train_homoscedastic(in_dim, X_tr, y_tr, X_val, y_val, seed=seed, **kw)
            m1, _ = train_probabilistic_member(in_dim, X_tr, y_tr, X_val, y_val, seed=seed, **kw)
            members = [
                train_probabilistic_member(in_dim, X_tr, y_tr, X_val, y_val, seed=seed * 100 + m, **kw)[0]
                for m in range(5)
            ]
            models = {
                "M0": m0,
                "M1": DeepEnsemble([m1]),
                "M2": DeepEnsemble(members),
                "M4": tune_blr(X_tr, y_tr, X_val, y_val, basis="linear"),
            }
            for name, model in models.items():
                rep, _, _ = evaluate_predictor(model, X_te, y_te, scaler_y)
                for key, val in (
                    ("nll", rep.nll), ("cov90", rep.coverage_90), ("crps", rep.crps),
                    ("sharp", rep.sharpness), ("calerr", rep.calibration_error),
                ):
                    res[name][key].append(val)
                per_seed_rows.append({
                    "dataset": dataset_name, "shift": shift, "seed": seed, "model": name,
                    "nll": rep.nll, "crps": rep.crps, "cov50": rep.coverage_50, "cov80": rep.coverage_80,
                    "cov90": rep.coverage_90, "cov95": rep.coverage_95, "sharpness": rep.sharpness,
                    "cal_err": rep.calibration_error, "ks_stat": rep.ks_statistic, "ks_p": rep.ks_pvalue,
                    "z_mean": rep.z_mean, "z_var": rep.z_var,
                })
            print(f"  seed {seed}: " + " | ".join(
                f"{m} NLL={res[m]['nll'][-1]:.3f} cov90={res[m]['cov90'][-1]:.3f}" for m in MODELS
            ))

        def stat(arr: list[float]) -> str:
            return f"{np.mean(arr):.3f} +/- {np.std(arr):.3f}"

        # H1: closer to nominal 90% coverage -> advantage = |cov-0.9|(M1) - |cov-0.9|(M2) > 0
        adv_cov = [abs(a - 0.9) - abs(b - 0.9) for a, b in zip(res["M1"]["cov90"], res["M2"]["cov90"])]
        h1_mean, h1_lo, h1_hi, h1_ok = hypothesis_verdict(adv_cov)
        # H2: lower NLL -> advantage = NLL(M1) - NLL(M2) > 0
        adv_nll = [a - b for a, b in zip(res["M1"]["nll"], res["M2"]["nll"])]
        h2_mean, h2_lo, h2_hi, h2_ok = hypothesis_verdict(adv_nll)

        for m in MODELS:
            print(f"{m}: NLL={stat(res[m]['nll']):<18} | Cov90={stat(res[m]['cov90'])} | "
                  f"CRPS={stat(res[m]['crps'])} | Sharp={stat(res[m]['sharp'])} | CalErr={stat(res[m]['calerr'])}")
        print(f"H1 coverage-error advantage: {h1_mean:+.4f} (95% CI [{h1_lo:+.4f}, {h1_hi:+.4f}]) "
              f"all-seeds-positive={all(d > 0 for d in adv_cov)} CONFIRMED={h1_ok}")
        print(f"H2 NLL advantage           : {h2_mean:+.4f} (95% CI [{h2_lo:+.4f}, {h2_hi:+.4f}]) "
              f"all-seeds-positive={all(d > 0 for d in adv_nll)} CONFIRMED={h2_ok}")

        row: dict = {"dataset": dataset_name, "shift": shift, "lr": hp["lr"], "weight_decay": hp["weight_decay"]}
        for m in MODELS:
            tag = m.lower()
            for key in ("nll", "cov90", "crps", "sharp", "calerr"):
                row[f"{tag}_{key}_mean"] = float(np.mean(res[m][key]))
                row[f"{tag}_{key}_std"] = float(np.std(res[m][key]))
        row.update({
            "h1_cov_err_advantage": h1_mean, "h1_ci_low": h1_lo, "h1_ci_high": h1_hi,
            "h1_all_seeds_positive": all(d > 0 for d in adv_cov), "h1_confirmed": h1_ok,
            "h2_nll_advantage": h2_mean, "h2_ci_low": h2_lo, "h2_ci_high": h2_hi,
            "h2_all_seeds_positive": all(d > 0 for d in adv_nll), "h2_confirmed": h2_ok,
        })
        summary_rows.append(row)

    return summary_rows, per_seed_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_all_uci_experiments(results_dir: str = "results/tabular", datasets: Optional[Sequence[str]] = None) -> None:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loaders = {
        "diabetes": load_diabetes_dataset,
        "energy": load_energy_dataset,
        "concrete": load_concrete_dataset,
    }
    chosen = list(datasets) if datasets else list(loaders)
    write_manifest(out_dir, "tabular", {"seeds": list(SEEDS), "datasets": chosen, "shifts": list(SHIFTS),
                                        "epochs": 100, "patience": 15, "ensemble_size": 5})
    all_summary: list[dict] = []
    all_per_seed: list[dict] = []
    for name in chosen:
        s_rows, p_rows = evaluate_tabular_dataset(name, loaders[name], results_dir=results_dir)
        all_summary.extend(s_rows)
        all_per_seed.extend(p_rows)

    _write_csv(out_dir / "tabular_summary.csv", all_summary)
    _write_csv(out_dir / "tabular_per_seed.csv", all_per_seed)
    print(f"\nAll tabular benchmark results saved to {(out_dir / 'tabular_summary.csv').absolute()}")


if __name__ == "__main__":
    run_all_uci_experiments(datasets=sys.argv[1:] or None)
