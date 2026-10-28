"""
Ablation studies (Diabetes, extrapolation shift, 5 seeds, mean +/- std over seeds):
1. Ensemble size M in {1, 3, 5, 10}
2. Network width in {16, 32, 64, 128}   (M=5 ensemble)
3. Weight decay in {0, 1e-4, 1e-2, 1e-1} (M=5 ensemble)
Base configuration: width 32, lr 0.01, weight_decay 0.01 (the value selected on validation for Diabetes in
run_uci.py). Test sets are never used for selection.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import evaluate_predictor, train_probabilistic_member  # noqa: E402

from ucnet.data import (  # noqa: E402
    StandardScaler,
    load_diabetes_dataset,
    make_extrapolation_shift_split,
)
from ucnet.ensemble import DeepEnsemble  # noqa: E402
from ucnet.runlog import write_manifest  # noqa: E402

SEEDS = (42, 123, 456, 789, 1024)
BASE = {"hidden": 32, "weight_decay": 1e-2, "lr": 0.01}


def _prepare(raw_X: np.ndarray, raw_y: np.ndarray, seed: int):
    split = make_extrapolation_shift_split(raw_X, raw_y, seed=seed)
    sx, sy = StandardScaler().fit(split.X_train), StandardScaler().fit(split.y_train)
    return (sx.transform(split.X_train), sy.transform(split.y_train).reshape(-1, 1),
            sx.transform(split.X_val), sy.transform(split.y_val).reshape(-1, 1),
            sx.transform(split.X_test), split.y_test.ravel(), sy)


def _metrics(model, X_te, y_te, sy) -> dict[str, float]:
    rep, _, _ = evaluate_predictor(model, X_te, y_te, sy)
    return {"nll": rep.nll, "crps": rep.crps, "cov90": rep.coverage_90, "sharpness": rep.sharpness,
            "calerr": rep.calibration_error}


def _aggregate(name: str, param, per_seed: list[dict[str, float]]) -> dict:
    row = {"ablation": name, "param": param, "n_seeds": len(per_seed)}
    for k in per_seed[0]:
        vals = [d[k] for d in per_seed]
        row[f"{k}_mean"], row[f"{k}_std"] = float(np.mean(vals)), float(np.std(vals))
    return row


def run_all_ablations(results_dir: str = "results/ablations", epochs: int = 100) -> None:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir, "ablations", {"seeds": list(SEEDS), "base": BASE, "epochs": epochs,
                                          "dataset": "diabetes", "shift": "extrapolation"})
    raw_X, raw_y, _ = load_diabetes_dataset()
    in_dim = raw_X.shape[1]
    rows: list[dict] = []

    def train(seed, member_idx, X_tr, y_tr, X_val, y_val, **over):
        cfg = {**BASE, **over}
        return train_probabilistic_member(in_dim, X_tr, y_tr, X_val, y_val, seed=seed * 100 + member_idx,
                                          epochs=epochs, **cfg)[0]

    # 1. ensemble size
    print("\n--- Ablation 1: ensemble size ---")
    sizes = (1, 3, 5, 10)
    per = {m: [] for m in sizes}
    for seed in SEEDS:
        X_tr, y_tr, X_val, y_val, X_te, y_te, sy = _prepare(raw_X, raw_y, seed)
        members = [train(seed, m, X_tr, y_tr, X_val, y_val) for m in range(max(sizes))]
        for m in sizes:
            per[m].append(_metrics(DeepEnsemble(members[:m]), X_te, y_te, sy))
    for m in sizes:
        rows.append(_aggregate("ensemble_size", m, per[m]))
        r = rows[-1]
        print(f"M={m:2d} | NLL={r['nll_mean']:.3f}+/-{r['nll_std']:.3f} | Cov90={r['cov90_mean']:.3f}+/-{r['cov90_std']:.3f}"
              f" | Sharp={r['sharpness_mean']:.1f} | CalErr={r['calerr_mean']:.4f}")

    # 2/3. width and weight decay (M=5)
    for name, key, values in (("width", "hidden", (16, 32, 64, 128)), ("weight_decay", "weight_decay", (0.0, 1e-4, 1e-2, 1e-1))):
        print(f"\n--- Ablation: {name} (M=5) ---")
        per = {v: [] for v in values}
        for seed in SEEDS:
            X_tr, y_tr, X_val, y_val, X_te, y_te, sy = _prepare(raw_X, raw_y, seed)
            for v in values:
                members = [train(seed, m, X_tr, y_tr, X_val, y_val, **{key: v}) for m in range(5)]
                per[v].append(_metrics(DeepEnsemble(members), X_te, y_te, sy))
        for v in values:
            rows.append(_aggregate(name, v, per[v]))
            r = rows[-1]
            print(f"{name}={v:<7} | NLL={r['nll_mean']:.3f}+/-{r['nll_std']:.3f} | Cov90={r['cov90_mean']:.3f}+/-{r['cov90_std']:.3f}"
                  f" | Sharp={r['sharpness_mean']:.1f} | CalErr={r['calerr_mean']:.4f}")

    out_file = out_dir / "ablations_summary.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for j, name in enumerate(("ensemble_size", "width", "weight_decay")):
        sel = [r for r in rows if r["ablation"] == name]
        x = np.arange(len(sel))
        labels = [str(r["param"]) for r in sel]
        for i, (key, title) in enumerate((("nll", "NLL (lower is better)"), ("cov90", "90% coverage (nominal 0.90)"))):
            ax = axes[i, j]
            ax.errorbar(x, [r[f"{key}_mean"] for r in sel], yerr=[r[f"{key}_std"] for r in sel], fmt="o-", capsize=4)
            if key == "cov90":
                ax.axhline(0.90, color="black", linestyle="--")
            ax.set_xticks(x, labels)
            ax.set_xlabel(name)
            ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle("Ablations on Diabetes (extrapolation shift): mean +/- std over 5 seeds")
    plt.tight_layout()
    fig.savefig(out_dir / "ablation_ensemble_size.png", dpi=200)
    plt.close(fig)
    print(f"Ablation results saved to {out_file.absolute()}")


if __name__ == "__main__":
    run_all_ablations()
