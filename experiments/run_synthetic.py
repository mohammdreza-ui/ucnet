"""
Experiment A: Synthetic Heteroscedastic Regression Benchmark (system validation with known truth).
y = sin(3x) + eps, eps ~ N(0, (0.05 + 0.2|x|)^2); train on [-1, 1], test in-distribution and on [-2,-1] U [1,2].
Checks noise recovery sigma(x) and the behaviour of epistemic uncertainty under extrapolation.
Compares M0 (homoscedastic), M1 (single heteroscedastic), M2 (deep ensemble, M=5) and M4 (RBF BLR) over 5 seeds.
Hyper-parameters are fixed a priori for the NN models (lr=0.01, weight_decay=1e-4); BLR (alpha, beta) is tuned on
validation NLL. Figures use seed 42.
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
)

from ucnet.data import StandardScaler, generate_synthetic_heteroscedastic_data  # noqa: E402
from ucnet.ensemble import DeepEnsemble  # noqa: E402
from ucnet.metrics import make_original_scale_cdf  # noqa: E402
from ucnet.runlog import write_manifest  # noqa: E402
from ucnet.viz import plot_1d_uncertainty, plot_calibration_curve, plot_pit_histogram  # noqa: E402

SEEDS = (42, 123, 456, 789, 1024)
MODELS = ("M0", "M1", "M2", "M4")
KEYS = ("nll", "crps", "cov90", "sharp", "calerr")


def run_synthetic_experiment(
    results_dir: str = "results/synthetic",
    seeds: tuple[int, ...] = SEEDS,
    num_ensemble_members: int = 5,
    epochs: int = 150,
) -> dict:
    out_dir = Path(results_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir, "synthetic", {"seeds": list(seeds), "epochs": epochs, "hidden": 64, "activation": "tanh",
                                          "lr": 0.01, "weight_decay": 1e-4, "ensemble_size": num_ensemble_members,
                                          "n_train": 600, "n_test_in": 300, "n_test_ood": 300})

    print("\n==================================================================")
    print(f"  Running Experiment A: Synthetic Benchmark (seeds={list(seeds)})")
    print("==================================================================")

    res = {m: {d: {k: [] for k in KEYS} for d in ("in", "ood")} for m in MODELS}
    extra = {"sigma_rmse_m1": [], "sigma_rmse_m2": [], "epi_growth_m2": []}
    fig_payload = None
    rows: list[dict] = []

    for seed in seeds:
        data = generate_synthetic_heteroscedastic_data(n_train=600, n_test_in=300, n_test_ood=300, seed=seed)
        in_split, ood_split = data["in_split"], data["ood_split"]
        sx, sy = StandardScaler().fit(in_split.X_train), StandardScaler().fit(in_split.y_train)
        X_tr, y_tr = sx.transform(in_split.X_train), sy.transform(in_split.y_train)
        X_val, y_val = sx.transform(in_split.X_val), sy.transform(in_split.y_val)
        X_in, X_ood = sx.transform(in_split.X_test), sx.transform(ood_split.X_test)
        kw = dict(hidden=64, activation="tanh", lr=0.01, weight_decay=1e-4, epochs=epochs, patience=25)

        m1, _ = train_probabilistic_member(1, X_tr, y_tr, X_val, y_val, seed=seed, **kw)
        members = [train_probabilistic_member(1, X_tr, y_tr, X_val, y_val, seed=seed * 100 + m, **kw)[0]
                   for m in range(num_ensemble_members)]
        models = {
            "M0": train_homoscedastic(1, X_tr, y_tr, X_val, y_val, seed=seed, **kw),
            "M1": DeepEnsemble([m1]),
            "M2": DeepEnsemble(members),
            "M4": tune_blr(X_tr, y_tr, X_val, y_val, basis="rbf", rbf_gamma=1.5),
        }
        y_scale = float(sy.scale_[0, 0])
        ens_cache = {}
        for name, model in models.items():
            for dom, X_ev, split in (("in", X_in, in_split), ("ood", X_ood, ood_split)):
                rep, mu, std = evaluate_predictor(model, X_ev, split.y_test, sy)
                for k, v in (("nll", rep.nll), ("crps", rep.crps), ("cov90", rep.coverage_90),
                             ("sharp", rep.sharpness), ("calerr", rep.calibration_error)):
                    res[name][dom][k].append(v)
                rows.append({"seed": seed, "model": name, "domain": dom, "nll": rep.nll, "crps": rep.crps,
                             "cov90": rep.coverage_90, "sharpness": rep.sharpness, "cal_err": rep.calibration_error,
                             "ks_stat": rep.ks_statistic, "ks_p": rep.ks_pvalue})
                if name in ("M1", "M2"):
                    p = model.predict(X_ev)
                    ens_cache[(name, dom)] = p
                if seed == seeds[0]:
                    cdf = make_original_scale_cdf(model.mixture_cdf, X_ev, float(sy.mean_[0, 0]), y_scale)
                    res.setdefault("_pit", {})[(name, dom)] = np.clip(cdf(split.y_test.ravel()), 1e-9, 1 - 1e-9)
                    res.setdefault("_pred", {})[(name, dom)] = (mu, std, np.sqrt(model.predict(X_ev).aleatoric) * y_scale
                                                               if name == "M2" else None)

        # noise recovery and epistemic growth (original units)
        for name in ("M1", "M2"):
            ale = np.sqrt(ens_cache[(name, "in")].aleatoric) * y_scale
            extra[f"sigma_rmse_{name.lower()}"].append(float(np.sqrt(np.mean((ale - data["true_sigma_in"]) ** 2))))
            if name == "M2":  # a single model has no epistemic term
                e_in = np.sqrt(ens_cache[(name, "in")].epistemic).mean()
                e_ood = np.sqrt(ens_cache[(name, "ood")].epistemic).mean()
                extra["epi_growth_m2"].append(float(e_ood / max(e_in, 1e-12)))
        if seed == seeds[0]:
            fig_payload = (in_split, ood_split, ens_cache[("M2", "in")], ens_cache[("M2", "ood")], y_scale)
        print(f"  seed {seed}: " + " | ".join(
            f"{m} in/ood cov90={res[m]['in']['cov90'][-1]:.2f}/{res[m]['ood']['cov90'][-1]:.2f}" for m in MODELS))

    def stat(a):
        return f"{np.mean(a):.3f}+/-{np.std(a):.3f}"

    print("\n" + "-" * 112)
    print(f"{'Model':<6} | {'Domain':<6} | {'NLL':<14} | {'CRPS':<14} | {'Cov90':<14} | {'Sharp':<14} | {'CalErr':<14}")
    print("-" * 112)
    summary = []
    for m in MODELS:
        for d in ("in", "ood"):
            r = res[m][d]
            print(f"{m:<6} | {d:<6} | {stat(r['nll']):<14} | {stat(r['crps']):<14} | {stat(r['cov90']):<14} | "
                  f"{stat(r['sharp']):<14} | {stat(r['calerr']):<14}")
            row = {"model": m, "domain": d}
            for k in KEYS:
                row[f"{k}_mean"], row[f"{k}_std"] = float(np.mean(r[k])), float(np.std(r[k]))
            summary.append(row)
    print("-" * 112)
    print(f"Aleatoric sigma(x) recovery RMSE (in-dist, orig units): M1 {stat(extra['sigma_rmse_m1'])}, "
          f"M2 {stat(extra['sigma_rmse_m2'])}")
    print(f"Epistemic std growth OOD/in-dist (M2): {stat(extra['epi_growth_m2'])}")

    for fname, rws in (("synthetic_summary.csv", summary), ("synthetic_per_seed.csv", rows)):
        with open(out_dir / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rws[0].keys()))
            w.writeheader()
            w.writerows(rws)
    with open(out_dir / "synthetic_extra.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std"])
        for k, v in extra.items():
            w.writerow([k, np.mean(v), np.std(v)])

    # ---- figures (seed 42) ----
    in_split, ood_split, p_in, p_ood, y_scale = fig_payload
    pred = res["_pred"]
    X_full = np.concatenate([in_split.X_test, ood_split.X_test])
    y_full = np.concatenate([in_split.y_test, ood_split.y_test])
    mu_full = np.concatenate([pred[("M2", "in")][0], pred[("M2", "ood")][0]])
    std_full = np.concatenate([pred[("M2", "in")][1], pred[("M2", "ood")][1]])
    ale_full = np.concatenate([pred[("M2", "in")][2], pred[("M2", "ood")][2]])
    fig = plot_1d_uncertainty(X_full, y_full, mu_full, std_full, aleatoric_std=ale_full, x_train=in_split.X_train,
                              y_train=in_split.y_train, save_path=figures_dir / "synthetic_ensemble_uncertainty_breakdown.png",
                              title="Deep Ensemble (M2): in-distribution and extrapolation")
    plt.close(fig)
    for name in ("M1", "M2"):
        fig = plot_pit_histogram(res["_pit"][(name, "ood")], save_path=figures_dir / f"synthetic_{name.lower()}_ood_pit_histogram.png",
                                 title=f"{name} PIT, extrapolation (seed {seeds[0]})")
        plt.close(fig)
    fig = plot_calibration_curve(res["_pit"][("M2", "ood")], save_path=figures_dir / "synthetic_m2_calibration_curve.png",
                                 title="Deep Ensemble reliability diagram (extrapolation)")
    plt.close(fig)
    print(f"Figures saved to {figures_dir.absolute()}")
    return {"summary": summary, "extra": extra}


if __name__ == "__main__":
    run_synthetic_experiment()
