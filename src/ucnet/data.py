"""
Data engineering, leak-free preprocessing, dataset loaders, and distribution shift generators.
Strictly guarantees no data leakage (scaling statistics computed only on train splits).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass
class DatasetSplit:
    """Container for train, validation, and test splits with features and targets."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: Optional[list[str]] = None
    shift_name: str = "in_distribution"
    # Optional held-out in-distribution test set (never used for fitting, early stopping or tuning)
    X_test_in: Optional[np.ndarray] = None
    y_test_in: Optional[np.ndarray] = None


class StandardScaler:
    """
    Standardize features and targets by removing the mean and scaling to unit variance.
    Strictly fit on training data only to prevent data snooping / leakage.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        self.mean_ = np.mean(X_arr, axis=0, keepdims=True)
        var = np.var(X_arr, axis=0, keepdims=True)
        self.scale_ = np.maximum(np.sqrt(var), self.eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler has not been fitted yet")

        orig_ndim = X.ndim
        X_arr = np.asarray(X, dtype=np.float64)
        if orig_ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        out = (X_arr - self.mean_) / self.scale_
        return out.ravel() if orig_ndim == 1 else out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler has not been fitted yet")

        orig_ndim = X.ndim
        X_arr = np.asarray(X, dtype=np.float64)
        if orig_ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        out = X_arr * self.scale_ + self.mean_
        return out.ravel() if orig_ndim == 1 else out


def make_in_distribution_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetSplit:
    """Standard random in-distribution train / validation / test split."""
    rng = np.random.default_rng(seed)
    N = len(X)
    indices = rng.permutation(N)

    n_train = int(train_ratio * N)
    n_val = int(val_ratio * N)

    idx_train = indices[:n_train]
    idx_val = indices[n_train : n_train + n_val]
    idx_test = indices[n_train + n_val :]

    return DatasetSplit(
        X_train=X[idx_train],
        y_train=y[idx_train],
        X_val=X[idx_val],
        y_val=y[idx_val],
        X_test=X[idx_test],
        y_test=y[idx_test],
        shift_name="in_distribution",
    )


def make_extrapolation_shift_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_idx: Optional[int] = None,
    split_quantile: float = 0.70,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> DatasetSplit:
    """
    Extrapolation distribution shift:
    Selects the feature with highest correlation to target (if not specified).
    Lower 70% of feature values form the train/val split.
    Upper 30% of feature values form the out-of-distribution shifted test set.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y).ravel()

    # If feature not given, choose continuous feature with highest absolute Pearson correlation
    if feature_idx is None:
        candidate_cols = [c for c in range(X_arr.shape[1]) if len(np.unique(X_arr[:, c])) > 5]
        if not candidate_cols:
            candidate_cols = list(range(X_arr.shape[1]))
        corrs = []
        for col in candidate_cols:
            c = float(np.corrcoef(X_arr[:, col], y_arr)[0, 1])
            corrs.append(abs(c) if not np.isnan(c) else 0.0)
        feature_idx = candidate_cols[int(np.argmax(corrs))]

    feat_vals = X_arr[:, feature_idx]
    # Sort indices by feature value to ensure strictly non-empty top fraction
    sorted_idx = np.argsort(feat_vals)
    n_train_val = int(split_quantile * len(feat_vals))

    train_val_idx = sorted_idx[:n_train_val]
    test_idx = sorted_idx[n_train_val:]

    X_train_val = X_arr[train_val_idx]
    y_train_val = y_arr[train_val_idx]
    X_test_shift = X_arr[test_idx]
    y_test_shift = y_arr[test_idx]

    # Split train_val into train and validation
    rng = np.random.default_rng(seed)
    N_tv = len(X_train_val)
    perm = rng.permutation(N_tv)
    n_val = int(val_ratio * N_tv)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    return DatasetSplit(
        X_train=X_train_val[train_idx],
        y_train=y_train_val[train_idx],
        X_val=X_train_val[val_idx],
        y_val=y_train_val[val_idx],
        X_test=X_test_shift,
        y_test=y_test_shift,
        shift_name=f"extrapolation_feat_{feature_idx}",
    )


def make_noise_shift_split(
    split: DatasetSplit,
    noise_level: float = 0.5,
    seed: int = 42,
) -> DatasetSplit:
    """
    Feature noise shift:
    Adds Gaussian noise to test features with amplitude proportional to training std.
    """
    rng = np.random.default_rng(seed)
    train_std = np.std(split.X_train, axis=0, keepdims=True)
    noise = rng.normal(0.0, noise_level * train_std, size=split.X_test.shape)

    return DatasetSplit(
        X_train=split.X_train.copy(),
        y_train=split.y_train.copy(),
        X_val=split.X_val.copy(),
        y_val=split.y_val.copy(),
        X_test=split.X_test + noise,
        y_test=split.y_test.copy(),
        feature_names=split.feature_names,
        shift_name=f"noise_shift_{noise_level}std",
    )


# ---------------------------------------------------------------------------
# Synthetic Dataset Generator (Experiment A)
# ---------------------------------------------------------------------------

def generate_synthetic_heteroscedastic_data(
    n_train: int = 500,
    n_test_in: int = 200,
    n_test_ood: int = 200,
    seed: int = 42,
) -> dict[str, Union[np.ndarray, DatasetSplit]]:
    """
    Generates synthetic 1D regression benchmark with heteroscedastic noise (Plan Section 7.1):
    f(x) = sin(3*x)
    sigma(x) = 0.05 + 0.2 * |x|
    epsilon ~ N(0, sigma(x)^2)
    y = f(x) + epsilon

    Train / Val region: x in [-1, 1]
    In-distribution test: x in [-1, 1]
    OOD extrapolation test: x in [1, 2] and [-2, -1]
    """
    rng = np.random.default_rng(seed)

    def sample_xy(x_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = np.sin(3.0 * x_vals)
        sigma = 0.05 + 0.2 * np.abs(x_vals)
        eps = rng.normal(0.0, sigma)
        y = f + eps
        return x_vals.reshape(-1, 1), y.reshape(-1, 1), sigma.reshape(-1, 1)

    # Train and Val
    x_train_val = rng.uniform(-1.0, 1.0, size=n_train)
    X_tv, y_tv, s_tv = sample_xy(x_train_val)

    n_val = int(0.2 * n_train)
    X_train, y_train = X_tv[n_val:], y_tv[n_val:]
    X_val, y_val = X_tv[:n_val], y_tv[:n_val]

    # In-distribution Test
    x_test_in = np.linspace(-1.0, 1.0, n_test_in)
    X_test_in, y_test_in, s_test_in = sample_xy(x_test_in)

    # OOD Extrapolation Test (positive branch [1, 2] and negative branch [-2, -1])
    half_ood = n_test_ood // 2
    x_ood = np.concatenate([
        np.linspace(-2.0, -1.0, half_ood),
        np.linspace(1.0, 2.0, n_test_ood - half_ood),
    ])
    X_test_ood, y_test_ood, s_test_ood = sample_xy(x_ood)

    in_split = DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test_in,
        y_test=y_test_in,
        shift_name="synthetic_in_distribution",
    )

    ood_split = DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test_ood,
        y_test=y_test_ood,
        shift_name="synthetic_ood_extrapolation",
    )

    return {
        "in_split": in_split,
        "ood_split": ood_split,
        "true_sigma_in": s_test_in,
        "true_sigma_ood": s_test_ood,
    }


# ---------------------------------------------------------------------------
# Tabular Dataset Loaders (Experiment B)
# ---------------------------------------------------------------------------

def load_diabetes_dataset(seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Loads scikit-learn Diabetes dataset (442 samples, 10 features)."""
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    X = data.data.astype(np.float64)
    y = data.target.astype(np.float64)
    names = list(data.feature_names)
    return X, y, names


DATA_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

ENERGY_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
CONCRETE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"


def _load_uci_table(url: str, cache_name: str, cache_dir: Optional[Path] = None) -> "pd.DataFrame":  # noqa: F821
    """
    Load a UCI table from the local CSV cache, downloading and caching it on first use.
    Failures raise: a missing dataset must never be replaced silently by simulated data.
    """
    import io
    import urllib.request

    import pandas as pd

    cache_path = (cache_dir or DATA_CACHE_DIR) / cache_name
    if cache_path.exists():
        return pd.read_csv(cache_path)

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = resp.read()
        df = pd.read_excel(io.BytesIO(payload))
    except Exception as exc:  # network, missing xlrd/openpyxl, malformed file
        raise RuntimeError(
            f"Could not load {cache_name} from cache or from {url} ({exc!r}). "
            f"Place the CSV at {cache_path} or install the optional readers (xlrd, openpyxl)."
        ) from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def load_energy_dataset(seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """UCI Energy Efficiency dataset (768 samples, 8 features, target: Heating Load Y1)."""
    feature_names = [
        "Relative_Compactness", "Surface_Area", "Wall_Area", "Roof_Area",
        "Overall_Height", "Orientation", "Glazing_Area", "Glazing_Area_Distribution"
    ]
    df = _load_uci_table(ENERGY_URL, "energy.csv")
    X = df.iloc[:, :8].values.astype(np.float64)
    y = df.iloc[:, 8].values.astype(np.float64)
    return X, y, feature_names


def load_concrete_dataset(seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """UCI Concrete Compressive Strength dataset (1030 samples, 8 features)."""
    feature_names = [
        "Cement", "Blast_Furnace_Slag", "Fly_Ash", "Water",
        "Superplasticizer", "Coarse_Aggregate", "Fine_Aggregate", "Age"
    ]
    df = _load_uci_table(CONCRETE_URL, "concrete.csv")
    X = df.iloc[:, :8].values.astype(np.float64)
    y = df.iloc[:, 8].values.astype(np.float64)
    return X, y, feature_names


# ---------------------------------------------------------------------------
# Spectral Dataset Generator & Contract (Experiment C)
# ---------------------------------------------------------------------------

def generate_simulated_spectral_dataset(
    n_samples: int = 600,
    scan_length: int = 500,
    n_devices: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Generates realistic TDLAS spectroscopic scan profiles based on Beer-Lambert law.
    Satisfies Project 1 & 2 data contract (Plan Section 7.3):
    Keys: X (N, L), y (N,), device (N,), scenario (N,), meta (json dict).

    Each device has distinct baseline drift, path length jitter, and detector noise.
    Concentration y is target in ppm.
    """
    rng = np.random.default_rng(seed)
    wavenumbers = np.linspace(1650.0, 1660.0, scan_length)
    peak_center = 1655.0
    gamma = 0.8  # HWHM of Lorentzian absorption peak

    # Generate concentrations y in [10, 500] ppm
    y_ppm = rng.uniform(10.0, 500.0, size=n_samples)
    device_ids = rng.integers(0, n_devices, size=n_samples)
    scenarios = np.zeros(n_samples, dtype=int)

    X_scans = np.empty((n_samples, scan_length), dtype=np.float64)

    # Device-specific optical properties (simulates instrument variability)
    device_path_lengths = 1.0 + 0.05 * rng.standard_normal(n_devices)  # Path length in meters
    device_noise_stds = 0.005 + 0.003 * rng.uniform(0.5, 1.5, size=n_devices)

    for i in range(n_samples):
        dev = device_ids[i]
        c = y_ppm[i]
        path_len = device_path_lengths[dev]

        # Beer-Lambert: I(nu) = I_0 * exp(-sigma(nu) * c * L)
        # Lorentzian cross section
        absorbance_peak = 0.0005 * c * path_len
        line_shape = (gamma ** 2) / ((wavenumbers - peak_center) ** 2 + gamma ** 2)
        transmission = np.exp(-absorbance_peak * line_shape)

        # Baseline polynomial drift
        baseline = 1.0 + 0.02 * np.sin(0.1 * wavenumbers) + 0.01 * (dev / n_devices)

        # Detector noise
        noise = rng.normal(0.0, device_noise_stds[dev], size=scan_length)

        # Observed intensity scan
        X_scans[i] = baseline * transmission + noise

    meta = {
        "scan_length": scan_length,
        "n_samples": n_samples,
        "n_devices": n_devices,
        "peak_center_cm_1": peak_center,
        "gamma_cm_1": gamma,
        "target_unit": "ppm",
        "description": "Simulated TDLAS absorption spectra meeting UCNet-TDLAS data contract",
    }

    return X_scans, y_ppm, device_ids, scenarios, meta


def save_spectral_contract_npz(
    filepath: Union[str, Path],
    X: np.ndarray,
    y: np.ndarray,
    device: np.ndarray,
    scenario: np.ndarray,
    meta: dict,
) -> None:
    """Save dataset according to Project 1 & 2 .npz contract."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        y=y,
        device=device,
        scenario=scenario,
        meta=json.dumps(meta),
    )


def load_spectral_contract_npz(filepath: Union[str, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load dataset following Project 1 & 2 .npz contract."""
    loaded = np.load(filepath, allow_pickle=False)
    meta = json.loads(str(loaded["meta"]))
    return loaded["X"], loaded["y"], loaded["device"], loaded["scenario"], meta


def split_by_device(
    X: np.ndarray,
    y: np.ndarray,
    device: np.ndarray,
    test_device: int = 0,
    val_ratio: float = 0.20,
    seed: int = 42,
    test_in_ratio: float = 0.0,
) -> DatasetSplit:
    """
    Device-level dataset split:
    Train and validation sets come strictly from non-test devices.
    Test set comes entirely from the held-out device (test_device).
    If test_in_ratio > 0, an additional in-distribution test set (same devices as training) is
    carved out before the train/validation split, so it is disjoint from both.
    Guarantees zero instrument leakage across domains.
    """
    test_mask = device == test_device
    train_val_mask = ~test_mask

    X_test = X[test_mask]
    y_test = y[test_mask]

    X_tv = X[train_val_mask]
    y_tv = y[train_val_mask]

    rng = np.random.default_rng(seed)
    N_tv = len(X_tv)
    perm = rng.permutation(N_tv)
    n_val = int(val_ratio * N_tv)

    n_test_in = int(test_in_ratio * N_tv)
    test_in_idx = perm[:n_test_in]
    val_idx = perm[n_test_in : n_test_in + n_val]
    train_idx = perm[n_test_in + n_val :]

    return DatasetSplit(
        X_train=X_tv[train_idx],
        y_train=y_tv[train_idx],
        X_val=X_tv[val_idx],
        y_val=y_tv[val_idx],
        X_test=X_test,
        y_test=y_test,
        shift_name=f"device_domain_shift_device_{test_device}",
        X_test_in=X_tv[test_in_idx] if n_test_in > 0 else None,
        y_test_in=y_tv[test_in_idx] if n_test_in > 0 else None,
    )
