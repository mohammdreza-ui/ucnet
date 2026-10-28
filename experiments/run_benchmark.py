"""
Runtime benchmark: comparison of execution speed between UCNet (pure NumPy) and PyTorch on CPU.

Methodology (both frameworks are treated identically):
- float64, the same shapes, the same number of CPU threads (set explicitly via torch.set_num_threads),
- untimed warm-up calls, then the median of several timed blocks,
- no allocation of new input tensors inside the timed region.
A single-process micro-benchmark on small shapes mostly measures framework overhead, not kernel speed.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as t_nn
import torch.optim as t_optim

from ucnet.nn.activations import Tanh
from ucnet.nn.containers import Sequential
from ucnet.nn.conv1d import Conv1D
from ucnet.nn.linear import Linear
from ucnet.nn.module import Parameter
from ucnet.optim import Adam
from ucnet.runlog import write_manifest
from ucnet.tensor import Tensor

NUM_THREADS = 1


def _timeit(fn, steps: int, warmup: int = 20, repeats: int = 7) -> float:
    """Median seconds per call over `repeats` timed blocks, after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(steps):
            fn()
        times.append((time.perf_counter() - start) / steps)
    return float(np.median(times))


def benchmark_mlp(batch_size: int = 64, width: int = 128, steps: int = 50) -> tuple[float, float]:
    """Forward + backward of a 2-layer tanh MLP (width -> width -> width), loss = sum of outputs."""
    x_np = np.random.randn(batch_size, width).astype(np.float64)

    net_uc = Sequential(Linear(width, width), Tanh(), Linear(width, width))
    x_uc = Tensor(x_np)

    def step_uc() -> None:
        net_uc.zero_grad()
        net_uc(x_uc).sum().backward()

    net_th = t_nn.Sequential(t_nn.Linear(width, width), t_nn.Tanh(), t_nn.Linear(width, width)).double()
    x_th = torch.tensor(x_np, dtype=torch.float64)

    def step_th() -> None:
        net_th.zero_grad()
        net_th(x_th).sum().backward()

    return _timeit(step_uc, steps) * 1e3, _timeit(step_th, steps) * 1e3


def benchmark_conv1d(batch_size: int = 32, in_c: int = 8, out_c: int = 16, length: int = 200, k: int = 7,
                     steps: int = 30) -> tuple[float, float]:
    """Conv1D forward + backward (weights, bias and input gradients)."""
    x_np = np.random.randn(batch_size, in_c, length).astype(np.float64)

    conv_uc = Conv1D(in_c, out_c, kernel_size=k, stride=1, padding=1)
    x_uc = Tensor(x_np, requires_grad=True)

    def step_uc() -> None:
        x_uc.zero_grad()
        conv_uc.zero_grad()
        conv_uc(x_uc).sum().backward()

    conv_th = t_nn.Conv1d(in_c, out_c, kernel_size=k, stride=1, padding=1).double()
    x_th = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)

    def step_th() -> None:
        x_th.grad = None
        conv_th.zero_grad()
        conv_th(x_th).sum().backward()

    return _timeit(step_uc, steps) * 1e3, _timeit(step_th, steps) * 1e3


def benchmark_adam(param_dim: int = 10000, steps: int = 100) -> tuple[float, float]:
    """One Adam step on a 10,000-parameter vector with a fixed, pre-allocated gradient."""
    p_np = np.random.randn(param_dim).astype(np.float64)
    g_np = np.random.randn(param_dim).astype(np.float64)

    p_uc = Parameter(p_np.copy())
    opt_uc = Adam([p_uc], lr=0.01)
    p_uc.grad = g_np

    def step_uc() -> None:
        opt_uc.step()

    p_th = torch.tensor(p_np.copy(), dtype=torch.float64, requires_grad=True)
    opt_th = t_optim.Adam([p_th], lr=0.01)
    p_th.grad = torch.tensor(g_np, dtype=torch.float64)

    def step_th() -> None:
        opt_th.step()

    return _timeit(step_uc, steps) * 1e3, _timeit(step_th, steps) * 1e3


def run_benchmarks(results_dir: str = "results") -> None:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / "runtime_benchmark.csv"
    torch.set_num_threads(NUM_THREADS)
    write_manifest(out_dir, "runtime_benchmark", {"torch": torch.__version__, "threads": NUM_THREADS,
                                                  "dtype": "float64"})

    print("\n==================================================================")
    print(f"  Running Runtime Benchmark: UCNet (NumPy) vs PyTorch on CPU ({NUM_THREADS} thread)")
    print("==================================================================")

    np.random.seed(0)
    results = []
    for name, fn in (("2-layer tanh MLP 128-128-128, batch 64, fwd+bwd", benchmark_mlp),
                     ("Conv1D (C8->16, L200, K7, batch 32) fwd+bwd", benchmark_conv1d),
                     ("Adam step (10,000 parameters)", benchmark_adam)):
        uc, th = fn()
        results.append({"Operation": name, "UCNet_ms": uc, "PyTorch_ms": th, "Ratio": uc / th})

    print(f"{'Operation':<52} | {'UCNet (ms)':<11} | {'PyTorch (ms)':<12} | {'UCNet/PyTorch':<13}")
    print("-" * 98)
    for r in results:
        print(f"{r['Operation']:<52} | {r['UCNet_ms']:11.3f} | {r['PyTorch_ms']:12.3f} | {r['Ratio']:12.2f}x")

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Benchmark results saved to {csv_file.absolute()}")


if __name__ == "__main__":
    run_benchmarks()
