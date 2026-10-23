#Run in the same workspace in order to have stored the multi_asset_option_pricer

"""Run-to-run variability: repeat d=2 and d=20 (dividend case) with different seeds
for path simulation and network initialisation."""
import sys
import numpy as np
import torch


K, r, q, sigma, T = 100.0, 0.05, 0.10, 0.2, 3.0
for d in (2, 20):
    rows = []
    for seed in range(3):
        torch.manual_seed(7000 + 100 * seed + d)
        C = np.eye(d)
        tr = simulate_gbm_paths([100.0] * d, r, sigma, C, T, 9, 100_000, q)
        te = simulate_gbm_paths([100.0] * d, r, sigma, C, T, 9, 100_000, q)
        l = price_classical_lsm(tr, te, K, r, T)[0]
        torch.manual_seed(seed); b = price_becker_deep_lsm(tr, te, K, r, T)[0]
        torch.manual_seed(seed); h = price_han_zheng_partitioned(tr, te, K, r, T)[0]
        rows.append((l, b, h))
        print(d, seed, f"LSM {l:.3f}  deep {b:.3f}  part {h:.3f}  deep-LSM {b - l:+.3f}  part-LSM {h - l:+.3f}", flush=True)
    a = np.array(rows)
    print(d, "mean", np.round(a.mean(0), 3), "std", np.round(a.std(0, ddof=1), 3),
          "mean diff deep-LSM", round(float((a[:, 1] - a[:, 0]).mean()), 3), flush=True)
