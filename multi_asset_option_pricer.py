"""
Pricing Bermudan max-call options on d assets with Longstaff-Schwartz-type
regression: classical polynomial LSM vs. neural-network continuation values.

    python multi_asset_option_pricer.py                  # benchmark with 10% dividend yield
    python multi_asset_option_pricer.py --no-dividend    # sanity check: American = European
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for complete numerical reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. Multi-Asset Geometric Brownian Motion Simulator
# ==============================================================================
def simulate_gbm_paths(S0, r, sigma, corr_matrix, T, num_steps, num_paths, q=0.0):
    """
    Simulates multi-asset stock price paths under the risk-neutral measure Q
    using Cholesky decomposition. q is a continuous dividend yield.
    """
    d = len(S0)
    dt = T / num_steps
    L = torch.linalg.cholesky(torch.tensor(corr_matrix, dtype=torch.float32))

    paths = torch.zeros((num_paths, num_steps + 1, d), dtype=torch.float32)
    paths[:, 0, :] = torch.tensor(S0, dtype=torch.float32)

    for t in range(1, num_steps + 1):
        Z = torch.randn((num_paths, d), dtype=torch.float32)
        correlated_Z = Z @ L.T
        drift = (r - q - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * correlated_Z
        paths[:, t, :] = paths[:, t - 1, :] * torch.exp(drift + diffusion)

    return paths

def max_call_payoff(S, K):
    """Payoff = max(max(S_1, S_2, ..., S_d) - K, 0)"""
    max_S, _ = torch.max(S, dim=-1)
    return torch.maximum(max_S - K, torch.tensor(0.0, dtype=S.dtype))

# ==============================================================================
# 2. Neural Network Architecture for Continuation Value Estimation
# ==============================================================================
class ContinuationValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(ContinuationValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def fit_network(X, Y, d, num_epochs, lr):
    """Full-batch Adam on inputs/targets already scaled by K."""
    model = ContinuationValueNet(input_dim=d, hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()
    model.eval()
    return model

# ==============================================================================
# Shared backward induction.
# The exercise rule is fitted on TRAINING paths and applied to independent
# TEST paths. The test-path average is an unbiased estimate of the value of the
# fitted rule, i.e. a lower bound for the true price (pricing on the training
# paths themselves is biased upwards).
# ==============================================================================
def backward_induction(train_paths, test_paths, K, r, T, fit_continuation):
    num_steps = train_paths.shape[1] - 1
    discount = float(np.exp(-r * T / num_steps))
    paths = (train_paths, test_paths)
    cashflows = [max_call_payoff(P[:, -1, :], K) for P in paths]

    for t in range(num_steps - 1, 0, -1):
        # cash flows are now valued at time t (all paths, in or out of the money)
        cashflows = [cf * discount for cf in cashflows]
        S = [P[:, t, :] for P in paths]
        immediate = [max_call_payoff(s, K) for s in S]
        itm = [ip > 0 for ip in immediate]
        if not torch.any(itm[0]):
            continue

        # regression on in-the-money training paths only
        predict = fit_continuation(S[0][itm[0]], cashflows[0][itm[0]])
        if predict is None:
            continue

        for j in range(2):
            if not torch.any(itm[j]):
                continue
            v_con = predict(S[j][itm[j]])
            exercise = immediate[j][itm[j]] > v_con
            idx = torch.where(itm[j])[0][exercise]
            cashflows[j][idx] = immediate[j][idx]

    # discount from t = 1 back to t = 0
    test_values = cashflows[1].double() * discount
    price = test_values.mean().item()
    std_err = (test_values.std() / np.sqrt(len(test_values))).item()
    return price, std_err

# ==============================================================================
# 3. Pricing Engine 1: Global Deep LSM (Becker, Cheridito & Jentzen, 2020)
# ==============================================================================
def price_becker_deep_lsm(train_paths, test_paths, K, r, T, num_epochs=300, lr=1e-2):
    start_time = time.time()
    d = train_paths.shape[2]

    def fit(X, Y):
        model = fit_network(X / K, Y / K, d, num_epochs, lr)
        def predict(Z):
            with torch.no_grad():
                return model(Z / K) * K
        return predict

    price, se = backward_induction(train_paths, test_paths, K, r, T, fit)
    return price, se, time.time() - start_time

# ==============================================================================
# 4. Pricing Engine 2: State-Space Partitioned Deep LSM
#    (a simplified two-region version inspired by Han & Zheng, 2024)
# ==============================================================================
def price_han_zheng_partitioned(train_paths, test_paths, K, r, T, num_epochs=300, lr=1e-2,
                                split=1.15):
    start_time = time.time()
    d = train_paths.shape[2]

    def region_masks(X):
        max_asset_vals, _ = torch.max(X, dim=-1)
        return [max_asset_vals <= K * split, max_asset_vals > K * split]

    def fit(X, Y):
        models = []
        for p_mask in region_masks(X):
            if torch.sum(p_mask) < 20:
                models.append(None)
                continue
            models.append(fit_network(X[p_mask] / K, Y[p_mask] / K, d, num_epochs, lr))

        def predict(Z):
            v_con = torch.zeros(len(Z))
            with torch.no_grad():
                for model, p_mask in zip(models, region_masks(Z)):
                    if model is not None and torch.any(p_mask):
                        v_con[p_mask] = model(Z[p_mask] / K) * K
            return v_con
        return predict

    price, se = backward_induction(train_paths, test_paths, K, r, T, fit)
    return price, se, time.time() - start_time

# ==============================================================================
# 5. Pricing Engine 3: Classical Polynomial LSM (Longstaff & Schwartz, 2001)
# ==============================================================================
def polynomial_features(X):
    """Constant, linear, quadratic and cross terms of X / K (scaled)."""
    n, d = X.shape
    feature_list = [np.ones(n)]
    for i_col in range(d):
        feature_list.append(X[:, i_col])
        feature_list.append(X[:, i_col] ** 2)
    for i_col in range(d):
        for j_col in range(i_col + 1, d):
            feature_list.append(X[:, i_col] * X[:, j_col])
    return np.column_stack(feature_list)

def price_classical_lsm(train_paths, test_paths, K, r, T):
    start_time = time.time()
    worst_cond = [0.0]

    def fit(X, Y):
        A = polynomial_features(X.double().numpy() / K)
        worst_cond[0] = max(worst_cond[0], np.linalg.cond(A))
        coeffs, _, _, _ = np.linalg.lstsq(A, Y.double().numpy(), rcond=None)
        def predict(Z):
            return torch.from_numpy(polynomial_features(Z.double().numpy() / K) @ coeffs).float()
        return predict

    price, se = backward_induction(train_paths, test_paths, K, r, T, fit)
    return price, se, time.time() - start_time, worst_cond[0]

# ==============================================================================
# 6. Reference values
# ==============================================================================
def european_price(d, S0, K, r, q, sigma, rho, T, num_paths=2_000_000, seed=0):
    """European max-call by Monte Carlo on terminal values only (antithetic).
    A lower bound for the Bermudan price; equal to it when q = 0."""
    g = np.random.default_rng(seed)
    C = np.full((d, d), rho)
    np.fill_diagonal(C, 1.0)
    Z = g.standard_normal((num_paths // 2, d)) @ np.linalg.cholesky(C).T
    Z = np.vstack([Z, -Z])
    ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return float(np.exp(-r * T) * np.maximum(ST.max(axis=1) - K, 0.0).mean())

def lattice_price_2d(S0, K, r, q, sigma, T, exercise_dates, steps_per_period=200):
    """Bermudan max-call on 2 independent assets with a 2-D binomial lattice.
    Used as the reference price for d = 2 (rho = 0)."""
    n = exercise_dates * steps_per_period
    dt = T / n
    u = sigma * np.sqrt(dt)
    p = 0.5 + 0.5 * (r - q - 0.5 * sigma**2) * np.sqrt(dt) / sigma
    disc = np.exp(-r * dt)
    grid = lambda step: S0 * np.exp((2 * np.arange(step + 1) - step) * u)
    s = grid(n)
    V = np.maximum(np.maximum.outer(s, s) - K, 0.0)
    for step in range(n - 1, -1, -1):
        V = disc * (p * p * V[1:, 1:] + p * (1 - p) * (V[1:, :-1] + V[:-1, 1:])
                    + (1 - p) ** 2 * V[:-1, :-1])
        if step % steps_per_period == 0 and step > 0:
            s = grid(step)
            V = np.maximum(V, np.maximum.outer(s, s) - K)
    return float(V[0, 0])

# ==============================================================================
# 7. Dimensional Scaling Benchmark
# ==============================================================================
def run_dimensional_benchmark(q=0.10, dimensions=(2, 3, 5, 10, 20)):
    S0_base = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    rho = 0.0
    T = 3.0
    num_steps = 9            # 9 exercise dates
    num_train = 100_000
    num_test = 100_000

    print("\n" + "=" * 104)
    print(f" BERMUDAN MAX-CALL: S0=K=100, r=5%, q={q:.0%}, sigma=20%, rho={rho}, T={T:g}, "
          f"{num_steps} exercise dates")
    print(f" exercise rules fitted on {num_train:,} paths, priced on {num_test:,} independent paths")
    print("=" * 104)
    print(f"{'d':>3} | {'European':>9} | {'Reference':>9} | {'Classical LSM':>15} | "
          f"{'Deep LSM':>15} | {'Partitioned':>15} | {'cond(A)':>8}")
    print("-" * 104)

    results = []
    for d in dimensions:
        S0 = [S0_base] * d
        corr_matrix = np.full((d, d), rho)
        np.fill_diagonal(corr_matrix, 1.0)

        torch.manual_seed(1000 + d)
        train = simulate_gbm_paths(S0, r, sigma, corr_matrix, T, num_steps, num_train, q)
        test = simulate_gbm_paths(S0, r, sigma, corr_matrix, T, num_steps, num_test, q)

        euro = european_price(d, S0_base, K, r, q, sigma, rho, T)
        if q == 0.0:
            ref = euro                      # no dividends: never optimal to exercise early
        elif d == 2 and rho == 0.0:
            ref = lattice_price_2d(S0_base, K, r, q, sigma, T, num_steps)
        else:
            ref = None

        p_lsm, se_lsm, t_lsm, cond = price_classical_lsm(train, test, K, r, T)
        torch.manual_seed(d)
        p_becker, se_becker, t_becker = price_becker_deep_lsm(train, test, K, r, T)
        torch.manual_seed(d)
        p_han, se_han, t_han = price_han_zheng_partitioned(train, test, K, r, T)

        ref_str = f"{ref:9.3f}" if ref is not None else f"{'-':>9}"
        print(f"{d:>3} | {euro:9.3f} | {ref_str} | {p_lsm:8.3f} ±{se_lsm:5.3f} | "
              f"{p_becker:8.3f} ±{se_becker:5.3f} | {p_han:8.3f} ±{se_han:5.3f} | {cond:8.1e}",
              flush=True)

        results.append({
            'd': d, 'euro_price': euro, 'reference': ref,
            'lsm_price': p_lsm, 'lsm_se': se_lsm, 'lsm_time': t_lsm, 'lsm_cond': cond,
            'becker_price': p_becker, 'becker_se': se_becker, 'becker_time': t_becker,
            'han_price': p_han, 'han_se': se_han, 'han_time': t_han,
        })

    print("-" * 104)
    print(" ± = Monte Carlo standard error. Each price is a lower bound: higher is a better exercise rule.\n")
    return results

# ==============================================================================
# 8. Visualization Generator (Static & Interactive Plotly Dashboard)
# ==============================================================================
COLORS = {'lsm': '#eb6834', 'becker': '#2a78d6', 'han': '#1baf7a', 'euro': '#6b6a63'}
LABELS = {'lsm': 'Classical polynomial LSM', 'becker': 'Deep LSM (Becker et al. 2020)',
          'han': 'Partitioned deep LSM (after Han & Zheng 2024)'}

def generate_visualizations(results, out_dir='docs', suffix=''):
    os.makedirs(out_dir, exist_ok=True)
    dims = [r['d'] for r in results]
    rel = lambda key: [100 * (r[f'{key}_price'] / r['euro_price'] - 1) for r in results]

    # --------------------------------------------------------------------------
    # A. Static figure: premium over European price, and runtime
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for key in ('lsm', 'becker', 'han'):
        ax1.plot(dims, rel(key), 'o-', color=COLORS[key], label=LABELS[key], linewidth=2, markersize=6)
        ax2.plot(dims, [r[f'{key}_time'] for r in results], 'o-', color=COLORS[key],
                 label=LABELS[key], linewidth=2, markersize=6)
    refs = [(r['d'], 100 * (r['reference'] / r['euro_price'] - 1)) for r in results
            if r['reference'] is not None]
    if refs:
        ax1.plot(*zip(*refs), 'D', color='#1f1f1e', markersize=8, label='Reference price', zorder=5)
    ax1.axhline(0, color=COLORS['euro'], linewidth=1, linestyle='--')
    ax1.set_title('Price above the European value (early-exercise premium)', fontsize=11, loc='left')
    ax1.set_xlabel('Number of assets d')
    ax1.set_ylabel('% above European price')
    ax2.set_title('Runtime (fit on train paths + price on test paths)', fontsize=11, loc='left')
    ax2.set_xlabel('Number of assets d')
    ax2.set_ylabel('seconds')
    for ax in (ax1, ax2):
        ax.set_xticks(dims)
        ax.grid(True, color='#e4e3dd')
        ax.spines[['top', 'right']].set_visible(False)
    ax1.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    static_path = os.path.join(out_dir, f'benchmark{suffix}.png')
    plt.savefig(static_path, dpi=150)
    plt.close()
    print(f"[+] Static figure saved to: {static_path}")

    # --------------------------------------------------------------------------
    # B. Interactive Plotly HTML version of the same two panels
    # --------------------------------------------------------------------------
    plotly_fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Price above the European value (%)", "Runtime (seconds)"))
    for key in ('lsm', 'becker', 'han'):
        plotly_fig.add_trace(go.Scatter(
            x=dims, y=rel(key), mode='lines+markers', name=LABELS[key],
            line=dict(color=COLORS[key], width=2), legendgroup=key,
            customdata=[[r[f'{key}_price'], r[f'{key}_se']] for r in results],
            hovertemplate='d=%{x}<br>price %{customdata[0]:.3f} ± %{customdata[1]:.3f}'
                          '<br>%{y:.2f}% above European<extra></extra>'), row=1, col=1)
        plotly_fig.add_trace(go.Scatter(
            x=dims, y=[r[f'{key}_time'] for r in results], mode='lines+markers',
            name=LABELS[key], line=dict(color=COLORS[key], width=2), legendgroup=key,
            showlegend=False), row=1, col=2)
    plotly_fig.update_layout(template="plotly_white", height=480, width=1200,
                             title_text="Bermudan max-call: classical vs. deep LSM")
    html_path = os.path.join(out_dir, f'interactive_benchmark{suffix}.html')
    plotly_fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"[+] Interactive figure saved to: {html_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-dividend', action='store_true',
                        help='q = 0: early exercise is never optimal, so every method should '
                             'return the European price')
    args = parser.parse_args()
    if args.no_dividend:
        benchmark_results = run_dimensional_benchmark(q=0.0)
        generate_visualizations(benchmark_results, suffix='_no_dividend')
    else:
        benchmark_results = run_dimensional_benchmark(q=0.10)
        generate_visualizations(benchmark_results)
