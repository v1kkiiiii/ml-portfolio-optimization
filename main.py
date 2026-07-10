"""
End-to-end pipeline: data -> features -> ML predictions -> risk model ->
optimization -> walk-forward backtest -> evaluation & plots.

Run with:  python3 main.py
Outputs land in ./outputs/
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_data import generate_universe
from src.features import compute_features
from src.models import train_predict_walk_forward, model_diagnostics
from src.optimizer import optimize_portfolio, efficient_frontier
from src.risk import estimate_covariance
from src.backtest import run_backtest, compute_metrics

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


def main():
    print("=" * 60)
    print("STEP 1: Generating market data")
    print("=" * 60)
    prices, sectors = generate_universe(n_assets=15, n_days=1500, seed=42)
    print(f"Universe: {prices.shape[1]} assets, {prices.shape[0]} trading days "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")

    print("\n" + "=" * 60)
    print("STEP 2: Feature engineering")
    print("=" * 60)
    panel = compute_features(prices, horizon=21)
    print(f"Feature panel: {panel.shape[0]} rows x {panel.shape[1]} cols")

    print("\n" + "=" * 60)
    print("STEP 3: Model diagnostics (single train/test split)")
    print("=" * 60)
    split_date = panel["date"].quantile(0.7)
    diag = model_diagnostics(panel, split_date)
    print(f"Train R^2: {diag['train_r2']:.4f}")
    print(f"Test  R^2: {diag['test_r2']:.4f}")
    print(f"Test Information Coefficient (Spearman): {diag['test_information_coefficient']:.4f}")
    print("Top features by importance:")
    for feat, imp in sorted(diag["feature_importance"].items(), key=lambda x: -x[1])[:5]:
        print(f"  {feat:15s} {imp:.4f}")

    print("\n" + "=" * 60)
    print("STEP 4: Walk-forward ML return predictions")
    print("=" * 60)
    all_dates = panel["date"].drop_duplicates().sort_values()
    rebalance_dates = pd.date_range(
        start=all_dates.iloc[300], end=all_dates.iloc[-22], freq="MS"
    )
    rebalance_dates = pd.DatetimeIndex(
        [all_dates[all_dates >= d].iloc[0] for d in rebalance_dates if (all_dates >= d).any()]
    ).unique()
    print(f"Rebalancing on {len(rebalance_dates)} monthly dates")

    preds = train_predict_walk_forward(panel, rebalance_dates, horizon=21, min_train_rows=500)
    print(f"Generated {len(preds)} (date, ticker) predictions")

    print("\n" + "=" * 60)
    print("STEP 5: Snapshot optimization (most recent rebalance date)")
    print("=" * 60)
    last_date = preds["date"].max()
    mu_snapshot = preds[preds["date"] == last_date].set_index("ticker")["pred_ret"] * (252 / 21)
    daily_ret_all = prices.pct_change().dropna()
    hist = daily_ret_all[daily_ret_all.index <= last_date].tail(126)
    sigma_snapshot = estimate_covariance(hist)

    w_opt = optimize_portfolio(mu_snapshot, sigma_snapshot, objective="max_sharpe", max_weight=0.25)
    print("Optimal weights (max Sharpe, capped at 25%):")
    for t, w in w_opt.sort_values(ascending=False).items():
        if w > 0.001:
            print(f"  {t:12s} {w:.1%}")

    frontier = efficient_frontier(mu_snapshot, sigma_snapshot, n_points=25, max_weight=0.25)

    print("\n" + "=" * 60)
    print("STEP 6: Full walk-forward backtest")
    print("=" * 60)
    result_ml, result_ew, result_iv = run_backtest(
        prices, preds, cov_lookback=126, max_weight=0.25, tc_bps=10.0
    )

    metrics_table = {}
    for res in (result_ml, result_ew, result_iv):
        metrics_table[res["label"]] = compute_metrics(res["daily_returns"])

    metrics_df = pd.DataFrame(metrics_table).T
    print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))
    metrics_df.to_csv(os.path.join(OUT, "backtest_metrics.csv"))

    print("\n" + "=" * 60)
    print("STEP 7: Generating plots")
    print("=" * 60)
    _plot_cumulative_returns(result_ml, result_ew, result_iv)
    _plot_drawdown(result_ml, result_ew, result_iv)
    _plot_efficient_frontier(frontier, mu_snapshot, sigma_snapshot, w_opt)
    _plot_weights_over_time(result_ml)
    _plot_feature_importance(diag["feature_importance"])

    print(f"\nAll outputs saved to {OUT}/")
    return metrics_df


def _plot_cumulative_returns(result_ml, result_ew, result_iv):
    fig, ax = plt.subplots(figsize=(10, 6))
    for res, color in zip((result_ml, result_ew, result_iv), ("#2563eb", "#6b7280", "#f59e0b")):
        cum = (1 + res["daily_returns"]).cumprod()
        ax.plot(cum.index, cum.values, label=res["label"], color=color, linewidth=2)
    ax.set_title("Cumulative Growth of $1 — Walk-Forward Backtest", fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cumulative_returns.png"), dpi=150)
    plt.close(fig)


def _plot_drawdown(result_ml, result_ew, result_iv):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for res, color in zip((result_ml, result_ew, result_iv), ("#2563eb", "#6b7280", "#f59e0b")):
        cum = (1 + res["daily_returns"]).cumprod()
        dd = cum / cum.cummax() - 1
        ax.fill_between(dd.index, dd.values * 100, 0, alpha=0.3, color=color, label=res["label"])
    ax.set_title("Drawdown (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "drawdown.png"), dpi=150)
    plt.close(fig)


def _plot_efficient_frontier(frontier, mu, sigma, w_opt):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(frontier["volatility"] * 100, frontier["target_return"] * 100,
            color="#2563eb", linewidth=2, label="Efficient Frontier")

    opt_ret = w_opt.values @ mu.values
    opt_vol = np.sqrt(w_opt.values @ sigma.loc[mu.index, mu.index].values @ w_opt.values)
    ax.scatter([opt_vol * 100], [opt_ret * 100], color="#ef4444", s=100, zorder=5,
               label="Max Sharpe Portfolio")

    n = len(mu)
    ew = np.ones(n) / n
    ew_ret = ew @ mu.values
    ew_vol = np.sqrt(ew @ sigma.loc[mu.index, mu.index].values @ ew)
    ax.scatter([ew_vol * 100], [ew_ret * 100], color="#6b7280", s=80, zorder=5,
               marker="s", label="Equal-Weight")

    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Expected Return (%)")
    ax.set_title("Efficient Frontier (ML-Predicted Returns)", fontsize=13, fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "efficient_frontier.png"), dpi=150)
    plt.close(fig)


def _plot_weights_over_time(result_ml):
    w = result_ml["weights"]
    monthly = w.resample("ME").last() if hasattr(w.index, "freq") or True else w
    fig, ax = plt.subplots(figsize=(11, 6))
    monthly.plot.area(ax=ax, linewidth=0, cmap="tab20", legend=False)
    ax.set_title("ML-Optimized Portfolio Weights Over Time", fontsize=13, fontweight="bold")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "weights_over_time.png"), dpi=150)
    plt.close(fig)


def _plot_feature_importance(importance: dict):
    items = sorted(importance.items(), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#2563eb")
    ax.set_title("ML Model Feature Importance", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
