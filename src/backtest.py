"""
Walk-forward backtest engine.

At each monthly rebalance date:
  1. Pull ML-predicted expected returns (already computed walk-forward,
     so no lookahead).
  2. Estimate covariance from the trailing `cov_lookback` days of returns.
  3. Solve for optimal weights (max Sharpe, long-only, capped).
  4. Hold those weights until the next rebalance, applying transaction
     costs on turnover.

Benchmarks computed alongside for comparison:
  - Equal-weight (1/N), rebalanced monthly
  - Inverse-volatility weighted, rebalanced monthly
"""
import numpy as np
import pandas as pd

from src.risk import estimate_covariance
from src.optimizer import optimize_portfolio


def run_backtest(
    prices: pd.DataFrame,
    pred_returns: pd.DataFrame,  # columns: date, ticker, pred_ret
    cov_lookback: int = 126,
    max_weight: float = 0.25,
    tc_bps: float = 10.0,        # transaction cost, basis points per unit turnover
) -> dict:
    daily_ret = prices.pct_change().dropna()
    rebalance_dates = sorted(pred_returns["date"].unique())

    tickers = prices.columns.tolist()
    n = len(tickers)

    weights_ml = {}
    weights_ew = {}
    weights_iv = {}

    prev_w_ml = np.ones(n) / n
    prev_w_iv = np.ones(n) / n

    for rdate in rebalance_dates:
        hist = daily_ret[daily_ret.index <= rdate].tail(cov_lookback)
        if len(hist) < cov_lookback // 2:
            continue

        sigma = estimate_covariance(hist)

        mu_row = pred_returns[pred_returns["date"] == rdate].set_index("ticker")["pred_ret"]
        mu_row = mu_row.reindex(tickers).fillna(0.0) * (252 / 21)  # annualize the 21d pred

        w_ml = optimize_portfolio(mu_row, sigma, objective="max_sharpe", max_weight=max_weight)

        # inverse-vol benchmark
        vol = np.sqrt(np.diag(sigma.values))
        inv_vol = 1 / np.where(vol == 0, np.nan, vol)
        w_iv = inv_vol / np.nansum(inv_vol)
        w_iv = pd.Series(w_iv, index=tickers)

        w_ew = pd.Series(np.ones(n) / n, index=tickers)

        weights_ml[rdate] = w_ml
        weights_iv[rdate] = w_iv
        weights_ew[rdate] = w_ew

    if not weights_ml:
        raise RuntimeError("Backtest produced no rebalances — check cov_lookback vs data length.")

    return _simulate(daily_ret, weights_ml, tc_bps, "ML-Optimized"), \
           _simulate(daily_ret, weights_ew, tc_bps, "Equal-Weight"), \
           _simulate(daily_ret, weights_iv, tc_bps, "Inverse-Vol")


def _simulate(daily_ret: pd.DataFrame, weights_by_date: dict, tc_bps: float, label: str) -> dict:
    rebal_dates = sorted(weights_by_date.keys())
    tickers = daily_ret.columns.tolist()

    port_daily_ret = []
    weight_history = []
    turnover_history = []

    current_w = np.zeros(len(tickers))
    dates_idx = daily_ret.index

    for i, rdate in enumerate(rebal_dates):
        target_w = weights_by_date[rdate].reindex(tickers).fillna(0).values
        turnover = np.abs(target_w - current_w).sum()
        turnover_history.append({"date": rdate, "turnover": turnover})
        tc_cost = turnover * (tc_bps / 10000)
        current_w = target_w

        end_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates_idx[-1]
        period = daily_ret[(daily_ret.index > rdate) & (daily_ret.index <= end_date)]
        if period.empty:
            continue

        first_day = True
        w = current_w.copy()
        for date, row in period.iterrows():
            r = row.values
            day_ret = w @ r
            if first_day:
                day_ret -= tc_cost  # apply cost on first day of new holding period
                first_day = False
            port_daily_ret.append({"date": date, "ret": day_ret})
            # drift weights with returns (no rebalancing intra-period)
            w = w * (1 + r)
            w = w / w.sum() if w.sum() != 0 else w
            weight_history.append({"date": date, **dict(zip(tickers, w))})

    ret_series = pd.DataFrame(port_daily_ret).set_index("date")["ret"]
    weight_df = pd.DataFrame(weight_history).set_index("date")
    turnover_df = pd.DataFrame(turnover_history).set_index("date")

    return {
        "label": label,
        "daily_returns": ret_series,
        "weights": weight_df,
        "turnover": turnover_df,
    }


def compute_metrics(ret_series: pd.Series, rf: float = 0.02) -> dict:
    ann_factor = 252
    mean_ret = ret_series.mean() * ann_factor
    vol = ret_series.std() * np.sqrt(ann_factor)
    sharpe = (mean_ret - rf) / vol if vol > 0 else np.nan

    downside = ret_series[ret_series < 0]
    downside_vol = downside.std() * np.sqrt(ann_factor) if len(downside) > 0 else np.nan
    sortino = (mean_ret - rf) / downside_vol if downside_vol and downside_vol > 0 else np.nan

    cum = (1 + ret_series).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = drawdown.min()

    total_return = cum.iloc[-1] - 1
    n_years = len(ret_series) / ann_factor
    cagr = (cum.iloc[-1]) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "CAGR": cagr,
        "Annual Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Total Return": total_return,
    }
