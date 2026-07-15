"""
Walk-forward backtest. At each monthly rebalance: pull the ML's
predicted returns (already walk-forward, no lookahead), estimate cov
from trailing 126 days, solve for weights, hold until next rebalance
with turnover-based transaction costs. Equal-weight and inverse-vol are
run alongside as benchmarks - any "smart" weighting scheme needs to
actually beat these to justify the extra complexity.
"""
import numpy as np
import pandas as pd

from src.risk import estimate_covariance
from src.optimizer import optimize_portfolio


def run_backtest(prices, pred_returns, cov_lookback=126, max_weight=0.25, tc_bps=10.0, optimizer_kwargs=None):
    """
    optimizer_kwargs, if given, is passed straight through to
    optimize_portfolio() for the ML leg - e.g. {"risk_profile": "conservative"}
    or {"sector_map": {...}, "sector_caps": {"Tech": 0.4}}. Lets you rerun
    the same backtest under different risk appetites/constraints without
    touching this function.
    """
    optimizer_kwargs = optimizer_kwargs or {"objective": "max_sharpe"}
    daily_ret = prices.pct_change().dropna()
    rebalance_dates = sorted(pred_returns["date"].unique())
    tickers = prices.columns.tolist()
    n = len(tickers)

    weights_ml, weights_ew, weights_iv = {}, {}, {}

    for rdate in rebalance_dates:
        hist = daily_ret[daily_ret.index <= rdate].tail(cov_lookback)
        if len(hist) < cov_lookback // 2:
            continue

        sigma = estimate_covariance(hist)

        mu_row = pred_returns[pred_returns["date"] == rdate].set_index("ticker")["pred_ret"]
        mu_row = mu_row.reindex(tickers).fillna(0.0) * (252 / 21)  # annualize the 21d pred

        weights_ml[rdate] = optimize_portfolio(mu_row, sigma, max_weight=max_weight, **optimizer_kwargs)

        vol = np.sqrt(np.diag(sigma.values))
        inv_vol = 1 / np.where(vol == 0, np.nan, vol)
        weights_iv[rdate] = pd.Series(inv_vol / np.nansum(inv_vol), index=tickers)
        weights_ew[rdate] = pd.Series(np.ones(n) / n, index=tickers)

    if not weights_ml:
        raise RuntimeError("backtest produced no rebalances - check cov_lookback vs data length")

    return (
        _simulate(daily_ret, weights_ml, tc_bps, "ML-Optimized"),
        _simulate(daily_ret, weights_ew, tc_bps, "Equal-Weight"),
        _simulate(daily_ret, weights_iv, tc_bps, "Inverse-Vol"),
    )


def _simulate(daily_ret, weights_by_date, tc_bps, label):
    rebal_dates = sorted(weights_by_date.keys())
    tickers = daily_ret.columns.tolist()

    port_daily_ret = []
    weight_history = []
    turnover_history = []
    current_w = np.zeros(len(tickers))

    for i, rdate in enumerate(rebal_dates):
        target_w = weights_by_date[rdate].reindex(tickers).fillna(0).values
        turnover = np.abs(target_w - current_w).sum()
        turnover_history.append({"date": rdate, "turnover": turnover})
        tc_cost = turnover * (tc_bps / 10000)
        current_w = target_w

        end_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else daily_ret.index[-1]
        period = daily_ret[(daily_ret.index > rdate) & (daily_ret.index <= end_date)]
        if period.empty:
            continue

        w = current_w.copy()
        first_day = True
        for date, row in period.iterrows():
            r = row.values
            day_ret = w @ r
            if first_day:
                day_ret -= tc_cost
                first_day = False
            port_daily_ret.append({"date": date, "ret": day_ret})
            w = w * (1 + r)
            w = w / w.sum() if w.sum() != 0 else w  # let weights drift with returns between rebalances
            weight_history.append({"date": date, **dict(zip(tickers, w))})

    return {
        "label": label,
        "daily_returns": pd.DataFrame(port_daily_ret).set_index("date")["ret"],
        "weights": pd.DataFrame(weight_history).set_index("date"),
        "turnover": pd.DataFrame(turnover_history).set_index("date"),
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
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()

    n_years = len(ret_series) / ann_factor
    cagr = cum.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "CAGR": cagr,
        "Annual Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Total Return": cum.iloc[-1] - 1,
    }
