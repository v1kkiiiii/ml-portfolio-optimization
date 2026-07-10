"""
Mean-variance portfolio optimization (Markowitz), using scipy SLSQP.

Given ML-predicted expected returns mu and a shrunk covariance matrix
Sigma, solve for weights w that:
  - maximize Sharpe ratio (mu'w - rf) / sqrt(w'Sigma w)   [primary]
  - or minimize variance w'Sigma w                          [defensive variant]
subject to:
  - sum(w) == 1  (fully invested)
  - 0 <= w_i <= max_weight  (long-only, diversification cap)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _neg_sharpe(w, mu, sigma, rf):
    port_ret = w @ mu
    port_vol = np.sqrt(w @ sigma @ w)
    if port_vol < 1e-9:
        return 1e6
    return -(port_ret - rf) / port_vol


def _variance(w, mu, sigma, rf):
    return w @ sigma @ w


def optimize_portfolio(
    mu: pd.Series,
    sigma: pd.DataFrame,
    objective: str = "max_sharpe",
    max_weight: float = 0.25,
    rf: float = 0.02,
) -> pd.Series:
    """
    mu    : expected annual return per asset (Series, index=tickers)
    sigma : annualized covariance matrix (DataFrame, index/cols=tickers)
    """
    tickers = mu.index.tolist()
    n = len(tickers)
    mu_arr = mu.values
    sigma_arr = sigma.loc[tickers, tickers].values

    w0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    fn = _neg_sharpe if objective == "max_sharpe" else _variance

    result = minimize(
        fn, w0, args=(mu_arr, sigma_arr, rf),
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if not result.success:
        # fall back to equal weight if optimizer fails to converge
        w = w0
    else:
        w = result.x
        w = np.clip(w, 0, None)
        w = w / w.sum()

    return pd.Series(w, index=tickers, name="weight")


def efficient_frontier(
    mu: pd.Series, sigma: pd.DataFrame, n_points: int = 30, max_weight: float = 0.25,
) -> pd.DataFrame:
    """Trace the efficient frontier by minimizing variance for a grid of target returns."""
    tickers = mu.index.tolist()
    n = len(tickers)
    mu_arr = mu.values
    sigma_arr = sigma.loc[tickers, tickers].values

    target_returns = np.linspace(mu_arr.min(), mu_arr.max(), n_points)
    frontier = []

    for target in target_returns:
        w0 = np.ones(n) / n
        bounds = [(0.0, max_weight)] * n
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mu_arr - t},
        ]
        result = minimize(
            lambda w: w @ sigma_arr @ w, w0,
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if result.success:
            w = np.clip(result.x, 0, None)
            w = w / w.sum()
            vol = np.sqrt(w @ sigma_arr @ w)
            frontier.append({"target_return": target, "volatility": vol})

    return pd.DataFrame(frontier)
