"""
Mean-variance optimization (scipy SLSQP), with configurable risk appetite
and constraints on top of the base Markowitz setup.

Risk appetite is exposed two ways:
  - `objective="risk_aversion"` with a `risk_aversion` (lambda) value:
    maximizes w'mu - (lambda/2) w'Sigma w, the classic Markowitz quadratic
    utility. Higher lambda = more conservative (penalizes variance harder).
  - preset RISK_PROFILES ("conservative" / "moderate" / "aggressive") that
    bundle a sensible lambda + per-asset cap together, so this can be
    exposed as a single dropdown in a UI instead of asking someone to pick
    a risk-aversion number out of the air.

Constraints supported on top of full-investment + long-only:
  - max_weight: uniform per-asset cap (existing behavior)
  - asset_bounds: dict of {ticker: (min, max)} to override the uniform cap
    for specific names
  - target_return: minimum required portfolio return (used with
    objective="target_return", or as an additional floor under any
    other objective)
  - sector_map + sector_caps: cap total exposure to a given sector,
    e.g. {"Tech": 0.40} to keep tech under 40% of the book
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


RISK_PROFILES = {
    "conservative": {"risk_aversion": 8.0, "max_weight": 0.15},
    "moderate":     {"risk_aversion": 3.0, "max_weight": 0.25},
    "aggressive":   {"risk_aversion": 1.0, "max_weight": 0.35},
}


def _neg_sharpe(w, mu, sigma, rf, risk_aversion):
    port_ret = w @ mu
    port_vol = np.sqrt(w @ sigma @ w)
    if port_vol < 1e-9:
        return 1e6
    return -(port_ret - rf) / port_vol


def _variance(w, mu, sigma, rf, risk_aversion):
    return w @ sigma @ w


def _neg_utility(w, mu, sigma, rf, risk_aversion):
    # classic Markowitz quadratic utility: return minus a risk penalty
    return -(w @ mu - 0.5 * risk_aversion * (w @ sigma @ w))


_OBJECTIVES = {
    "max_sharpe": _neg_sharpe,
    "min_variance": _variance,
    "risk_aversion": _neg_utility,
    "target_return": _variance,  # target_return just adds an equality constraint below
}


def _build_bounds(tickers, max_weight, asset_bounds):
    asset_bounds = asset_bounds or {}
    bounds = []
    for t in tickers:
        lo, hi = asset_bounds.get(t, (0.0, max_weight))
        bounds.append((lo, hi))
    return bounds


def _build_constraints(tickers, mu_arr, target_return, sector_map, sector_caps):
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if target_return is not None:
        constraints.append({"type": "ineq", "fun": lambda w: w @ mu_arr - target_return})

    if sector_map is not None and sector_caps:
        for sector, cap in sector_caps.items():
            mask = np.array([sector_map.get(t) == sector for t in tickers])
            if mask.any():
                constraints.append({"type": "ineq", "fun": lambda w, m=mask, c=cap: c - w[m].sum()})

    return constraints


def optimize_portfolio(
    mu: pd.Series,
    sigma: pd.DataFrame,
    objective: str = "max_sharpe",
    max_weight: float = 0.25,
    rf: float = 0.02,
    risk_aversion: float = 3.0,
    target_return: float = None,
    asset_bounds: dict = None,
    sector_map: dict = None,
    sector_caps: dict = None,
    risk_profile: str = None,
) -> pd.Series:
    """
    risk_profile, if given ("conservative"/"moderate"/"aggressive"),
    overrides objective/risk_aversion/max_weight with a preset - pass
    individual args instead if you want finer control.
    """
    if risk_profile is not None:
        preset = RISK_PROFILES[risk_profile]
        objective = "risk_aversion"
        risk_aversion = preset["risk_aversion"]
        max_weight = preset["max_weight"]

    tickers = mu.index.tolist()
    n = len(tickers)
    mu_arr = mu.values
    sigma_arr = sigma.loc[tickers, tickers].values

    w0 = np.ones(n) / n
    bounds = _build_bounds(tickers, max_weight, asset_bounds)
    constraints = _build_constraints(tickers, mu_arr, target_return, sector_map, sector_caps)
    fn = _OBJECTIVES[objective]

    result = minimize(
        fn, w0, args=(mu_arr, sigma_arr, rf, risk_aversion),
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if not result.success:
        w = w0  # fall back to equal weight if it doesn't converge
    else:
        w = np.clip(result.x, 0, None)
        w = w / w.sum()

    return pd.Series(w, index=tickers, name="weight")


def efficient_frontier(mu: pd.Series, sigma: pd.DataFrame, n_points=30, max_weight=0.25) -> pd.DataFrame:
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
