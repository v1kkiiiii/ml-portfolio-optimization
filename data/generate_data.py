"""
Simulates a small multi-asset market since this sandbox can't hit the
internet to pull real prices. Swap this out for a yfinance/Bloomberg pull
later - everything downstream just expects a (date x ticker) price
DataFrame so nothing else needs to change.

Roughly how it works: 3 macro factors drive all assets (like a mini
Barra model), plus each asset has its own slow-moving alpha state (AR1,
~2 week half life) that's what actually makes momentum features
predictive. Without that the "signal" the ML model finds would just be
noise. Also throwing in vol regime switching and occasional jumps so the
backtest isn't just smooth drift.
"""
import numpy as np
import pandas as pd


def generate_universe(n_assets=15, n_days=1500, start_date="2019-01-01", seed=42):
    rng = np.random.default_rng(seed)

    tickers = [f"ASSET_{i:02d}" for i in range(n_assets)]
    sectors = ["Tech", "Financials", "Energy", "Healthcare", "Consumer"]
    sector_map = pd.DataFrame({
        "ticker": tickers,
        "sector": rng.choice(sectors, size=n_assets),
    }).set_index("ticker")

    dates = pd.bdate_range(start=start_date, periods=n_days)

    # macro factors, slightly autocorrelated day to day
    n_factors = 3
    factor_returns = np.zeros((n_days, n_factors))
    phi = 0.05
    factor_vol = np.array([0.006, 0.008, 0.010])
    for t in range(1, n_days):
        factor_returns[t] = phi * factor_returns[t - 1] + rng.normal(0, factor_vol, size=n_factors)

    # simple 2-state vol regime (calm vs stressed)
    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        if regime[t - 1] == 0:
            regime[t] = 1 if rng.random() < 0.01 else 0
        else:
            regime[t] = 0 if rng.random() < 0.05 else 1
    regime_mult = np.where(regime == 1, 2.2, 1.0)

    betas = rng.normal(0.5, 0.4, size=(n_assets, n_factors))
    idio_vol = rng.uniform(0.010, 0.022, size=n_assets)
    drift = rng.uniform(-0.0001, 0.0005, size=n_assets)

    # per-asset alpha state - this is the piece that gives momentum
    # features something real to pick up on. phi=0.95 -> about a
    # 2-week half life, small enough that it doesn't dominate returns
    # but persistent enough to show up in 21/63/126d momentum windows.
    alpha_phi = 0.95
    alpha_innov_vol = rng.uniform(0.0006, 0.0013, size=n_assets)
    alpha_state = np.zeros((n_days, n_assets))
    for t in range(1, n_days):
        alpha_state[t] = alpha_phi * alpha_state[t - 1] + rng.normal(0, alpha_innov_vol, size=n_assets)

    returns = np.zeros((n_days, n_assets))
    for t in range(n_days):
        idio_shock = rng.normal(0, idio_vol, size=n_assets) * regime_mult[t]
        jump = np.where(rng.random(n_assets) < 0.003, rng.normal(0, 0.05, size=n_assets), 0.0)
        returns[t] = drift + betas @ factor_returns[t] + idio_shock + alpha_state[t] + jump

    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    price_df = pd.DataFrame(prices, index=dates, columns=tickers)
    return price_df, sector_map


if __name__ == "__main__":
    prices, sectors = generate_universe()
    prices.to_csv("data/synthetic_prices.csv")
    sectors.to_csv("data/sectors.csv")
    print(prices.tail())
    print(f"\n{prices.shape[1]} assets, {prices.shape[0]} trading days")
