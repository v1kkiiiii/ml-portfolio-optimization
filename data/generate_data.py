"""
Synthetic multi-asset market data generator.

Why synthetic data? This sandbox has no internet access, so we can't pull
live data from yfinance/Bloomberg/etc. Instead we simulate a realistic
multi-factor market so the full pipeline (features -> ML -> optimization ->
backtest) can be demonstrated end to end.

To use REAL data instead, replace `load_prices()` in main.py with e.g.:

    import yfinance as yf
    prices = yf.download(tickers, start="2015-01-01")["Adj Close"]

everything downstream (features, models, optimizer, backtest) is written
against a plain (date x ticker) price DataFrame, so it doesn't care where
the prices came from.

Simulation design:
- 3 latent macro factors (rates/growth/risk-sentiment) that drive all assets
  via random factor loadings (like a Barra-style risk model).
- Each asset also gets idiosyncratic momentum: returns are weakly
  autocorrelated over a 20-60 day lookback, which is what gives the ML
  model something real to learn (pure random walk prices would make any
  "prediction" model equivalent to noise).
- Regime shifts (volatility clusters) and fat-tailed shocks are added so
  the backtest has realistic drawdowns to manage, not just smooth drift.
"""
import numpy as np
import pandas as pd


def generate_universe(
    n_assets: int = 15,
    n_days: int = 1500,
    start_date: str = "2019-01-01",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    prices : DataFrame (date x ticker)
    sector_map : DataFrame mapping ticker -> sector, for reference
    """
    rng = np.random.default_rng(seed)

    tickers = [f"ASSET_{i:02d}" for i in range(n_assets)]
    sectors = ["Tech", "Financials", "Energy", "Healthcare", "Consumer"]
    sector_map = pd.DataFrame({
        "ticker": tickers,
        "sector": rng.choice(sectors, size=n_assets),
    }).set_index("ticker")

    dates = pd.bdate_range(start=start_date, periods=n_days)

    # ---- 3 latent macro factors (daily innovations, mildly autocorrelated) ----
    n_factors = 3
    factor_returns = np.zeros((n_days, n_factors))
    phi = 0.05  # slight factor autocorrelation (macro persistence)
    factor_vol = np.array([0.006, 0.008, 0.010])
    for t in range(1, n_days):
        factor_returns[t] = (
            phi * factor_returns[t - 1]
            + rng.normal(0, factor_vol, size=n_factors)
        )

    # ---- volatility regime (2-state Markov-ish clustering) ----
    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        if regime[t - 1] == 0:
            regime[t] = 1 if rng.random() < 0.01 else 0   # rare shift to high-vol
        else:
            regime[t] = 0 if rng.random() < 0.05 else 1   # high-vol mean-reverts faster
    regime_mult = np.where(regime == 1, 2.2, 1.0)

    # ---- per-asset factor loadings + idiosyncratic params ----
    betas = rng.normal(0.5, 0.4, size=(n_assets, n_factors))
    idio_vol = rng.uniform(0.010, 0.022, size=n_assets)
    drift = rng.uniform(-0.0001, 0.0005, size=n_assets)  # small daily alpha differences

    # Slow-moving "alpha state" per asset: an AR(1) process with high
    # persistence (~40 trading day half-life). This is what makes
    # medium-term momentum (21-126d) an actually learnable signal for a
    # 21-day-forward prediction horizon -- mirrors real equity momentum,
    # where trends persist over weeks-to-months, not single days.
    alpha_phi = 0.95           # ~13.5 trading day half-life
    alpha_innov_vol = rng.uniform(0.0006, 0.0013, size=n_assets)
    alpha_state = np.zeros((n_days, n_assets))
    for t in range(1, n_days):
        alpha_state[t] = (
            alpha_phi * alpha_state[t - 1]
            + rng.normal(0, alpha_innov_vol, size=n_assets)
        )

    returns = np.zeros((n_days, n_assets))
    for t in range(n_days):
        idio_shock = rng.normal(0, idio_vol, size=n_assets) * regime_mult[t]
        # fat tails: occasional jump shocks
        jump = np.where(rng.random(n_assets) < 0.003,
                         rng.normal(0, 0.05, size=n_assets), 0.0)
        r = drift + betas @ factor_returns[t] + idio_shock + alpha_state[t] + jump
        returns[t] = r

    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    price_df = pd.DataFrame(prices, index=dates, columns=tickers)
    return price_df, sector_map


if __name__ == "__main__":
    prices, sectors = generate_universe()
    prices.to_csv("/home/claude/portfolio-ml/data/synthetic_prices.csv")
    sectors.to_csv("/home/claude/portfolio-ml/data/sectors.csv")
    print(prices.tail())
    print(f"\nGenerated {prices.shape[1]} assets over {prices.shape[0]} trading days")
