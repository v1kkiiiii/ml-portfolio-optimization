# ML-Driven Portfolio Optimization

An end-to-end quantitative pipeline that combines a machine learning
return forecaster with classical mean-variance optimization, evaluated
through a walk-forward backtest against standard benchmarks.

Built to demonstrate: factor-style feature engineering, avoiding
lookahead bias, shrinkage-based risk estimation, constrained convex
optimization, and honest out-of-sample evaluation — the core toolkit for
quant research and applied ML roles.

## Pipeline

```
Prices  →  Features  →  ML Model  →  Expected Returns (μ)
                                            │
Prices  →  Rolling Window  →  Ledoit-Wolf  →  Covariance (Σ)
                                            │
                              Mean-Variance Optimizer
                              max Sharpe, long-only, 25% cap per asset
                                            │
                              Walk-Forward Backtest
                              monthly rebalance, transaction costs
                                            │
                    Sharpe / Sortino / Max DD / Calmar vs benchmarks
```

## Why each design choice

**Data.** This sandbox has no internet access, so `data/generate_data.py`
simulates a 15-asset universe with a 3-factor macro structure, volatility
regime switching, fat-tailed jump risk, and an AR(1) "alpha state" per
asset that gives medium-term momentum (21–126 day windows) genuine,
learnable predictive power — mirroring the empirical equity momentum
literature. **To run on real data**, swap `load_prices()` for a
`yfinance` (or Bloomberg/Refinitiv) pull — every downstream module only
depends on a plain `(date × ticker)` price DataFrame, so nothing else
changes.

**Features.** Momentum (21/63/126d), realized vol (21/63d), vol-of-vol
(regime signal), price-to-moving-average, RSI(14), and cross-sectional
momentum rank. All are computed strictly from information available at
time *t*.

**Model.** Gradient Boosted Trees (`sklearn.GradientBoostingRegressor`).
Chosen over linear regression because it captures nonlinear
feature interactions, and over deep nets because with ~15 assets and a
few thousand rows, a shallow tree ensemble generalizes far better and
is what's actually used in most tabular quant-factor work.

**No lookahead, anywhere.** The model is retrained from scratch at every
monthly rebalance date, using only rows whose forward-return label window
closes *before* that date. This is the single most common bug in retail
backtests — it's handled explicitly in `src/models.py::train_predict_walk_forward`.

**Risk model.** Sample covariance is notoriously noisy with limited
history (classic Markowitz estimation-error problem). We use Ledoit-Wolf
shrinkage toward a structured target, the standard practical fix.

**Optimizer.** Constrained mean-variance optimization (`scipy.SLSQP`):
maximize Sharpe subject to full investment, long-only, and a 25% single-name
cap (prevents the optimizer from concentrating entirely in whichever
asset the model likes most — a real risk-management constraint, not
just a modeling nicety).

**Backtest.** Monthly rebalancing, 10bps transaction cost per unit of
turnover, weights drift with returns between rebalances (no artificial
daily rebalancing). Benchmarked against **equal-weight (1/N)** and
**inverse-volatility** portfolios — the two benchmarks any "smart"
portfolio construction method needs to beat to justify its complexity.

## Results (synthetic universe, 2019–2024)

| Strategy | CAGR | Ann. Vol | Sharpe | Max DD |
|---|---|---|---|---|
| ML-Optimized | 33.5% | 17.9% | 1.59 | -35.0% |
| Equal-Weight | 15.1% | 14.0% | 0.94 | -32.3% |
| Inverse-Vol | 14.7% | 13.8% | 0.92 | -31.6% |

Out-of-sample Information Coefficient (Spearman rank correlation between
predicted and realized forward returns): **~0.07**, consistent with
genuinely useful — not overfit — equity factors in practice (top
quantitative signals typically run 0.03–0.08 IC; anything above ~0.15
on real data should be treated with suspicion, not celebrated).

**Important honesty note:** these results are on a *controlled synthetic
environment* built specifically to contain a learnable signal. Real
markets are far more adversarial — this project's value is in the
methodology (no-lookahead walk-forward evaluation, shrinkage risk
estimation, constrained optimization, benchmark comparison), not in the
specific Sharpe ratio, which would compress substantially on live data.
Say this explicitly in an interview — it signals maturity, not weakness.

## Project structure

```
portfolio-ml/
├── data/
│   └── generate_data.py     # synthetic market simulator (swap for real data loader)
├── src/
│   ├── features.py          # feature engineering, no-lookahead labels
│   ├── models.py            # walk-forward ML training/prediction
│   ├── risk.py               # Ledoit-Wolf covariance estimation
│   ├── optimizer.py          # mean-variance optimization + efficient frontier
│   └── backtest.py           # walk-forward backtest engine + metrics
├── main.py                   # runs the full pipeline end to end
├── requirements.txt
└── outputs/                  # generated charts + metrics CSV
```

## Running it

```bash
pip install -r requirements.txt
python3 main.py
```

Outputs (`outputs/`):
- `cumulative_returns.png` — strategy vs benchmarks growth of $1
- `drawdown.png` — underwater equity curve
- `efficient_frontier.png` — risk/return frontier with optimal portfolio marked
- `weights_over_time.png` — portfolio composition through time
- `feature_importance.png` — which signals the model actually uses
- `backtest_metrics.csv` — full metrics table

## Extensions worth mentioning in an interview

- Swap Gradient Boosting for an LSTM/Transformer on sequential price
  data, or a linear factor model as a simpler baseline to compare against
- Add a Black-Litterman layer to blend ML views with a market-cap prior
  (reduces the optimizer's sensitivity to noisy point estimates of μ)
- Risk parity or CVaR-based objectives instead of variance
- Purged/embargoed cross-validation (López de Prado) instead of a simple
  chronological split, to more rigorously bound leakage from overlapping
  labels
- Multi-horizon ensembling (predict 5d/21d/63d returns and blend)
