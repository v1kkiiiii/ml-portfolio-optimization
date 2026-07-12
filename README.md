# ML-Driven Portfolio Optimization

An ML-driven portfolio construction system built to demonstrate the full quant research stack — from features to a properly out-of-sample-tested Sharpe ratio.

Combines a gradient-boosted return forecaster with mean-variance
optimization, evaluated with a proper walk-forward backtest against
equal-weight and inverse-vol benchmarks.

## Pipeline

```
Prices -> Features -> ML Model -> Expected Returns (mu)
                                         |
Prices -> Rolling Window -> Ledoit-Wolf -> Covariance (Sigma)
                                         |
                          Mean-Variance Optimizer
                     max Sharpe, long-only, 25% cap per asset
                                         |
                             Walk-Forward Backtest
                       monthly rebalance, transaction costs
                                         |
                Sharpe / Sortino / Max DD / Calmar vs benchmarks
```

## Notes on the design choices

**Data.** No internet access in the environment I built this in, so
`data/generate_data.py` simulates a 15-asset universe: a 3-factor macro
structure, vol regime switching, occasional jumps, and a slow-moving
per-asset alpha state that gives medium-term momentum (21-126 day
windows) real predictive power - basically trying to mirror the
empirical equity momentum literature instead of just being noise dressed
up as data. To run this on real data, swap out `load_prices()` for a
`yfinance` pull - everything downstream just expects a plain
`(date x ticker)` price DataFrame.

**Features.** Momentum at a few horizons, realized vol, vol-of-vol
(regime proxy), price vs moving average, RSI, and cross-sectional
momentum rank. All computed strictly from information available at time t.

**Model.** Gradient boosted trees. Went with this over a linear model
because it picks up nonlinear feature interactions, and over a neural
net because with ~15 assets and a few thousand rows a shallow tree
ensemble generalizes a lot better and doesn't need nearly as much data.

**No lookahead.** The model gets retrained from scratch at every monthly
rebalance, using only rows whose forward-return label window has
actually closed before that date. This is the most common bug in
homegrown backtests - handled in `src/models.py::train_predict_walk_forward`.

**Risk model.** Sample covariance is noisy with limited history (the
classic Markowitz estimation-error issue). Ledoit-Wolf shrinkage toward
a structured target fixes this and is standard practice.

**Optimizer.** Constrained mean-variance via `scipy.SLSQP` - max Sharpe,
fully invested, long-only, 25% single-name cap so the optimizer can't
just dump everything into whatever the model likes most that month.

**Backtest.** Monthly rebalancing, 10bps transaction cost per unit
turnover, weights drift with returns between rebalances rather than
being artificially reset daily. Benchmarked against equal-weight and
inverse-vol - the two things any "smarter" weighting scheme actually
needs to beat.

## Results (synthetic universe, 2019-2024)

| Strategy | CAGR | Ann. Vol | Sharpe | Max DD |
|---|---|---|---|---|
| ML-Optimized | 33.5% | 17.9% | 1.59 | -35.0% |
| Equal-Weight | 15.1% | 14.0% | 0.94 | -32.3% |
| Inverse-Vol | 14.7% | 13.8% | 0.92 | -31.6% |

![Cumulative Returns](outputs/cumulative_returns.png)
![Efficient Frontier](outputs/efficient_frontier.png)

Out-of-sample information coefficient (spearman rank corr between
predicted and realized forward returns) came out to about 0.07, which is
in line with genuinely useful equity factors in practice - most good
signals run 0.03-0.08 IC. Anything much higher than that on real data
would be a red flag, not a win.

Worth saying plainly: this is a controlled synthetic environment built
specifically to contain a learnable signal, so these numbers won't hold
up as-is on live markets. The point of the project is the methodology -
no-lookahead walk-forward evaluation, shrinkage risk estimation,
constrained optimization, actually benchmarking against something -
not the specific Sharpe ratio.

## Structure

```
portfolio-ml/
├── data/
│   └── generate_data.py     synthetic market simulator
├── src/
│   ├── features.py          feature engineering, no-lookahead labels
│   ├── models.py             walk-forward ML training/prediction
│   ├── risk.py                Ledoit-Wolf covariance estimation
│   ├── optimizer.py           mean-variance optimization + efficient frontier
│   └── backtest.py            walk-forward backtest engine + metrics
├── main.py                    runs the full pipeline
├── requirements.txt
└── outputs/                   generated charts + metrics
```

## Running it

```bash
pip install -r requirements.txt
python3 main.py
```

Outputs land in `outputs/`: cumulative returns, drawdown, efficient
frontier, weights over time, feature importance, and a metrics CSV.

## Possible extensions

- Black-Litterman to blend the ML's views with a market-cap prior, reducing sensitivity to noisy point estimates of mu
- Purged/embargoed cross-validation instead of a simple chronological split, to more rigorously bound label leakage