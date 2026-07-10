"""
Risk model: covariance estimation.

Sample covariance is extremely noisy with limited history (classic
Markowitz problem — estimation error dominates for anything beyond a
handful of assets). We use Ledoit-Wolf shrinkage, which blends the sample
covariance with a structured target (scaled identity), and is the
standard practical fix used in industry.
"""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def estimate_covariance(
    returns_window: pd.DataFrame,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    returns_window : DataFrame (date x ticker) of daily returns, most
                      recent `lookback` days only (pass in pre-sliced).
    """
    lw = LedoitWolf()
    lw.fit(returns_window.values)
    cov = lw.covariance_
    if annualize:
        cov = cov * 252
    return pd.DataFrame(cov, index=returns_window.columns, columns=returns_window.columns)
