"""
Covariance estimation. Sample covariance gets noisy fast once you have
more than a handful of assets relative to your history length - this is
the classic Markowitz estimation-error problem. Ledoit-Wolf shrinkage
toward a structured target is the standard practical fix, so that's
what we use here instead of raw sample cov.
"""
import pandas as pd
from sklearn.covariance import LedoitWolf


def estimate_covariance(returns_window: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    lw = LedoitWolf()
    lw.fit(returns_window.values)
    cov = lw.covariance_
    if annualize:
        cov = cov * 252
    return pd.DataFrame(cov, index=returns_window.columns, columns=returns_window.columns)
