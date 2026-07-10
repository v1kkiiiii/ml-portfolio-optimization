"""
Feature engineering for the return prediction model. Everything here
only looks backward from time t - the label (fwd_ret) is the only
forward-looking column, and it's the thing we're trying to predict, not
a feature.
"""
import numpy as np
import pandas as pd


def compute_features(prices: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    daily_ret = prices.pct_change()
    rows = []

    for ticker in prices.columns:
        p = prices[ticker]
        r = daily_ret[ticker]

        mom_21 = p.pct_change(21)
        mom_63 = p.pct_change(63)
        mom_126 = p.pct_change(126)
        vol_21 = r.rolling(21).std() * np.sqrt(252)
        vol_63 = r.rolling(63).std() * np.sqrt(252)
        vol_of_vol = vol_21.rolling(21).std()

        ma_50 = p.rolling(50).mean()
        ma_ratio = p / ma_50 - 1

        # RSI(14)
        delta = p.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_14 = 100 - (100 / (1 + rs))

        fwd_ret = p.shift(-horizon) / p - 1

        rows.append(pd.DataFrame({
            "date": p.index,
            "ticker": ticker,
            "mom_21": mom_21.values,
            "mom_63": mom_63.values,
            "mom_126": mom_126.values,
            "vol_21": vol_21.values,
            "vol_63": vol_63.values,
            "vol_of_vol": vol_of_vol.values,
            "ma_ratio": ma_ratio.values,
            "rsi_14": rsi_14.values,
            "fwd_ret": fwd_ret.values,
        }))

    panel = pd.concat(rows, ignore_index=True)

    # relative strength vs the rest of the universe on a given date
    panel["xsec_mom_rank"] = panel.groupby("date")["mom_63"].rank(pct=True)

    return panel.dropna().reset_index(drop=True)


FEATURE_COLS = [
    "mom_21", "mom_63", "mom_126",
    "vol_21", "vol_63", "vol_of_vol",
    "ma_ratio", "rsi_14", "xsec_mom_rank",
]
