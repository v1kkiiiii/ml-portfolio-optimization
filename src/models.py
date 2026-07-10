"""
Walk-forward ML model for expected returns. The important part here is
avoiding lookahead: at each rebalance date we only train on rows whose
label window has actually closed by that point. It's a common mistake
to just do a single train/test split and call it a day, which quietly
leaks future information into a "backtest."

Using gradient boosted trees rather than something fancier - with ~15
assets and a few thousand rows a shallow tree ensemble generalizes
better than a neural net would, and it's honestly what most tabular
quant factor work actually uses.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLS


def train_predict_walk_forward(panel, rebalance_dates, horizon=21, min_train_rows=500):
    panel = panel.sort_values("date").reset_index(drop=True)
    preds = []

    for rdate in rebalance_dates:
        cutoff = rdate - pd.tseries.offsets.BDay(horizon)
        train = panel[panel["date"] <= cutoff]

        if len(train) < min_train_rows:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[FEATURE_COLS].values)
        y_train = train["fwd_ret"].values

        model = GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        model.fit(X_train, y_train)

        # use the most recent row available per ticker as of rdate
        asof = panel[panel["date"] <= rdate].groupby("ticker").tail(1)
        if asof.empty:
            continue
        X_pred = scaler.transform(asof[FEATURE_COLS].values)
        pred_ret = model.predict(X_pred)

        preds.append(pd.DataFrame({
            "date": rdate,
            "ticker": asof["ticker"].values,
            "pred_ret": pred_ret,
        }))

    if not preds:
        raise RuntimeError("no predictions generated - check min_train_rows vs data length")
    return pd.concat(preds, ignore_index=True)


def model_diagnostics(panel, cutoff_date):
    """One-off train/test split just to sanity check the model - not used in the backtest itself."""
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr

    train = panel[panel["date"] <= cutoff_date]
    test = panel[panel["date"] > cutoff_date]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS])
    y_train = train["fwd_ret"].values

    model = GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42,
    )
    model.fit(X_train, y_train)

    X_test = scaler.transform(test[FEATURE_COLS])
    y_test = test["fwd_ret"].values
    y_pred = model.predict(X_test)

    ic, _ = spearmanr(y_pred, y_test)
    return {
        "train_r2": r2_score(y_train, model.predict(X_train)),
        "test_r2": r2_score(y_test, y_pred),
        "test_information_coefficient": ic,
        "feature_importance": dict(zip(FEATURE_COLS, model.feature_importances_)),
    }
