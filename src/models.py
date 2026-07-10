"""
ML expected-return model, trained walk-forward to avoid lookahead bias.

At each rebalance date t, we train ONLY on data with fwd_ret labels that
are fully resolved before t (i.e. label windows that don't peek into the
future relative to t), then predict expected returns for every asset as of t.

Model: Gradient Boosted Trees (sklearn). Chosen over linear regression
because it captures nonlinear interactions between momentum/vol/RSI
features, and over deep nets because with ~15 assets x a few thousand
rows, a boosted-tree ensemble is far less prone to overfitting and is
standard in quant research for tabular factor data.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLS


def train_predict_walk_forward(
    panel: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    horizon: int = 21,
    min_train_rows: int = 500,
) -> pd.DataFrame:
    """
    For each rebalance date, train a fresh model on all label-resolved
    history strictly before that date, then predict expected returns for
    every asset as of that date.

    Returns a DataFrame indexed by (date, ticker) with column 'pred_ret'.
    """
    panel = panel.sort_values("date").reset_index(drop=True)
    preds = []

    for rdate in rebalance_dates:
        # A label fwd_ret at date d uses prices up to d + horizon days.
        # To avoid lookahead, only train on rows whose label window closes
        # before rdate.
        cutoff = rdate - pd.tseries.offsets.BDay(horizon)
        train = panel[panel["date"] <= cutoff]

        if len(train) < min_train_rows:
            continue  # not enough history yet, skip this rebalance date

        X_train = train[FEATURE_COLS].values
        y_train = train["fwd_ret"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train_s, y_train)

        # predict using the most recent feature row available for each
        # ticker as of rdate (last observation on/before rdate)
        asof = panel[panel["date"] <= rdate].groupby("ticker").tail(1)
        if asof.empty:
            continue
        X_pred = scaler.transform(asof[FEATURE_COLS].values)
        pred_ret = model.predict(X_pred)

        out = pd.DataFrame({
            "date": rdate,
            "ticker": asof["ticker"].values,
            "pred_ret": pred_ret,
        })
        preds.append(out)

    if not preds:
        raise RuntimeError(
            "No predictions generated — increase n_days in data generation "
            "or lower min_train_rows."
        )
    return pd.concat(preds, ignore_index=True)


def model_diagnostics(panel: pd.DataFrame, cutoff_date) -> dict:
    """
    Quick in-sample vs out-of-sample R^2 / IC (information coefficient)
    check on a single split, useful for a README/report screenshot.
    """
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr

    train = panel[panel["date"] <= cutoff_date]
    test = panel[panel["date"] > cutoff_date]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS])
    y_train = train["fwd_ret"].values

    model = GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
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
