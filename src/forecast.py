"""Features, models and backtest for day-ahead price forecasting.

The forecast for day d is made in the morning of d-1, before the auction closes at 12:00.
At that time prices are known up to day d-1.
"""
import holidays
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from data import to_panel

LINEAR_FEATURES = ["p_d1", "p_d2", "p_d7", "p_d1_min", "p_d1_max", "p_d1_last",
                   "demand_fc", "wind_fc", "pv_fc", "holiday",
                   "dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6"]
GB_FEATURES = ["p_d1", "p_d2", "p_d7", "p_d1_mean", "p_d1_min", "p_d1_max", "p_d1_last",
               "demand_fc", "wind_fc", "pv_fc", "residual_load_fc", "hour", "dow", "holiday",
               "month_sin", "month_cos"]


def build_features(hourly, fc_lag=0):
    """One row per (day, hour).

    fc_lag=0 uses the PSE plan for day d (published on d-1, maybe after 12:00).
    fc_lag=1 uses the plan for day d-1, which is surely known before 12:00.
    """
    P = to_panel(hourly["price"])
    D = to_panel(hourly["demand_fc"]).shift(fc_lag)
    W = to_panel(hourly["wind_fc"]).shift(fc_lag)
    S = to_panel(hourly["pv_fc"]).shift(fc_lag)

    rows = []
    for h in range(24):
        f = pd.DataFrame(index=P.index)
        f["hour"] = h
        f["target"] = P[h]
        f["p_d1"] = P[h].shift(1)          # same hour yesterday
        f["p_d2"] = P[h].shift(2)
        f["p_d7"] = P[h].shift(7)          # same hour a week ago
        f["p_d1_mean"] = P.mean(axis=1).shift(1)
        f["p_d1_min"] = P.min(axis=1).shift(1)
        f["p_d1_max"] = P.max(axis=1).shift(1)
        f["p_d1_last"] = P[23].shift(1)    # last hour of yesterday
        f["demand_fc"] = D[h]
        f["wind_fc"] = W[h]
        f["pv_fc"] = S[h]
        f["residual_load_fc"] = D[h] - W[h] - S[h]
        rows.append(f)

    X = pd.concat(rows).reset_index(names="day")
    pl_holidays = pd.to_datetime(list(holidays.Poland(years=range(2024, 2028)).keys()))
    X["dow"] = X["day"].dt.dayofweek
    X["holiday"] = X["day"].isin(pl_holidays).astype(int)
    X["month_sin"] = np.sin(2 * np.pi * X["day"].dt.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["day"].dt.month / 12)
    for k in range(7):
        X[f"dow_{k}"] = (X["dow"] == k).astype(int)
    return X.dropna().sort_values(["day", "hour"]).reset_index(drop=True)


def naive_weekly(test):
    return test["p_d7"].to_numpy()


def naive_similar_day(test):
    # Monday and weekend: price from a week ago, other days: yesterday's price
    use_week = test["dow"].isin([0, 5, 6])
    return np.where(use_week, test["p_d7"], test["p_d1"])


def fit_predict_linear(train, test):
    """Separate Ridge regression for each hour."""
    out = np.empty(len(test))
    for h in range(24):
        tr, te = train[train.hour == h], test[test.hour == h]
        if te.empty:
            continue
        Xtr = tr[LINEAR_FEATURES].to_numpy(float)
        Xte = te[LINEAR_FEATURES].to_numpy(float)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        model = Ridge(alpha=1.0).fit((Xtr - mu) / sd, tr["target"])
        out[(test.hour == h).to_numpy()] = model.predict((Xte - mu) / sd)
    return out


def make_gb(quantile=None):
    params = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=False, random_state=0)
    if quantile is None:
        return HistGradientBoostingRegressor(**params)
    return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **params)


def fit_predict_gb(train, test, quantiles=True):
    Xtr, Xte, y = train[GB_FEATURES], test[GB_FEATURES], train["target"]
    out = {"gb": make_gb().fit(Xtr, y).predict(Xte)}
    if quantiles:
        out["q10"] = make_gb(0.1).fit(Xtr, y).predict(Xte)
        out["q90"] = make_gb(0.9).fit(Xtr, y).predict(Xte)
    return out


def walk_forward(X, test_start, retrain_days=28, quantiles=True):
    """Train on all data before each block of `retrain_days` days, then forecast the block."""
    days = np.sort(X.loc[X.day >= test_start, "day"].unique())
    blocks = [days[i:i + retrain_days] for i in range(0, len(days), retrain_days)]
    out = []
    for block in blocks:
        train = X[X.day < block[0]]
        test = X[X.day.isin(block)].copy()
        test["naive_weekly"] = naive_weekly(test)
        test["naive_similar_day"] = naive_similar_day(test)
        test["linear"] = fit_predict_linear(train, test)
        for name, pred in fit_predict_gb(train, test, quantiles).items():
            test[name] = pred
        out.append(test)
    return pd.concat(out).reset_index(drop=True)


def empirical_interval(res, model="gb", window_days=56, level=0.8):
    """Prediction interval from the model's errors in the last `window_days` days (same hour)."""
    res = res.sort_values(["day", "hour"]).copy()
    err = res["target"] - res[model]
    lo = np.full(len(res), np.nan)
    hi = np.full(len(res), np.nan)
    a = (1 - level) / 2
    for h in range(24):
        idx = res.index[res.hour == h]
        # the error for day d is known on d+1, and the forecast for d is made on d-1
        past = err.loc[idx].shift(2)
        pos = res.index.get_indexer(idx)
        lo[pos] = (res.loc[idx, model] + past.rolling(window_days, min_periods=21).quantile(a)).to_numpy()
        hi[pos] = (res.loc[idx, model] + past.rolling(window_days, min_periods=21).quantile(1 - a)).to_numpy()
    res["int_lo"], res["int_hi"] = lo, hi
    return res


def metrics(res, models):
    y = res["target"]
    base = (res["naive_similar_day"] - y).abs().mean()
    rows = {}
    for m in models:
        e = res[m] - y
        rows[m] = {"MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
                   "rMAE": e.abs().mean() / base, "bias": e.mean()}
    return pd.DataFrame(rows).T


def diebold_mariano(res, a, b):
    """Diebold-Mariano test on daily mean absolute errors. Negative statistic: model a is better."""
    la = (res[a] - res["target"]).abs().groupby(res["day"]).mean()
    lb = (res[b] - res["target"]).abs().groupby(res["day"]).mean()
    d = (la - lb).to_numpy()
    stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    p_value = 2 * stats.norm.sf(abs(stat))
    return float(stat), float(p_value)
