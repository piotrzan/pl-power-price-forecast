"""Run everything: backtest, battery evaluation, summary and charts.

    python3 src/run.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.inspection import permutation_importance

import battery
from data import load
from forecast import (GB_FEATURES, build_features, diebold_mariano, empirical_interval, make_gb,
                      metrics, walk_forward)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
FIG = OUT / "figures"
TEST_START = "2025-07-01"
MODELS = ["naive_weekly", "naive_similar_day", "linear", "gb"]
LABELS = {"naive_weekly": "Naiwny (tydzień temu)", "naive_similar_day": "Naiwny (podobny dzień)",
          "linear": "Regresja (24 modele)", "gb": "Gradient boosting"}
FEATURE_NAMES = {
    "p_d1": "cena wczoraj (ta godzina)", "p_d2": "cena 2 dni temu", "p_d7": "cena tydzień temu",
    "p_d1_mean": "średnia cena wczoraj", "p_d1_min": "min. cena wczoraj", "p_d1_max": "maks. cena wczoraj",
    "p_d1_last": "cena wczoraj o 23:00", "demand_fc": "prognoza zapotrzebowania",
    "wind_fc": "prognoza wiatru", "pv_fc": "prognoza fotowoltaiki",
    "residual_load_fc": "zapotrzebowanie minus wiatr i PV", "hour": "godzina", "dow": "dzień tygodnia",
    "holiday": "święto", "month_sin": "pora roku (sin)", "month_cos": "pora roku (cos)",
}

DARK, GREY, BLUE = "#1f2a33", "#7b8794", "#2459d6"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e8ec", "axes.axisbelow": True,
                     "figure.dpi": 130, "savefig.bbox": "tight"})


def backtest(X, name, **kwargs):
    """Run the backtest once and keep the result in reports/ (it takes about a minute)."""
    path = OUT / f"forecasts_{name}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["day"])
    res = walk_forward(X, TEST_START, **kwargs)
    res.to_csv(path, index=False)
    return res


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    hourly = load()
    prices = hourly["price"]

    # forecasts with the PSE plan for day d, and with the plan for day d-1 only
    X = build_features(hourly, fc_lag=0)
    res = empirical_interval(backtest(X, "plan_d"))
    res_d1 = backtest(build_features(hourly, fc_lag=1), "plan_d1", quantiles=False)

    m = metrics(res, MODELS)
    m_d1 = metrics(res_d1, MODELS)
    dm_stat, dm_p = diebold_mariano(res, "gb", "linear")
    dm_stat_d1, dm_p_d1 = diebold_mariano(res_d1, "gb", "linear")

    inside_q = (res.target >= res.q10) & (res.target <= res.q90)
    ok = res.int_lo.notna()
    inside = ((res.target >= res.int_lo) & (res.target <= res.int_hi))[ok]
    width = (res.int_hi - res.int_lo)[ok].mean()

    err = res.gb - res.target
    by_quarter = {}
    for q, g in res.groupby(res.day.dt.to_period("Q").astype(str)):
        by_quarter[q] = {"MAE_gb": round((g.gb - g.target).abs().mean(), 1),
                         "MAE_naive": round((g.naive_similar_day - g.target).abs().mean(), 1),
                         "bias_gb": round((g.gb - g.target).mean(), 1),
                         "mean_price": round(g.target.mean(), 1)}
    mae_by_hour = err.abs().groupby(res.hour).mean()

    # feature importance: model trained on data until the end of 2025, tested on 2026
    train, test = X[X.day < "2026-01-01"], X[X.day >= "2026-01-01"]
    model = make_gb().fit(train[GB_FEATURES], train.target)
    pi = permutation_importance(model, test[GB_FEATURES], test.target, n_repeats=5, random_state=0,
                                scoring="neg_mean_absolute_error")
    importance = pd.Series(pi.importances_mean, index=GB_FEATURES).sort_values()

    negative_hours = (prices < 0).groupby(prices.index.year).sum()
    profile = prices.groupby([prices.index.year, prices.index.hour]).mean().unstack(0)

    pnl = battery.evaluate(res, MODELS)
    pnl_total = pnl.sum()
    share_of_max = (pnl_total / pnl_total["perfect"]).drop("perfect")
    monthly_max = battery.monthly_perfect_value(prices)

    summary = {
        "data_period": [str(hourly.index.min().date()), str(hourly.index.max().date())],
        "test_period": [str(res.day.min().date()), str(res.day.max().date())],
        "test_days": int(res.day.nunique()),
        "mean_price_test": round(float(res.target.mean()), 1),
        "metrics_plan_d": m.round(2).to_dict(orient="index"),
        "metrics_plan_d1": m_d1.round(2).to_dict(orient="index"),
        "dm_gb_vs_linear_plan_d": {"stat": round(dm_stat, 2), "p_value": round(dm_p, 4)},
        "dm_gb_vs_linear_plan_d1": {"stat": round(dm_stat_d1, 2), "p_value": round(dm_p_d1, 4)},
        "interval_quantile_gb_coverage": round(float(inside_q.mean()), 3),
        "interval_empirical": {"coverage": round(float(inside.mean()), 3), "mean_width": round(float(width), 1)},
        "by_quarter": by_quarter,
        "mae_gb_by_hour": mae_by_hour.round(1).to_dict(),
        "importance_mae_increase": importance.sort_values(ascending=False).round(2).to_dict(),
        "negative_price_hours_by_year": {int(k): int(v) for k, v in negative_hours.items()},
        "battery_pnl_per_day_pln": (pnl_total / len(pnl)).round(0).to_dict(),
        "battery_share_of_max": share_of_max.round(3).to_dict(),
        "battery_max_per_day_by_month": {str(k): v for k, v in monthly_max.round(0).items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    colors = [GREY, GREY, DARK, BLUE]

    # 1. error of each model
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    y = np.arange(len(MODELS))
    ax.barh(y + 0.2, m.loc[MODELS, "MAE"], height=0.38, color=colors)
    ax.barh(y - 0.2, m_d1.loc[MODELS, "MAE"], height=0.38, color=colors, alpha=0.4)
    for i, k in enumerate(MODELS):
        ax.text(m.loc[k, "MAE"] + 1, i + 0.2, f"{m.loc[k, 'MAE']:.0f}", va="center", fontsize=9)
        ax.text(m_d1.loc[k, "MAE"] + 1, i - 0.2, f"{m_d1.loc[k, 'MAE']:.0f}", va="center", fontsize=9, color=GREY)
    ax.set_yticks(y, [LABELS[k] for k in MODELS])
    ax.set_xlabel("Średni błąd bezwzględny, MAE [PLN/MWh]")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=DARK, label="z prognozą PSE na dzień d"),
                       Patch(color=DARK, alpha=0.4, label="tylko dane znane przed 12:00 (prognoza PSE na d−1)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8.5)
    ax.set_title(f"Błąd prognozy, test {summary['test_period'][0]} – {summary['test_period'][1]}",
                 loc="left", pad=34)
    fig.savefig(FIG / "mae_by_model.png")
    plt.close(fig)

    # 2. last two weeks of forecasts
    last = res[res.day > res.day.max() - pd.Timedelta(days=14)]
    t = last.day + pd.to_timedelta(last.hour, unit="h")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.fill_between(t, last.int_lo, last.int_hi, color=BLUE, alpha=0.15, lw=0, label="Przedział 80%")
    ax.plot(t, last.target, color=DARK, lw=1.3, label="Cena rzeczywista")
    ax.plot(t, last.gb, color=BLUE, lw=1.3, label="Prognoza (gradient boosting)")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.set_ylabel("PLN/MWh")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=3)
    ax.set_title("Prognoza na dzień następny, ostatnie dwa tygodnie", loc="left", pad=28)
    fig.savefig(FIG / "example_forecast.png")
    plt.close(fig)

    # 3. feature importance
    top = importance.tail(8)
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.barh([FEATURE_NAMES[k] for k in top.index], top.values, color=BLUE)
    ax.set_xlabel("Wzrost MAE po przetasowaniu zmiennej [PLN/MWh]")
    ax.grid(axis="y", visible=False)
    ax.set_title("Co wpływa na cenę (2026)", loc="left")
    fig.savefig(FIG / "importance.png")
    plt.close(fig)

    # 4. average price by hour of day
    first_year, last_year = profile.columns.min(), profile.columns.max()
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for year, color in zip(profile.columns, [GREY, DARK, BLUE]):
        if year == first_year:
            label = f"{year} (od czerwca)"
        elif year == last_year:
            label = f"{year} (do {hourly.index.max():%d.%m})"
        else:
            label = str(year)
        ax.plot(profile.index, profile[year], color=color, lw=1.8, label=label)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Godzina doby")
    ax.set_ylabel("Średnia cena [PLN/MWh]")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Średnia cena w ciągu doby", loc="left")
    fig.savefig(FIG / "price_profile.png")
    plt.close(fig)

    # 5. battery
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.1), gridspec_kw={"width_ratios": [1, 1.25]})
    share = share_of_max.loc[MODELS] * 100
    a1.barh([LABELS[k] for k in MODELS], share, color=colors)
    for i, v in enumerate(share):
        a1.text(v + 1, i, f"{v:.0f}%", va="center")
    a1.set_xlim(0, 100)
    a1.set_xlabel("% maksymalnego zysku")
    a1.grid(axis="y", visible=False)
    a1.set_title("Magazyn 1 MW / 2 MWh", loc="left")
    a2.bar(range(len(monthly_max)), monthly_max.values, color=BLUE, width=0.8)
    ticks = [i for i, k in enumerate(monthly_max.index) if k.month in (1, 7)]
    a2.set_xticks(ticks, [monthly_max.index[i].strftime("%m.%Y") for i in ticks])
    a2.set_ylabel("PLN dziennie")
    a2.grid(axis="x", visible=False)
    a2.set_title("Maksymalny zysk dzienny, średnio w miesiącu", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "battery.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
