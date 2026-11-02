"""Daily arbitrage of a 1 MW / 2 MWh battery.

The schedule is planned on forecast prices and settled at real prices.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

POWER_MW = 1.0
ENERGY_MWH = 2.0
ROUND_TRIP = 0.88          # round-trip efficiency
DEGRADATION = 20.0         # wear cost, PLN per MWh discharged
ETA = np.sqrt(ROUND_TRIP)  # efficiency of charging and of discharging


def schedule(prices):
    """Best charge and discharge plan for one day (linear programming).

    Variables: charge (n), discharge (n), state of charge (n). The battery is empty at the end of the day.
    """
    n = len(prices)
    cost = np.concatenate([prices, -(prices - DEGRADATION), np.zeros(n)])

    # soc[t] = soc[t-1] + ETA * charge[t] - discharge[t] / ETA
    A = np.zeros((n, 3 * n))
    for t in range(n):
        A[t, t] = -ETA
        A[t, n + t] = 1 / ETA
        A[t, 2 * n + t] = 1
        if t > 0:
            A[t, 2 * n + t - 1] = -1
    bounds = [(0, POWER_MW)] * (2 * n) + [(0, ENERGY_MWH)] * (n - 1) + [(0, 0)]

    r = linprog(cost, A_eq=A, b_eq=np.zeros(n), bounds=bounds, method="highs")
    return r.x[:n], r.x[n:2 * n]


def settle(real_prices, charge, discharge):
    """Profit in PLN for a given plan at real prices."""
    return float(real_prices @ (discharge - charge) - DEGRADATION * discharge.sum())


def evaluate(res, models):
    """Daily profit when planning with each model, and with perfect knowledge of prices."""
    out = []
    for day, g in res.sort_values("hour").groupby("day"):
        real = g["target"].to_numpy()
        row = {"day": day, "perfect": settle(real, *schedule(real))}
        for m in models:
            row[m] = settle(real, *schedule(g[m].to_numpy()))
        out.append(row)
    return pd.DataFrame(out).set_index("day")


def monthly_perfect_value(prices):
    """Maximum possible daily profit, averaged by month."""
    daily = {}
    for day, g in prices.groupby(prices.index.normalize()):
        p = g.to_numpy()
        if len(p) == 24:
            daily[day] = settle(p, *schedule(p))
    s = pd.Series(daily)
    return s.groupby(s.index.to_period("M")).mean()
