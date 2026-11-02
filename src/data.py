"""Read data/pse_hourly.csv and make sure every day has 24 hours."""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "pse_hourly.csv"
COLUMNS = ["price", "demand_fc", "wind_fc", "pv_fc", "demand", "wind", "pv"]


def load(path=DATA):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run src/download_pse.py first")
    d = pd.read_csv(path)
    # local time without time zone; on DST days drop the repeated hour and fill the missing one
    local = pd.to_datetime(d["start_local"], utc=True).dt.tz_convert("Europe/Warsaw").dt.tz_localize(None)
    s = d.assign(start=local).set_index("start")[COLUMNS].astype(float)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    full = pd.date_range(s.index.min().normalize(), s.index.max().normalize() + pd.Timedelta(hours=23), freq="h")
    s = s.reindex(full).interpolate(limit=3)
    hours_per_day = s["price"].groupby(s.index.normalize()).count()
    full_days = hours_per_day[hours_per_day == 24].index
    return s[s.index.normalize().isin(full_days)]


def to_panel(s):
    """Hourly series to a table: one row per day, 24 columns."""
    df = s.to_frame("v")
    df["day"] = df.index.normalize()
    df["hour"] = df.index.hour
    return df.pivot(index="day", columns="hour", values="v")


if __name__ == "__main__":
    h = load()
    print(h.index.min(), "-", h.index.max(), len(h) // 24, "days")
    print(h.describe().round(1).T)
