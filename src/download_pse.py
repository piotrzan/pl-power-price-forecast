"""Download hourly data from the PSE API (api.raporty.pse.pl, no key needed).

    python3 src/download_pse.py              # from 2024-06-14 to yesterday
    python3 src/download_pse.py 2025-01-01   # from a chosen date

Saves data/pse_hourly.csv with one row per hour (Polish time):
    price                    day-ahead price for Poland, PLN/MWh (report csdac-pln)
    demand_fc, wind_fc, pv_fc  PSE plan for the next day, MW (report pdgopkd)
    demand, wind, pv         actual values, MW (report his-wlk-cal)

The API gives 15-minute values, they are averaged to hours. Uses only the standard library.
"""
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = "https://api.raporty.pse.pl/api"
API_START = date(2024, 6, 14)
WARSAW = ZoneInfo("Europe/Warsaw")
OUT = Path(__file__).resolve().parents[1] / "data" / "pse_hourly.csv"

# report: {field in the API: column in the csv}
REPORTS = {
    "csdac-pln": {"csdac_pln": "price"},
    "pdgopkd": {"kse_pow_dem": "demand_fc", "gen_wi": "wind_fc", "gen_fv": "pv_fc"},
    "his-wlk-cal": {"demand": "demand", "wi": "wind", "pv": "pv"},
}
COLUMNS = ["price", "demand_fc", "wind_fc", "pv_fc", "demand", "wind", "pv"]


def get_json(url, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                       "User-Agent": "pl-power-price-forecast"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if attempt == retries - 1:
                raise
            print("    retry after error:", e)
            time.sleep(5 * (attempt + 1))


def records(report, fields, first, last):
    """All rows of one report between two dates (the API splits long answers into pages)."""
    query = {
        "$filter": "business_date ge '{}' and business_date le '{}'".format(first, last),
        "$select": ",".join(["dtime_utc", "business_date"] + list(fields)),
        "$first": 100000,
    }
    url = "{}/{}?{}".format(BASE, report, urllib.parse.urlencode(query, quote_via=urllib.parse.quote))
    out = []
    while url:
        payload = get_json(url)
        out.extend(payload.get("value", []))
        url = payload.get("nextLink")
    return out


def hour_start_local(dtime_utc):
    """The API gives the end of each 15-minute interval in UTC. Return the start of its hour in Polish time."""
    end = datetime.fromisoformat(dtime_utc.strip()[:19]).replace(tzinfo=timezone.utc)
    start = (end - timedelta(minutes=15)).astimezone(WARSAW)
    return start.replace(minute=0, second=0, microsecond=0)


def aggregate(rows, fields, table):
    """Add 15-minute values to hourly sums: table[hour][column] = [sum, count]."""
    for row in rows:
        h = hour_start_local(row["dtime_utc"])
        for api_field, col in fields.items():
            v = row.get(api_field)
            if v is None or v == "":
                continue
            cell = table[h][col]
            cell[0] += float(v)
            cell[1] += 1


def months(first, last):
    m = date(first.year, first.month, 1)
    while m <= last:
        nxt = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
        yield max(m, first), min(nxt - timedelta(days=1), last)
        m = nxt


def main():
    first = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else API_START
    last = date.today() - timedelta(days=1)
    table = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for report, fields in REPORTS.items():
        print(report)
        for a, b in months(first, last):
            rows = records(report, fields, a, b)
            aggregate(rows, fields, table)
            print("  {} .. {}: {} rows".format(a, b, len(rows)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_local"] + COLUMNS)
        for h in sorted(table):
            vals = []
            for c in COLUMNS:
                s, n = table[h].get(c, (0.0, 0))
                vals.append("{:.2f}".format(s / n) if n else "")
            w.writerow([h.isoformat()] + vals)
    print("saved {} hours to {}".format(len(table), OUT))


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            print("SSL certificate error. On macOS with Python from python.org run once:\n"
                  "  /Applications/Python\\ 3*/Install\\ Certificates.command\n"
                  "or use the system Python: /usr/bin/python3 src/download_pse.py")
        raise
