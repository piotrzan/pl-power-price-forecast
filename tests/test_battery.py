import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import battery


def test_flat_prices_do_nothing():
    c, d = battery.schedule(np.full(24, 200.0))
    assert battery.settle(np.full(24, 200.0), c, d) == 0


def test_buys_low_sells_high():
    p = np.array([100.0] * 12 + [400.0] * 12)
    c, d = battery.schedule(p)
    assert c[:12].sum() > 0 and d[12:].sum() > 0
    assert battery.settle(p, c, d) > 0


def test_energy_limits_respected():
    p = np.array([50.0] * 6 + [500.0] * 18)
    c, d = battery.schedule(p)
    assert c.max() <= battery.POWER_MW + 1e-9
    assert d.sum() <= battery.ENERGY_MWH + 1e-6
