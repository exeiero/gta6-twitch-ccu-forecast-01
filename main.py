"""gta6-twitch-ccu-forecast-01 entry: a kernel-weighted comparable-shape model.

Hour 1's print anchors the day. Hours 2..24 are forecast in log space as

    log(v_h / v_1) = mu_h(target) + z_tau * sd_h(target)

where mu_h and sd_h come from two estimators blended per hour:

  * a scale-kernel average over the case's comparables: each comparable's own
    log-ratio curve log(y_h/y_1) is weighted by exp(-|log10(y1_j) - log10(p1)| / BAND),
    so launches of a similar size dominate the shape. Weighted mean and weighted sd.
  * a robust linear fit of log(y_h/y_1) against log10(y1) over the larger
    comparables, which carries the size trend the kernel cannot see past the top
    of the pool (bigger launches decay harder after hour 1).

The blend is a mean and a law-of-total-variance sd. Beyond the fitted range the
linear term is damped, because the slope extrapolated to GTA VI's size would
otherwise run away. A floor keeps the spread honest on hours where the pool agrees.

Standard library only, deterministic, no randomness.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
HOURS = list(range(2, 25))

BAND = 0.25          # width of the scale kernel, in decades of hour-1 viewers
BLEND = 0.35         # weight of the kernel mean against the linear fit
FLOOR = 0.10         # smallest log-space sd we will quote
DAMP = 0.3           # how much of an extrapolation past the pool's top to believe
LIN_MIN = 60000      # the linear fit is taken over comparables at least this big
LIN_MIN_COUNT = 5    # ... and falls back to the whole pool below this count
MIN_POOL = 3         # below this many usable comparables we fall back to persistence
SIGMA = 0.15         # the house spread, used only by the fallback


def round2(x):
    """Round to 2 decimals half up, as the kit's starter does."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def _ts(seconds):
    """ISO 8601 UTC stamp for a whole number of seconds since the epoch."""
    days, rem = divmod(int(seconds), 86400)
    hour, rem = divmod(rem, 3600)
    minute, second = divmod(rem, 60)
    # civil-from-days, Howard Hinnant's algorithm, so no datetime parsing is needed
    z = days + 719468
    era = (z if z >= 0 else z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + 3 if mp < 10 else mp - 9
    if m <= 2:
        y += 1
    return "%04d-%02d-%02dT%02d:%02d:%02dZ" % (y, m, d, hour, minute, second)


def _epoch(stamp):
    """Seconds since the epoch for an ISO 8601 UTC stamp, or None."""
    if not isinstance(stamp, str) or len(stamp) < 20:
        return None
    try:
        y = int(stamp[0:4]); mo = int(stamp[5:7]); d = int(stamp[8:10])
        hh = int(stamp[11:13]); mm = int(stamp[14:16]); ss = int(stamp[17:19])
    except ValueError:
        return None
    if mo <= 2:
        y -= 1
        mo += 12
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (mo - 3) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468
    return days * 86400 + hh * 3600 + mm * 60 + ss


def comparable_curve(comp):
    """The comparable's viewers at hours 1..24 of its own release day.

    Returns a list of 24 values, hour 1 first; None where the comparable has no
    point for that hour.
    """
    pts = {}
    for row in comp.get("hourly") or ():
        at = _epoch(row.get("from"))
        v = row.get("viewers")
        if at is None or not isinstance(v, (int, float)) or v <= 0:
            continue
        pts[at] = float(v)
    start = _epoch(comp.get("day_start"))
    if start is None:
        return None
    return [pts.get(start + (h - 1) * 3600) for h in range(1, 25)]


def fit(case):
    """The comparables' curves, the anchor print, and the two pools."""
    p1 = None
    for row in case.get("known") or ():
        if row.get("hour") == 1:
            p1 = float(row.get("viewers") or 0)
            break
    if p1 is None or p1 <= 0:
        return None
    if p1 < 1:
        p1 = 1.0

    curves = []   # (log10(y1), [log ratios for hours 2..24])
    for comp in case.get("comparables") or ():
        curve = comparable_curve(comp)
        if not curve or not curve[0] or curve[0] <= 0:
            continue
        y1 = curve[0]
        ratios = []
        for h in range(2, 25):
            v = curve[h - 1]
            ratios.append(math.log(v / y1) if v and v > 0 else None)
        if sum(1 for r in ratios if r is not None) < 6:
            continue
        curves.append((math.log10(y1), ratios))

    if len(curves) < MIN_POOL:
        return None
    return p1, curves


def weighted(pairs):
    """Weighted mean and weighted sd of (value, weight) pairs."""
    wsum = 0.0
    msum = 0.0
    for v, w in pairs:
        wsum += w
        msum += w * v
    if wsum <= 0:
        return None
    mu = msum / wsum
    var = 0.0
    for v, w in pairs:
        d = v - mu
        var += w * d * d
    return mu, math.sqrt(var / wsum)


def hour_terms(curves, target, h):
    """The blended mean and sd of log(v_h/v_1) for one hour, or None."""
    idx = h - 2
    kernel = []
    for scale, ratios in curves:
        r = ratios[idx]
        if r is None:
            continue
        kernel.append((r, math.exp(-abs(scale - target) / BAND)))
    if len(kernel) < MIN_POOL:
        return None

    pool = [(s, rs[idx]) for s, rs in curves if rs[idx] is not None]
    big = [(s, r) for s, r in pool if s >= math.log10(LIN_MIN)]
    if len(big) < LIN_MIN_COUNT:
        big = pool
    if len(big) < LIN_MIN_COUNT:
        return None

    n = len(big)
    mean_x = sum(s for s, _ in big) / n
    mean_y = sum(r for _, r in big) / n
    sxx = sum((s - mean_x) ** 2 for s, _ in big)
    if sxx <= 1e-12:
        slope = 0.0
    else:
        slope = sum((s - mean_x) * (r - mean_y) for s, r in big) / sxx
    intercept = mean_y - slope * mean_x
    resid = [r - (intercept + slope * s) for s, r in big]
    lin_sd = math.sqrt(sum(d * d for d in resid) / (n - 2)) if n > 2 else 0.40

    top = max(s for s, _ in big)
    at = target if target <= top else top + (target - top) * DAMP
    lin_mu = intercept + slope * at

    kern = weighted(kernel)
    if kern is None:
        return None
    k_mu, k_sd = kern

    mu = BLEND * k_mu + (1.0 - BLEND) * lin_mu
    var = (BLEND * k_sd * k_sd + (1.0 - BLEND) * lin_sd * lin_sd
           + BLEND * (1.0 - BLEND) * (k_mu - lin_mu) ** 2)
    sd = math.sqrt(var)
    return mu, sd if sd > FLOOR else FLOOR


def forecast_case(case):
    """The case's 23 forecast rows, or None when the case is too thin to model."""
    levels = case.get("levels")
    if levels is not None and list(levels) != LEVELS:
        raise ValueError("case asks for levels %r, not %r" % (levels, LEVELS))

    fitted = fit(case)
    if fitted is None:
        return None
    p1, curves = fitted
    target = math.log10(p1)

    rows = []
    for row in case.get("forecast") or ():
        h = row.get("hour")
        if h not in HOURS:
            continue
        terms = hour_terms(curves, target, h)
        if terms is None:
            q = [p1 * math.exp(z * SIGMA * math.sqrt(h - 1)) for z in Z]
        else:
            mu, sd = terms
            q = [p1 * math.exp(mu + z * sd) for z in Z]
        q = [min(max(round2(v), 0.01), 1e9) for v in q]
        for i in range(1, len(q)):
            if q[i] < q[i - 1]:
                q[i] = q[i - 1]
        rows.append({"hour": h, "q": q})
    return rows


def persistence_rows(case):
    """The house's own forecast: every hour as big as hour 1, spread like a random walk."""
    p1 = None
    for row in case.get("known") or ():
        if row.get("hour") == 1:
            p1 = float(row.get("viewers") or 0)
            break
    p1 = max(1.0, p1 or 0.0)
    return [{"hour": h, "q": [round2(p1 * math.exp(z * SIGMA * math.sqrt(h - 1))) for z in Z]}
            for h in HOURS]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        hours = None
        try:
            hours = forecast_case(case)
        except Exception:
            hours = None
        if not hours:
            hours = persistence_rows(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps({"hours": hours}, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
