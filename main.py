"""gta6-trophy-rarity-forecast-01 v3: curve-family forecaster with a
dilution branch for frozen pages.

Design (tuned on the practice cases and on fresh synthetic final-like
cases built with the kit's own generator family; the five held-out
synthetic cases were only ever used for validation, never for tuning):

- The house reads Exophase's trophy page on a clock, but the page only
  changes when Exophase refreshes it (about every 4 hours), so a case's
  history is a step function: many reads, few distinct rate vectors.
  Each trophy's distinct vectors, at their first-seen times, are the
  observations. The target is the rate at the case's target instant.
- Trending trophies (the history moves): fit two curve families to the
  observations and let a short-range extrapolation test pick one.
    * saturating exponential  r(t) = rinf - (rinf - r0) * exp(-t/tau),
      r0 fixed to the first observed rate, tau chosen by unweighted
      squared error, rinf by player-count-weighted least squares;
    * power law  r(t) = r0 * (1 + t/tau) ** gamma, (tau, gamma) chosen
      by player-count-weighted squared error over a grid;
  The pick is leave-last-4-out cross-validation in logit space: fit
  each family on the earlier points, score it on the last four, take
  the lower error. The picked family's value at the target is the
  median. (In-sample fit alone cannot tell the families apart: both
  fit the window, they only diverge past it. The hold-out test is
  what separates them.)
- Frozen pages (every read identical, as in the practice cases): the
  page has not refreshed since launch night, so the target sits just
  after a refresh that dilutes every rate. The refresh takes the
  tracked players from 638 to about 3,868 (the jump the rules describe)
  while the newly tracked players have earned almost nothing, so the
  forecast rate is (638 * r + 3230 * c(r)) / 3868 with c(r) the new
  cohort's earn rate, modelled as 0.05 * r ** 4: near zero for rare
  trophies, rising steeply for the common ones the new players reach
  first.
- Quantiles are the median on the logit scale plus/minus z * sigma,
  with sigma 0.15 on the frozen branch and a small, cross-validation
  aware spread on the trending branch (see SIG_TREND_* below).

Standard library only, deterministic, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
LAUNCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# Saturating-exponential time constants (hours) and power-law grid.
TAU_SAT = [10.0 * (1.35 ** k) for k in range(14)]
TAU_POW = [5.0 * (1.35 ** k) for k in range(13)]
GAM_POW = [0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.2, 1.5]

# Frozen-page dilution: players 638 -> 3868 at the refresh; the new
# cohort's earn rate is DIL_A * r ** DIL_B (percent).
DIL_OLD_N = 638.0
DIL_NEW_N = 3230.0
DIL_TOTAL_N = 3868.0
DIL_A = 0.05
DIL_B = 4.0
SIG_FLAT = 0.15

# Trending-branch spread (logit units): a floor plus a term in the
# picked family's cross-validation error, so trophies the families
# cannot pin down get wider quantiles.
SIG_TREND_BASE = 0.06
SIG_TREND_CV = 0.15


def parse_time(text):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError("unparsed time: %r" % text)


def hours_since_launch(when):
    return (when - LAUNCH).total_seconds() / 3600.0


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit_of(pct):
    q = clamp_rate(pct) / 100.0
    return math.log(q / (1.0 - q))


def inv_logit_of(x):
    return 100.0 / (1.0 + math.exp(-x))


def round4(x):
    """Round to 4 decimals, half up (as JavaScript's Math.round does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def states_of(case, index):
    """A trophy's distinct (hours, rate, players) observations, in time order."""
    seen = set()
    out = []
    for read in case["history"]:
        rate = read["rates"][index]
        if rate is None:
            continue
        key = round(rate, 2)
        if key in seen:
            continue
        seen.add(key)
        out.append((hours_since_launch(parse_time(read["at"])), rate, read["players_tracked"]))
    return out


def fit_saturating(states, tau):
    """r(t) = rinf - (rinf - r0) * exp(-t/tau), r0 the first rate.

    rinf by player-count-weighted least squares.
    """
    r0 = states[0][1]
    num = 0.0
    den = 0.0
    for t, rate, n in states:
        e = math.exp(-max(t, 0.0) / tau)
        u = 1.0 - e
        num += n * u * (rate - r0 * e)
        den += n * u * u
    rinf = num / den if den > 0 else r0
    return max(r0, min(100.0, rinf)), r0


def saturating_at(rinf, r0, tau, t):
    return rinf - (rinf - r0) * math.exp(-max(t, 0.0) / tau)


def best_saturating(states, weighted):
    best = None
    for tau in TAU_SAT:
        rinf, r0 = fit_saturating(states, tau)
        if weighted:
            err = sum(n * (saturating_at(rinf, r0, tau, t) - r) ** 2 for t, r, n in states)
        else:
            err = sum((saturating_at(rinf, r0, tau, t) - r) ** 2 for t, r, n in states)
        if best is None or err < best[0]:
            best = (err, tau, rinf, r0)
    return best


def fit_power(states, tau, gamma):
    """r(t) = r0 * (1 + t/tau) ** gamma, r0 by player-count-weighted least squares."""
    num = 0.0
    den = 0.0
    for t, rate, n in states:
        b = (1.0 + max(t, 0.0) / tau) ** gamma
        num += n * b * rate
        den += n * b * b
    return max(0.0, num / den) if den > 0 else states[0][1]


def power_at(r0, tau, gamma, t):
    return min(95.0, r0 * (1.0 + max(t, 0.0) / tau) ** gamma)


def best_power(states, weighted):
    best = None
    for tau in TAU_POW:
        for gamma in GAM_POW:
            r0 = fit_power(states, tau, gamma)
            if weighted:
                err = sum(n * (power_at(r0, tau, gamma, t) - r) ** 2 for t, r, n in states)
            else:
                err = sum((power_at(r0, tau, gamma, t) - r) ** 2 for t, r, n in states)
            if best is None or err < best[0]:
                best = (err, tau, gamma, r0)
    return best


def cv_errors(states, hold=4):
    """Leave-last-`hold`-out error (mean squared logit error) per family.

    Each family is refit on the earlier states only; returns
    (sat_error, pow_error), or (None, None) when too few states.
    """
    if len(states) <= hold + 2:
        return None, None
    train, test = states[:-hold], states[-hold:]
    sat = best_saturating(train, weighted=False)
    pow_ = best_power(train, weighted=False)
    err_sat = sum(
        (logit_of(saturating_at(sat[2], sat[3], sat[1], t)) - logit_of(r)) ** 2
        for t, r, _ in test
    ) / len(test)
    err_pow = sum(
        (logit_of(power_at(pow_[3], pow_[1], pow_[2], t)) - logit_of(r)) ** 2
        for t, r, _ in test
    ) / len(test)
    return err_sat, err_pow


def trend_median_and_sigma(states, target_hours):
    """Pick a curve family by cross-validation; return (median, sigma)."""
    sat = best_saturating(states, weighted=False)
    pow_ = best_power(states, weighted=True)
    p_sat = saturating_at(sat[2], sat[3], sat[1], target_hours)
    p_pow = power_at(pow_[3], pow_[1], pow_[2], target_hours)
    err_sat, err_pow = cv_errors(states)
    if err_sat is None or err_pow is None:
        # Too few states to cross-validate: the families are
        # indistinguishable here, so hedge between them on the logit
        # scale instead of committing to saturation.
        return inv_logit_of((logit_of(p_sat) + logit_of(p_pow)) / 2.0), SIG_TREND_BASE
    if err_sat <= err_pow:
        return p_sat, SIG_TREND_BASE + SIG_TREND_CV * math.sqrt(err_sat)
    return p_pow, SIG_TREND_BASE + SIG_TREND_CV * math.sqrt(err_pow)


def dilution_median(last_rate):
    new_cohort = DIL_A * (max(last_rate, 0.0) ** DIL_B)
    return (DIL_OLD_N * last_rate + DIL_NEW_N * new_cohort) / DIL_TOTAL_N


def quantiles(median, sigma_down, sigma_up):
    centre = logit_of(median)
    out = []
    for z in Z:
        out.append(inv_logit_of(centre + z * (sigma_down if z < 0 else sigma_up)))
    out = [min(100.0, max(0.01, q)) for q in out]
    out.sort()
    return [round4(q) for q in out]


def forecast_case(case):
    target_hours = hours_since_launch(parse_time(case["target"]))
    history = case["history"]
    frozen = len({tuple(read["rates"]) for read in history}) == 1
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last = history[-1]["rates"][i]
        if last is None:
            last = 0.0
        if last >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.9, 99.99, 100.0, 100.0, 100.0]})
            continue
        if frozen:
            median = dilution_median(last)
            rows.append({"id": trophy["id"], "q": quantiles(median, SIG_FLAT, SIG_FLAT)})
            continue
        states = states_of(case, i)
        if len(states) < 2:
            rows.append({"id": trophy["id"], "q": quantiles(last, SIG_TREND_BASE, SIG_TREND_BASE)})
            continue
        median, sigma = trend_median_and_sigma(states, target_hours)
        rows.append({"id": trophy["id"], "q": quantiles(median, sigma, sigma)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
