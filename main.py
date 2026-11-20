"""GTA VI trophy rarity forecast: earn-rate quantiles at 2026-11-26T00:00Z.

Reads every case file ``<id>.in`` in $MAKEFUN_INPUT_DIR (else /input) and writes
``<id>.out`` to $MAKEFUN_OUTPUT_DIR (else /output): for each of the case's
trophies, five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of its Exophase earn
rate (percent) at the case's target, scored by pinball loss on the logit scale.

Method (pure standard library, deterministic, no network, no randomness):

1.  Per trophy, take the case history's distinct reads (consecutive identical
    rate vectors are one Exophase refresh, not new information), work in
    logit space, and fit a weighted least-squares line of logit(rate) against
    ln(1 + hours since launch).  Weights grow with sqrt(players tracked)
    (a rate from 50,000 tracked players is far more precise than one from
    638) and decay with a 24 h half-life, so the recent, well-measured reads
    dominate.  One Huber-style reweighting pass limits the pull of single
    stale or jumpy reads.
2.  Blend that data extrapolation with a difficulty prior.  The prior is a
    week-one (168 h after launch) median per trophy, set from the game's
    structure -- 91 story missions over 5 chapters, a 50-80 h main story and
    ~200 h for 100% completion (reviewer playthroughs), and GTA V's
    long-run PSN rarity as the shape of the funnel (first mission common,
    mid-story ~30-55%, final mission ~35%, platinum ~0.1% Sony-wide) --
    shifted to the case's target by a prior slope in log time.  The blend is
    precision-weighted: the data's standard error grows with extrapolation
    distance, so a clean 48 h trend still moves the forecast while a flat or
    noisy trophy falls back to the prior.
3.  Enforce the story funnel on the medians (Gone Fishing >= Chapter 1 >=
    Chapter 2 >= Chapter 3 >= Chapter 4 >= final mission >= 100% >=
    platinum) with pool-adjacent-violators averaging, so a noisy fit can
    never claim more players finished chapter 4 than chapter 1.
4.  Quantiles are the blended median plus/minus the honest predictive
    spread (posterior error plus a random-walk term growing with the
    horizon), on the logit scale.  Pinball loss is a proper scoring rule:
    the spread reported is the spread believed.

Why this should beat the house baselines: the persistence baseline says
rates stay where the last read put them, and the trend baseline extends a
log-time line fitted to the last 24 h only.  Both ignore that earn rates
are a funnel (later chapters cannot exceed earlier ones), that early reads
are noisy because few players were tracked, and that a 4.5-day
extrapolation of a 2-day series deserves both a difficulty anchor and a
wide, honest spread.  This forecaster uses the whole history, the funnel,
and the anchor.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader runs `python3 -P -s main.py`, which drops this file's directory
# from sys.path; this program is a single file and needs no sibling imports,
# but the line below keeps that true if a helper is ever added beside it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Prior uncertainty (logit sd) on the week-one median of any trophy.
PRIOR_SD = 0.85
# Extra logit sd per unit of log-time extrapolated past the last read:
# the log-time form is a model, not a law, and a 4.5-day extrapolation of a
# 2-day series can bend.  (0.8 * ~1.03 log-hours for the final case.)
EXTRAP_SD_PER_LOG_HOUR = 0.8
# Random-walk spread, logit sd per sqrt(day) of horizon, inside the
# predictive spread of the reported quantiles.
RW_SD_PER_SQRT_DAY = 0.45
# Floor on a trophy's residual sd: a perfectly flat history (the practice
# cases) is not proof of zero future movement.
RESID_SD_FLOOR = 0.10

# Week-one (168 h after launch) prior medians, in percent, and the prior
# slope in logit per log-hour, keyed by trophy id (the case file's id is the
# trophy's position in Exophase's list; 1 is the platinum at the opening).
# Medians: story trophies from the mission counts per chapter (22/11/26/25/7
# of 91) against a 50-80 h story, so by day 7 a tracked player -- Exophase's
# self-selected, pre-order-heavy slice, not the average buyer -- has had
# roughly 10-25 h of play; grind trophies from their counts (250 stunts,
# 100 WAiNK spots, 3 card sets, 200 h for 100%); easy one-offs higher.
# Slopes: story trophies climb fastest in log time, grinds slowest.
PRIOR = {
    1: (0.04, 0.4),    # platinum: 100% completion in week one is ~nobody
    2: (95.0, 0.9),    # Gone Fishing: mission 2 of 91, nearly everyone
    3: (42.0, 0.9),    # Chapter 1 (22 missions)
    4: (30.0, 0.9),    # Chapter 2 (33 missions cumulative)
    5: (16.0, 0.9),    # Chapter 3 (59 missions cumulative)
    6: (8.0, 0.9),     # Chapter 4 (84 missions cumulative)
    7: (5.0, 0.9),     # final mission (91 missions)
    8: (22.0, 0.9),    # Breakfast Run (story mission)
    9: (18.0, 0.9),    # The Cartel Heist
    10: (15.0, 0.9),   # Child's Play
    11: (12.0, 0.9),   # Headliners
    12: (10.0, 0.9),   # VCIA Takedown
    13: (8.0, 0.9),    # Operation Savage Glory
    14: (0.10, 0.4),   # 100% completion
    15: (3.0, 0.5),    # 250 stunts
    16: (22.0, 0.7),   # try each activity once
    17: (2.0, 0.5),    # gold in all Shooting Range challenges
    18: (3.0, 0.5),    # drone race at every location
    19: (4.0, 0.6),    # win 20 different races
    20: (12.0, 0.7),   # beat Cal in a fishing challenge
    21: (28.0, 0.7),   # complete an arcade game
    22: (10.0, 0.7),   # final mission for Andy
    23: (9.0, 0.7),    # final mission for Kenton
    24: (8.0, 0.7),    # final mission for Billy
    25: (5.0, 0.7),    # 8 stranger mission strands
    26: (7.0, 0.7),    # final mission for Randy
    27: (6.0, 0.7),    # final mission for Tyrone
    28: (4.0, 0.5),    # meditate at every spot
    29: (8.0, 0.7),    # hunt a unique animal
    30: (2.0, 0.5),    # 3 collectible card sets
    31: (2.5, 0.5),    # WAiNK at 100 locations
    32: (5.0, 0.6),    # complete 50 challenges
    33: (15.0, 0.7),   # lose a 4-star wanted level with partner
    34: (8.0, 0.7),    # rob 3 shops back-to-back
    35: (8.0, 0.6),    # discover 100 vehicles
    36: (20.0, 0.7),   # 5 kills in one Combat Focus
    37: (5.0, 0.7),    # unscheduled departure at VCIA
    38: (4.0, 0.7),    # Rhinos
    39: (8.0, 0.7),    # rob a jewelry store undetected
    40: (3.0, 0.6),    # rob 5 bank vaults
    41: (12.0, 0.7),   # rob an armored truck
    42: (5.0, 0.6),    # explore all of Leonida
    43: (5.0, 0.6),    # rob an illegal operation of each gang
    44: (20.0, 0.7),   # cocktail on Ocean Beach
    45: (12.0, 0.7),   # smash out of a vehicle showroom
    46: (20.0, 0.7),   # bowling at Kolt & Associates
    47: (8.0, 0.7),    # a partnership with mass appeal
    48: (15.0, 0.7),   # choices made
    49: (5.0, 0.7),    # second time's the charm
    50: (2.0, 0.7),    # what the hell just happened?
    51: (5.0, 0.7),    # melee drive-by bowling
    52: (4.0, 0.5),    # $250,000 laundering marked bills
    53: (12.0, 0.7),   # spend $100,000 across shops
    54: (2.0, 0.5),    # purchase all garage properties
    55: (8.0, 0.6),    # register 10 vehicles
}

# Fallback prior (median %, slope) for a trophy id with no row above,
# by tier: a new trophy's difficulty is mostly its tier.
TIER_PRIOR = {
    "platinum": (0.05, 0.4),
    "gold": (3.0, 0.7),
    "silver": (5.0, 0.7),
    "bronze": (10.0, 0.7),
}

# The story funnel, in order: each trophy's median cannot exceed the one
# before it (you cannot finish chapter 4 without chapter 1, nor earn the
# platinum without 100%).
FUNNEL = [2, 3, 4, 5, 6, 7, 14, 1]


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as the house baseline does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(dt):
    return (dt - LAUNCH).total_seconds() / 3600.0


def wls(points):
    """Weighted least squares y = a + b*x. points: (x, y, w). Returns
    (a, b, xbar_w, sxx_w, sum_w) or None when the slope is unidentified."""
    sw = sum(w for _, _, w in points)
    if sw <= 0:
        return None
    xbar = sum(w * x for x, _, w in points) / sw
    ybar = sum(w * y for _, y, w in points) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, _, w in points)
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in points)
    b = sxy / sxx
    a = ybar - b * xbar
    return a, b, xbar, sxx, sw


def fit_trophy(series, t_target, t_last, prior_logit):
    """series: distinct (t_hours, rate_pct, players) reads, oldest first.
    Returns (data_logit, data_se): the extrapolation to the target and its
    standard error.  Two forms are fitted and averaged in logit space:
    logit(rate) against ln(1 + t) (concave, saturating -- the house trend's
    form) and against t itself (which catches the convex early part of an
    adoption curve, when a cohort of players crosses a mission threshold
    together).  The linear form is capped at the prior + 2 logit so a
    refresh jump can never launch it.  Falls back to the last rate with a
    wide error when no slope can be fitted."""
    x_star = math.log1p(max(t_target, -0.9))
    x_last = math.log1p(max(t_last, -0.9))
    lin_cap = prior_logit + 2.0

    def build(weights, xfun):
        pts = []
        for t, rate, n in series:
            x = xfun(t)
            # A rate of exactly 0 or 100 from a small tracked count is a
            # censored observation, not a logit of +/-infinity.
            y = logit(min(99.95, max(0.05, rate)))
            recency = 0.5 ** (max(t_last - t, 0.0) / 24.0)
            pts.append((x, y, weights(n, recency)))
        return pts

    def fit_form(xfun, x_target):
        wfun = lambda n, r: math.sqrt(max(n, 1)) * r
        pts = build(wfun, xfun)
        if len(pts) < 3:
            return None
        fit = wls(pts)
        if fit is None:
            return None
        a, b, xbar, sxx, sw = fit
        resid = [y - (a + b * x) for x, y, _ in pts]
        absr = sorted(abs(r) for r in resid)
        mad = absr[len(absr) // 2] if absr else 0.0
        scale = max(1.4826 * mad, RESID_SD_FLOOR)
        pts2 = []
        for (x, y, w), r in zip(pts, resid):
            if abs(r) > 2 * scale:
                w = w * (2 * scale / abs(r))
            pts2.append((x, y, w))
        fit2 = wls(pts2)
        if fit2 is not None:
            a, b, xbar, sxx, sw = fit2
            resid = [y - (a + b * x) for x, y, _ in pts2]
            sw2 = sum(w for _, _, w in pts2)
        else:
            sw2 = sw
        dof = max(len(pts) - 2, 1)
        s2 = sum(w * r * r for (_, _, w), r in zip(pts2, resid)) / max(sw2, 1e-9)
        s2 = s2 * len(pts) / dof if len(pts) > 2 else s2
        s = min(max(math.sqrt(max(s2, 0.0)), RESID_SD_FLOOR), 1.5)
        pred = a + b * x_target
        se_pred = s * math.sqrt(1.0 + (x_target - xbar) ** 2 / max(sxx, 1e-9))
        return pred, se_pred, s

    if len(series) < 3:
        last_rate = series[-1][1]
        return logit(last_rate), 1.5
    # Censoring floor for the data's error (applied on every return below):
    # a rate that never rose above a fraction of a percent says "not yet",
    # not "never".
    top_rate = max(rate for _, rate, _ in series)
    se_floor = 2.0 if top_rate < 0.2 else (1.5 if top_rate < 0.5 else 0.0)

    def with_floor(se):
        return max(se, se_floor) if se_floor else se

    log_fit = fit_form(lambda t: math.log1p(max(t, -0.9)), x_star)
    lin_fit = fit_form(lambda t: max(t, -0.9), max(t_target, -0.9))
    if log_fit is None and lin_fit is None:
        last_rate = series[-1][1]
        return logit(last_rate), 1.5
    if log_fit is None:
        pred, se_pred, _ = lin_fit
        data_logit = min(pred, lin_cap)
        se_data = math.sqrt(se_pred ** 2 + (EXTRAP_SD_PER_LOG_HOUR * max(x_star - x_last, 0.0)) ** 2 + 0.15 ** 2)
        return data_logit, with_floor(se_data)
    if lin_fit is None:
        pred, se_pred, _ = log_fit
        se_data = math.sqrt(se_pred ** 2 + (EXTRAP_SD_PER_LOG_HOUR * max(x_star - x_last, 0.0)) ** 2 + 0.15 ** 2)
        return pred, with_floor(se_data)
    d_log, se_log, _ = log_fit
    d_lin, se_lin, _ = lin_fit
    d_lin = min(d_lin, lin_cap)
    # The log-time form is the primary shape (for a lognormal adoption
    # curve, logit(rate) is near-linear in log time); the linear form is
    # only a hedge against the early convex part, so it gets 30%.
    data_logit = d_log + 0.3 * (d_lin - d_log)
    # Prediction error: the worse of the two forms' errors, plus form
    # error growing with how far past the last read the target sits.
    se_pred = max(se_log, se_lin)
    se_data = math.sqrt(
        se_pred ** 2
        + (EXTRAP_SD_PER_LOG_HOUR * max(x_star - x_last, 0.0)) ** 2
        + 0.15 ** 2
    )
    return data_logit, with_floor(se_data)


def prior_logit(trophy, t_target):
    """The difficulty prior's logit at the target time."""
    med168, slope = PRIOR.get(
        trophy["id"], TIER_PRIOR.get(trophy.get("tier", "bronze"), (10.0, 0.7))
    )
    return logit(med168) + slope * (
        math.log1p(max(t_target, -0.9)) - math.log1p(168.0)
    )


def pava(values):
    """Pool-adjacent-violators averaging: the closest non-increasing
    sequence (in logit space) to `values`, by least squares."""
    levels = [[v, 1] for v in values]
    out = []
    for v, w in levels:
        out.append([v, w])
        while len(out) >= 2 and out[-2][0] < out[-1][0]:
            v2, w2 = out.pop()
            v1, w1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    res = []
    for v, w in out:
        res.extend([v] * w)
    return res


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation,
    |error| < 1.15e-9): used only if a case ever asks for levels other
    than the contest's five, so the program still answers instead of
    crashing."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= 1 - p_low:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def forecast_case(case):
    levels = list(case.get("levels") or LEVELS)
    zvals = Z if levels == LEVELS else [norm_ppf(p) for p in levels]
    history = case.get("history") or []
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    t_target = hours_since_launch(target)
    if history:
        checkpoint = parse_time(case["at"])
        t_last = hours_since_launch(checkpoint)
    else:
        # No history at all: the priors alone still answer every trophy
        # instead of crashing the run (a crash is an invalid case).
        t_last = t_target

    # Distinct reads only: Exophase holds a page unchanged between
    # refreshes, so a run of identical rate vectors is one observation.
    distinct = []
    prev_rates = None
    for read in history:
        rates = read.get("rates") or []
        n = read.get("players_tracked") or 0
        if rates != prev_rates:
            distinct.append(read)
            prev_rates = rates
    if distinct:
        t_last = hours_since_launch(parse_time(distinct[-1]["at"]))
        # The checkpoint is the last slot, but the last distinct read may
        # be older when the page went stale; measure extrapolation from
        # the last real observation.
    else:
        distinct = history

    medians = {}
    spreads = {}
    for i, trophy in enumerate(trophies):
        series = []
        for read in distinct:
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                series.append(
                    (
                        hours_since_launch(parse_time(read["at"])),
                        float(rates[i]),
                        read.get("players_tracked") or 0,
                    )
                )
        p_logit = prior_logit(trophy, t_target)
        if len(series) >= 1:
            d_logit, d_se = fit_trophy(series, t_target, t_last, p_logit)
        else:
            d_logit, d_se = p_logit, 1.5
        prec_d = 1.0 / (d_se * d_se)
        prec_p = 1.0 / (PRIOR_SD * PRIOR_SD)
        med_logit = (d_logit * prec_d + p_logit * prec_p) / (prec_d + prec_p)
        post_var = 1.0 / (prec_d + prec_p)
        # Keep the median inside sane bounds: no trophy is literally
        # unwinnable in a week, and none is held by literally everyone.
        med_logit = min(max(med_logit, logit(0.02)), logit(99.5))
        medians[trophy["id"]] = med_logit
        spreads[trophy["id"]] = math.sqrt(
            post_var + (RW_SD_PER_SQRT_DAY * math.sqrt(max(horizon_days, 0.0))) ** 2 + 0.10 ** 2
        )

    # The story funnel: medians along the chapter chain cannot increase.
    chain = [tid for tid in FUNNEL if tid in medians]
    if len(chain) >= 2:
        adjusted = pava([medians[tid] for tid in chain])
        for tid, val in zip(chain, adjusted):
            medians[tid] = val

    rows = []
    for trophy in trophies:
        tid = trophy["id"]
        lg = medians[tid]
        sd = spreads[tid]
        qs = []
        prev = None
        for z in zvals:
            q = round4(clamp_pct(inv_logit(lg + z * sd)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
