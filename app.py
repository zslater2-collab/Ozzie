import os
import sys
import csv
import math
import bisect
import pickle
import unicodedata
import requests
import time
import pandas as pd
import pytz
from datetime import datetime, date as _date
from collections import Counter
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import gdown
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeoutError

# Under gunicorn, stdout to a non-tty defaults to block-buffered, not line-buffered -- print()
# calls from inside request handlers can sit in the buffer indefinitely instead of reaching
# Render's logs, while only the handful of print()s that happen to fire together at worker
# boot (FG profiles/PQ prior/offense prior) show up reliably. Confirmed June 2026: an entire
# 53-minute window with 10+ requests, including two that definitely ran the pitcher-quality
# population build, produced zero diagnostic output. Force line buffering so every print() is
# visible in logs immediately, not just the ones that happen to fill a buffer.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

DOWNLOAD_TIMEOUT_SECONDS = 20

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'zach-picks-secret-2026')
APP_PASSWORD = os.environ.get('APP_PASSWORD', 'picks2026')

FILE_IDS = {
    'all_pitcher_arch_scores': '1qA9iSXEv1ONRlXvm0O5TzLSQTUKFVQGn',
    'arch_hitter_map_combined': '15wE3HfjaR4g68YnwcSCQV76_t0F4I7YG',
    'archetypes_combined':      '1WNBjJ98Q8n20oDjF5AMVFeeUIJiqkz4h',
    'all_parks':                '1Wch81FIHxpoJXboFOV3p3C_06PFnUNB7',
    'team_bullpen_scores':      '1-W_hgheeGdMeSDA6enW2EJgvXLXtzqlP',
    'batter_power_map':         '1ZHGuGMmkjd-uW2sbKq1wMMq3p4zSMJbg',
    'pitcher_tendency_map':     '1u328HojWnhcQWlgx0SW5DX-WrsThBxon',
    'batter_ev_profiles_lr':    '1dnFavwW5CPXoJy3IBZueAYOokb3oxXyE',
    'pitcher_contact_rates':    '1DXNF_rvjHBk31Ja6Tle3i8LrjwD5iXec',
    'negbin_model_params':      '122sd0M7XFhb-JlU2qE7_wQey9iTntIv9',
    'hitter_splits':            '1mJA898UJaO62azlrw2-l1qeBto5hV_fj',
    'pitcher_bb_gb_rates':      '1kaCr_b0Zw9_4nAYMk84kCh8mTbOE5chC',
}

LEAGUE_AVG        = 3.88
JUICY_THRESHOLD   = 5.0
MIN_SENSOR_PA     = 20
STARTER_WEIGHT    = 0.80
BULLPEN_WEIGHT    = 0.20
ASSUMED_PAS       = 4
TB3_MULTIPLIER    = 1.3
LEAGUE_AVG_KRATE  = 0.229

HEATMAP_MEAN            = 0.9984
HEATMAP_STD             = 0.0549
HEATMAP_UNDER_THRESHOLD = HEATMAP_MEAN - 1.0 * HEATMAP_STD

# ── UNDER SIGNAL — Ozzie Scoring System v3 ───────────────────────────────
# Gate: std_from_mean <= -1.0 AND (away + K>=26%) >= 2
# Score: park (0/1/3) + K>=28% (1) + away (1) + sigma<=-2.0 (1) = max 6
# Tiers: Gold >=6, Silver 4-5, Bronze 3
# BB rate removed (p=0.810, no signal). U1.5 removed (no market edge on real lines).
# Validated 2025 Pinnacle U2.5: all flagged ROI +0.552, Silver +0.551, Gold +0.664
# Sigma alone: sigma<=0.0 n=258 ROI +0.215, sigma<=-0.5 n=172 ROI +0.263

PARK_GOOD = {'TEX', 'BAL', 'MIL', 'CIN', 'DET', 'COL', 'LAD', 'NYY', 'CWS'}
PARK_BAD  = {'ATH', 'CHC', 'MIA', 'NYM', 'SF', 'HOU', 'TB', 'STL', 'BOS', 'SEA', 'PIT', 'MIN', 'CLE'}

FG_BLEND_MEAN        = 0.9984
FG_BLEND_STD         = 0.0353
FG_STARTER_Z_THRESH  = -0.5
FG_BP_BAND_LOW       =  0.5
FG_BP_BAND_HIGH      =  1.0
FG_VALID_START_MONTH =  6
FG_VALID_START_DAY   =  1
FG_VALID_END_MONTH   =  7
FG_VALID_END_DAY     = 31

DFS_HOT_Z           =  1.5
DFS_FADE_Z          = -1.5
DFS_STACK_MIN_Z     =  0.0
DFS_ACE_K           =  0.28
DFS_VALUE_K         =  0.22
DFS_FADE_PITCH      =  0.5
DFS_SPLIT_THRESHOLD =  0.5


_model_cache      = None
_model_cache_time = None
MODEL_CACHE_TTL   = 3600

_FG_OFF_Z     = {}
_FG_BP_WEAK_Z = {}
_FG_PROFILES_LOADED = False

# ── PITCHER QUALITY COMPOSITE (research / primary tracking signal, NOT validated for betting) ──
# Built June 17, 2026. K/BB/HR rate only (DIPS-stable inputs — contact-quality-allowed metrics
# like sigma/PX have ~zero YoY stability for individual pitchers, see CLAUDE.md). Career prior
# (2022-2025 pooled, pitcher_quality_prior_2026.csv) Marcel-blended with current-season-to-date
# stats (Carleton phantom-PA constants: K=70 BF, BB=170 BF, HR=1320 BF). Weights fit via logistic
# regression of under_hit on standardized K/BB/HR at the 1.5 line (2025 sample):
#   score = 0.1186*k_z - 0.1561*bb_z - 0.1945*hr_z   (higher = better pitcher = more under-favorable)
#
# VALIDATION STATUS — read before trusting this:
#   2025 (where found): top quartile @ 1.5-line -> 61.1% U, +18.1% ROI after juice, n=185, p=0.011
#   2026 OOS (true forward test): much weaker pooled (gap 0.08-0.12, p=0.35-0.72, NOT significant).
#   BUT June 2026 specifically: gap=0.251, +22.4% ROI (n=36, p=0.16) — nearly identical to June
#   2025's gap=0.271. Same weak-April/May -> strong-June seasonal shape replicated independently
#   in two different years (likely because the Marcel blend itself gets more accurate as more
#   current-season data accumulates). Promising but underpowered — NOT a proven, deployable edge.
#
#   THRESHOLD UPDATE (June 19, 2026, TIER 1 finding #3): the ≥75th-percentile (quartile) cutoff
#   contains a hidden dead zone at the 1.5 line — the 74th-87th percentile sub-band is actually
#   NEGATIVE (2025: -6.5% ROI; 2026 OOS: -11.4% ROI, both replicated). A narrower 87-92nd "sweet
#   spot" looked best in-sample but failed 2026 OOS (-5.2%, overfit, rejected). The cutoff that
#   DOES replicate cleanly OOS is ~80th percentile (top quintile): 2025 n=310, 57.4% hit, +12.7%
#   ROI, p=0.0006; 2026 OOS n=266, 56.0% hit, +9.4% ROI, p=0.0024 — both beat the old 75th-cut's
#   OOS showing. Moved PQ_Q4_PCTILE from 75 to 80 accordingly. See CLAUDE.md TIER 1 FINDINGS.
#
# CRITICAL — this ONLY works at the 1.5 F5 team-total line specifically. At 2.5 lines, the effect
# is flat-to-backward. This app does not fetch live odds/lines, so it cannot gate on line level —
# you must check the actual posted F5 line yourself before treating this as anything actionable.
# Offense-side K/BB/SLG composites (with and without platoon-matching) were tested extensively and
# showed ZERO signal anywhere — pitcher-only by design, this is not an oversight.

PQ_WEIGHTS       = {'k': 0.1186068959664584, 'bb': -0.156144018411192, 'hr': -0.1944839999359043}
PQ_PHANTOM_BF    = {'k_rate': 70, 'bb_rate': 170, 'hr_rate': 1320}
PQ_PRIOR_CSV     = 'pitcher_quality_prior_2026.csv'
PQ_Q4_PCTILE     = 80.0  # top quintile, not quartile -- see THRESHOLD UPDATE note above
PQ_SEASON_TTL    = 21600  # 6h — current-season cumulative stats don't need hourly refresh

_pq_prior        = {}
_pq_prior_loaded = False
_pq_population_cache      = None
_pq_population_cache_time = None


def load_pitcher_quality_prior():
    global _pq_prior, _pq_prior_loaded
    if _pq_prior_loaded:
        return
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, PQ_PRIOR_CSV)
    if not os.path.exists(path):
        print(f"Warning: {PQ_PRIOR_CSV} not found — pitcher quality composite disabled")
        return
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                _pq_prior[int(row['mlbID'])] = {
                    'k_rate':     float(row['k_rate']),
                    'bb_rate':    float(row['bb_rate']),
                    'hr_rate':    float(row['hr_rate']),
                    'is_starter': row.get('is_starter', 'True').strip().lower() == 'true',
                }
        n_starters = sum(1 for v in _pq_prior.values() if v['is_starter'])
        _pq_prior_loaded = True
        print(f"Pitcher quality prior loaded: {len(_pq_prior)} pitchers ({n_starters} starters)")
    except Exception as e:
        print(f"Warning: pitcher quality prior load failed: {e}")


def _pq_fetch_current_season():
    """
    Current-season BF/K/BB/HR per pitcher via the official MLB Stats API bulk season-stats
    endpoint -- one request for the whole league (playerPool=ALL, not just qualifiers).
    Replaces a pybaseball Baseball-Reference scrape that started failing 100% of the time in
    production (June 2026: "list index out of range" on every call) while working fine
    locally -- consistent with Baseball-Reference rate-limiting/blocking Render's IP after
    repeated automated requests. MLB Stats API is the same official, keyless, un-blocked
    source already used reliably elsewhere in this app for lineups/boxscores.
    """
    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/stats",
            params={'stats': 'season', 'group': 'pitching', 'season': datetime.now().year,
                    'sportId': 1, 'limit': 3000, 'playerPool': 'ALL'},
            timeout=20)
        r.raise_for_status()
        splits = r.json().get('stats', [{}])[0].get('splits', [])
        season = {}
        for s in splits:
            stat = s.get('stat', {})
            bf = stat.get('battersFaced', 0) or 0
            pid = s.get('player', {}).get('id')
            if bf <= 0 or pid is None:
                continue
            season[int(pid)] = {
                'bf': bf, 'so': stat.get('strikeOuts', 0) or 0,
                'bb': stat.get('baseOnBalls', 0) or 0, 'hr': stat.get('homeRuns', 0) or 0,
                'gs': stat.get('gamesStarted', 0) or 0,   # for per-pitcher expected outing length
            }
        print(f"Pitcher quality current-season fetch: {len(season)} pitchers with BF>0")
        return season
    except Exception as e:
        print(f"Warning: pitcher quality current-season fetch failed: {e}")
        return {}


def get_pitcher_quality_population(force=False):
    """
    Returns dict: mlbID -> {score, percentile, quartile, k_rate, bb_rate, hr_rate}.
    Blends career prior with current-season-to-date (Marcel, leak-free as of right now).
    Cached PQ_SEASON_TTL seconds since current-season stats don't change minute to minute.
    force=True bypasses this cache (see /api/picks?refresh=1).
    """
    global _pq_population_cache, _pq_population_cache_time
    now = datetime.now().timestamp()
    if not force and _pq_population_cache and _pq_population_cache_time and \
            (now - _pq_population_cache_time < PQ_SEASON_TTL):
        return _pq_population_cache

    load_pitcher_quality_prior()
    if not _pq_prior:
        return {}

    season = _pq_fetch_current_season()
    blended = {}
    current_bf_by_pid = {}
    exp_bf_by_pid = {}
    n_zero_current = 0
    for pid, prior in _pq_prior.items():
        s = season.get(pid, {'bf': 0, 'so': 0, 'bb': 0, 'hr': 0, 'gs': 0})
        bf = s['bf']
        current_bf_by_pid[pid] = bf
        exp_bf_by_pid[pid] = expected_bf_per_start(bf, s.get('gs', 0))
        if bf == 0:
            n_zero_current += 1
        season_rates = {
            'k_rate':  s['so'] / bf if bf > 0 else 0,
            'bb_rate': s['bb'] / bf if bf > 0 else 0,
            'hr_rate': s['hr'] / bf if bf > 0 else 0,
        }
        blend = {}
        for stat in ('k_rate', 'bb_rate', 'hr_rate'):
            ph = PQ_PHANTOM_BF[stat]
            blend[stat] = (ph * prior[stat] + bf * season_rates[stat]) / (ph + bf)
        blended[pid] = blend

    if n_zero_current > len(_pq_prior) * 0.5:
        print(f"Warning: pitcher quality current-season data missing for {n_zero_current}/{len(_pq_prior)} "
              f"pitchers -- most blended scores this cycle are degraded to pure career prior")

    if len(blended) < 10:
        return {}

    # fip_like = -2*k + 3*bb + 13*hr (lower = better pitcher), ranked against all pitchers.
    # Replaced the weighted z-score with starters-only reference pool (June 26, 2026): the
    # starters-only cutoff shifted Q4 enough to capture ~61 extra starts at -33% ROI that the
    # all-pitcher fip_like ranking excluded (backtest_live_model.py, 2026 OOS comparison).
    ref_pool = list(blended.keys())

    def compute_score(b):
        return -2 * b['k_rate'] + 3 * b['bb_rate'] + 13 * b['hr_rate']

    # Sort ascending (lowest fip_like = best pitcher). pctile = fraction with a HIGHER (worse)
    # fip_like value, so higher pctile = better pitcher, Q4 = top quintile (pctile >= 80).
    ref_scores_sorted = sorted(compute_score(blended[pid]) for pid in ref_pool)
    n_ref = len(ref_scores_sorted)

    population = {}
    for pid, b in blended.items():
        score  = compute_score(b)
        pctile = 100.0 * (n_ref - bisect.bisect_right(ref_scores_sorted, score)) / n_ref if n_ref > 1 else 50.0
        population[pid] = {
            'score':       round(score, 4),
            'percentile':  round(pctile, 1),
            'quartile':    'Q4' if pctile >= PQ_Q4_PCTILE else ('Q1' if pctile < 25 else ('Q2' if pctile < 50 else 'Q3')),
            'k_rate':      round(blended[pid]['k_rate'], 4),
            'bb_rate':     round(blended[pid]['bb_rate'], 4),
            'hr_rate':     round(blended[pid]['hr_rate'], 4),
            # How much of THIS pitcher's score is real current-season data vs pure career prior --
            # 0 means the current-season fetch had nothing for them (network blip, BRef lag, etc.)
            # and the blend silently fell back to the prior alone. See incident notes, June 2026.
            'current_bf':  current_bf_by_pid.get(pid, 0),
            # this pitcher's expected batters-faced for a start (season BF/start, shrunk toward the
            # league-average outing) -- drives the K projection, see project_starter_ks.
            'exp_bf':      exp_bf_by_pid.get(pid, K_PROJ_EXPECTED_BF),
            # season-to-date games started (point-in-time prior-start count, pregame). Gates the
            # K-prop SHARP tier -- see KPROP_SHARP_MIN_STARTS.
            'gs':          season.get(pid, {}).get('gs', 0),
        }

    print(f"Pitcher quality population built: {len(population)} pitchers, "
          f"{n_zero_current} with zero current-season BF")
    _pq_population_cache      = population
    _pq_population_cache_time = now
    return population


# ── FG (FULL-GAME) BULLPEN COMPOSITE — TEAM-TOTAL UNDER (June 22, 2026) ─────────────────
# Built same day as signal #4's full-game extension. Same K/BB/HR DIPS composite, same
# recency-weighted prior (pitcher_quality_prior_2026.csv), same PQ_WEIGHTS/PQ_PHANTOM_BF — but
# aggregated by TEAM instead of by individual pitcher, because (unlike a starter) you don't know
# which specific relievers will pitch before the game. This is the honest live equivalent of the
# backtest's "actual relievers who pitched, BF-weighted" methodology — a live signal can't have
# that hindsight, so it uses the team's bullpen-as-a-whole current-season performance instead.
# Expect some degradation vs. the backtest for this reason, same as every other signal here going
# from backtest to live.
#
# VALIDATION STATUS:
#   Opponent's bullpen quality predicts THIS team's full-game total going under — the opposing
#   STARTER alone is flat (p=0.158), the entire edge is bullpen-specific. Validated at FG lines
#   3.5/4.5/5.5 (2.5 is flat, not gated on). Recency-weighted prior, pooled across those three
#   lines: 2025 n=852, 65.0% hit, +22.7% ROI, p<0.001; 2026 n=39, 66.7% hit, +27.8% ROI, p=0.027.
#   Threshold sweep is clean and monotonic (no hidden dead zone like the starter composite needed
#   to route around) — 75th percentile is well-supported by both years, not just a leftover
#   default. Holds across line levels (3.5/4.5/5.5), not offense-confounded (r=0.05 vs batting
#   team's own off_pctile), not park-confounded once the join was corrected (r=-0.03), and holds
#   at Coors specifically (72.0% hit vs. Coors' unconditional 49.1% under rate). See CLAUDE.md
#   "SIGNAL #5" sections for full validation history, including a backwards-join bug found and
#   fixed mid-session — current numbers are post-fix.
#
# CRITICAL: this is a TRACKING-ONLY signal, same bar as pq_q4/off_q3 — not yet proven on a full
# second season at the live (recency-weighted, team-aggregated) methodology. Gate on Pinnacle's
# full-game team_totals line being exactly 3.5, 4.5, or 5.5 for this team.

FG_BP_PCTILE_GATE = 75.0
FG_TT_VALID_LINES = {3.5, 4.5, 5.5}
BP_SEASON_TTL      = 21600  # same cadence as pq/off — current-season cumulative stats don't churn

# ── FG JOINT (COMBINED GAME) TOTAL — BULLPEN COMPOSITE (June 22, 2026) ────────────────
# Flip + extension of the FG team-total signal above to the JOINT/combined game total. Both
# teams' bullpens suppress a joint total (each team's runs are held down by the OPPONENT's
# pen, and the joint total = sum of both teams' runs), so the game-level composite is the
# AVERAGE of the two teams' bullpen percentiles. Validated in CLAUDE.md "SIGNAL #5 — OVER SIDE
# + JOINT TOTAL REOPENED" (the joint total had been wrongly "closed" using a starter-heavy
# blend; with a BULLPEN-ONLY composite it's the strongest cross-year result in the project):
#   STRONG combined bullpen -> UNDER: 2025 n=605 60.3%/+14.8%, 2026 n=193 63.2%/+20.5%
#     (both DraftKings, p<0.001; theScore Bet 2025 n=604 63.9%/+21.8%)
#   WEAK   combined bullpen -> OVER : 2025 n=604 59.3%/+12.9%, 2026 n=192 57.3%/+9.6%
# Both directions, both years, real sample, two usable books; survives offense/park/month
# confounds. TRACKING ONLY — same bar as every other signal here, do NOT bet yet, and note it
# is CORRELATED with the fg_tt_under team-total signal (same composite family) — don't stack.
# THRESHOLD: the backtest cut on the top/bottom QUARTILE OF GAMES by the mean of two teams'
# bullpen z-scores. The live `percentile` field is each team's rank among 30 teams; the mean of
# two ~uniform team percentiles is triangular, whose 75th/25th percentiles sit at ~64.6/~35.4
# (NOT 75/25) — so 65/35 on the combined percentile approximates the validated quartile-of-games
# cut. Both team percentiles + the combined value are logged to the sheet so this can be re-tuned
# empirically against outcomes later (same spirit as the live-vs-backtest degradation everywhere).
# LINE GATE: the edge holds across game totals 7.0–9.5 (strongest at 7.5/8.5/9.5); 6.5 was flat
# and 10.0+ thin/negative in the backtest. Gated on DraftKings' game total (the book it was
# validated on + full coverage), mirroring fg_tt's Pinnacle-gate shape.
# CONVICTION TIERS (June 22, 2026) — the combined-percentile threshold sweep is cleanly monotonic
# both years, so the joint signal is tiered Gold/Silver/Bronze with suggested unit sizing (tracking
# guidance, NOT auto-bets). Validated DraftKings, both years (CLAUDE.md "JOINT BULLPEN AGGREGATION
# METHODOLOGY SWEEP" + "TIERS"). The UNDER side tiers monotonically; the OVER side does NOT (its
# 10-20 combined band reverses in 2026), so over is left as a single tier — forcing medals on it
# would misrepresent the data. Floors are tighter than the original flat 65/35 gate (the 60-70
# under band and 30-40 over band were inconsistent/negative in 2026, so they're dropped).
FG_JOINT_UNDER_TIERS = [   # (min_combined, max_combined, tier_label, units)
    (90.0, 100.01, 'Gold',   1.5),   # 2025 n=47  70.2%/+33.1%; 2026 n=10 90.0%/+70.9% (thin)
    (80.0, 90.0,   'Silver', 1.0),   # 2025 n=106 63.2%/+20.5%; 2026 n=34 70.6%/+34.3%
    (70.0, 80.0,   'Bronze', 0.5),   # 2025 n=200 56.0%/+6.6%;  2026 n=84 59.5%/+14.1%
]
FG_JOINT_OVER_MAX     = 30.0   # combined <= 30 -> OVER (2025 65.4%/+24.9%, 2026 57.4%/+10.1%); single tier
FG_JOINT_OVER_UNITS   = 1.0
FG_JOINT_LINE_MIN     = 7.0
FG_JOINT_LINE_MAX     = 9.5
FG_JOINT_GATE_BOOK    = 'DraftKings'

_bp_population_cache      = None
_bp_population_cache_time = None


def _bp_fetch_current_season_by_team():
    """
    Same MLB Stats API bulk season-stats call as _pq_fetch_current_season, but also captures
    each pitcher's CURRENT team (the 'team' object on every split, same response, no extra
    request) — needed to aggregate relievers by team for the bullpen composite. Kept as a
    separate function rather than modifying _pq_fetch_current_season itself, since that one is
    already depended on by the starter composite and this app has a history of production
    incidents from touching shared fetch paths (see TRACKING-ONLY MODE notes above).
    """
    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/stats",
            params={'stats': 'season', 'group': 'pitching', 'season': datetime.now().year,
                    'sportId': 1, 'limit': 3000, 'playerPool': 'ALL'},
            timeout=20)
        r.raise_for_status()
        splits = r.json().get('stats', [{}])[0].get('splits', [])
        season = {}
        n_no_team = 0
        for s in splits:
            stat = s.get('stat', {})
            bf = stat.get('battersFaced', 0) or 0
            pid = s.get('player', {}).get('id')
            team_name = s.get('team', {}).get('name')
            team_abb  = NAME_TO_ABB.get(team_name)
            if bf <= 0 or pid is None:
                continue
            if team_abb is None:
                n_no_team += 1
                continue
            season[int(pid)] = {
                'bf': bf, 'so': stat.get('strikeOuts', 0) or 0,
                'bb': stat.get('baseOnBalls', 0) or 0, 'hr': stat.get('homeRuns', 0) or 0,
                'team_abb': team_abb,
            }
        print(f"Bullpen quality current-season fetch: {len(season)} pitchers with BF>0 and a "
              f"resolvable team ({n_no_team} skipped, unresolvable team name)")
        return season
    except Exception as e:
        print(f"Warning: bullpen quality current-season fetch failed: {e}")
        return {}


def get_bullpen_quality_population(force=False):
    """
    Returns dict: team_abb -> {score, percentile, n_relievers, bf_total}.
    Aggregates every RELIEVER (is_starter=False in the prior) currently on each team, blends
    each one's career prior with current-season-to-date (same Marcel formula as the starter
    composite), then takes the BF-weighted average score across that team's relievers. z-scored/
    percentile-ranked across all teams with enough relievers to trust.
    Cached BP_SEASON_TTL seconds. force=True bypasses this cache (see /api/picks?refresh=1).
    """
    global _bp_population_cache, _bp_population_cache_time
    now = datetime.now().timestamp()
    if not force and _bp_population_cache and _bp_population_cache_time and \
            (now - _bp_population_cache_time < BP_SEASON_TTL):
        return _bp_population_cache

    load_pitcher_quality_prior()
    if not _pq_prior:
        return {}

    season = _bp_fetch_current_season_by_team()
    reliever_scores = {}  # pid -> (score, bf, team_abb)
    for pid, prior in _pq_prior.items():
        if prior.get('is_starter'):
            continue
        s = season.get(pid)
        if not s or s['bf'] <= 0:
            continue
        bf = s['bf']
        season_rates = {
            'k_rate':  s['so'] / bf, 'bb_rate': s['bb'] / bf, 'hr_rate': s['hr'] / bf,
        }
        blend = {}
        for stat in ('k_rate', 'bb_rate', 'hr_rate'):
            ph = PQ_PHANTOM_BF[stat]
            blend[stat] = (ph * prior[stat] + bf * season_rates[stat]) / (ph + bf)
        reliever_scores[pid] = (blend, bf, s['team_abb'])

    if len(reliever_scores) < 30:  # need at least ~1/team on average to trust anything
        print(f"Bullpen quality: only {len(reliever_scores)} relievers with current-season data, "
              f"skipping (need current-season fetch to have actually worked)")
        return {}

    # Reference distribution for standardizing K/BB/HR — same starters-only-style reasoning as
    # the pitcher composite doesn't apply here (this population IS already relievers-only by
    # construction), so just use the full reliever pool directly.
    means = {stat: sum(b[stat] for b, _, _ in reliever_scores.values()) / len(reliever_scores)
             for stat in ('k_rate', 'bb_rate', 'hr_rate')}
    stds  = {stat: (sum((b[stat] - means[stat]) ** 2 for b, _, _ in reliever_scores.values())
                     / len(reliever_scores)) ** 0.5
             for stat in ('k_rate', 'bb_rate', 'hr_rate')}

    def compute_score(b):
        z = {stat: (b[stat] - means[stat]) / stds[stat] if stds[stat] > 0 else 0.0
             for stat in ('k_rate', 'bb_rate', 'hr_rate')}
        return PQ_WEIGHTS['k'] * z['k_rate'] + PQ_WEIGHTS['bb'] * z['bb_rate'] + PQ_WEIGHTS['hr'] * z['hr_rate']

    by_team = {}
    for pid, (blend, bf, team_abb) in reliever_scores.items():
        score = compute_score(blend)
        by_team.setdefault(team_abb, {'wsum': 0.0, 'wtot': 0.0, 'n': 0})
        by_team[team_abb]['wsum'] += score * bf
        by_team[team_abb]['wtot'] += bf
        by_team[team_abb]['n']   += 1

    team_scores = {team: d['wsum'] / d['wtot'] for team, d in by_team.items() if d['wtot'] > 0}
    if len(team_scores) < 10:
        print(f"Bullpen quality: only {len(team_scores)} teams with a usable bullpen score, skipping")
        return {}

    scores_sorted = sorted(team_scores.values())
    n_ref = len(scores_sorted)
    population = {}
    for team, score in team_scores.items():
        pctile = 100.0 * bisect.bisect_left(scores_sorted, score) / n_ref if n_ref > 1 else 50.0
        population[team] = {
            'score':       round(score, 4),
            'percentile':  round(pctile, 1),
            'n_relievers': by_team[team]['n'],
            'bf_total':    by_team[team]['wtot'],
        }

    print(f"Bullpen quality population built: {len(population)} teams, "
          f"{len(reliever_scores)} relievers with current-season data")
    _bp_population_cache      = population
    _bp_population_cache_time = now
    return population


def _fg_tt_note(fg_flag, gate_reason, pinnacle_point):
    """fg_tt_note shown on the dashboard card -- same three-state shape as _pq_note/_off_note."""
    if not fg_flag:
        return None
    if gate_reason == 'line_match':
        return (f'Opposing bullpen rates top-quartile — Pinnacle full-game line confirmed at '
                 f'{pinnacle_point} for this team, this historically favors UNDER (2025 n=852, '
                 '65.0% hit, +22.7% ROI, p<0.001; 2026 n=39, 66.7% hit, +27.8% ROI, p=0.027 — '
                 'tracking signal only, do not bet)')
    return ('Opposing bullpen rates top-quartile — IF the full-game line for this team is 3.5, '
             '4.5, or 5.5, this historically favors UNDER (Pinnacle gate inactive, no odds feed '
             'configured — check the line yourself; tracking signal only, do not bet)')


# ── OFFENSE QUALITY COMPOSITE (research / tracking signal, NOT validated for betting) ──
# Built June 18, 2026. Lineup-weighted K/BB/SLG composite per batting team, blended the same
# way as the pitcher composite. Career prior (2022-2025 pooled, batter_offense_prior_2026.csv)
# Marcel-blended with current-season-to-date stats (phantom-PA/AB constants recovered exactly
# from the 2025 research file: K=60 PA, BB=120 PA, SLG=320 AB — different stat, different
# constants than the pitcher side's K=70/BB=170/HR=1320, this is correct, not a typo).
#   off_z = -k_rate_z + bb_rate_z + slg_z   (higher = stronger offense)
# Team-game value = each lineup batter's percentile, weighted by LINEUP_SLOT_WEIGHTS (PA-share
# by batting slot), normalized over available batters — exact formula verified against the 2025
# research file's off_pctile column to <1e-9 residual before being trusted here.
#
# VALIDATION STATUS — read before trusting this:
#   Found June 18, 2026 testing pitcher/offense quartile interactions. The gate below (mid-tier
#   off_pctile Q3 AND low o_bb_rate) at the 1.5 F5 line: 2025 n=89, 57.3% U, +17.9% ROI, p=0.026;
#   2026 OOS n=85, 58.8% U, +17.2% ROI, p=0.016 — tight replication, best OOS result of that
#   session. Book under odds nearly identical between the high/low o_bb_rate groups (1.059 vs
#   1.064 decimal) despite the 15-point hit-rate gap — book does not appear to price this split.
#   Still only 2 years / 1 line level / n~85-90 — same "needs more data before deploying" bar as
#   every other signal here. Flipped to the over side: hit rate mirrors as expected, but over ROI
#   is much weaker (+3.0% vs the under side's +17.9%) — inefficiency is concentrated on the under
#   side specifically, do not assume an equivalent over edge on the patient-offense side.
#
# CRITICAL — like the pitcher composite, this ONLY works at the 1.5 F5 line specifically (2.5
# line: p=0.12 OOS, opposite-signed in 2025). This app does not fetch live odds/lines, so it
# cannot gate on line level — check the actual posted F5 line yourself before treating this as
# anything actionable. Independent of the pitcher quality composite — a game can show either,
# both, or neither flag; they describe different sides of the same matchup.

OFF_PHANTOM        = {'k_rate': 60.0, 'bb_rate': 120.0, 'slg': 320.0}
OFF_PRIOR_CSV       = 'batter_offense_prior_2026.csv'
LINEUP_WEIGHTS_CSV  = 'lineup_slot_pa_weights.csv'
# NOTE: this is NOT a generic "50th-75th percentile" Q3 — it's the actual empirical off_pctile
# range covered by the Q3 quartile WITHIN 1.5-line games specifically in the research sample
# (1.5-line games are pre-selected toward certain off_pctile values, same population-selection
# effect documented for the pitcher composite's line-level splits). Stable across both years:
# 2025 Q3 band = 66.6-72.6, 2026 OOS Q3 band = 67.7-73.5 — using the union, rounded.
OFF_Q3_LOW_PCTILE   = 66.5
OFF_Q3_HIGH_PCTILE  = 73.5
# o_bb_rate median within that Q3/1.5-line band: 2025=0.0855, 2026=0.0866 — averaged.
OFF_BB_RATE_MEDIAN  = 0.0860
OFF_SEASON_TTL      = 21600  # 6h, same cadence as the pitcher composite
# Fraction of prior batters with ZERO current-season PA above which we treat the current-season
# hitting fetch as having FAILED (transient MLB API blip) rather than just normally sparse. Under
# normal mid-season conditions ~44% of the 883-batter prior has no 2026 PA (bench/minors/2022-25
# players no longer up) -- measured June 23, 2026 -- so the old 0.5 warning threshold was only ~6
# points above baseline. A real fetch failure degrades ~95-100% of batters to pure career prior;
# 0.85 sits cleanly between the two. See get_offense_quality_population for why this now BAILS
# (returns empty, uncached) instead of just warning.
OFF_FETCH_FAIL_FRAC = 0.85

_off_prior            = {}
_off_prior_loaded      = False
_off_population_cache      = None
_off_population_cache_time = None
_lineup_slot_weights   = {}
_lineup_weights_loaded = False
# Rolling median batter K-rate from the live Marcel blend — recomputed each time
# get_offense_quality_population() rebuilds. Logged for diagnostics (printed when
# population rebuilds). No longer used in k_prop_flag — the K-composite handles
# opp K-rate via its fitted coefficient (K_COMP_O_K = +15.192).
_off_k_rate_median    = None


def load_lineup_slot_weights():
    global _lineup_slot_weights, _lineup_weights_loaded
    if _lineup_weights_loaded:
        return
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, LINEUP_WEIGHTS_CSV)
    if not os.path.exists(path):
        print(f"Warning: {LINEUP_WEIGHTS_CSV} not found — offense quality composite disabled")
        return
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                _lineup_slot_weights[int(float(row['batting_slot']))] = float(row['weight'])
        _lineup_weights_loaded = True
    except Exception as e:
        print(f"Warning: lineup slot weights load failed: {e}")


def load_batter_offense_prior():
    global _off_prior, _off_prior_loaded
    if _off_prior_loaded:
        return
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, OFF_PRIOR_CSV)
    if not os.path.exists(path):
        print(f"Warning: {OFF_PRIOR_CSV} not found — offense quality composite disabled")
        return
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                _off_prior[int(row['mlbID'])] = {
                    'k_rate':  float(row['k_rate']),
                    'bb_rate': float(row['bb_rate']),
                    'slg':     float(row['slg']),
                }
        _off_prior_loaded = True
        print(f"Batter offense prior loaded: {len(_off_prior)} batters")
    except Exception as e:
        print(f"Warning: batter offense prior load failed: {e}")


def _off_fetch_current_season():
    """
    Current-season PA/AB/K/BB/TB per batter via the official MLB Stats API bulk season-stats
    endpoint -- same fix and same reasoning as _pq_fetch_current_season above (the prior
    Baseball-Reference scrape was failing 100% of the time in production).
    """
    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/stats",
            params={'stats': 'season', 'group': 'hitting', 'season': datetime.now().year,
                    'sportId': 1, 'limit': 3000, 'playerPool': 'ALL'},
            timeout=20)
        r.raise_for_status()
        splits = r.json().get('stats', [{}])[0].get('splits', [])
        season = {}
        for s in splits:
            stat = s.get('stat', {})
            pa = stat.get('plateAppearances', 0) or 0
            pid = s.get('player', {}).get('id')
            if pa <= 0 or pid is None:
                continue
            season[int(pid)] = {
                'pa': pa, 'ab': stat.get('atBats', 0) or 0, 'so': stat.get('strikeOuts', 0) or 0,
                'bb': stat.get('baseOnBalls', 0) or 0, 'tb': stat.get('totalBases', 0) or 0,
            }
        print(f"Offense quality current-season fetch: {len(season)} batters with PA>0")
        return season
    except Exception as e:
        print(f"Warning: offense quality current-season fetch failed: {e}")
        return {}


def get_offense_quality_population(force=False):
    """
    Returns dict: mlbID -> {score, percentile, k_rate, bb_rate, slg}.
    Blends career prior with current-season-to-date (Marcel). Cached OFF_SEASON_TTL seconds.
    force=True bypasses this cache (see /api/picks?refresh=1).
    """
    global _off_population_cache, _off_population_cache_time
    now = datetime.now().timestamp()
    if not force and _off_population_cache and _off_population_cache_time and \
            (now - _off_population_cache_time < OFF_SEASON_TTL):
        return _off_population_cache

    load_batter_offense_prior()
    if not _off_prior:
        return {}

    season = _off_fetch_current_season()
    blended = {}
    current_pa_by_bid = {}
    n_zero_current = 0
    for bid, prior in _off_prior.items():
        s = season.get(bid, {'pa': 0, 'ab': 0, 'so': 0, 'bb': 0, 'tb': 0})
        pa, ab = s['pa'], s['ab']
        current_pa_by_bid[bid] = pa
        if pa == 0:
            n_zero_current += 1
        season_rates = {
            'k_rate':  s['so'] / pa if pa > 0 else 0,
            'bb_rate': s['bb'] / pa if pa > 0 else 0,
            'slg':     s['tb'] / ab if ab > 0 else 0,
        }
        blend = {
            'k_rate':  (OFF_PHANTOM['k_rate']  * prior['k_rate']  + pa * season_rates['k_rate'])  / (OFF_PHANTOM['k_rate']  + pa),
            'bb_rate': (OFF_PHANTOM['bb_rate'] * prior['bb_rate'] + pa * season_rates['bb_rate']) / (OFF_PHANTOM['bb_rate'] + pa),
            'slg':     (OFF_PHANTOM['slg']     * prior['slg']     + ab * season_rates['slg'])     / (OFF_PHANTOM['slg']     + ab),
        }
        blended[bid] = blend

    # Population-wide current-season fetch failure guard (added June 23, 2026). If nearly every
    # batter degraded to pure 2022-2025 career prior, the hitting fetch failed (transient MLB API
    # blip -- the exact incident shape the comments above warn about). Do NOT build, cache, or serve
    # offense flags off this: a cached degraded population pins every lineup's avg PA to ~0 for
    # OFF_SEASON_TTL (6h) and fires offense-family signals (off_q3_gate, off_fade, joint) on stale
    # priors -- including straight to Telegram, since off_fade has no per-flag stale gate the way the
    # pitcher composite got one after the Bieber incident (see api_notify PQ_BF_STALE_THRESHOLD).
    # Return empty WITHOUT caching so the next cycle retries once the fetch recovers; get_lineup_
    # offense_quality already returns None on an empty population, cleanly suppressing every offense
    # flag this cycle rather than emitting an untrustworthy one. Threshold = OFF_FETCH_FAIL_FRAC.
    if n_zero_current > len(_off_prior) * OFF_FETCH_FAIL_FRAC:
        print(f"Offense quality: current-season data missing for {n_zero_current}/{len(_off_prior)} "
              f"batters (>{OFF_FETCH_FAIL_FRAC:.0%}) -- treating as fetch failure, returning empty and "
              f"NOT caching so offense signals are suppressed this cycle and retried next cycle")
        return {}

    if len(blended) < 10:
        return {}

    means = {stat: sum(blended[bid][stat] for bid in blended) / len(blended) for stat in ('k_rate', 'bb_rate', 'slg')}
    stds  = {stat: (sum((blended[bid][stat] - means[stat]) ** 2 for bid in blended) / len(blended)) ** 0.5
             for stat in ('k_rate', 'bb_rate', 'slg')}

    def compute_score(b):
        z = {stat: (b[stat] - means[stat]) / stds[stat] if stds[stat] > 0 else 0.0
             for stat in ('k_rate', 'bb_rate', 'slg')}
        return -z['k_rate'] + z['bb_rate'] + z['slg']

    scores_sorted = sorted(compute_score(blended[bid]) for bid in blended)
    n_ref = len(scores_sorted)

    population = {}
    for bid, b in blended.items():
        score  = compute_score(b)
        pctile = 100.0 * bisect.bisect_left(scores_sorted, score) / n_ref if n_ref > 1 else 50.0
        population[bid] = {
            'score':      round(score, 4),
            'percentile': round(pctile, 1),
            'k_rate':     round(b['k_rate'], 4),
            'bb_rate':    round(b['bb_rate'], 4),
            'slg':        round(b['slg'], 4),
            # Same diagnostic as the pitcher side: 0 means this batter's current-season fetch came
            # up empty and their score is pure career prior. See incident notes, June 2026.
            'current_pa': current_pa_by_bid.get(bid, 0),
        }

    # Compute rolling median batter K-rate from the blended population — stored for
    # diagnostics and printed each rebuild cycle. The K-composite model (k_prop_flag)
    # incorporates o_k_rate directly via a fitted coefficient rather than using this
    # median as a threshold, so this is informational only.
    global _off_k_rate_median
    import statistics as _stats
    k_rates = [v['k_rate'] for v in population.values() if v.get('k_rate') is not None]
    if k_rates:
        _off_k_rate_median = round(_stats.median(k_rates), 4)
        print(f"Offense quality population built: {len(population)} batters, "
              f"{n_zero_current} with zero current-season PA, "
              f"k_rate median={_off_k_rate_median:.4f}")
    else:
        print(f"Offense quality population built: {len(population)} batters, "
              f"{n_zero_current} with zero current-season PA")
    _off_population_cache      = population
    _off_population_cache_time = now
    return population


def get_lineup_offense_quality(lineup, population):
    """
    PA-weights each lineup batter's percentile and bb_rate by batting slot (lineup_slot_pa_weights),
    normalized over batters with population data available. Formula verified against the 2025
    research file's off_pctile column to <1e-9 residual before being trusted live.
    Returns None if fewer than 5 lineup batters have population data.
    """
    load_lineup_slot_weights()
    if not _lineup_slot_weights or not population:
        return None
    wsum_pctile = wsum_bb = wsum_k = wsum_pa = wtot = 0.0
    n_matched = 0
    for batter in lineup:
        info = population.get(batter.get('id'))
        slot = batter.get('batting_order')
        if info is None or slot not in _lineup_slot_weights:
            continue
        w = _lineup_slot_weights[slot]
        wsum_pctile += info['percentile']         * w
        wsum_bb     += info['bb_rate']            * w
        wsum_k      += info['k_rate']             * w
        wsum_pa     += info.get('current_pa', 0)  * w
        wtot        += w
        n_matched   += 1
    if n_matched < 5 or wtot <= 0:
        return None
    return {
        'off_pctile':      round(wsum_pctile / wtot, 1),
        'o_bb_rate':       round(wsum_bb / wtot, 4),
        'o_k_rate':        round(wsum_k / wtot, 4),   # PA-weighted lineup K rate (for K projection)
        'current_pa_avg':  round(wsum_pa / wtot, 1),
    }


# ── PROJECTED STRIKEOUTS (informational, June 23, 2026) ───────────────────────────────
# Log5 matchup K projection shown on pitcher-quality flags so the starter's expected K total is
# visible at a glance (e.g. for eyeballing K props). NOT a bet signal -- this projection agrees
# with sharp books on most starts; any edge lives only in rare divergences, and individual-pitcher
# input noise (thin priors, outing-length) drives most big gaps. See expected_k_validation research.
K_PROJ_LEAGUE_K_RATE = 0.221   # 2026 league K per plate appearance
K_PROJ_EXPECTED_BF   = 22.5    # league-average start's batters faced (~5.2 IP); fallback outing
K_PROJ_BF_PHANTOM    = 4       # shrink a pitcher's season BF/start toward the league average
# ── K-PROP COMPOSITE (June 25, 2026) ─────────────────────────────────────────
# OLS fit on 2025 proj_error (actual_k - proj_k) using pitcher and opponent rates.
# o_k_rate dominates (+15.2) because the baseline proj (p_k_rate * 22) ignores opponent
# K-rate entirely. p_k_rate is negative (-4.8) because it's already baked into proj_k —
# we're predicting the RESIDUAL. Threshold = 80th pctile of this score in 2025 U1.5 starts.
#
# VALIDATION: 2025 in-sample n=138, +0.71K, 60% beat; 2026 true OOS n=44, +0.67K, 66%.
# Combined 2025+2026: n=182, +0.70K, 61.5%, p=0.0000 vs 0. Flag vs non-flag p=0.0002.
# Much stronger than the original pq_q4 composite for K props (which collapses in 2026 OOS).
# No separate opp_k_hi filter needed — o_k_rate is the dominant term in the composite,
# so a pitcher only clears the threshold when facing a legitimately high-K lineup.
# Eyeball the posted K prop line vs projected_k; no automated odds pull yet.
K_COMP_P_K       = -4.817    # pitcher K-rate (negative: already in proj_k baseline)
K_COMP_P_BB      = -11.906   # pitcher BB-rate
K_COMP_P_HR      = -9.897    # pitcher HR-rate
K_COMP_O_K       = +15.192   # opposing lineup K-rate (dominant predictor)
K_COMP_INTERCEPT = -0.884
K_COMP_THRESH    = 0.5839    # 80th pctile from 2025 calibration; applied unchanged to 2026 OOS

def expected_bf_per_start(bf, gs):
    """Per-pitcher expected batters faced in a start: season BF/start, shrunk toward the league
    average (K_PROJ_BF_PHANTOM phantom starts) so thin samples don't swing it, clamped to a sane
    range. Falls back to the league average when the pitcher has no starts yet. Note: `bf` is the
    season total across all appearances, so for a swingman this slightly overcounts -- fine for the
    announced starters this is used on."""
    if not bf or gs < 1:
        return K_PROJ_EXPECTED_BF
    est = (K_PROJ_BF_PHANTOM * K_PROJ_EXPECTED_BF + bf) / (K_PROJ_BF_PHANTOM + gs)
    return round(max(16.0, min(28.0, est)), 1)

def k_composite_score(p_k_rate, p_bb_rate, p_hr_rate, o_k_rate):
    """K-prop composite score: OLS on 2025 proj_error, applied unchanged to 2026 OOS.
    Returns None if any input is missing (flag suppressed, not fired)."""
    if None in (p_k_rate, p_bb_rate, p_hr_rate, o_k_rate):
        return None
    return (K_COMP_INTERCEPT
            + K_COMP_P_K  * p_k_rate
            + K_COMP_P_BB * p_bb_rate
            + K_COMP_P_HR * p_hr_rate
            + K_COMP_O_K  * o_k_rate)


def project_starter_ks(p_k_rate, o_k_rate, exp_bf=None):
    """Log5 of the pitcher's blended K rate vs the opposing lineup's K rate, x this pitcher's
    expected outing length. Returns expected strikeouts (1 decimal), or None if a rate is missing."""
    if not p_k_rate or not o_k_rate:
        return None
    matchup_k_per_bf = p_k_rate * o_k_rate / K_PROJ_LEAGUE_K_RATE
    return round(matchup_k_per_bf * (exp_bf or K_PROJ_EXPECTED_BF), 1)


def load_fg_profiles():
    global _FG_OFF_Z, _FG_BP_WEAK_Z, _FG_PROFILES_LOADED
    if _FG_PROFILES_LOADED:
        return
    base = os.path.dirname(os.path.abspath(__file__))
    year = datetime.now().year
    for y in [year, year - 1]:
        off_path = os.path.join(base, f"team_offense_baseline_{y}.csv")
        bp_path  = os.path.join(base, f"bullpen_profile_{y}.csv")
        if os.path.exists(off_path) and os.path.exists(bp_path):
            try:
                with open(off_path) as f:
                    for row in csv.DictReader(f):
                        _FG_OFF_Z[row['team']] = float(row['off_z'])
                with open(bp_path) as f:
                    for row in csv.DictReader(f):
                        _FG_BP_WEAK_Z[row['team']] = float(row['bp_weak_z'])
                _FG_PROFILES_LOADED = True
                print(f"FG profiles loaded: {y} ({len(_FG_OFF_Z)} teams)")
                return
            except Exception as e:
                print(f"Warning: FG profile load failed for {y}: {e}")
    print("Warning: FG profiles not found")


def is_fg_valid_window():
    today = _date.today()
    start = _date(today.year, FG_VALID_START_MONTH, FG_VALID_START_DAY)
    end   = _date(today.year, FG_VALID_END_MONTH,   FG_VALID_END_DAY)
    return start <= today <= end


def compute_fg_under_flag(batting_team, opp_team, blend_70_30_score):
    in_window = is_fg_valid_window()
    starter_z = round((blend_70_30_score - FG_BLEND_MEAN) / FG_BLEND_STD, 3)
    off_z         = _FG_OFF_Z.get(batting_team)
    opp_bp_weak_z = _FG_BP_WEAK_Z.get(opp_team)
    if opp_bp_weak_z is None:
        return {'fg_under_signal': False, 'starter_z': starter_z,
                'opp_bp_weak_z': None, 'off_z': off_z,
                'bp_in_band': False, 'in_valid_window': in_window,
                'reason': 'missing_profile_data'}
    starter_flag = starter_z <= FG_STARTER_Z_THRESH
    bp_band_flag = FG_BP_BAND_LOW <= opp_bp_weak_z <= FG_BP_BAND_HIGH
    signal       = in_window and starter_flag and bp_band_flag
    if signal:                reason = 'all_conditions_met'
    elif not in_window:       reason = 'outside_valid_window'
    elif not starter_flag:    reason = f'starter_not_suppressed (z={starter_z:.2f})'
    else:                     reason = f'bp_outside_band (z={opp_bp_weak_z:.2f})'
    return {'fg_under_signal': signal, 'starter_z': starter_z,
            'opp_bp_weak_z': round(opp_bp_weak_z, 3),
            'off_z': round(off_z, 3) if off_z is not None else None,
            'bp_in_band': bp_band_flag, 'in_valid_window': in_window, 'reason': reason}


def download_pkl(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        tmp_path = tmp.name
    gdown.download(url, tmp_path, quiet=True)
    with open(tmp_path, 'rb') as f:
        obj = pickle.load(f)
    os.unlink(tmp_path)
    return obj


def load_model():
    global _model_cache, _model_cache_time
    now = datetime.now().timestamp()
    if _model_cache and _model_cache_time and (now - _model_cache_time < MODEL_CACHE_TTL):
        return _model_cache
    model = {}
    for name, fid in FILE_IDS.items():
        # Each download runs in its own thread with a hard timeout -- gdown has no native
        # timeout and a stalled Google Drive request used to hang the whole worker process
        # until Gunicorn's timeout force-killed it. shutdown(wait=False) is deliberate: on
        # timeout we abandon the stuck thread rather than blocking the request on it too.
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            model[name] = ex.submit(download_pkl, fid).result(timeout=DOWNLOAD_TIMEOUT_SECONDS)
        except _FutureTimeoutError:
            print(f"Warning: could not load {name}: download exceeded {DOWNLOAD_TIMEOUT_SECONDS}s, skipping")
            model[name] = None
        except Exception as e:
            print(f"Warning: could not load {name}: {e}")
            model[name] = None
        finally:
            ex.shutdown(wait=False)
    model['arch_hitter_map'] = model['arch_hitter_map_combined']
    model['archetypes']      = model['archetypes_combined']

    splits_lookup = {}
    try:
        splits_df = model.get('hitter_splits')
        if splits_df is not None and isinstance(splits_df, pd.DataFrame):
            for _, row in splits_df.iterrows():
                diff = float(row['split_diff']) if pd.notna(row['split_diff']) else 0.0
                if abs(diff) >= DFS_SPLIT_THRESHOLD:
                    splits_lookup[int(row['batter'])] = {
                        'diff': round(diff, 2),
                        'flag': 'favors_RHP' if diff > 0 else 'favors_LHP',
                    }
    except Exception as e:
        print(f"Warning: hitter splits build failed: {e}")
    model['splits_lookup'] = splits_lookup

    _model_cache      = model
    _model_cache_time = now
    return model


def get_pitcher_avg_score(pitcher_id, pitcher_scores, archetypes):
    if pitcher_id not in pitcher_scores:
        return None, 'Unknown'
    pitcher    = pitcher_scores[pitcher_id]
    all_scores = [pitcher['archetypes'].get(ak, {}).get('shrunk_rate', LEAGUE_AVG)
                  for ak in archetypes]
    avg = sum(all_scores) / len(all_scores)
    if avg >= 5.5:   category = 'Very Juicy'
    elif avg >= 4.5: category = 'Juicy'
    elif avg >= 3.5: category = 'Average'
    elif avg >= 3.0: category = 'Safe'
    else:            category = 'Very Safe'
    return round(avg, 2), category


def get_lineups_and_starters(game_date):
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?sportId=1&date={game_date}&hydrate=probablePitcher,lineups")
    try:
        data = requests.get(url, timeout=15).json()
    except Exception:
        return []
    games = []
    for date_data in data.get('dates', []):
        for g in date_data.get('games', []):
            gid = g['gamePk']
            game_status = g.get('status', {}).get('abstractGameState', '')
            if game_status not in ('Preview', ''):
                continue
            home = g['teams']['home']['team'].get('abbreviation') or g['teams']['home']['team'].get('name', 'HOME')
            away = g['teams']['away']['team'].get('abbreviation') or g['teams']['away']['team'].get('name', 'AWAY')
            hp   = g['teams']['home'].get('probablePitcher', {})
            ap   = g['teams']['away'].get('probablePitcher', {})
            home_lineup, away_lineup = [], []
            try:
                bs = requests.get(
                    f"https://statsapi.mlb.com/api/v1/game/{gid}/boxscore",
                    timeout=15).json()
                for team_key, lineup in [('home', home_lineup), ('away', away_lineup)]:
                    td = bs.get('teams', {}).get(team_key, {})
                    batting_order = td.get('battingOrder', [])[:9]
                    for slot_idx, pid in enumerate(batting_order, start=1):
                        p = td.get('players', {}).get(f'ID{pid}', {})
                        lineup.append({
                            'id':            pid,
                            'name':          p.get('person', {}).get('fullName', ''),
                            'hand':          p.get('batSide', {}).get('code', 'R'),
                            'batting_order': slot_idx,
                            'position':      p.get('position', {}).get('abbreviation', ''),
                        })
            except Exception:
                pass
            game_time = None
            try:
                raw_time = g.get('gameDate', '')
                if raw_time:
                    utc_dt  = datetime.strptime(raw_time, '%Y-%m-%dT%H:%M:%SZ')
                    utc_dt  = utc_dt.replace(tzinfo=pytz.utc)
                    et_dt   = utc_dt.astimezone(pytz.timezone('America/New_York'))
                    game_time = et_dt.strftime('%-I:%M %p') + ' ET'
            except Exception:
                pass
            games.append({
                'game_id':           gid,
                'home_team':         home,
                'away_team':         away,
                'home_pitcher_id':   hp.get('id'),
                'away_pitcher_id':   ap.get('id'),
                'home_pitcher_name': hp.get('fullName', 'TBD'),
                'away_pitcher_name': ap.get('fullName', 'TBD'),
                'home_lineup':       home_lineup,
                'away_lineup':       away_lineup,
                'game_time':         game_time,
            })
    return games


# ── LIVE ODDS (The Odds API) — F5 team totals, line-level data for tracking signals ──
# Added June 18, 2026. Both tracking signals (Pitcher Quality Composite, Offense Quality
# Composite) only work at the 1.5 F5 line specifically — until now this app had no live odds
# feed and required manually checking the posted line. Pulls real lines/prices from the three
# books Zach can actually bet PLUS Pinnacle.
#
# Pinnacle added June 19, 2026, originally display-only, then upgraded to a HARD GATE the same
# day per Zach's explicit request — Zach cannot access Pinnacle from the US (see CLAUDE.md
# "IMPORTANT CORRECTION — VALIDATION BOOK"), but it's what signal #4 and the offense composite
# were actually validated against historically, and the three bettable books don't always agree
# with each other on the line level itself (only ~91-93% line-match rate vs Pinnacle historically
# — see CLAUDE.md "TEAM-LEVEL DEEP DIVE" Pinnacle-gate research). So: a pq_q4/off_q3_gate
# candidate is now only surfaced as a flag at all if Pinnacle's posted F5 line for that team is
# exactly 1.5 — see PINNACLE_GATE block in get_tracking_only_flags. Bettable-book lines (and any
# Pinnacle/bettable-book divergence) are still attached for shopping the best price once a flag
# clears the gate. PINNACLE_BOOK_LABEL is rendered first/separately by the frontend (see
# formatOddsLines in index.html) rather than mixed in with the bettable books.
#
# SAFETY FALLBACK: if ODDS_API_KEY isn't configured at all (no live odds feed exists to check
# against), the gate is skipped rather than silently suppressing every tracking flag app-wide —
# the June 19 production incident was exactly this failure shape (a quiet upstream outage making
# the whole app look empty with no visible cause), not worth risking again for a feed that's
# allowed to not exist. Per-game/per-book gaps (feed configured but Pinnacle didn't post this
# specific game) still suppress that one flag, logged via the "Pinnacle gate:" print each run.
#
# Market key 'team_totals_1st_5_innings' requires the per-event odds endpoint (not the cheaper
# bulk endpoint), so this costs real API credits: cost = markets x regions per event, now
# 2 markets x 3 regions = 6 credits/game (was 1 market x 3 regions = 3 credits/game before
# adding the joint-total market below, June 20, 2026 -- ~90-174 credits/day at a typical
# 15-29 game slate, was ~45-87 before). regions=us covers DraftKings/Fanatics/FanDuel/BetMGM/
# Caesars (all free additions, same region, June 19, 2026 -- the latter three are inconsistent
# game to game since they don't always post this specific market), regions=us2 covers theScore
# Bet (formerly ESPN Bet — its bookmaker key may still be 'espnbet' on this API; TARGET_BOOKS
# below tries both so a rename doesn't silently break this), regions=eu covers Pinnacle.
# Cached in Redis on ODDS_API_TTL so /api/picks and /api/notify don't double-spend credits when
# both run close together.

ODDS_API_KEY        = os.environ.get('ODDS_API_KEY', '')
ODDS_API_BASE       = 'https://api.the-odds-api.com/v4'
ODDS_F5_MARKET      = 'team_totals_1st_5_innings'
# Joint/combined F5 total (both teams summed) -- a different market from the per-team one
# above. Added June 20, 2026 for the joint-offense tracking signal (see JOINT_OFF_PCTILE_MAX
# below). Bundled into the same per-event request as ODDS_F5_MARKET (one extra market, not an
# extra request) -- see cost note above.
ODDS_JOINT_MARKET   = 'totals_1st_5_innings'
# Full-GAME per-team total (no _1st_5_innings suffix) -- added June 22, 2026 for the FG bullpen
# tracking signal (see FG_BP_PCTILE_GATE above). Same per-event request as the other two markets,
# one more market on top -- pushes the per-event cost up further (see cost note above ODDS_API_KEY),
# still well within the live key's monthly budget at typical slate sizes.
ODDS_FG_MARKET      = 'team_totals'
# Full-GAME joint/combined total (whole-game game total, no _1st_5_innings suffix) -- added
# June 22, 2026 for the FG JOINT bullpen tracking signal (see FG_JOINT_* below). Same per-event
# request as the other three markets, one more market on top. This is the standard MLB game
# total ('totals'), distinct from ODDS_JOINT_MARKET which is the FIRST-5 joint total.
ODDS_FG_JOINT_MARKET = 'totals'
# Pitcher strikeout props (player prop). Added June 27, 2026 to drive the rebuilt K-prop signal
# (the over-favorite price bias — see _kprop_over_fav below). Bundled into get_odds_api_lines'
# per-event call (June 27, 2026) so all books' strikeout prices come back with the F5/FG totals in
# one request — multi-book line shopping at no extra per-event cost. Player-prop outcomes carry the
# pitcher in 'description' and Over/Under in 'name' (same shape pull_k_props_2026.py proved).
ODDS_KPROP_MARKET   = 'pitcher_strikeouts'
# Over-favorite band: the edge (2026 DK, replicates May+June) is betting the OVER when the book
# prices it a moderate favorite. Classified on DraftKings (where it was measured); best price
# across books is used for the actual bet. Refinement tier: opposing offense's F5 team total < 2.0
# sharpened ROI to +17-21% and held both months.
KPROP_OVER_FAV_LO   = -160   # most negative over price still in the band
KPROP_OVER_FAV_HI   = -120   # least negative (closest to even)
KPROP_SHARP_F5_TOTAL = 2.0   # opp F5 total below this = 'sharp' tier
# SHARP-pocket prior-start gate (June 27, 2026). Within the over-favorite pocket, the +12.8% ROI
# is concentrated in pitchers with >= 7 prior starts this season (n=118, 68.6% over, +18.4%);
# the <=3-prior slice is ~breakeven (-0.9%, n=50) and the out-of-pocket thin-prior plays bled
# heavily (Mar/Apr -38%). The driver is estimate stability, not the pitcher: too few starts ->
# noisy point-in-time rates. So a would-be SHARP play with < 7 prior starts is demoted to 'base'
# (the signal still fires; it just loses the SHARP-conviction label). K-rate < 22% carries an
# independent ~-8pt ROI drag even with stable priors (surfaced in the note, not gated).
KPROP_SHARP_MIN_STARTS = 7
ODDS_REGIONS        = 'us,us2,eu'
ODDS_API_TTL        = 14400  # 4h
PINNACLE_BOOK_LABEL = 'Pinnacle'
# FanDuel/BetMGM/Caesars added June 19, 2026 per Zach -- zero extra cost, they're already in the
# same 'us' region response DraftKings/Fanatics come from, just weren't being kept before. He's
# largely ignored these historically since they don't always post the F5 1st-5-innings team
# totals market, so expect them to show up inconsistently game to game -- that's the book simply
# not offering this specific market, not a bug. Caesars' key has changed before (William Hill US
# rebrand), so both candidates are mapped defensively, same pattern as theScore Bet below.
TARGET_BOOKS = {
    'draftkings':    'DraftKings',
    'fanatics':      'Fanatics',
    'espnbet':       'theScore Bet',
    'thescorebet':   'theScore Bet',   # in case the API renames this key
    'fanduel':       'FanDuel',
    'betmgm':        'BetMGM',
    'caesars':       'Caesars',
    'williamhill_us': 'Caesars',       # legacy key from the William Hill US -> Caesars rebrand
    'pinnacle':      PINNACLE_BOOK_LABEL,
}

# ── PARK GATE EXCEPTION (June 20, 2026) ──────────────────────────────────────────────
# The Pinnacle hard gate below only let pq_q4/off_q3 through at exactly F5 1.5 -- but a
# matchup that's genuinely 1.5-quality often gets bumped to 2.5+ purely because it's in a
# hitter-favorable park (Coors being the obvious case), and the line itself hasn't caught up
# to that. Tested (CLAUDE.md "PARK-INFLATED LINE FINDING" + same-day cutoff sweep): among
# elevated (2.5+) games, pq_q4/off_q3 predicts the under specifically inside these 6 parks --
# 2025 n=42, 71.4% hit, +26.0% ROI, p=0.004; 2026 OOS n=21, 71.4% hit, +25.1% ROI, p=0.039.
# Swept 7 cutoffs (decile through half) the same day -- this top-quintile set is the tightest
# year-over-year replication of all of them (identical 71.4% hit rate both years), not just the
# first cutoff tried; narrower cutoffs look better in 2025 alone but fail 2026 OOS the same way
# other narrow-band findings in this project have. Source: park_factor_woba_current.pkl
# (Statcast actual wOBA/xwOBA on balls in play, 2021-2025, leak-free), top quintile of 30 teams.
PARK_GATE_TEAMS = {'COL', 'CIN', 'BOS', 'PHI', 'AZ', 'TB'}
PARK_GATE_MIN_POINT = 2.5

# ── JOINT OFFENSE TRACKING SIGNAL (June 20, 2026) ────────────────────────────────────
# Different market, different mechanism from pq_q4/off_q3 above -- this one is about the
# JOINT/combined F5 total (both teams' runs summed), not a per-team total. Validated in
# CLAUDE.md "FOLLOW-UP, SAME DAY: DIGGING INTO #6": when both lineups project weak (combined
# average lineup off_pctile in the bottom quartile of games) AND the joint F5 line is exactly
# 3.5, the OVER hits more than the market implies. Tight cross-year match, FanDuel-specific:
# 2025 n=34, 67.6% hit, +21.8% ROI; 2026 OOS n=27, 70.4% hit, +22.2% ROI. Gated on FanDuel's
# own line (JOINT_GATE_BOOK), not Pinnacle's -- deliberate deviation from the pq_q4/off_q3
# Pinnacle-gate convention, because Pinnacle posts this specific market on only ~15% of events
# (too thin to gate on), while FanDuel posts it on essentially every event and is also the book
# this finding was actually validated against. NOT extended to elevated lines in hitter-
# favorable parks the way pq_q4/off_q3 was -- tested same day, does not replicate (2025 n=29
# +24.2% ROI vs 2026 n=23 -7.3% ROI, opposite signs). Coors/COL games never post a 3.5 joint
# line in two years of data (mean 5.6, range 4.5-6.5 across 22 games) so this signal will
# essentially never fire there -- expected, not a bug. JOINT_OFF_PCTILE_MAX (63.5) is NOT a
# round number -- it's the empirical 25th-percentile cutoff of combined lineup off_pctile
# across both validation years (63.1 in 2025, 63.9 in 2026 -- averaged), the same way
# OFF_Q3_LOW/HIGH_PCTILE above are research-derived, not guessed.
JOINT_OFF_PCTILE_MAX = 63.5
JOINT_LINE_TARGET    = 3.5
JOINT_GATE_BOOK      = 'FanDuel'

# ── OFFENSE-OVERPRICED FG JOINT UNDER (June 22, 2026) ── the ONE clean leak-free lead from the
# post-bullpen-leak search (see CLAUDE.md "CLEAN LEAK-FREE SEARCH"): combined offense (both lineups)
# is OVERPRICED in the FULL-GAME joint total, so a strong combined offense fades to the UNDER. Same
# combined-offense measure as joint_offense above, but the TOP tail and the FULL-GAME joint market.
# Threshold = empirical 75th pctile of combined off_pctile (73.1 in 2025, 73.2 in 2026 -> 73.0).
# Leak-free (offense = announced lineups + blended prior rates), confound-checked (NOT a line proxy;
# corr(offense,residual) -0.12/-0.34 both years). TRACKING LEAD, NOT a bet trigger: 2025 n=114
# 62.3%/+18.9% p=0.006, 2026 n=24 66.7%/+26.4% (thin). Gated on the DraftKings full-game total >= 7.0.
OFF_FADE_PCTILE_MIN = 73.0
OFF_FADE_LINE_MIN   = 7.0


def _pq_note(pq_q4, gate_reason, pinnacle_point):
    """pq_note shown on the dashboard card -- three states: gate inactive, confirmed at the
    base 1.5 line, or passed via the validated park exception at an elevated line."""
    if not pq_q4:
        return None
    if gate_reason == 'line_1_5':
        return ('Top-quintile pitcher quality — Pinnacle line confirmed at F5 1.5 for this team, '
                 'this historically favors UNDER (not yet proven on 2026 — tracking signal only, '
                 'do not bet)')
    if gate_reason == 'park_exception':
        return (f'Top-quintile pitcher quality — Pinnacle line is F5 {pinnacle_point} in a '
                 'hitter-favorable park (COL/CIN/BOS/PHI/AZ/TB), a validated extension of the '
                 '1.5-line signal (2025 n=42, 71.4% hit, +26.0% ROI; 2026 OOS n=21, 71.4% hit, '
                 '+25.1% ROI, p<0.05 both years — thinner sample than the base 1.5-line signal, '
                 'tracking signal only, do not bet)')
    return ('Top-quintile pitcher quality — IF the F5 line for this team is 1.5 (or 2.5+ in a '
             'hitter-favorable park), this historically favors UNDER (Pinnacle gate inactive, no '
             'odds feed configured — check the line yourself; not yet proven on 2026 — tracking '
             'signal only, do not bet)')


def _kprop_note(ofav, kal_k):
    """K-prop signal note for the rebuilt OVER-FAVORITE bias signal (replaced the leaked
    proj_gap>0.5 Kalshi signal June 27, 2026). Kalshi price kept as a reference only — it does
    NOT drive the signal (exchange pricing, the recreational-shading mechanism likely absent).
    Labeled TRACKING: one season, one book, threshold-searched — not yet OOS/forward-validated."""
    if not ofav or not ofav.get('signal'):
        return None
    dk     = ofav.get('dk_over')
    best   = ofav.get('best_over')
    bk     = ofav.get('best_book')
    line_s = f"{ofav['line']:g}" if ofav.get('line') is not None else '—'
    f5_s   = f"{ofav['opp_f5_total']:g}" if ofav.get('opp_f5_total') is not None else '—'
    tier_s = 'SHARP (opp F5<2.0, >=7 starts)' if ofav.get('tier') == 'sharp' else 'base'
    best_s = f"{best:+d} ({bk})" if best is not None else '—'
    kal_s  = ''
    if kal_k and kal_k.get('implied_line') is not None:
        kal_s = f" [ref: Kalshi line {kal_k['implied_line']}, {kal_k.get('yes_bid','—')}¢]"
    thin_s = ''
    if ofav.get('sharp_blocked_thin'):
        thin_s = (f" [opp F5<2.0 but only {ofav.get('prior_starts')} prior starts (<7) — held at "
                  f"base; point-in-time rates not yet stable, the SHARP premium needs >=7]")
    return (f'K-PROP OVER [{tier_s}] — TRACKING: BUY Over {line_s}K. '
            f'DK over {dk}, best {best_s}, {ofav.get("n_books", 0)} books. '
            f'Opp F5 total {f5_s}.{thin_s}{kal_s} '
            f'(2026 DK over-favorite bias: ~+9% ROI base / +18% sharp [opp F5<2.0 & >=7 starts], '
            f'replicates May+June; UNVALIDATED OOS — tracking only, shop the best over price.)')


def _kprop_tg_bet(f):
    """One-line K-prop OVER bet for Telegram, from a flag dict. Used both for standalone K-prop
    pitchers and for pitchers that ALSO fired pitcher-quality (so the actual bet always shows
    instead of a bare tag)."""
    tier = 'SHARP' if f.get('k_prop_tier') == 'sharp' else 'base'
    line = f.get('kprop_line')
    dk   = f.get('kprop_dk_over')
    best = f.get('kprop_best_over')
    book = f.get('kprop_best_book') or '—'
    f5   = f.get('kprop_opp_f5')
    best_s = f"{best:+d}" if best is not None else '—'
    # ✅ confirms the gate that DEFINES this signal: DraftKings over price in the favorite band
    # (KPROP_OVER_FAV_LO..HI, -160..-120), where the edge was measured.
    dk_s  = f"{dk:+d}" if dk is not None else '—'
    dk_ok = dk is not None and KPROP_OVER_FAV_LO <= dk <= KPROP_OVER_FAV_HI
    dk_mark = '✅' if dk_ok else ''
    # surface why a low-F5 game is still 'base': too few prior starts for the SHARP premium
    thin = ''
    if f.get('kprop_sharp_blocked_thin'):
        thin = f" (F5<2.0 but {f.get('kprop_prior_starts')} starts <7 — held at base)"
    return (f"🎯 K-PROP OVER {line if line is not None else '—'} [{tier}] — "
            f"DK {dk_s}{dk_mark}, best {best_s} ({book}), "
            f"opp F5 {f5 if f5 is not None else '—'}{thin}")


def _off_note(off_q3_gate, gate_reason, pinnacle_point):
    """off_note shown on the dashboard card -- same three states as _pq_note."""
    if not off_q3_gate:
        return None
    if gate_reason == 'line_1_5':
        return ('Mid-tier offense (Q3) with a low walk rate — Pinnacle line confirmed at F5 1.5 '
                 'for this team, this historically favors UNDER (2 years OOS-replicated but still '
                 'tracking signal only, do not bet)')
    if gate_reason == 'park_exception':
        return (f'Mid-tier offense (Q3) with a low walk rate — Pinnacle line is F5 {pinnacle_point} '
                 'in a hitter-favorable park (COL/CIN/BOS/PHI/AZ/TB), a validated extension of the '
                 '1.5-line signal (2025 n=42, 71.4% hit, +26.0% ROI; 2026 OOS n=21, 71.4% hit, '
                 '+25.1% ROI, p<0.05 both years — thinner sample than the base 1.5-line signal, '
                 'tracking signal only, do not bet)')
    return ('Mid-tier offense (Q3) with a low walk rate — IF the F5 line for this team is 1.5 (or '
             '2.5+ in a hitter-favorable park), this historically favors UNDER (Pinnacle gate '
             'inactive, no odds feed configured — check the line yourself; 2 years OOS-replicated '
             'but still tracking signal only, do not bet)')


def _joint_note(fanduel_point):
    """joint_note shown on the dashboard card -- only one state, since this signal has no
    Pinnacle-gate / park-exception machinery (see JOINT OFFENSE TRACKING SIGNAL notes above)."""
    return (f'Bottom-quartile combined lineup offense (both teams), FanDuel joint F5 total '
            f'confirmed at {fanduel_point} — historically favors OVER (2025 n=34, 67.6% hit, '
            '+21.8% ROI; 2026 OOS n=27, 70.4% hit, +22.2% ROI — tight cross-year match, '
            'FanDuel-specific, gated on FanDuel not Pinnacle — see CLAUDE.md. Tracking signal '
            'only, do not bet)')


def fg_joint_tier(combined_pctile):
    """(direction, tier_label, units) for a game's combined bullpen percentile, or (None,None,None).
    UNDER tiers Gold/Silver/Bronze (monotonic both years); OVER is a single tier (doesn't tier)."""
    for lo, hi, label, units in FG_JOINT_UNDER_TIERS:
        if lo <= combined_pctile < hi:
            return ('under', label, units)
    if combined_pctile <= FG_JOINT_OVER_MAX:
        return ('over', 'Flagged', FG_JOINT_OVER_UNITS)
    return (None, None, None)


# Per-tier validated track record, for the note text shown on the card / Telegram.
_FG_JOINT_TIER_STATS = {
    'Gold':   '2025 70.2% hit/+33.1% ROI, 2026 90.0%/+70.9% (thin n=10)',
    'Silver': '2025 63.2%/+20.5%, 2026 70.6%/+34.3%',
    'Bronze': '2025 56.0%/+6.6%, 2026 59.5%/+14.1%',
}


def _off_fade_note(dk_point):
    """offense-overpriced FG joint UNDER note (the clean leak-free lead). Honestly labeled a
    tracking LEAD, not a bet trigger — modest magnitude + thin 2026."""
    return (f'Strong COMBINED offense (both lineups top-quartile) — DraftKings full-game total '
            f'{dk_point}. Clean leak-free LEAD: offense is OVERPRICED in the joint total, so a strong '
            f'offense fades to the UNDER (2025 n=114 62.3% hit/+18.9% ROI p=0.006; 2026 n=24 66.7%/'
            f'+26.4%, thin — confound-checked, NOT a line proxy). TRACKING LEAD — accumulating live '
            f'data, do NOT bet yet.')


def _fg_joint_note(direction, gate_point, tier, units):
    """fg_joint_note shown on the dashboard card / Telegram. Combined (both teams') bullpen
    quality vs. the whole-game combined total; conviction-tiered with suggested unit sizing."""
    if direction == 'under':
        return (f'{tier} UNDER ({units}u) — top-tier COMBINED bullpens (both teams), '
                f'{FG_JOINT_GATE_BOOK} full-game total {gate_point}. Tier record: '
                f'{_FG_JOINT_TIER_STATS.get(tier, "")}. Strongest cross-year result in the project, '
                f'but CORRELATED with the FG team-total signal (same composite). Tracking only, do not bet.')
    return (f'OVER ({units}u) — bottom-tier COMBINED bullpens (both teams), {FG_JOINT_GATE_BOOK} '
            f'full-game total {gate_point} (2025 65.4% hit/+24.9% ROI, 2026 57.4%/+10.1%). The over '
            f'side does not separate into clean conviction tiers, so it is a single tier. '
            f'Tracking only, do not bet.')


NAME_TO_ABB = {
    'Philadelphia Phillies': 'PHI', 'Atlanta Braves': 'ATL',
    'New York Mets': 'NYM',         'Miami Marlins': 'MIA',
    'Washington Nationals': 'WSH',  'Chicago Cubs': 'CHC',
    'Milwaukee Brewers': 'MIL',     'St. Louis Cardinals': 'STL',
    'Cincinnati Reds': 'CIN',       'Pittsburgh Pirates': 'PIT',
    'Los Angeles Dodgers': 'LAD',   'San Francisco Giants': 'SF',
    'San Diego Padres': 'SD',       'Colorado Rockies': 'COL',
    'Arizona Diamondbacks': 'ARI',  'New York Yankees': 'NYY',
    'Boston Red Sox': 'BOS',        'Toronto Blue Jays': 'TOR',
    'Tampa Bay Rays': 'TB',         'Baltimore Orioles': 'BAL',
    'Cleveland Guardians': 'CLE',   'Minnesota Twins': 'MIN',
    'Chicago White Sox': 'CWS',     'Detroit Tigers': 'DET',
    'Kansas City Royals': 'KC',     'Houston Astros': 'HOU',
    'Texas Rangers': 'TEX',         'Seattle Mariners': 'SEA',
    'Los Angeles Angels': 'LAA',    'Oakland Athletics': 'ATH',
    'Athletics': 'ATH',
}


def _odds_american_to_profit(price):
    price = float(price)
    return price / 100.0 if price > 0 else 100.0 / abs(price)


def get_odds_api_lines(games, force=False):
    """
    Returns a tuple (team_lines, joint_lines, fg_lines, fg_joint_lines, kprop_lines):
      team_lines:  team_abb -> {book_key: {'point', 'over', 'under', 'over_profit', 'under_profit'}}
                   (ODDS_F5_MARKET -- per-team F5 total, used by pq_q4/off_q3)
      joint_lines: (home_abb, away_abb) -> {book_key: {'point', 'over', 'under',
                   'over_profit', 'under_profit'}} (ODDS_JOINT_MARKET -- combined F5 total,
                   used by the joint-offense signal, added June 20, 2026)
      fg_lines:    team_abb -> {book_key: {...same shape as team_lines}} (ODDS_FG_MARKET --
                   per-team FULL-GAME total, used by the FG bullpen signal, added June 22, 2026)
      fg_joint_lines: (home_abb, away_abb) -> {book_key: {...same shape as joint_lines}}
                   (ODDS_FG_JOINT_MARKET -- whole-game combined total, used by the FG JOINT
                   bullpen signal, added June 22, 2026)
      kprop_lines: norm_pitcher_name -> {book_label: {'point','over','under','over_profit'}}
                   (ODDS_KPROP_MARKET -- pitcher strikeout props, drives the over-favorite K
                   signal, bundled June 27, 2026 to save a second per-event call)
    All five markets are requested in the same per-event call (see cost note above ODDS_API_KEY).
    Cached together ODDS_API_TTL seconds in Redis (shared across /api/picks and /api/notify)
    since each call costs real API credits. force=True bypasses this cache (see
    /api/picks?refresh=1) -- spends real credits, only use when actually needed, not on every
    normal request. NOTE: the Odds API simply omits a market it can't price for an event rather
    than erroring the whole request, so adding the K-prop market can't break the F5 lines.
    """
    if not ODDS_API_KEY or not games:
        return {}, {}, {}, {}, {}

    import json as _json
    today     = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
    cache_key = f"ozzie:odds_lines:{today}"
    cached    = None if force else redis_get(cache_key)
    if cached:
        try:
            parsed = _json.loads(cached)
            # joint_lines keys are tuples, which JSON can't represent -- stored as "HOME|AWAY"
            # strings and reconstructed here.
            joint = {tuple(k.split('|', 1)): v for k, v in parsed.get('joint', {}).items()}
            fg_joint = {tuple(k.split('|', 1)): v for k, v in parsed.get('fg_joint', {}).items()}
            return parsed.get('team', {}), joint, parsed.get('fg', {}), fg_joint, parsed.get('kprop', {})
        except Exception:
            pass

    try:
        ev_resp = requests.get(
            f"{ODDS_API_BASE}/sports/baseball_mlb/events",
            params={'apiKey': ODDS_API_KEY}, timeout=15)
        print(f"Odds API events fetch: status={ev_resp.status_code}, ok={ev_resp.ok}")
        events = ev_resp.json() if ev_resp.ok else []
    except Exception as e:
        print(f"Odds API events fetch error: {e}")
        return {}, {}, {}, {}, {}

    # games' home_team/away_team come back as full names (e.g. "Detroit Tigers") whenever MLB
    # Stats API's schedule response doesn't include the 'abbreviation' field for a team -- which
    # turned out to be EVERY team in this call context, not occasional. Normalize both sides
    # through NAME_TO_ABB (".get(x, x)" passes already-abbreviated values through unchanged) so
    # this matches regardless of which format either side happens to use. Confirmed June 19,
    # 2026: without this, matched was 0 of 25 events, every single time.
    wanted = {(NAME_TO_ABB.get(g['home_team'], g['home_team']),
               NAME_TO_ABB.get(g['away_team'], g['away_team'])) for g in games}
    matched_events = []
    for ev in events:
        home_abb = NAME_TO_ABB.get(ev.get('home_team', ''))
        away_abb = NAME_TO_ABB.get(ev.get('away_team', ''))
        if (home_abb, away_abb) in wanted:
            matched_events.append((ev['id'], home_abb, away_abb))

    print(f"Odds API debug: {len(events)} events returned, wanted={wanted}, "
          f"matched={len(matched_events)} of {len(events)} "
          f"(unmatched sample: {[(NAME_TO_ABB.get(e.get('home_team','')), NAME_TO_ABB.get(e.get('away_team',''))) for e in events[:5]]})")

    lines = {}
    joint_lines = {}
    fg_lines = {}
    fg_joint_lines = {}
    kprop = {}
    all_bookmaker_keys = set()
    for event_id, home_abb, away_abb in matched_events:
        try:
            r = requests.get(
                f"{ODDS_API_BASE}/sports/baseball_mlb/events/{event_id}/odds",
                params={'apiKey': ODDS_API_KEY, 'regions': ODDS_REGIONS,
                        'markets': f'{ODDS_F5_MARKET},{ODDS_JOINT_MARKET},{ODDS_FG_MARKET},{ODDS_FG_JOINT_MARKET},{ODDS_KPROP_MARKET}',
                        'oddsFormat': 'american'},
                timeout=15)
            if not r.ok:
                continue
            data = r.json()
        except Exception as e:
            print(f"Odds API event {event_id} fetch error: {e}")
            continue

        for bm in data.get('bookmakers', []):
            book_key = bm.get('key', '')
            all_bookmaker_keys.add(book_key)
            if book_key not in TARGET_BOOKS:
                continue
            book_label = TARGET_BOOKS[book_key]
            for market in bm.get('markets', []):
                mkey = market.get('key')
                if mkey == ODDS_F5_MARKET or mkey == ODDS_FG_MARKET:
                    # Same per-team outcome shape for both markets -- F5 and full-game team
                    # totals are parsed identically, just written into separate dicts.
                    target = lines if mkey == ODDS_F5_MARKET else fg_lines
                    by_team = {}
                    for outcome in market.get('outcomes', []):
                        team_abb = NAME_TO_ABB.get(outcome.get('description', ''))
                        if not team_abb:
                            continue
                        by_team.setdefault(team_abb, {})[outcome['name'].lower()] = outcome
                    for team_abb, sides in by_team.items():
                        over, under = sides.get('over'), sides.get('under')
                        if not over or not under:
                            continue
                        target.setdefault(team_abb, {})[book_label] = {
                            'point':         over.get('point'),
                            'over':          int(over['price']),
                            'under':         int(under['price']),
                            'over_profit':   round(_odds_american_to_profit(over['price']), 3),
                            'under_profit':  round(_odds_american_to_profit(under['price']), 3),
                        }
                elif mkey == ODDS_JOINT_MARKET or mkey == ODDS_FG_JOINT_MARKET:
                    # Joint/combined total -- one line for the whole game, no per-team
                    # 'description' on the outcomes (unlike ODDS_F5_MARKET above). The F5 joint
                    # and full-game joint markets are parsed identically, just written into
                    # separate dicts (same idiom as the F5/FG per-team markets above).
                    target_j = joint_lines if mkey == ODDS_JOINT_MARKET else fg_joint_lines
                    sides = {o['name'].lower(): o for o in market.get('outcomes', [])}
                    over, under = sides.get('over'), sides.get('under')
                    if not over or not under:
                        continue
                    target_j.setdefault((home_abb, away_abb), {})[book_label] = {
                        'point':         over.get('point'),
                        'over':          int(over['price']),
                        'under':         int(under['price']),
                        'over_profit':   round(_odds_american_to_profit(over['price']), 3),
                        'under_profit':  round(_odds_american_to_profit(under['price']), 3),
                    }
                elif mkey == ODDS_KPROP_MARKET:
                    # Pitcher strikeout props -- outcomes carry the pitcher in 'description' and
                    # Over/Under in 'name' (player-prop shape). Keyed by normalized pitcher name so
                    # the statcast "Last, First" lookup in _kprop_over_fav can find it.
                    by_p = {}
                    for o in market.get('outcomes', []):
                        nm = _kprop_norm_name(o.get('description', ''))
                        if not nm:
                            continue
                        by_p.setdefault(nm, {})[o.get('name', '').lower()] = o
                    for nm, sides in by_p.items():
                        over, under = sides.get('over'), sides.get('under')
                        if not over or not under:
                            continue
                        kprop.setdefault(nm, {})[book_label] = {
                            'point':       over.get('point'),
                            'over':        int(over['price']),
                            'under':       int(under['price']),
                            'over_profit': round(_odds_american_to_profit(over['price']), 3),
                        }

    print(f"Odds API bookmaker keys seen across all matched events: {sorted(all_bookmaker_keys)}")
    print(f"K-prop lines parsed for {len(kprop)} pitchers (bundled in get_odds_api_lines)")
    try:
        cache_payload = {
            'team':  lines,
            'joint': {f"{h}|{a}": v for (h, a), v in joint_lines.items()},
            'fg':    fg_lines,
            'fg_joint': {f"{h}|{a}": v for (h, a), v in fg_joint_lines.items()},
            'kprop': kprop,
        }
        redis_set(cache_key, _json.dumps(cache_payload), ex=ODDS_API_TTL)
    except Exception as e:
        print(f"Odds lines cache write error: {e}")
    return lines, joint_lines, fg_lines, fg_joint_lines, kprop


def _kprop_norm_name(name):
    """Normalize a pitcher name to 'first last' lowercase for cross-source matching.
    Statcast gives 'Last, First'; the Odds API gives 'First Last'. Strips suffixes/accents
    loosely. Not bulletproof (suffixes, Jr., accents) — matching is best-effort."""
    if not name:
        return ''
    n = name.strip()
    if ',' in n:                       # 'Last, First' -> 'First Last'
        parts = [p.strip() for p in n.split(',', 1)]
        n = f"{parts[1]} {parts[0]}" if len(parts) == 2 else parts[0]
    # Strip accents so Statcast's 'Sánchez, Cristopher' matches the odds feed's plain
    # 'Cristopher Sanchez'. The docstring always promised this but the code never did it,
    # silently dropping ~10 accented-name pitchers (Sánchez, Rodón, Luzardo, Eury Pérez,
    # Germán Márquez, ...) from the K-prop join for all of 2026. Safe both ways: a no-op
    # when neither side has accents; never breaks an already-matching pair.
    n = ''.join(c for c in unicodedata.normalize('NFKD', n) if not unicodedata.combining(c))
    return ' '.join(n.lower().replace('.', '').split())


def _kprop_over_fav(pitcher_name, kprop_lines, opp_team, team_lines, prior_starts=None):
    """The rebuilt K-prop signal: bet the OVER when the book prices it a moderate favorite
    (KPROP_OVER_FAV_LO..HI on DraftKings, where the edge was measured), sharpened when the
    opposing offense's F5 team total < KPROP_SHARP_F5_TOTAL AND the pitcher has >= 7 prior
    starts this season (KPROP_SHARP_MIN_STARTS -- a would-be SHARP play with too few priors is
    demoted to 'base' because its point-in-time rates aren't stable yet).
    Returns dict (always) with 'signal' bool + display fields, or signal=False if no qualifying
    over price. Classifies on DK; reports the BEST over price across books for the actual bet."""
    out = {'signal': False, 'dk_over': None, 'best_over': None, 'best_book': None,
           'line': None, 'opp_f5_total': None, 'tier': None, 'n_books': 0,
           'prior_starts': prior_starts, 'sharp_blocked_thin': False}
    books = kprop_lines.get(_kprop_norm_name(pitcher_name)) or {}
    if not books:
        return out
    dk = books.get('DraftKings')
    dk_over  = dk['over'] if dk else None
    dk_line  = dk.get('point') if dk else None
    # Shop the best over price ONLY among books posting the SAME line as DK. Comparing across
    # different strikes is meaningless — e.g. over 4.5 at -157 vs another book's over 6.5 at +240
    # are different bets, and naively taking max(price) recommended the +240 (a totally different
    # prop). Fall back to DK if no other book matches the line.
    same_line = {b: v for b, v in books.items()
                 if dk_line is not None and v.get('point') == dk_line and v.get('over') is not None}
    if same_line:
        best_book = max(same_line, key=lambda b: same_line[b]['over'])
        best_over = same_line[best_book]['over']
    else:
        best_book, best_over = ('DraftKings', dk_over) if dk_over is not None else (None, None)
    out.update({'dk_over': dk_over, 'best_over': best_over, 'best_book': best_book,
                'line': dk_line, 'n_books': len(same_line)})
    # opposing offense's F5 team total (median across books). team_lines is keyed by ABBREVIATION
    # ('BOS'); opp_team arrives as a full name ('Boston Red Sox') -> convert first, else it never
    # resolves and the SHARP tier (opp F5 < 2.0) can never fire.
    opp_abb = NAME_TO_ABB.get(opp_team, opp_team)
    opp_pts = [v.get('point') for v in (team_lines.get(opp_abb) or {}).values()
               if v.get('point') is not None]
    opp_f5 = round(sorted(opp_pts)[len(opp_pts) // 2], 1) if opp_pts else None
    out['opp_f5_total'] = opp_f5
    # classify on DK over price (the measured edge)
    if dk_over is not None and KPROP_OVER_FAV_LO <= dk_over <= KPROP_OVER_FAV_HI:
        out['signal'] = True
        f5_sharp = opp_f5 is not None and opp_f5 < KPROP_SHARP_F5_TOTAL
        # prior-start gate: only demote when we positively know the count is too low. Unknown
        # (prior_starts is None, e.g. missing current-season fetch) keeps the F5-based tier.
        thin = prior_starts is not None and prior_starts < KPROP_SHARP_MIN_STARTS
        out['sharp_blocked_thin'] = bool(f5_sharp and thin)
        out['tier'] = 'sharp' if (f5_sharp and not thin) else 'base'
    return out


# ── KALSHI (CFTC-regulated event exchange — a line-shopping venue bettable in many US states) ──
# Public market-data API, no auth. Pulls MLB full-game JOINT (KXMLBTOTAL) and per-team
# (KXMLBTEAMTOTAL) totals so Kalshi shows alongside the sportsbook lines on the FG joint + team-total
# signals. Each event = one game (ticker encodes date+teams), each market = one strike (floor_strike
# = line), YES = OVER / NO = UNDER. Prices are $-denominated strings (the *_dollars fields). Bet UNDER
# = buy NO at no_ask -> decimal 1/no_ask; OVER = buy YES at yes_ask. Fully fail-safe: any error returns
# empty and the existing book lines are unaffected. Redis-cached KALSHI_TTL.
KALSHI_BASE         = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_TTL          = 14400   # 4h, same cadence as the odds cache
KALSHI_JOINT_SERIES = "KXMLBTOTAL"
KALSHI_TEAM_SERIES  = "KXMLBTEAMTOTAL"
KALSHI_MON = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
KALSHI_ABB = {'AZ': 'ARI'}    # Kalshi uses AZ; we use ARI (NAME_TO_ABB). Everything else matches.
KALSHI_KS_SERIES = "KXMLBKS"  # pitcher strikeout prop ladder
KALSHI_KS_TTL    = 1200        # 20min — K prop prices move significantly pre-game


def _kalshi_get(path, **params):
    """GET Kalshi market-data with throttle + 429 backoff. Returns {} on any non-OK (fail-safe)."""
    for attempt in range(5):
        time.sleep(0.12)
        try:
            r = requests.get(f"{KALSHI_BASE}{path}", params=params, timeout=15)
        except Exception:
            return {}
        if r.status_code == 429:
            time.sleep(0.5 * (2 ** attempt))
            continue
        if not r.ok:
            return {}
        return r.json()
    return {}


def _kalshi_dec_to_american(d):
    if not d or d <= 1:
        return None
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def _kalshi_market_row(m):
    """One Kalshi market -> the same {point,over,under,*_profit} shape as the sportsbook dicts."""
    def f(v):
        try:
            return float(v) if v not in (None, '') else None
        except (TypeError, ValueError):
            return None
    ya, na = f(m.get('yes_ask_dollars')), f(m.get('no_ask_dollars'))
    line   = f(m.get('floor_strike'))
    if line is None:
        return None
    over_dec  = (1.0 / ya) if ya and ya > 0 else None
    under_dec = (1.0 / na) if na and na > 0 else None
    return {
        'point':        line,
        'over':         _kalshi_dec_to_american(over_dec),
        'under':        _kalshi_dec_to_american(under_dec),
        'over_profit':  round(over_dec - 1, 3) if over_dec else None,
        'under_profit': round(under_dec - 1, 3) if under_dec else None,
        'volume':       f(m.get('volume_fp')) or 0,
    }


def _kalshi_event_meta(ev):
    """(game_date 'YYYY-MM-DD', abb_a, abb_b) from a Kalshi event, or None. Date is the game's ET
    date (matches our convention); teams from sub_title ('ATH vs SF (Jun 23)'), normalized to ARI."""
    try:
        tail = ev['event_ticker'].split('-', 1)[1]
        gdate = f"20{int(tail[0:2]):02d}-{KALSHI_MON[tail[2:5]]:02d}-{int(tail[5:7]):02d}"
    except Exception:
        return None
    head = (ev.get('sub_title') or '').split('(')[0].strip()
    if ' vs ' not in head:
        return None
    a, b = [KALSHI_ABB.get(t.strip(), t.strip()) for t in head.split(' vs ', 1)]
    return gdate, a, b


def _kalshi_event_markets(event_ticker):
    out, cursor = [], None
    while True:
        mj = _kalshi_get('/markets', event_ticker=event_ticker, limit=200,
                         **({'cursor': cursor} if cursor else {}))
        out.extend(mj.get('markets', []))
        cursor = mj.get('cursor')
        if not cursor:
            break
    return out


def _kalshi_series_ladders(series, today_et, joint):
    """{pairkey: [market_row,...]} for joint; {f'{pairkey}@@{team}': [rows]} for team totals.
    pairkey = '|'.join(sorted([abb_a, abb_b])) in our convention."""
    out, cursor, events = {}, None, []
    while True:
        js = _kalshi_get('/events', series_ticker=series, status='open', limit=200,
                         **({'cursor': cursor} if cursor else {}))
        events.extend(js.get('events', []))
        cursor = js.get('cursor')
        if not cursor:
            break
    for ev in events:
        meta = _kalshi_event_meta(ev)
        if not meta or meta[0] != today_et:
            continue
        _, a, b = meta
        pk = '|'.join(sorted([a, b]))
        for m in _kalshi_event_markets(ev['event_ticker']):
            row = _kalshi_market_row(m)
            if not row:
                continue
            if joint:
                out.setdefault(pk, []).append(row)
            else:
                team = ''.join(ch for ch in m.get('ticker', '').rsplit('-', 1)[-1] if ch.isalpha())
                team = KALSHI_ABB.get(team, team)
                out.setdefault(f"{pk}@@{team}", []).append(row)
    return out


def get_kalshi_lines(games, force=False):
    """(joint, team) Kalshi ladders for today's slate. Redis-cached; ({}, {}) on any error."""
    if not games:
        return {}, {}
    import json as _json
    today_et  = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
    cache_key = f"ozzie:kalshi:{today_et}"
    if not force:
        cached = redis_get(cache_key)
        if cached:
            try:
                p = _json.loads(cached)
                return p.get('joint', {}), p.get('team', {})
            except Exception:
                pass
    try:
        joint = _kalshi_series_ladders(KALSHI_JOINT_SERIES, today_et, joint=True)
        team  = _kalshi_series_ladders(KALSHI_TEAM_SERIES,  today_et, joint=False)
    except Exception as e:
        print(f"Kalshi fetch error: {e}")
        return {}, {}
    try:
        redis_set(cache_key, _json.dumps({'joint': joint, 'team': team}), ex=KALSHI_TTL)
    except Exception as e:
        print(f"Kalshi cache write error: {e}")
    print(f"Kalshi: {len(joint)} joint game(s), {len(team)} team-total entries")
    return joint, team


def get_kalshi_k_props(games, force=False):
    """Pitcher strikeout props from Kalshi KXMLBKS series.
    Returns {pair_key: {pitcher_name_lower: {implied_line, bet_threshold, yes_bid, yes_ask}}}.
    pair_key = '|'.join(sorted([home_abb, away_abb])).
    pitcher_name_lower = 'first last' lowercase (e.g. 'roki sasaki').
    implied_line = floor_strike (Kalshi decimal) where yes_bid_dollars crosses 0.50.
    yes_bid is in cents (0-100).  ({}, {}) on any error."""
    if not games:
        return {}
    import json as _json
    today_et  = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
    cache_key = f"ozzie:kalshi_ks:{today_et}"
    if not force:
        cached = redis_get(cache_key)
        if cached:
            try:
                return _json.loads(cached)
            except Exception:
                pass
    out = {}
    try:
        all_events, cursor = [], None
        while True:
            js = _kalshi_get('/events', series_ticker=KALSHI_KS_SERIES, status='open', limit=200,
                             **({'cursor': cursor} if cursor else {}))
            all_events.extend(js.get('events', []))
            cursor = js.get('cursor')
            if not cursor:
                break
        for ev in all_events:
            meta = _kalshi_event_meta(ev)
            if not meta or meta[0] != today_et:
                continue
            _, a, b = meta
            pk = '|'.join(sorted([a, b]))
            markets = _kalshi_event_markets(ev['event_ticker'])
            pitcher_ladders = {}  # {name_lower: [(floor_strike, yes_bid_dollars, yes_ask_dollars)]}
            for m in markets:
                title = m.get('yes_sub_title') or m.get('title', '')
                if ':' not in title:
                    continue
                name_lower = title.split(':')[0].strip().lower()
                # Extract integer threshold from "Name: 7+" or "Name: 7+ strikeouts?"
                try:
                    threshold_str = title.split(':')[1].strip().split()[0].rstrip('+')
                    threshold = int(threshold_str)
                except (IndexError, ValueError):
                    continue
                try:
                    floor = float(m.get('floor_strike') or 0)
                except (TypeError, ValueError):
                    continue
                if floor <= 0:
                    continue
                def _fv(v):
                    try: return float(v) if v not in (None, '') else None
                    except: return None
                pitcher_ladders.setdefault(name_lower, []).append(
                    (floor, threshold, _fv(m.get('yes_ask_dollars'))))
            for name_lower, ladder in pitcher_ladders.items():
                ladder.sort(key=lambda x: x[0])  # sort by floor_strike ascending
                # Implied line: highest floor_strike where yes_ask >= 0.50
                implied_line = None
                implied_thr   = None
                implied_ya    = None
                for fl, thr, ya in ladder:
                    if ya is not None and ya >= 0.50:
                        implied_line = fl
                        implied_thr  = thr
                        implied_ya   = ya
                if implied_line is None and ladder:
                    implied_line = ladder[0][0] - 1.0
                    implied_thr  = ladder[0][1]
                    implied_ya   = ladder[0][2]
                if implied_line is None:
                    continue
                # Bet IS the implied-line threshold: proj_gap > 0.5 means we think
                # there's >50% chance of hitting the implied-line strike, so buy it.
                out.setdefault(pk, {})[name_lower] = {
                    'implied_line':  implied_line,
                    'bet_threshold': implied_thr,
                    'yes_bid':       round(implied_ya * 100) if implied_ya else None,
                }
    except Exception as e:
        print(f"Kalshi K props fetch error: {e}")
    try:
        redis_set(cache_key, _json.dumps(out), ex=KALSHI_KS_TTL)
    except Exception as e:
        print(f"Kalshi K props cache write error: {e}")
    print(f"Kalshi K props: {sum(len(v) for v in out.values())} pitcher props across {len(out)} games")
    return out


def _lookup_kalshi_k(kalshi_k_props, pair_key, pitcher_name):
    """Match a pitcher (MLB 'First Last' format) to their Kalshi K prop entry.
    Falls back to last-name match for accented / hyphenated names.
    Returns the prop dict or None if not found."""
    game_props = kalshi_k_props.get(pair_key, {})
    if not game_props or not pitcher_name or pitcher_name == 'TBD':
        return None
    name_lower = pitcher_name.strip().lower()
    if name_lower in game_props:
        return game_props[name_lower]
    last = name_lower.split()[-1]
    for key, val in game_props.items():
        if key.split()[-1] == last:
            return val
    return None


def _pick_kalshi(ladder, target_pt):
    """From a Kalshi market ladder, pick the strike matching target_pt (else nearest; else the
    most-liquid) so it's an apples-to-apples line vs. the signal's gate line."""
    if not ladder:
        return None
    if target_pt is not None:
        exact = [r for r in ladder if r.get('point') == target_pt]
        if exact:
            return exact[0]
        cand = [r for r in ladder if r.get('point') is not None]
        if cand:
            return min(cand, key=lambda r: abs(r['point'] - target_pt))
    return max(ladder, key=lambda r: r.get('volume') or 0)


def pa_rate_to_game_odds(pa_rate_pct):
    pa_rate   = pa_rate_pct / 100
    game_prob = 1 - (1 - pa_rate) ** ASSUMED_PAS
    if game_prob <= 0:
        return '+9999'
    fair = round((1 / game_prob - 1) * 100)
    return f'+{fair}' if fair > 0 else str(fair)


def format_odds(n):
    return f'+{n}' if n > 0 else str(n)


def get_bullpen_score(team, arch_key, model):
    bs = model.get('team_bullpen_scores', {})
    if team in bs and arch_key in bs[team]:
        return bs[team][arch_key].get('shrunk_rate', LEAGUE_AVG)
    return LEAGUE_AVG


def get_park_factor(fielding_team, batter_hand, model):
    all_parks = model.get('all_parks', {})
    if fielding_team not in all_parks:
        return 1.0
    try:
        dims   = all_parks[fielding_team].get('dimensions', {})
        valid  = [p for p in all_parks.values() if 'dimensions' in p]
        avg_lf = sum(p['dimensions'].get('left_field',  330) for p in valid) / len(valid)
        avg_cf = sum(p['dimensions'].get('center',      400) for p in valid) / len(valid)
        avg_rf = sum(p['dimensions'].get('right_field', 330) for p in valid) / len(valid)
        lf = dims.get('left_field',  avg_lf)
        cf = dims.get('center',      avg_cf)
        rf = dims.get('right_field', avg_rf)
        if batter_hand == 'L':
            pull, oppo, avg_pull, avg_oppo = rf, lf, avg_rf, avg_lf
        else:
            pull, oppo, avg_pull, avg_oppo = lf, rf, avg_lf, avg_rf
        pf = 1.0 - 0.003 * (
            (pull - avg_pull) / avg_pull +
            (oppo - avg_oppo) / avg_oppo +
            (cf   - avg_cf)   / avg_cf
        )
        return round(pf, 4)
    except Exception:
        return 1.0


def get_expected_f5_runs(overlap_k_mod, avg_ip=None, model_bundle=None):
    if model_bundle is None:
        return None
    try:
        mb     = model_bundle
        coeffs = mb['coefficients']
        sp     = mb['scaler_params']
        feats  = [f for f in mb['feature_raw_names'][1:] if f in sp]
        defaults = {
            'overlap_k_mod':    overlap_k_mod,
            'lineup_walk_rate': sp['lineup_walk_rate']['mean'] if 'lineup_walk_rate' in sp else 0.09,
            'bp_overlap_prob':  sp['bp_overlap_prob']['mean']  if 'bp_overlap_prob'  in sp else 1.00,
            'avg_ip':           avg_ip if avg_ip is not None else mb.get('league_avg_ip', 5.43),
        }
        z_vals = [(defaults[f] - sp[f]['mean']) / sp[f]['std'] for f in feats]
        X      = [1.0] + z_vals
        log_mu = sum(c * x for c, x in zip(coeffs, X))
        raw    = math.exp(log_mu)
        return round(raw + mb.get('bias_correction', 0.0), 3)
    except Exception:
        return None


def poisson_cdf(k, lam):
    if lam <= 0:
        return 1.0
    total = 0.0
    term  = math.exp(-lam)
    for i in range(k + 1):
        total += term
        if i < k:
            term *= lam / (i + 1)
    return min(total, 1.0)


def prob_to_american(p):
    if p is None or p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        odds = round(-100 * p / (1 - p))
    else:
        odds = round(100 * (1 - p) / p)
    return f'+{odds}' if odds > 0 else str(odds)


def get_f5_fair_odds(expected_runs):
    if expected_runs is None:
        return {'fair_under_2_5': None, 'fair_over_2_5': None}
    lam = expected_runs
    p_under_2_5 = poisson_cdf(2, lam)
    return {
        'fair_under_2_5': prob_to_american(p_under_2_5),
        'fair_over_2_5':  prob_to_american(1 - p_under_2_5),
    }


def get_ozzie_score(std_from_mean, is_away, k_rate, park, bb_gb_rates):
    """
    Under signal scoring system v3.
    Gate: std_from_mean <= -1.0 AND (away + K>=26%) >= 2
    Score: park (0/1/3) + K>=28% (1) + away (1) + sigma<=-2.0 (1) = max 6
    Tiers: Gold >=6, Silver 4-5, Bronze 3
    BB rate removed (p=0.810, no signal). U1.5 removed (no market edge on real lines).
    Validated 2025 Pinnacle U2.5: all flagged ROI +0.552, Silver +0.551, Gold +0.664
    """
    q_away     = 1 if is_away else 0
    q_krate_lo = 1 if (k_rate is not None and k_rate >= 0.26) else 0
    q_krate_hi = 1 if (k_rate is not None and k_rate >= 0.28) else 0

    if std_from_mean > -1.0:
        return None
    if (q_away + q_krate_lo + q_krate_hi) < 2:
        return None

    if park in PARK_GOOD:
        park_pts, park_tier = 3, 'good'
    elif park in PARK_BAD:
        park_pts, park_tier = 0, 'bad'
    else:
        park_pts, park_tier = 1, 'neutral'

    # BB rate: tracked for logging only — NOT included in score (p=0.810, no signal)
    bb_rate  = bb_gb_rates.get('bb_rate') if bb_gb_rates else None
    q_bb_vlo = 1 if (bb_rate is not None and bb_rate < 0.06) else 0

    # GB rate: tracked for logging only
    gb_rate  = bb_gb_rates.get('gb_rate') if bb_gb_rates else None
    q_gb_mid = 1 if (gb_rate is not None and 0.42 <= gb_rate < 0.48) else 0

    q_sigma_20 = 1 if std_from_mean <= -2.0 else 0

    total_score = park_pts + q_krate_hi + q_away + q_sigma_20

    if total_score < 3:
        return None

    if total_score >= 6:
        tier, color = 'Gold', 'gold'
    elif total_score >= 4:
        tier, color = 'Silver', 'silver'
    else:
        tier, color = 'Bronze', 'bronze'

    if tier == 'Gold':
        u25_min, u25_units = '-500', '3u'
    elif tier == 'Silver':
        u25_min, u25_units = '-300', '1.5u'
    else:
        u25_min, u25_units = '-210', '0.5u'

    return {
        'ozzie_score':      total_score,
        'confidence_label': tier,
        'confidence_color': color,
        'park_tier':        park_tier,
        'q_away':           bool(q_away),
        'q_krate_lo':       bool(q_krate_lo),
        'q_krate_hi':       bool(q_krate_hi),
        'q_bb_vlo':         bool(q_bb_vlo),
        'q_gb_mid':         bool(q_gb_mid),
        'q_sigma_20':       bool(q_sigma_20),
        'bb_rate':          round(bb_rate, 4) if bb_rate is not None else None,
        'gb_rate':          round(gb_rate, 4) if gb_rate is not None else None,
        'min_u25':          u25_min,
        'unit_u25':         u25_units,
    }


def compute_batter_overlap(batter_id, pitcher_id, batter_hand, pitcher_hand, model):
    batter_profiles    = model.get('batter_ev_profiles_lr', {})
    pitcher_tendencies = model.get('pitcher_tendency_map', {})
    archetypes         = model.get('archetypes', {})
    if batter_id not in batter_profiles:     return None
    if pitcher_id not in pitcher_tendencies: return None
    if pitcher_hand not in batter_profiles[batter_id]: return None
    tends = pitcher_tendencies[pitcher_id].get(batter_hand, {})
    if not tends: return None
    evs  = batter_profiles[batter_id][pitcher_hand]
    wsum = wtot = 0.0
    for ak in archetypes:
        if (ak.endswith('_L')) != (batter_hand == 'L'): continue
        t     = tends.get(ak, 0.0)
        e     = evs.get(ak, 1.0)
        wsum += e * t
        wtot += t
    return wsum / wtot if wtot > 0 else None


def apply_k_modifier(overlap_raw, pitcher_k_rate):
    if pitcher_k_rate is None:
        return overlap_raw
    return overlap_raw * (1 - pitcher_k_rate) / (1 - LEAGUE_AVG_KRATE)


# ── TRACKING-ONLY MODE (June 2026 incident) ──────────────────────────────────────────
# Google Drive started blocking gdown's public-link access to the ~12 sigma/heatmap model
# files ("Cannot retrieve the public link... may have had many accesses") -- every worker
# restart re-attempted all 12 downloads, which kept hammering Drive and worsening the block,
# crash-looping the whole app (worker timeout / OOM -> SIGKILL -> cold restart -> repeat).
# Per Zach: pause sigma/heatmap, FG Under, HR picks, and DFS picks entirely (all depend on
# load_model() / Google Drive) and surface ONLY the two F5 TT U1.5 tracking signals that
# don't touch Drive at all -- Pitcher Quality Composite (pq_q4) and Offense Quality
# Composite (off_q3_gate), both sourced from local CSV priors + live pybaseball pulls.
# To restore full functionality once Drive access is fixed: swap get_tracking_only_flags(games)
# back to model = load_model(); get_heatmap_flags(games, model) in api_picks/api_notify, and
# restore get_hr_picks/get_dfs_picks.

# DISABLED (June 22, 2026, see CLAUDE.md "PER-GAME BULLPEN_Z IS OUTCOME-CONTAMINATED"): both FG
# bullpen-totals signals were proven to be outcome leakage. BUG FOUND June 23, 2026 -- this filter
# was originally only applied inside get_heatmap_flags(), which has been paused (Drive incident,
# see comment block above) since before the disable was even written, so it never actually reached
# the live request path. get_tracking_only_flags() is the function the live dashboard/Telegram
# code actually calls (see api_picks/api_notify) -- the filter must live here to take effect.
DISABLED_SIGNALS = {'fg_joint_total', 'fg_tt_under'}


def get_tracking_only_flags(games, force=False):
    flags = []
    pinnacle_suppressed_wrong   = 0  # Pinnacle posted a line, but it wasn't 1.5 or a park exception
    pinnacle_suppressed_missing = 0  # gate active but no Pinnacle line found for this team/game
    pinnacle_park_exceptions    = 0  # passed via the park gate exception, not the base 1.5 line
    joint_suppressed            = 0  # weak combined offense, but FanDuel joint line wasn't 3.5
    joint_signals                = 0  # joint-offense flags actually surfaced
    fg_suppressed                = 0  # opposing bullpen rates top-quartile, but FG line wasn't 3.5/4.5/5.5
    fg_signals                   = 0  # FG TT under flags actually surfaced
    f5_over_signals               = 0  # derived F5 TT over flags actually surfaced
    off_fade_signals              = 0  # offense-overpriced FG joint UNDER (clean lead) flags surfaced
    off_fade_suppressed           = 0  # strong offense but DK joint line missing/below range
    fg_joint_signals              = 0  # FG joint (combined game total) bullpen flags surfaced
    fg_joint_suppressed           = 0  # combined bullpen qualified, but DK game total wasn't in range
    fg_no_bp_info                 = 0  # fielding team name didn't resolve to a bullpen population entry
    fg_pctile_checked            = []  # (fielding_team, fielding_abb, pctile_or_None) -- every matchup
    # checked today, regardless of gate result -- diagnostic so "0 flags shown" in the summary print
    # below is distinguishable between "real, no team cleared the bar today" (expected on small
    # slates -- ~75th-percentile-of-30-teams means only ~7-8 teams qualify on ANY given day) vs "the
    # team-name lookup is silently broken." Added June 22, 2026 after the first live cycle showed
    # 0/0 with no way to tell which case it was from the logs alone.

    try:
        team_lines, joint_lines, fg_lines, fg_joint_lines, kprop_lines = get_odds_api_lines(games, force=force)
    except Exception as e:
        print(f"Odds API lookup error: {e}")
        team_lines, joint_lines, fg_lines, fg_joint_lines, kprop_lines = {}, {}, {}, {}, {}

    # Kalshi line-shopping source (separate exchange API) — attached to the FG joint + team-total
    # flags alongside the sportsbook lines. Fully fail-safe: empty on any error, flags unaffected.
    try:
        kalshi_joint, kalshi_team = get_kalshi_lines(games, force=force)
    except Exception as e:
        print(f"Kalshi lookup error: {e}")
        kalshi_joint, kalshi_team = {}, {}

    # Kalshi pitcher K prop ladder — kept as a REFERENCE price only (no longer drives the signal;
    # the rebuilt over-favorite signal uses sportsbook prices from kprop_lines, bundled into the
    # get_odds_api_lines call above).
    try:
        kalshi_k_props = get_kalshi_k_props(games, force=force)
    except Exception as e:
        print(f"Kalshi K props lookup error: {e}")
        kalshi_k_props = {}

    # See "SAFETY FALLBACK" note above get_odds_api_lines: only gate on Pinnacle if a live odds
    # feed is actually configured. Without this, a missing/unset ODDS_API_KEY would silently
    # suppress every tracking flag instead of just not gating them -- same failure shape as the
    # June 19 Drive/BRef incidents, worth avoiding deliberately rather than by luck.
    pinnacle_gate_active = bool(ODDS_API_KEY)

    try:
        pq_population = get_pitcher_quality_population(force=force)
    except Exception as e:
        print(f"Pitcher quality population error: {e}")
        pq_population = {}

    try:
        off_population = get_offense_quality_population(force=force)
    except Exception as e:
        print(f"Offense quality population error: {e}")
        off_population = {}

    try:
        bp_population = get_bullpen_quality_population(force=force)
    except Exception as e:
        print(f"Bullpen quality population error: {e}")
        bp_population = {}

    # Tracks, per game, which team (by full name) actually got a confirmed F5 TT under flag
    # (pq_q4, Pinnacle-gate-passed) -- used below to derive the F5 TT OVER signal: when a game
    # is flagged for joint F5 over but exactly ONE team is flagged for F5 under, the OTHER team
    # is the over candidate (Zach's hypothesis, June 22, 2026 -- e.g. BOS@SEA: BOS flagged F5
    # under, joint flagged over -> SEA alone should be an F5 TT over play, not a joint play).
    pq_under_fired_team = {}

    # K-prop diagnostic (printed in the summary below) so a log paste confirms the signal at a
    # glance: how many starters we checked, how many matched a sportsbook prop line (name match),
    # how many fired the over-favorite signal, and any unmatched names to investigate.
    _kprop_diag = {'checked': 0, 'matched': 0, 'fired': 0, 'unmatched': []}

    for game in games:
        home_abb = NAME_TO_ABB.get(game['home_team'], game['home_team'])
        matchups = [
            (game['away_lineup'], game['home_pitcher_id'],
             game['home_pitcher_name'], game['away_team'],
             game['home_team'], f"{game['away_team']}@{game['home_team']}"),
            (game['home_lineup'], game['away_pitcher_id'],
             game['away_pitcher_name'], game['home_team'],
             game['away_team'], f"{game['away_team']}@{game['home_team']}"),
        ]
        for lineup, pitcher_id, pitcher_name, batting_team, fielding_team, game_str in matchups:
            # EARLY-FIRE (June 29, 2026): only the probable pitcher is required, NOT a posted lineup.
            # The K-prop signal needs just the starter + market lines (DK K prop, opp F5 total) +
            # prior starts -- all available in the morning, hours before lineups post. The lineup-
            # dependent signals (pq_q4, off_q3, joint) stay effectively lineup-gated below (pq_q4 and
            # off_info both require `lineup`), so they fire post-lineup exactly as before -- this
            # change ONLY lets K plays fire earlier. See project_kprop_timing_edge (open F5 is the
            # best gate, so firing early on the morning line is safe / arguably better).
            if not pitcher_id:
                continue

            pq_info = pq_population.get(pitcher_id) if pitcher_id else None
            # gated on `lineup` so PQ keeps firing only post-lineup (unchanged behavior); pre-lineup
            # runs surface K-only plays, where pq_q4 is False and the flag carries pitcher data only.
            pq_q4   = bool(pq_info and pq_info['quartile'] == 'Q4' and lineup)

            off_info = get_lineup_offense_quality(lineup, off_population) if (off_population and lineup) else None
            off_q3_gate = bool(
                off_info and
                OFF_Q3_LOW_PCTILE <= off_info['off_pctile'] <= OFF_Q3_HIGH_PCTILE and
                off_info['o_bb_rate'] < OFF_BB_RATE_MEDIAN
            )

            # ── FG (FULL-GAME) BULLPEN COMPOSITE ── independent of pq_q4/off_q3 above: different
            # market (full-game team total, not F5), different mechanism (the OPPOSING TEAM's
            # bullpen-as-a-whole, not this team's opposing starter or this team's own offense).
            # Computed here since it shares the same batting_team/fielding_team framing as the
            # matchup loop, but gated and appended independently -- see FG_BP_PCTILE_GATE notes.
            fielding_abb = NAME_TO_ABB.get(fielding_team, fielding_team)
            bp_info      = bp_population.get(fielding_abb)
            fg_bp_pctile = bp_info['percentile'] if bp_info else None
            fg_bp_gate   = bool(bp_info and fg_bp_pctile >= FG_BP_PCTILE_GATE)
            fg_pctile_checked.append((fielding_team, fielding_abb, fg_bp_pctile))
            if bp_info is None:
                fg_no_bp_info += 1
            if fg_bp_gate:
                fg_team_abb    = NAME_TO_ABB.get(batting_team, batting_team)
                fg_team_odds   = dict(fg_lines.get(fg_team_abb, {}))
                fg_pinnacle_pt = fg_team_odds.get(PINNACLE_BOOK_LABEL, {}).get('point')
                fg_line_match  = fg_pinnacle_pt in FG_TT_VALID_LINES
                # attach Kalshi at the gate line (this team's full-game total)
                _pk = '|'.join(sorted([home_abb, NAME_TO_ABB.get(game.get('away_team',''), game.get('away_team',''))]))
                _kal = _pick_kalshi(kalshi_team.get(f"{_pk}@@{fg_team_abb}", []), fg_pinnacle_pt)
                if _kal:
                    fg_team_odds['Kalshi'] = _kal
                if not pinnacle_gate_active or fg_line_match:
                    fg_gate_reason = None if not pinnacle_gate_active else 'line_match'
                    fg_signals += 1
                    flags.append({
                        'game':              game_str,
                        'game_id':           game.get('game_id'),
                        'batting_team':      batting_team,
                        'fielding_team':     fielding_team,
                        'pitcher_name':      pitcher_name,
                        'signal':            'fg_tt_under',
                        'game_time':         game.get('game_time'),
                        'fg_bp_score':       bp_info['score'],
                        'fg_bp_pctile':      fg_bp_pctile,
                        'fg_bp_n_relievers': bp_info['n_relievers'],
                        'fg_odds_lines':     fg_team_odds,
                        'fg_pinnacle_point': fg_pinnacle_pt,
                        'fg_gate_reason':    fg_gate_reason,
                        'fg_note':           _fg_tt_note(True, fg_gate_reason, fg_pinnacle_pt),
                    })
                else:
                    fg_suppressed += 1

            # K-composite kept for historical tracking only; no longer drives the signal.
            _kc = k_composite_score(
                pq_info['k_rate']  if pq_info else None,
                pq_info['bb_rate'] if pq_info else None,
                pq_info['hr_rate'] if pq_info else None,
                off_info.get('o_k_rate') if off_info else None,
            )

            # K-prop signal REBUILT June 27, 2026 — over-favorite price bias (see _kprop_over_fav).
            # The old proj_gap>0.5 vs Kalshi signal was leaked (its "59.5% OOS" used full-season K
            # rates to predict in-season games); leak-free it's no edge. Now: bet the OVER when the
            # book prices it a moderate favorite (DK -120..-160), sharper when opp F5 total < 2.0.
            # _proj_k_val / _kal_k kept only for the reference columns the tracker still logs.
            _proj_k_val = project_starter_ks(pq_info['k_rate'] if pq_info else None,
                                             off_info['o_k_rate'] if off_info else None,
                                             pq_info.get('exp_bf') if pq_info else None)
            _away_abb_ks = NAME_TO_ABB.get(game.get('away_team', ''), game.get('away_team', ''))
            _pk_ks       = '|'.join(sorted([home_abb, _away_abb_ks]))
            _kal_k       = _lookup_kalshi_k(kalshi_k_props, _pk_ks, pitcher_name)
            _ofav        = _kprop_over_fav(pitcher_name, kprop_lines, batting_team, team_lines,
                                           prior_starts=pq_info.get('gs') if pq_info else None)
            _k_prop_signal = _ofav['signal']
            _kprop_diag['checked'] += 1
            if _kprop_norm_name(pitcher_name) in kprop_lines:
                _kprop_diag['matched'] += 1
            elif kprop_lines:
                _kprop_diag['unmatched'].append(pitcher_name)
            if _k_prop_signal:
                _kprop_diag['fired'] += 1

            if not pq_q4 and not off_q3_gate and not _k_prop_signal:
                continue

            # ── PINNACLE GATE ── surface this candidate if Pinnacle's posted F5 line for this
            # team is exactly 1.5 (see PINNACLE GATE notes above get_odds_api_lines), OR if the
            # line is 2.5+ AND the game is in a hitter-favorable park (PARK_GATE_TEAMS, see notes
            # above that constant) -- the validated park-inflated-line exception.
            # K-prop-only flags bypass this gate: they're betting Ks on Kalshi, not the F5 total.
            team_abb       = NAME_TO_ABB.get(batting_team, batting_team)
            team_odds      = team_lines.get(team_abb, {})
            pinnacle_point = team_odds.get(PINNACLE_BOOK_LABEL, {}).get('point')
            park_exception = (pinnacle_point is not None and pinnacle_point >= PARK_GATE_MIN_POINT
                               and home_abb in PARK_GATE_TEAMS)
            if (pq_q4 or off_q3_gate) and pinnacle_gate_active and pinnacle_point != 1.5 and not park_exception:
                if pinnacle_point is None:
                    pinnacle_suppressed_missing += 1
                else:
                    pinnacle_suppressed_wrong += 1
                continue
            if park_exception and pinnacle_point != 1.5:
                pinnacle_park_exceptions += 1

            gate_reason = (None if not pinnacle_gate_active else
                            'park_exception' if (park_exception and pinnacle_point != 1.5) else
                            'line_1_5')

            flag = {
                'game':             game_str,
                'game_id':          game.get('game_id'),
                'batting_team':     batting_team,
                'fielding_team':    fielding_team,
                'pitcher_name':     pitcher_name,
                'pitcher_id':       pitcher_id,
                'signal':           'pitcher_quality_only' if pq_q4 else ('k_prop_only' if (_k_prop_signal and not off_q3_gate) else 'offense_quality_only'),
                'game_time':        game.get('game_time'),
                'pq_score':         pq_info['score']      if pq_info else None,
                'pq_percentile':    pq_info['percentile'] if pq_info else None,
                'pq_quartile':      pq_info['quartile']   if pq_info else None,
                'pq_k_rate':        pq_info['k_rate']      if pq_info else None,
                'pq_bb_rate':       pq_info['bb_rate']     if pq_info else None,
                'pq_hr_rate':       pq_info['hr_rate']     if pq_info else None,
                'projected_k':      _proj_k_val,
                'pq_q4':            pq_q4,
                # K-prop signal REBUILT June 27, 2026 — over-favorite price bias (over-fav fields
                # below drive it). k_comp_score / kalshi_* kept as reference only.
                'k_comp_score':     _kc,
                'k_prop_flag':      _k_prop_signal,
                'k_prop_tier':      _ofav['tier'],
                'kprop_dk_over':    _ofav['dk_over'],
                'kprop_best_over':  _ofav['best_over'],
                'kprop_best_book':  _ofav['best_book'],
                'kprop_line':       _ofav['line'],
                'kprop_opp_f5':     _ofav['opp_f5_total'],
                'kprop_n_books':    _ofav['n_books'],
                'kprop_prior_starts':      _ofav['prior_starts'],
                'kprop_sharp_blocked_thin': _ofav['sharp_blocked_thin'],
                'kalshi_k_line':    _kal_k['implied_line'] if _kal_k else None,
                'kalshi_k_strike':  _kal_k.get('bet_threshold') if _kal_k else None,
                'kalshi_k_yes_bid': _kal_k.get('yes_bid') if _kal_k else None,
                'k_prop_note':      _kprop_note(_ofav, _kal_k),
                'pq_current_bf':    pq_info['current_bf'] if pq_info else None,
                'pq_note':          _pq_note(pq_q4, gate_reason, pinnacle_point),
                'o_k_rate':         off_info.get('o_k_rate') if off_info else None,
                'off_pctile':       off_info['off_pctile'] if off_info else None,
                'off_bb_rate':      off_info['o_bb_rate']  if off_info else None,
                'off_q3_gate':      off_q3_gate,
                # Same diagnostic, lineup-weighted: how much real current-season PA backs this
                # team's offense score vs. falling back toward the career prior.
                'off_current_pa_avg': off_info['current_pa_avg'] if off_info else None,
                'off_note':         _off_note(off_q3_gate, gate_reason, pinnacle_point),
                # odds_lines is keyed by abbreviation (team_abb), batting_team is the full name --
                # normalize here too, same mismatch and same fix as get_odds_api_lines' matching.
                'odds_lines':       team_odds,
                'odds_has_1_5':     any(b.get('point') == 1.5 for b in team_odds.values()),
                'pinnacle_point':   pinnacle_point,
                # 'line_1_5' (base signal), 'park_exception' (validated elevated-line extension,
                # thinner sample -- see PARK_GATE_TEAMS), or None if the gate itself is inactive.
                'pinnacle_gate_reason': gate_reason,
            }
            flags.append(flag)
            if pq_q4:
                # confirmed, line-gate-passed F5 TT under -- feeds the F5 TT OVER derivation below
                pq_under_fired_team[batting_team] = True

        # ── JOINT OFFENSE TRACKING SIGNAL ── game-level (not per-team), so computed once per
        # game after the matchups loop above, not once per matchup side. See notes above
        # JOINT_OFF_PCTILE_MAX.
        home_lineup = game.get('home_lineup')
        away_lineup = game.get('away_lineup')
        home_off_info = get_lineup_offense_quality(home_lineup, off_population) if (off_population and home_lineup) else None
        away_off_info = get_lineup_offense_quality(away_lineup, off_population) if (off_population and away_lineup) else None
        if home_off_info and away_off_info:
            combined_off_pctile = (home_off_info['off_pctile'] + away_off_info['off_pctile']) / 2.0

            # ── OFFENSE-OVERPRICED FG JOINT UNDER (clean leak-free lead) ── strong combined offense
            # (top tail of the SAME measure) fades to the FULL-GAME joint total UNDER. Independent of
            # the weak-offense F5-over block below. Gated on the DraftKings full-game total. See
            # OFF_FADE_* constants. Tracking lead, not a bet trigger.
            if combined_off_pctile >= OFF_FADE_PCTILE_MIN:
                _away_abb = NAME_TO_ABB.get(game['away_team'], game['away_team'])
                _ofj_odds = dict(fg_joint_lines.get((home_abb, _away_abb), {}))
                _ofj_pt   = _ofj_odds.get('DraftKings', {}).get('point')
                _kalj = _pick_kalshi(kalshi_joint.get('|'.join(sorted([home_abb, _away_abb])), []), _ofj_pt)
                if _kalj:
                    _ofj_odds['Kalshi'] = _kalj
                _ofj_line_ok = (_ofj_pt is not None and _ofj_pt >= OFF_FADE_LINE_MIN)
                if not pinnacle_gate_active or _ofj_line_ok:
                    off_fade_signals += 1
                    flags.append({
                        'game':              f"{game['away_team']}@{game['home_team']}",
                        'game_id':           game.get('game_id'),
                        'batting_team':      'Joint Total (Off Fade)',  # sentinel — game-level flag
                        'fielding_team':     '',
                        'pitcher_name':      '',
                        'home_team':         game['home_team'],
                        'away_team':         game['away_team'],
                        'signal':            'fg_joint_off_under',
                        'game_time':         game.get('game_time'),
                        'ofj_combined_off':  round(combined_off_pctile, 1),
                        'ofj_home_off':      round(home_off_info['off_pctile'], 1),
                        'ofj_away_off':      round(away_off_info['off_pctile'], 1),
                        # Lineup current-season PA backing each side's offense score (Marcel blend
                        # vs. pure career prior). ofj_min_pa_avg is the weaker-data side; logged so
                        # stale-prior rows can be filtered out of the track record (WHERE min_pa_avg
                        # >= 20). (Off-fade left Telegram June 28, 2026, so the old PA stale gate that
                        # used this for notification suppression is gone -- it's a Sheets filter now.)
                        'ofj_home_pa_avg':   home_off_info['current_pa_avg'],
                        'ofj_away_pa_avg':   away_off_info['current_pa_avg'],
                        'ofj_min_pa_avg':    min(home_off_info['current_pa_avg'],
                                                 away_off_info['current_pa_avg']),
                        'ofj_odds_lines':    _ofj_odds,
                        'ofj_dk_point':      _ofj_pt,
                        'ofj_note':          _off_fade_note(_ofj_pt),
                    })
                else:
                    off_fade_suppressed += 1

            if combined_off_pctile <= JOINT_OFF_PCTILE_MAX:
                away_abb   = NAME_TO_ABB.get(game['away_team'], game['away_team'])
                joint_odds = joint_lines.get((home_abb, away_abb), {})
                fanduel_pt = joint_odds.get(JOINT_GATE_BOOK, {}).get('point')
                # Gated on FanDuel's own line, not Pinnacle's -- see JOINT_GATE_BOOK notes above.
                joint_line_ok = (not pinnacle_gate_active) or (fanduel_pt == JOINT_LINE_TARGET)
                if not joint_line_ok:
                    joint_suppressed += 1
                else:
                    joint_signals += 1
                    flags.append({
                        'game':                  f"{game['away_team']}@{game['home_team']}",
                        'game_id':               game.get('game_id'),
                        # sentinel, not a real team -- this is a game-level flag, but reuses
                        # flag_key()/sheet keying unchanged (both key off game+batting_team).
                        'batting_team':          'Joint Total',
                        'fielding_team':         '',
                        'pitcher_name':          '',
                        'home_team':             game['home_team'],
                        'away_team':             game['away_team'],
                        'signal':                'joint_offense_over',
                        'joint_signal':          True,
                        'game_time':             game.get('game_time'),
                        'joint_off_pctile':      round(combined_off_pctile, 1),
                        'joint_home_off_pctile': round(home_off_info['off_pctile'], 1),
                        'joint_away_off_pctile': round(away_off_info['off_pctile'], 1),
                        'joint_odds_lines':      joint_odds,
                        'joint_fanduel_point':   fanduel_pt,
                        'joint_note':            _joint_note(fanduel_pt),
                    })

                # ── DERIVED F5 TT OVER ── Zach's hypothesis, June 22, 2026: a joint-F5-over flag
                # where exactly ONE team is also flagged F5-under means the OTHER team is the real
                # play, not the joint total (e.g. BOS@SEA: BOS confirmed F5 under, joint flagged
                # over -> SEA alone should be F5 TT over). Reuses the same joint-fires condition
                # (weak combined offense + FanDuel line == 3.5) as the gate; only fires when
                # exactly one side has a confirmed pq_under flag, not zero or both (no clean
                # "other side" if neither or both teams are already explained by F5 under).
                if joint_line_ok:
                    home_flagged = pq_under_fired_team.get(game['home_team'], False)
                    away_flagged = pq_under_fired_team.get(game['away_team'], False)
                    if home_flagged != away_flagged:  # exactly one, not zero or both
                        other_team   = game['away_team'] if home_flagged else game['home_team']
                        flagged_team = game['home_team'] if home_flagged else game['away_team']
                        other_abb    = NAME_TO_ABB.get(other_team, other_team)
                        f5_over_signals += 1
                        flags.append({
                            'game':                 f"{game['away_team']}@{game['home_team']}",
                            'game_id':              game.get('game_id'),
                            'batting_team':         other_team,
                            'fielding_team':        flagged_team,
                            'pitcher_name':         '',
                            'home_team':            game['home_team'],
                            'away_team':            game['away_team'],
                            'signal':               'f5_tt_over',
                            'game_time':            game.get('game_time'),
                            'f5_over_flagged_team': flagged_team,
                            'joint_off_pctile':     round(combined_off_pctile, 1),
                            'joint_fanduel_point':  fanduel_pt,
                            # the OTHER team's own F5 odds (this signal bets THEIR F5 total over,
                            # not the flagged-under team's) -- embedded directly on the flag, same
                            # pattern as pq_q4/fg_tt_under's odds_lines, so the sheet-append
                            # function doesn't need to re-fetch or cache anything separately.
                            'f5_over_odds_lines':   team_lines.get(other_abb, {}),
                            'f5_over_note': (
                                f'Joint F5 over flagged, but {flagged_team} alone is already '
                                f'F5-under flagged (elite opposing pitcher) — hypothesis: the '
                                f'OTHER team ({other_team}) is the real F5 TT over play here, not '
                                f'the joint total (the joint total likely fires because '
                                f'{flagged_team} is suppressed while {other_team} is expected to '
                                f'score heavily). New, untested signal — tracking only, do not bet.'
                            ),
                        })

        # ── FG JOINT (COMBINED GAME) TOTAL BULLPEN SIGNAL ── game-level like joint-offense
        # above. Both teams' bullpens suppress the combined game total, so the composite is the
        # MEAN of the two teams' bullpen percentiles: strong combined -> UNDER, weak -> OVER.
        # Gated on DraftKings' full-game game total being in FG_JOINT_LINE_MIN..MAX. See the
        # FG_JOINT_* constant block. Independent of every per-team flag above. Tracking only.
        home_bp_j  = bp_population.get(home_abb)
        away_abb_j = NAME_TO_ABB.get(game['away_team'], game['away_team'])
        away_bp_j  = bp_population.get(away_abb_j)
        if home_bp_j and away_bp_j:
            combined_bp_pctile = (home_bp_j['percentile'] + away_bp_j['percentile']) / 2.0
            fgj_direction, fgj_tier, fgj_units = fg_joint_tier(combined_bp_pctile)
            if fgj_direction:
                fgj_odds = dict(fg_joint_lines.get((home_abb, away_abb_j), {}))
                fgj_gate_pt = fgj_odds.get(FG_JOINT_GATE_BOOK, {}).get('point')
                fgj_line_ok = (fgj_gate_pt is not None and
                               FG_JOINT_LINE_MIN <= fgj_gate_pt <= FG_JOINT_LINE_MAX)
                # attach Kalshi's combined-game total at the gate line
                _kalj = _pick_kalshi(kalshi_joint.get('|'.join(sorted([home_abb, away_abb_j])), []), fgj_gate_pt)
                if _kalj:
                    fgj_odds['Kalshi'] = _kalj
                if not pinnacle_gate_active or fgj_line_ok:
                    fg_joint_signals += 1
                    flags.append({
                        'game':                  f"{game['away_team']}@{game['home_team']}",
                        'game_id':               game.get('game_id'),
                        # direction+tier sentinel so flag_key (game|batting_team) is unique per
                        # direction and distinct from joint-offense's 'Joint Total'.
                        'batting_team':          f"Joint Total {fgj_direction.capitalize()}",
                        'fielding_team':         '',
                        'pitcher_name':          '',
                        'home_team':             game['home_team'],
                        'away_team':             game['away_team'],
                        'signal':                'fg_joint_total',
                        'fg_joint_signal':       True,
                        'fg_joint_direction':    fgj_direction,
                        'fg_joint_tier':         fgj_tier,
                        'fg_joint_units':        fgj_units,
                        'game_time':             game.get('game_time'),
                        'fgj_combined_pctile':   round(combined_bp_pctile, 1),
                        'fgj_home_bp_pctile':    round(home_bp_j['percentile'], 1),
                        'fgj_away_bp_pctile':    round(away_bp_j['percentile'], 1),
                        'fgj_home_bp_score':     home_bp_j['score'],
                        'fgj_away_bp_score':     away_bp_j['score'],
                        'fgj_odds_lines':        fgj_odds,
                        'fgj_gate_point':        fgj_gate_pt,
                        'fg_joint_note':         _fg_joint_note(fgj_direction, fgj_gate_pt, fgj_tier, fgj_units),
                    })
                else:
                    fg_joint_suppressed += 1

    print(f"Pinnacle gate ({'active' if pinnacle_gate_active else 'INACTIVE -- no ODDS_API_KEY'}): "
          f"{len(flags)} flags shown ({pinnacle_park_exceptions} via park exception), "
          f"{pinnacle_suppressed_wrong} suppressed (Pinnacle line != 1.5, not a park exception), "
          f"{pinnacle_suppressed_missing} suppressed (no Pinnacle line found)")
    print(f"Joint offense gate: {joint_signals} flag(s) shown, {joint_suppressed} suppressed "
          f"(weak combined offense but FanDuel joint line wasn't {JOINT_LINE_TARGET})")
    print(f"Offense-fade FG joint UNDER (clean lead): {off_fade_signals} flag(s) shown, "
          f"{off_fade_suppressed} suppressed (strong combined offense >= {OFF_FADE_PCTILE_MIN} but "
          f"DraftKings full-game total missing/below {OFF_FADE_LINE_MIN})")
    print(f"FG bullpen gate: {fg_signals} flag(s) shown, {fg_suppressed} suppressed "
          f"(opposing bullpen top-quartile but FG line wasn't in {sorted(FG_TT_VALID_LINES)}), "
          f"{fg_no_bp_info}/{len(fg_pctile_checked)} matchups had no bullpen population entry for "
          f"the fielding team (name-resolution failure, not just below-threshold)")
    print(f"FG joint (combined game total) bullpen gate: {fg_joint_signals} flag(s) shown, "
          f"{fg_joint_suppressed} suppressed (combined bullpen qualified but {FG_JOINT_GATE_BOOK} "
          f"game total wasn't in {FG_JOINT_LINE_MIN}-{FG_JOINT_LINE_MAX})")
    print(f"FG bullpen pctile by matchup checked today: "
          f"{[(t, a, p) for t, a, p in fg_pctile_checked]}")
    print(f"F5 TT over (derived): {f5_over_signals} flag(s) shown")
    print(f"K-prop over-favorite: {_kprop_diag['fired']} fired | "
          f"{_kprop_diag['matched']}/{_kprop_diag['checked']} starters matched a sportsbook prop "
          f"line | {len(_kprop_diag['unmatched'])} no match"
          + (f" (no line posted OR name mismatch): {_kprop_diag['unmatched'][:8]}"
             if _kprop_diag['unmatched'] else ""))

    # SURFACED SIGNALS (June 23, 2026, per Zach): collapse the live output to exactly the three
    # signals actively tracked right now -- Q4 pitcher F5 U1.5, the derived F5 TT over, and the
    # full-game joint OFFENSE FADE. Everything else still COMPUTES above and is deliberately NOT
    # surfaced, so it reaches neither the dashboard, Telegram, nor the Sheets:
    #   * offense_quality_only (off_q3) -- muted, but its offense population/percentiles still feed
    #                                      the offense-fade signal.
    #   * joint_offense_over            -- muted, but its weak-combined-offense CONDITION still
    #                                      drives the derived f5_tt_over below: Zach wants the
    #                                      joint-over to FIRE the over-derivation without the joint
    #                                      flag itself being shown or logged.
    #   * fg_tt_under / fg_joint_total  -- disproven bullpen leakage (see DISABLED_SIGNALS).
    # NOTE: fg_joint_off_under was previously computed but accidentally left out of this assembly,
    # so it never reached the dashboard/Telegram/Sheets despite the full plumbing existing for it
    # (append_off_fade_to_sheet, the api_notify off-fade block, build_picks_payload's off_fade_flags).
    # Fixed here by including it.
    pq_only_flags       = sorted([f for f in flags if f['signal'] == 'pitcher_quality_only'],
                                  key=lambda x: x.get('pq_percentile', 0), reverse=True)
    kprop_only_flags    = sorted([f for f in flags if f['signal'] == 'k_prop_only'],
                                  key=lambda x: x.get('k_comp_score') or 0, reverse=True)
    f5_over_only_flags  = [f for f in flags if f['signal'] == 'f5_tt_over']
    off_fade_only_flags = [f for f in flags if f['signal'] == 'fg_joint_off_under']
    result = pq_only_flags + kprop_only_flags + f5_over_only_flags + off_fade_only_flags
    return [f for f in result if f.get('signal') not in DISABLED_SIGNALS]


def get_heatmap_flags(games, model):
    pitcher_scores  = model.get('all_pitcher_arch_scores', {})
    nb_bundle       = model.get('negbin_model_params')
    contact_rates   = model.get('pitcher_contact_rates', {})
    bb_gb_rates_all = model.get('pitcher_bb_gb_rates', {}) or {}
    flags           = []

    try:
        odds_lines, _joint_lines_unused, _fg_lines_unused, _fg_joint_unused, kprop_lines = get_odds_api_lines(games)
    except Exception as e:
        print(f"Odds API lookup error: {e}")
        odds_lines, kprop_lines = {}, {}

    # Mirrors the Pinnacle gate in get_tracking_only_flags (see notes above get_odds_api_lines) --
    # kept in sync here even though this function is currently paused (Drive incident) so the
    # gate isn't silently lost whenever this gets restored. Only gates the two standalone
    # tracking signals (pq_q4/off_q3_gate); the sigma 'under' signal is unaffected.
    pinnacle_gate_active = bool(ODDS_API_KEY)

    for game in games:
        home_abb = NAME_TO_ABB.get(game['home_team'], game['home_team'])
        matchups = [
            (game['away_lineup'], game['home_pitcher_id'],
             game['home_pitcher_name'], game['away_team'],
             game['home_team'], f"{game['away_team']}@{game['home_team']}"),
            (game['home_lineup'], game['away_pitcher_id'],
             game['away_pitcher_name'], game['home_team'],
             game['away_team'], f"{game['away_team']}@{game['home_team']}"),
        ]
        for lineup, pitcher_id, pitcher_name, batting_team, fielding_team, game_str in matchups:
            if not lineup or not pitcher_id:
                continue
            pitcher_hand = 'R'
            if pitcher_id in pitcher_scores:
                pitcher_hand = pitcher_scores[pitcher_id].get('p_throws', 'R')
            pitcher_k_rate = None
            if contact_rates and pitcher_id in contact_rates:
                pitcher_k_rate = contact_rates[pitcher_id].get('k_rate')

            bb_gb_rates = bb_gb_rates_all.get(pitcher_id) if pitcher_id else None

            scores = []
            scored_batters = 0
            for batter in lineup:
                s = compute_batter_overlap(
                    batter['id'], pitcher_id,
                    batter.get('hand', 'R'), pitcher_hand, model)
                if s is not None:
                    scores.append(s)
                    scored_batters += 1
            if scored_batters < 3:
                continue

            overlap_raw   = sum(scores) / len(scores)
            team_score    = apply_k_modifier(overlap_raw, pitcher_k_rate)
            std_from_mean = (team_score - HEATMAP_MEAN) / HEATMAP_STD

            expected_f5   = get_expected_f5_runs(overlap_k_mod=team_score, model_bundle=nb_bundle)
            fair_odds     = get_f5_fair_odds(expected_f5)
            fg_flag       = compute_fg_under_flag(batting_team, fielding_team, team_score)

            is_away       = (batting_team == game['away_team'])

            # ── PITCHER QUALITY COMPOSITE (research/tracking only, see notes above) ──
            pq_info = None
            try:
                pq_population = get_pitcher_quality_population()
                pq_info = pq_population.get(pitcher_id) if pitcher_id else None
            except Exception as e:
                print(f"Pitcher quality lookup error: {e}")
            pq_q4 = bool(pq_info and pq_info['quartile'] == 'Q4')

            # ── OFFENSE QUALITY COMPOSITE (research/tracking only, see notes above) ──
            off_info = None
            try:
                off_population = get_offense_quality_population()
                off_info = get_lineup_offense_quality(lineup, off_population) if off_population else None
            except Exception as e:
                print(f"Offense quality lookup error: {e}")
            off_q3_gate = bool(
                off_info and
                OFF_Q3_LOW_PCTILE <= off_info['off_pctile'] <= OFF_Q3_HIGH_PCTILE and
                off_info['o_bb_rate'] < OFF_BB_RATE_MEDIAN
            )

            # ── PINNACLE GATE (tracking signals only) ── line 1.5, or 2.5+ in a hitter-favorable
            # park (validated park exception, see PARK_GATE_TEAMS notes above).
            team_abb       = NAME_TO_ABB.get(batting_team, batting_team)
            team_odds      = odds_lines.get(team_abb, {})
            pinnacle_point = team_odds.get(PINNACLE_BOOK_LABEL, {}).get('point')
            park_exception = (pinnacle_point is not None and pinnacle_point >= PARK_GATE_MIN_POINT
                               and home_abb in PARK_GATE_TEAMS)
            pinnacle_ok    = (not pinnacle_gate_active) or (pinnacle_point == 1.5) or park_exception
            pq_q4          = pq_q4 and pinnacle_ok
            off_q3_gate    = off_q3_gate and pinnacle_ok
            gate_reason    = (None if not pinnacle_gate_active else
                               'park_exception' if (park_exception and pinnacle_point != 1.5) else
                               'line_1_5')

            # ── SIGNAL DETERMINATION ──────────────────────────────────────
            signal = None
            under_confidence = None

            if team_score <= HEATMAP_UNDER_THRESHOLD:
                under_confidence = get_ozzie_score(
                    std_from_mean=std_from_mean,
                    is_away=is_away,
                    k_rate=pitcher_k_rate,
                    park=fielding_team,
                    bb_gb_rates=bb_gb_rates)
                if under_confidence:
                    signal = 'under'

            if signal is None:
                if fg_flag['fg_under_signal']:
                    signal = 'fg_under_only'

            # Pitcher quality Q4 surfaces even without any other signal — tracking only, NOT a
            # validated bet signal. pq_q4 is already Pinnacle-gated to a confirmed 1.5 F5 line
            # above (when the gate is active) — see PINNACLE GATE block.
            if signal is None and pq_q4:
                signal = 'pitcher_quality_only'

            # Offense quality Q3-band+low-BB gate, same tracking-only treatment. Independent of
            # pq_q4 — both describe different sides of the same matchup and can co-occur (visible
            # via the off_q3_gate / pq_q4 boolean fields below even when 'signal' picks one).
            if signal is None and off_q3_gate:
                signal = 'offense_quality_only'

            if signal is None:
                continue

            _kc = k_composite_score(
                pq_info['k_rate']  if pq_info else None,
                pq_info['bb_rate'] if pq_info else None,
                pq_info['hr_rate'] if pq_info else None,
                off_info.get('o_k_rate') if off_info else None,
            )
            # K-prop signal REBUILT June 27, 2026 — over-favorite price bias (same as block 1 in
            # get_tracking_only_flags). This get_heatmap_flags path is paused (Drive incident) and
            # previously referenced an undefined kalshi_k_props here; _kal_k2 set None (Kalshi is
            # reference-only now and not fetched in this paused path).
            _proj_k_val2 = project_starter_ks(pq_info['k_rate'] if pq_info else None,
                                              off_info['o_k_rate'] if off_info else None,
                                              pq_info.get('exp_bf') if pq_info else None)
            _kal_k2         = None
            _ofav2          = _kprop_over_fav(pitcher_name, kprop_lines, batting_team, odds_lines,
                                              prior_starts=pq_info.get('gs') if pq_info else None)
            _k_prop_signal2 = _ofav2['signal']

            flag = {
                'game':             game_str,
                'game_id':          game.get('game_id'),
                'batting_team':     batting_team,
                'fielding_team':    fielding_team,
                'pitcher_name':     pitcher_name,
                'pitcher_hand':     pitcher_hand,
                'pitcher_id':       pitcher_id,
                'overlap_score':    round(team_score, 4),
                'std_from_mean':    round(std_from_mean, 2),
                'signal':           signal,
                'batters_scored':   scored_batters,
                'lineup_complete':  len(lineup) >= 8,
                'expected_f5_runs': expected_f5,
                'fair_under_2_5':   fair_odds['fair_under_2_5'],
                'fair_over_2_5':    fair_odds['fair_over_2_5'],
                'pitcher_k_rate':   round(pitcher_k_rate, 3) if pitcher_k_rate else None,
                'fg_under_signal':  fg_flag['fg_under_signal'],
                'fg_starter_z':     fg_flag['starter_z'],
                'fg_opp_bp_weak_z': fg_flag['opp_bp_weak_z'],
                'fg_off_z':         fg_flag['off_z'],
                'fg_bp_in_band':    fg_flag['bp_in_band'],
                'fg_in_window':     fg_flag['in_valid_window'],
                'fg_reason':        fg_flag['reason'],
                'game_time':        game.get('game_time'),
                'pq_score':         pq_info['score']      if pq_info else None,
                'pq_percentile':    pq_info['percentile'] if pq_info else None,
                'pq_quartile':      pq_info['quartile']   if pq_info else None,
                'pq_k_rate':        pq_info['k_rate']      if pq_info else None,
                'pq_bb_rate':       pq_info['bb_rate']     if pq_info else None,
                'pq_hr_rate':       pq_info['hr_rate']     if pq_info else None,
                'projected_k':      _proj_k_val2,
                'pq_q4':            pq_q4,
                'k_comp_score':     _kc,
                'k_prop_flag':      _k_prop_signal2,
                'k_prop_tier':      _ofav2['tier'],
                'kprop_dk_over':    _ofav2['dk_over'],
                'kprop_best_over':  _ofav2['best_over'],
                'kprop_best_book':  _ofav2['best_book'],
                'kprop_line':       _ofav2['line'],
                'kprop_opp_f5':     _ofav2['opp_f5_total'],
                'kprop_n_books':    _ofav2['n_books'],
                'kprop_prior_starts':      _ofav2['prior_starts'],
                'kprop_sharp_blocked_thin': _ofav2['sharp_blocked_thin'],
                'kalshi_k_line':    _kal_k2['implied_line'] if _kal_k2 else None,
                'kalshi_k_strike':  _kal_k2.get('bet_threshold') if _kal_k2 else None,
                'kalshi_k_yes_bid': _kal_k2.get('yes_bid') if _kal_k2 else None,
                'k_prop_note':      _kprop_note(_ofav2, _kal_k2),
                'pq_current_bf':    pq_info['current_bf'] if pq_info else None,
                'pq_note':          _pq_note(pq_q4, gate_reason, pinnacle_point),
                'o_k_rate':         off_info.get('o_k_rate') if off_info else None,
                'off_pctile':       off_info['off_pctile'] if off_info else None,
                'off_bb_rate':      off_info['o_bb_rate']  if off_info else None,
                'off_q3_gate':      off_q3_gate,
                'off_current_pa_avg': off_info['current_pa_avg'] if off_info else None,
                'off_note':         _off_note(off_q3_gate, gate_reason, pinnacle_point),
                # Live F5 odds (DraftKings/Fanatics/theScore Bet) — see get_odds_api_lines.
                # pq_q4/off_q3_gate are already Pinnacle-gated above (see PINNACLE GATE block) --
                # this is just the shoppable-book display data, same as before.
                'odds_lines':       team_odds,
                'odds_has_1_5':     any(b.get('point') == 1.5 for b in team_odds.values()),
                'pinnacle_point':   pinnacle_point,
                'pinnacle_gate_reason': gate_reason,
            }

            # Add under-specific fields
            if signal == 'under' and under_confidence:
                flag.update({
                    'ozzie_score':      under_confidence['ozzie_score'],
                    'confidence_label': under_confidence['confidence_label'],
                    'confidence_color': under_confidence['confidence_color'],
                    'park_tier':        under_confidence['park_tier'],
                    'q_away':           under_confidence['q_away'],
                    'q_krate_lo':       under_confidence['q_krate_lo'],
                    'q_krate_hi':       under_confidence['q_krate_hi'],
                    'q_bb_vlo':         under_confidence['q_bb_vlo'],
                    'q_gb_mid':         under_confidence['q_gb_mid'],
                    'q_sigma_20':       under_confidence['q_sigma_20'],
                    'bb_rate':          under_confidence['bb_rate'],
                    'gb_rate':          under_confidence['gb_rate'],
                    'min_u25':          under_confidence['min_u25'],
                    'unit_u25':         under_confidence['unit_u25'],
                })

            flags.append(flag)

    # Sort: unders by ozzie_score desc, pq_only by percentile desc, off_only by off_pctile desc, others last
    under_flags     = sorted([f for f in flags if f['signal'] == 'under'],
                             key=lambda x: x.get('ozzie_score', 0), reverse=True)
    pq_only_flags   = sorted([f for f in flags if f['signal'] == 'pitcher_quality_only'],
                             key=lambda x: x.get('pq_percentile', 0), reverse=True)
    off_only_flags  = sorted([f for f in flags if f['signal'] == 'offense_quality_only'],
                             key=lambda x: x.get('off_pctile', 0), reverse=True)
    other_flags     = [f for f in flags if f['signal'] not in
                       ('under', 'pitcher_quality_only', 'offense_quality_only')]
    flags = under_flags + pq_only_flags + off_only_flags + other_flags

    # ── Combined F5 signal — two tiers + Diamond+ upgrade ────────────────────
    # Diamond: asymmetric gate, 74% U4.5, ~51/yr. Bet U4.5 @ -200 or better.
    # Diamond+: Diamond + both pitchers deep, 82% U4.5, ~28/yr. Bet aggressively.
    # Watch: min_sigma <= -1.5, 61% U4.5, ~199/yr. BE=-157. Bet only if U4.5 <= -130.
    game_flags = {}
    for f in flags:
        if f['signal'] != 'under':
            continue
        g = f['game']
        if g not in game_flags:
            game_flags[g] = []
        game_flags[g].append(f)

    for f in flags:
        f['combined_f5_signal']    = False
        f['combined_f5_tier']      = None
        f['combined_f5_min_sigma'] = None

    for game_str, gflags in game_flags.items():
        away_flag = next((f for f in gflags if f.get('q_away')), None)
        home_flag = next((f for f in gflags if not f.get('q_away')), None)
        if not away_flag or not home_flag:
            continue

        away_std = away_flag['std_from_mean']
        home_std = home_flag['std_from_mean']
        away_k   = away_flag.get('pitcher_k_rate') or 1.0
        home_k   = home_flag.get('pitcher_k_rate') or 1.0
        min_sigma = min(away_std, home_std)
        max_sigma = max(away_std, home_std)

        # Diamond gate: asymmetric — one side very deep, other moderate, home low-K
        diamond = (
            away_std <= -1.5 and
            home_std > -1.5 and
            home_k < 0.28 and
            (away_std <= -2.0 or home_std <= -2.0)
        )

        # Diamond+ upgrade: Diamond AND both pitchers suppressed
        diamond_plus = diamond and (min_sigma <= -2.0) and (max_sigma <= -0.5)

        # Watch gate: either pitcher very deep
        watch = min_sigma <= -1.5

        if diamond or watch:
            if diamond_plus:
                tier = 'diamond_plus'
            elif diamond:
                tier = 'diamond'
            else:
                tier = 'watch'

            for f in [away_flag, home_flag]:
                f['combined_f5_signal']    = True
                f['combined_f5_tier']      = tier
                f['combined_f5_min_sigma'] = round(min_sigma, 2)
                f['combined_f5_max_sigma'] = round(max_sigma, 2)

    # ── DISABLED (June 22, 2026): the FULL-GAME bullpen totals signals were proven to be outcome
    # leakage — the backtest's per-game bullpen_z was built from the relievers who ACTUALLY pitched,
    # which is decided by the game itself; the clean leak-free re-test (live roster-quality bullpen)
    # was a coin toss (~50%, see CLAUDE.md "DEFINITIVE LEAK-FREE TEST"). Removed from firing /
    # notifying / logging / display. The F5 signals (pq_q4, f5_tt_over, off_q3, joint_offense) are
    # NOT affected — they're built on the announced starter + offense (pre-game, leak-free) — and
    # are left live. They still compute above; this just drops them before they surface anywhere.
    # (Reuses the module-level DISABLED_SIGNALS constant -- see get_tracking_only_flags, the
    # function actually on the live path, for why this copy here wasn't sufficient on its own.)
    flags = [f for f in flags if f.get('signal') not in DISABLED_SIGNALS]
    return flags


def get_hr_picks(games, model):
    arch_hitter_map = model['arch_hitter_map']
    pitcher_scores  = model['all_pitcher_arch_scores']
    archetypes      = model['archetypes']
    picks           = []
    for game in games:
        if not game['home_lineup'] or not game['away_lineup']:
            continue
        matchups = [
            (game['away_lineup'], game['home_pitcher_id'], game['home_team'],
             f"{game['away_team']}@{game['home_team']}", game['home_pitcher_name'], game['away_team']),
            (game['home_lineup'], game['away_pitcher_id'], game['away_team'],
             f"{game['away_team']}@{game['home_team']}", game['away_pitcher_name'], game['home_team']),
        ]
        for lineup, pitcher_id, fielding_team, game_str, pitcher_name, batting_team in matchups:
            if not pitcher_id or pitcher_id not in pitcher_scores:
                continue
            pitcher = pitcher_scores[pitcher_id]
            for batter in lineup:
                batter_id   = batter['id']
                batter_hand = batter.get('hand', 'R')
                for arch_key in archetypes:
                    if ('_L' in arch_key) != (batter_hand == 'L'):
                        continue
                    pool = arch_hitter_map.get(arch_key, set())
                    if isinstance(pool, pd.DataFrame):
                        pool = set(pool['batter_id'].tolist())
                    if batter_id not in pool:
                        continue
                    arch_data = pitcher['archetypes'].get(arch_key, {})
                    if arch_data.get('sensor_pa', 0) < MIN_SENSOR_PA:
                        continue
                    starter_rate = arch_data.get('shrunk_rate', LEAGUE_AVG)
                    bullpen_rate = get_bullpen_score(fielding_team, arch_key, model)
                    combined     = STARTER_WEIGHT * starter_rate + BULLPEN_WEIGHT * bullpen_rate
                    if combined < JUICY_THRESHOLD:
                        continue
                    pf          = get_park_factor(fielding_team, batter_hand, model)
                    adjusted    = combined * pf
                    power_map   = model.get('batter_power_map', {})
                    power_mult  = power_map.get(batter_id, 1.0)
                    hr_adjusted = adjusted * power_mult
                    hr_odds_str = pa_rate_to_game_odds(hr_adjusted)
                    hr_odds_num = int(hr_odds_str.replace('+', ''))
                    picks.append({
                        'player_name':   batter['name'],
                        'player_id':     batter_id,
                        'pitcher_name':  pitcher_name,
                        'pitcher_id':    pitcher_id,
                        'game':          game_str,
                        'batting_team':  batting_team,
                        'fielding_team': fielding_team,
                        'combined':      round(combined, 2),
                        'hr_fair':       hr_odds_str,
                        'hr_odds_num':   hr_odds_num,
                        'tb3_fair':      pa_rate_to_game_odds(adjusted * TB3_MULTIPLIER),
                        'arch_name':     archetypes[arch_key]['name'],
                    })
    seen = {}
    for p in picks:
        pid = p['player_id']
        if pid not in seen or p['combined'] > seen[pid]['combined']:
            seen[pid] = p
    return sorted(seen.values(), key=lambda x: x['combined'], reverse=True)


def get_dfs_picks(games, model):
    pitcher_scores  = model.get('all_pitcher_arch_scores', {})
    contact_rates   = model.get('pitcher_contact_rates', {})
    arch_hitter_map = model.get('arch_hitter_map', {})
    splits_lookup   = model.get('splits_lookup', {})

    batter_arch_pool = {}
    for arch_key, pool in arch_hitter_map.items():
        ids = list(pool['batter_id'].tolist()) if isinstance(pool, pd.DataFrame) else list(pool)
        for bid in ids:
            bid = int(bid)
            batter_arch_pool.setdefault(bid, []).append(arch_key)

    pitcher_plays = []
    stack_targets = []

    for game in games:
        game_str  = f"{game['away_team']}@{game['home_team']}"
        game_time = game.get('game_time', '')

        matchups = [
            (game['away_lineup'], game['home_pitcher_id'],
             game['home_pitcher_name'], game['away_team'], game['home_team']),
            (game['home_lineup'], game['away_pitcher_id'],
             game['away_pitcher_name'], game['home_team'], game['away_team']),
        ]

        for lineup, pitcher_id, pitcher_name, batting_team, fielding_team in matchups:
            if not pitcher_id or not lineup:
                continue

            k_rate = None
            if contact_rates and pitcher_id in contact_rates:
                k_rate = contact_rates[pitcher_id].get('k_rate')

            pitcher_hand = 'R'
            if pitcher_id in pitcher_scores:
                pitcher_hand = pitcher_scores[pitcher_id].get('p_throws', 'R')

            scores = []
            for batter in lineup:
                s = compute_batter_overlap(
                    batter['id'], pitcher_id,
                    batter.get('hand', 'R'), pitcher_hand, model)
                if s is not None:
                    scores.append(s)

            if len(scores) < 3:
                continue

            overlap_raw = sum(scores) / len(scores)
            team_score  = apply_k_modifier(overlap_raw, k_rate)
            overlap_z   = round((team_score - HEATMAP_MEAN) / HEATMAP_STD, 2)
            k_pct       = round(float(k_rate) * 100, 1) if k_rate else None

            if k_rate and k_rate >= DFS_ACE_K and overlap_z <= -1.0:
                tier, tier_color = 'Ace',     'gold'
            elif k_rate and k_rate >= DFS_VALUE_K and overlap_z <= 0.0:
                tier, tier_color = 'Value',   'silver'
            elif overlap_z >= DFS_FADE_PITCH:
                tier, tier_color = 'Fade',    'red'
            else:
                tier, tier_color = 'Neutral', 'gray'

            pitcher_plays.append({
                'pitcher_name':  str(pitcher_name),
                'pitcher_id':    int(pitcher_id),
                'fielding_team': str(fielding_team),
                'batting_team':  str(batting_team),
                'game':          str(game_str),
                'game_time':     str(game_time),
                'k_rate':        k_pct,
                'overlap_z':     float(overlap_z),
                'tier':          str(tier),
                'tier_color':    str(tier_color),
            })

            if overlap_z >= DFS_STACK_MIN_Z:
                stack_batters = []
                for batter in lineup:
                    bid   = int(batter['id'])
                    slot  = int(batter.get('batting_order', 9))
                    archs = batter_arch_pool.get(bid, [])

                    arch_match = False
                    if pitcher_id in pitcher_scores:
                        p_archs = pitcher_scores[pitcher_id].get('archetypes', {})
                        for ak in archs:
                            hand_ok = ak.endswith('_L') == (batter.get('hand') == 'L')
                            if hand_ok and p_archs.get(ak, {}).get('shrunk_rate', 0) >= JUICY_THRESHOLD:
                                arch_match = True
                                break

                    split_info = splits_lookup.get(bid)
                    split_diff = None
                    split_favorable = None
                    if split_info:
                        diff = split_info['diff']
                        flag = split_info['flag']
                        if flag == 'favors_RHP':
                            split_favorable = (pitcher_hand == 'R')
                        else:
                            split_favorable = (pitcher_hand == 'L')
                        split_diff = diff

                    stack_batters.append({
                        'name':            str(batter['name']),
                        'id':              bid,
                        'slot':            slot,
                        'position':        str(batter.get('position', '')),
                        'arch_match':      bool(arch_match),
                        'priority':        bool(slot <= 4),
                        'split_diff':      split_diff,
                        'split_favorable': split_favorable,
                    })

                stack_batters.sort(key=lambda x: x['slot'])
                stack_targets.append({
                    'batting_team':  str(batting_team),
                    'fielding_team': str(fielding_team),
                    'pitcher_name':  str(pitcher_name),
                    'game':          str(game_str),
                    'game_time':     str(game_time),
                    'overlap_z':     float(overlap_z),
                    'is_hot':        bool(overlap_z >= DFS_HOT_Z),
                    'batters':       stack_batters,
                })

    tier_order = {'Ace': 0, 'Value': 1, 'Neutral': 2, 'Fade': 3}
    pitcher_plays.sort(key=lambda x: (tier_order.get(x['tier'], 9), x['overlap_z']))
    stack_targets.sort(key=lambda x: x['overlap_z'], reverse=True)

    return {'pitcher_plays': pitcher_plays, 'stack_targets': stack_targets}


@app.route('/')
def index():
    if not session.get('authenticated'):
        return redirect(url_for('login'))
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form.get('password') == APP_PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('index'))
        error = 'Wrong password. Try again.'
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


def build_picks_payload(today, games, heatmap_flags):
    """Shared payload shape for /api/picks' response and its Redis cache entry. Also called
    from /api/notify so a Telegram-triggered flag computation can refresh the dashboard's
    cache immediately instead of leaving it to show stale data until the cache's own TTL
    naturally expires (up to 30 min later, or never if the live odds gate closes again before
    then -- see PICKS CACHE / NOTIFY SYNC note above api_notify)."""
    # TRACKING-ONLY MODE (see comment above get_tracking_only_flags): Google Drive model
    # loading is paused, so load_model()/get_hr_picks()/get_dfs_picks() are not called --
    # all three need the Drive-loaded model. picks/dfs_picks stay empty until restored.
    complete  = sum(1 for g in games if g['home_lineup'] and g['away_lineup'])
    games_out = [{'away': g['away_team'], 'home': g['home_team'],
                  'complete': bool(g['home_lineup'] and g['away_lineup']),
                  'game_time': g.get('game_time')}
                 for g in games]
    fg_under_flags  = [f for f in heatmap_flags if f.get('fg_under_signal')]
    over_info_flags = [f for f in heatmap_flags if f.get('signal') == 'over_info']
    pq_flags        = [f for f in heatmap_flags if f.get('pq_q4')]
    kprop_only_flags = [f for f in heatmap_flags if f.get('signal') == 'k_prop_only']
    fg_tt_flags     = [f for f in heatmap_flags if f.get('signal') == 'fg_tt_under']
    f5_over_flags   = [f for f in heatmap_flags if f.get('signal') == 'f5_tt_over']
    fg_joint_flags  = [f for f in heatmap_flags if f.get('signal') == 'fg_joint_total']
    off_fade_flags  = [f for f in heatmap_flags if f.get('signal') == 'fg_joint_off_under']
    # VISIBILITY, per Zach (June 22, 2026): keep computing/logging off_q3_gate and the raw
    # joint_offense_over signal exactly as before (see api_notify's Sheets-append calls, which
    # are NOT gated on this) -- just don't surface them on the dashboard or in Telegram anymore.
    # Concentrate visible signals on F5 TT under (pq_flags), FG TT under (fg_tt_flags, new), and
    # F5 TT over (f5_over_flags, new, derived). Sent as empty lists rather than omitted so the
    # frontend's `(data.off_flags||[]).filter(...)` pattern degrades to "section just stays
    # hidden" instead of needing a template change.
    off_flags       = []
    joint_flags     = []
    return {
        'date':              today,
        'complete':          complete,
        'total':             len(games),
        'picks':             [],
        'dfs_picks':         {},
        'heatmap_flags':     heatmap_flags,
        'fg_under_flags':    fg_under_flags,
        'over_info_flags':   over_info_flags,
        'pq_flags':          pq_flags,
        'kprop_only_flags':  kprop_only_flags,
        'off_flags':         off_flags,
        'joint_flags':       joint_flags,
        'fg_tt_flags':       fg_tt_flags,
        'f5_over_flags':     f5_over_flags,
        'fg_joint_flags':    fg_joint_flags,
        'off_fade_flags':    off_fade_flags,
        'fg_in_window':      is_fg_valid_window(),
        'games':             games_out,
    }


def picks_cache_ttl(now_et=None):
    """Seconds until the picks cache should expire -- capped at 30 min, or sooner if 1am ET
    (the next day's slate boundary) is closer than that."""
    from datetime import timedelta
    now_et = now_et or datetime.now(pytz.timezone('America/New_York'))
    expire_et = now_et.replace(hour=1, minute=0, second=0, microsecond=0)
    if now_et >= expire_et:
        expire_et = expire_et + timedelta(days=1)
    return min(1800, int((expire_et - now_et).total_seconds()))


@app.route('/api/picks')
def api_picks():
    if not session.get('authenticated'):
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        import json as _json
        et_now    = datetime.now(pytz.timezone('America/New_York'))
        today     = et_now.strftime('%Y-%m-%d')
        cache_key = f"ozzie:picks:{today}"
        # ?refresh=1 bypasses ALL THREE cache layers, not just this one: the picks-response
        # Redis cache here, plus the 6h in-memory PQ/offense population caches and the 4h Redis
        # odds-lines cache (both inside get_tracking_only_flags). Forcing only this top-level
        # cache used to silently still serve the inner caches' data with zero fresh fetch
        # happening -- discovered June 19, 2026 debugging why diagnostic print()s never fired.
        force = request.args.get('refresh') == '1'

        cached = None if force else redis_get(cache_key)
        if cached:
            try:
                return jsonify(_json.loads(cached))
            except Exception:
                pass

        games         = get_lineups_and_starters(today)
        heatmap_flags = get_tracking_only_flags(games, force=force)
        payload       = build_picks_payload(today, games, heatmap_flags)

        ttl = picks_cache_ttl(et_now)
        if ttl > 0 and heatmap_flags:
            try:
                redis_set(cache_key, _json.dumps(payload), ex=ttl)
            except Exception as cache_err:
                print(f"Cache write failed: {cache_err}")
                try:
                    safe_payload = {k: v for k, v in payload.items() if k != 'dfs_picks'}
                    redis_set(cache_key, _json.dumps(safe_payload), ex=ttl)
                except Exception:
                    pass

        return jsonify(payload)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


NOTIFY_SECRET       = os.environ.get('NOTIFY_SECRET', '')
TELEGRAM_BOT_TOKEN  = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID    = os.environ.get('TELEGRAM_CHAT_ID', '')
UPSTASH_URL         = os.environ.get('UPSTASH_REDIS_REST_URL', '')
UPSTASH_TOKEN       = os.environ.get('UPSTASH_REDIS_REST_TOKEN', '')
SHEETS_CREDS        = os.environ.get('GOOGLE_SHEETS_CREDENTIALS', '')
SHEETS_ID           = '1AKalzsMqSDmLe5j26de3R6_YVrpptJVZY9tWrZNCbHs'
SHEETS_TAB          = 'Sheet1'


def redis_set(key, value, ex=86400):
    try:
        from urllib.parse import quote
        encoded_key   = quote(key,   safe='')
        encoded_value = quote(value, safe='')
        r = requests.post(
            f"{UPSTASH_URL}/set/{encoded_key}/{encoded_value}?ex={ex}",
            headers={'Authorization': f'Bearer {UPSTASH_TOKEN}'},
            timeout=5
        )
        print(f"Redis SET {key}: {r.status_code} {r.text}")
    except Exception as e:
        print(f"Redis SET error: {e}")


def redis_get(key):
    try:
        from urllib.parse import quote
        encoded_key = quote(key, safe='')
        r = requests.get(
            f"{UPSTASH_URL}/get/{encoded_key}",
            headers={'Authorization': f'Bearer {UPSTASH_TOKEN}'},
            timeout=5
        )
        data = r.json()
        return data.get('result')
    except Exception as e:
        print(f"Redis GET error: {e}")
        return None


def send_telegram(message):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'},
            timeout=10
        )
    except Exception as e:
        print(f"Telegram send error: {e}")


def flag_key(flag):
    return f"{flag['game']}|{flag['batting_team']}"


def format_odds_lines(odds_lines, gate_book=None, gate_point=None):
    """e.g. 'DraftKings 1.5 -150 | Pinnacle 1.5✅ -119' — empty string if no data.

    A ✅ confirms ONLY the gating book at the gate point -- i.e. the book the signal is actually
    gated on (Pinnacle at 1.5 for Pitcher Quality), not every book that happens to sit at 1.5.
    Other books are line-shopping context and never get a checkmark. Callers with no line gate
    (e.g. the F5-over other-side play) pass no gate_book, yielding a clean checkmark-free row.
    Updated June 28, 2026 per Zach: a ✅ should mean "the gate is met," not "this book is 1.5.\""""
    if not odds_lines:
        return ''
    parts = []
    for book, info in odds_lines.items():
        pt = info.get('point')
        marker = '✅' if (gate_book is not None and book == gate_book and pt == gate_point) else ''
        price = info.get('under')
        price_str = '' if price is None else f" {f'+{price}' if price > 0 else price}"
        parts.append(f"{book} {pt}{marker}{price_str}")
    return ' | '.join(parts)


def format_joint_odds_lines(odds_lines):
    """Same shape as format_odds_lines, but shows the OVER price (this signal bets over, not
    under) and marks JOINT_LINE_TARGET (3.5) instead of 1.5 -- e.g. 'FanDuel 3.5✅ -114'."""
    if not odds_lines:
        return ''
    parts = []
    for book, info in odds_lines.items():
        pt = info.get('point')
        marker = '✅' if pt == JOINT_LINE_TARGET else ''
        price = info.get('over')
        price_str = '' if price is None else f" {f'+{price}' if price > 0 else price}"
        parts.append(f"{book} {pt}{marker}{price_str}")
    return ' | '.join(parts)


def format_fg_joint_odds_lines(odds_lines, direction):
    """FG joint/combined total odds, per book, showing the side this flag bets (under or over)
    and marking lines inside FG_JOINT_LINE_MIN..MAX with ✅ — e.g. 'DraftKings 8.5✅ -110'."""
    if not odds_lines:
        return ''
    side = 'over' if direction == 'over' else 'under'
    parts = []
    for book, info in odds_lines.items():
        pt = info.get('point')
        marker = '✅' if (pt is not None and FG_JOINT_LINE_MIN <= pt <= FG_JOINT_LINE_MAX) else ''
        price = info.get(side)
        price_str = '' if price is None else f" {f'+{price}' if price > 0 else price}"
        parts.append(f"{book} {pt}{marker}{price_str}")
    return ' | '.join(parts)


# Books tracked as their own Line/Odds column pair on the PitcherQuality/OffenseQuality sheet
# tabs (June 20, 2026, per Zach -- U1 at +100 is a very different bet than U1.5 at -150, so the
# single combined "books_lines" text column wasn't enough to compute real ROI from the sheet).
SHEET_TRACKED_BOOKS = ['Pinnacle', 'DraftKings', 'theScore Bet']


def _book_line_odds(team_odds, book_label):
    """
    (point, under_odds) for one book from a team's odds_lines dict, '' for either if that book
    didn't post this game. Under odds specifically -- both tracking signals bet the under side,
    not the over -- formatted with an explicit '+' on positive American odds (e.g. '+100') to
    match standard sportsbook display, since a bare 'point' column can't distinguish U1.5 -150
    from U1.5 +100 and that difference is the whole point of tracking odds at all.
    """
    info  = (team_odds or {}).get(book_label) or {}
    point = info.get('point', '')
    price = info.get('under')
    odds  = '' if price is None else (f"+{price}" if price > 0 else str(price))
    return point, odds


def _open_sheet():
    """Authorize gspread from SHEETS_CREDS and open the Ozzie spreadsheet once.
    Returns (gspread_module, spreadsheet). Callers keep their own try/except, so each
    sheet's ImportError/Exception handling (and backfill's return-on-error) is unchanged --
    this only removes the identical auth boilerplate copy-pasted into every appender."""
    import gspread
    import json as _json
    from google.oauth2.service_account import Credentials
    creds_dict = _json.loads(SHEETS_CREDS)
    scopes = ['https://www.googleapis.com/auth/spreadsheets',
              'https://www.googleapis.com/auth/drive']
    creds  = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc     = gspread.authorize(creds)
    return gspread, gc.open_by_key(SHEETS_ID)


def _get_or_create_ws(gspread, sh, tab, header):
    """Return worksheet `tab`, migrating its header row to `header` if it drifted, or
    creating the tab (seeded with `header`) if it doesn't exist yet. Identical behavior to
    the get-or-create block that was duplicated across the tracking-sheet appenders."""
    try:
        ws = sh.worksheet(tab)
        cur = ws.row_values(1)
        if cur != header:
            # SAFE MIGRATION ONLY (June 28, 2026): relabel/extend the header in place only when the
            # existing header is a PREFIX of the new one -- i.e. columns were appended at the END, so
            # every already-written row still lines up. If columns were inserted or reordered mid-list,
            # overwriting A1 silently shifts every old row under the wrong labels (the KProp June 2026
            # incident). In that case, on a tab that already holds data, REFUSE and warn so it's caught,
            # not buried -- the fix is to archive/realign the old rows, then append new cols at the end.
            if header[:len(cur)] == cur or len(ws.col_values(1)) <= 1:
                ws.update('A1', [header])
            else:
                print(f"[sheet] {tab}: header changed mid-list with existing data — NOT overwriting A1 "
                      f"(would shift old rows). Reconcile that tab, then only append new columns at the END.")
        return ws
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab, rows=1000, cols=len(header))
        ws.append_row(header, value_input_option='USER_ENTERED')
        return ws


def append_to_sheet(flags):
    if not SHEETS_CREDS or not flags:
        return
    try:
        _, sh  = _open_sheet()
        ws     = sh.worksheet(SHEETS_TAB)
        existing      = ws.get_all_values()
        existing_keys = set()
        for row in existing[1:]:
            if len(row) >= 3:
                existing_keys.add(f"{row[0]}|{row[1]}|{row[2]}")
        today      = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
        rows_added = 0
        for f in flags:
            if f.get('signal') not in ('under', 'fg_under_only'):
                continue
            key = f"{today}|{f.get('game','')}|{f.get('batting_team','')}"
            if key in existing_keys:
                continue
            r = len(existing) + rows_added + 1
            tier_label = f.get('confidence_label', '')
            score_val  = f.get('ozzie_score', '')
            ws.append_row([
                today,
                f.get('game', ''),
                f.get('batting_team', ''),
                f.get('pitcher_name', ''),
                f.get('signal', ''),
                tier_label,
                score_val,
                f.get('std_from_mean', ''),
                f.get('pitcher_k_rate', ''),
                'Y' if f.get('q_away')             else 'N',
                'Y' if f.get('q_krate_hi')         else 'N',
                'Y' if f.get('q_bb_vlo')           else 'N',
                'Y' if f.get('q_gb_mid')           else 'N',
                'Y' if f.get('q_sigma_20')         else 'N',
                'Y' if f.get('fg_under_signal')    else 'N',
                'Y' if f.get('combined_f5_signal') else 'N',
                f.get('game_time', ''),
                '', '', '',
                '', '', '',
                '',
                f'=IF(AND(S{r}<>"",W{r}<>""),IF(W{r}<S{r},"Yes","No"),"")'.format(r=r),
                f'=IF(AND(V{r}<>"",W{r}<>""),IF(W{r}<V{r},"Yes","No"),"")'.format(r=r),
                f'=IF(W{r}="","",IF(T{r}<>"",IF(X{r}="Yes",IF(S{r}<0,T{r}*(100/ABS(S{r})),T{r}*(S{r}/100)),-T{r}),0)+IF(U{r}<>"",IF(Y{r}="Yes",IF(V{r}<0,U{r}*(100/ABS(V{r})),U{r}*(V{r}/100)),-U{r}),0))'.format(r=r),
            ], value_input_option='USER_ENTERED')
            existing_keys.add(key)
            rows_added += 1
        print(f"Google Sheets: {rows_added} rows added")
    except ImportError:
        print("gspread not installed — skipping sheet append")
    except Exception as e:
        print(f"Google Sheets error: {e}")


PQ_SHEET_TAB    = 'PitcherQuality'
PQ_SHEET_HEADER = [
    'date', 'game', 'batting_team', 'pitcher_name', 'pq_score', 'pq_percentile',
    'pq_k_rate', 'pq_bb_rate', 'pq_hr_rate', 'game_time',
    # one Line/Odds pair per tracked book (June 20, 2026), auto-filled from live odds when
    # available -- see SHEET_TRACKED_BOOKS/_book_line_odds. Replaces the old combined
    # 'books_lines'/'actual_f5_line' text columns, which couldn't separate line from price.
    'pinnacle_line', 'pinnacle_odds', 'draftkings_line', 'draftkings_odds',
    'thescore_line', 'thescore_odds',
    'actual_f5_runs', 'under_hit',    # auto-filled by backfill_sheet_outcomes() once F5 is final -- see below
    'game_id',                        # MLB gamePk, added June 19, 2026 -- lets backfill find the right boxscore
    'projected_k',                    # informational log5 K projection (June 23, 2026) -- forward ledger vs K props
    'current_bf',                     # pitcher's current-season BF behind the blend (June 23, 2026) -- < 20 means
                                      # mostly career prior (Bieber case), Telegram-suppressed; log it so stale rows
                                      # are filterable out of the track record (WHERE current_bf >= 20). Appended at
                                      # END so existing rows + backfill column positions are unaffected.
    'k_prop_flag',                    # K-prop tracking: K-composite >= 0.5839 (June 25, 2026). 2025 n=138
                                      # +0.71K/60%; 2026 OOS n=44 +0.67K/66%; combined p=0.0000. Eyeball K prop.
]


def append_pq_to_sheet(flags):
    """
    Tracking-only log for the Pitcher Quality Composite (NOT a validated bet signal — see
    PQ_WEIGHTS comment block). Separate tab from the main Under sheet because this signal's
    fields (and the manual line/outcome columns you'd fill in) don't match that sheet's schema.
    """
    if not SHEETS_CREDS or not flags:
        return
    try:
        gspread, sh = _open_sheet()
        ws = _get_or_create_ws(gspread, sh, PQ_SHEET_TAB, PQ_SHEET_HEADER)

        existing = ws.get_all_values()
        existing_keys = set()
        for row in existing[1:]:
            if len(row) >= 3:
                existing_keys.add(f"{row[0]}|{row[1]}|{row[2]}")
        today      = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
        rows_added = 0
        for f in flags:
            if not f.get('pq_q4'):
                continue
            key = f"{today}|{f.get('game','')}|{f.get('batting_team','')}"
            if key in existing_keys:
                continue
            team_odds = f.get('odds_lines') or {}
            book_cells = []
            for book_label in SHEET_TRACKED_BOOKS:
                line, odds = _book_line_odds(team_odds, book_label)
                book_cells.extend([line, odds])
            ws.append_row([
                today, f.get('game', ''), f.get('batting_team', ''), f.get('pitcher_name', ''),
                f.get('pq_score', ''), f.get('pq_percentile', ''),
                f.get('pq_k_rate', ''), f.get('pq_bb_rate', ''), f.get('pq_hr_rate', ''),
                f.get('game_time', ''),
                *book_cells,
                '', '',  # actual_f5_runs / under_hit — filled in later by backfill_sheet_outcomes()
                f.get('game_id', ''),
                f.get('projected_k', ''),
                f.get('pq_current_bf', ''),  # staleness column -- see PQ_SHEET_HEADER 'current_bf' note
                'yes' if f.get('k_prop_flag') else '',
            ], value_input_option='USER_ENTERED')
            existing_keys.add(key)
            rows_added += 1
        print(f"Google Sheets (PitcherQuality): {rows_added} rows added")
    except ImportError:
        print("gspread not installed — skipping PQ sheet append")
    except Exception as e:
        print(f"Google Sheets (PitcherQuality) error: {e}")


KPROP_SHEET_TAB    = 'KProp'
# Rebuilt June 27, 2026 for the over-favorite price-bias signal (replaced the leaked proj_gap
# columns). 'k_hit' now = 1 if actual_k > kprop_line (over wins). Kalshi/projected_k kept as
# reference. F5 team-total odds columns are the OPPOSING-OFFENSE F5 total context (sharp tier).
KPROP_SHEET_HEADER = [
    'date', 'game', 'batting_team', 'pitcher_name',
    'k_prop_tier',                   # 'base' or 'sharp' (opp F5 total < 2.0)
    'kprop_line',                    # strikeout line (strike) the over is bet at
    'kprop_dk_over',                 # DraftKings over price — signal classified here (-120..-160)
    'kprop_best_over',               # best over price across books — SHOP this
    'kprop_best_book',               # book offering the best over price
    'kprop_n_books',                 # # books priced
    'kprop_opp_f5',                  # opposing offense F5 team total (sharp if < 2.0)
    'kalshi_k_line',                 # REFERENCE only — Kalshi KXMLBKS implied line
    'projected_k',                   # REFERENCE only — log5 K projection
    'o_k_rate',                      # opposing lineup K-rate
    'p_k_rate', 'p_bb_rate', 'p_hr_rate',
    'game_time',
    'pinnacle_line', 'pinnacle_odds',
    'draftkings_line', 'draftkings_odds',
    'thescore_line', 'thescore_odds',
    'actual_k',                      # actual Ks thrown — fill in after game
    'k_hit',                         # 1 if actual_k > kprop_line (over won) — fill in after game
    'game_id',
    'kprop_prior_starts',            # season-to-date prior starts (SHARP needs >=7); appended last
                                     # to avoid shifting the manually-filled actual_k/k_hit columns
]


def append_kprop_to_sheet(flags):
    """Dedicated K-prop tracking tab. Logs every k_prop_flag=True start for manual follow-up."""
    if not SHEETS_CREDS or not flags:
        return
    try:
        gspread, sh = _open_sheet()
        ws = _get_or_create_ws(gspread, sh, KPROP_SHEET_TAB, KPROP_SHEET_HEADER)

        existing = ws.get_all_values()
        existing_keys = set()
        for row in existing[1:]:
            if len(row) >= 3:
                existing_keys.add(f"{row[0]}|{row[1]}|{row[2]}")
        today      = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
        rows_added = 0
        for f in flags:
            if not f.get('k_prop_flag'):
                continue
            key = f"{today}|{f.get('game','')}|{f.get('batting_team','')}"
            if key in existing_keys:
                continue
            team_odds = f.get('odds_lines') or {}
            def _lo(book):
                line, odds = _book_line_odds(team_odds, book)
                return [line, odds]
            ws.append_row([
                today, f.get('game', ''), f.get('batting_team', ''), f.get('pitcher_name', ''),
                f.get('k_prop_tier', ''),
                f.get('kprop_line', ''),
                f.get('kprop_dk_over', ''),
                f.get('kprop_best_over', ''),
                f.get('kprop_best_book', ''),
                f.get('kprop_n_books', ''),
                f.get('kprop_opp_f5', ''),
                f.get('kalshi_k_line', ''),
                f.get('projected_k', ''),
                round(f.get('o_k_rate') or 0, 4) or '',
                f.get('pq_k_rate', ''), f.get('pq_bb_rate', ''), f.get('pq_hr_rate', ''),
                f.get('game_time', ''),
                *_lo('pinnacle'), *_lo('draftkings'), *_lo('thescore'),
                '', '',  # actual_k / k_hit — filled manually after game
                f.get('game_id', ''),
                f.get('kprop_prior_starts', ''),
            ], value_input_option='USER_ENTERED')
            existing_keys.add(key)
            rows_added += 1
        print(f"Google Sheets (KProp): {rows_added} rows added")
    except ImportError:
        print("gspread not installed — skipping KProp sheet append")
    except Exception as e:
        print(f"Google Sheets (KProp) error: {e}")


OFF_SHEET_TAB    = 'OffenseQuality'
OFF_SHEET_HEADER = [
    'date', 'game', 'batting_team', 'pitcher_name', 'off_pctile', 'off_bb_rate',
    'game_time',
    # one Line/Odds pair per tracked book (June 20, 2026) -- see PQ_SHEET_HEADER note above
    'pinnacle_line', 'pinnacle_odds', 'draftkings_line', 'draftkings_odds',
    'thescore_line', 'thescore_odds',
    'actual_f5_runs', 'under_hit',  # auto-filled by backfill_sheet_outcomes() once F5 is final
    'game_id',                      # MLB gamePk, added June 19, 2026 -- see PQ_SHEET_HEADER note
]


def append_off_to_sheet(flags):
    """
    Tracking-only log for the Offense Quality Composite (NOT a validated bet signal — see
    OFF_PHANTOM comment block). Separate tab, same pattern as append_pq_to_sheet.
    """
    if not SHEETS_CREDS or not flags:
        return
    try:
        gspread, sh = _open_sheet()
        ws = _get_or_create_ws(gspread, sh, OFF_SHEET_TAB, OFF_SHEET_HEADER)

        existing = ws.get_all_values()
        existing_keys = set()
        for row in existing[1:]:
            if len(row) >= 3:
                existing_keys.add(f"{row[0]}|{row[1]}|{row[2]}")
        today      = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
        rows_added = 0
        for f in flags:
            if not f.get('off_q3_gate'):
                continue
            key = f"{today}|{f.get('game','')}|{f.get('batting_team','')}"
            if key in existing_keys:
                continue
            team_odds = f.get('odds_lines') or {}
            book_cells = []
            for book_label in SHEET_TRACKED_BOOKS:
                line, odds = _book_line_odds(team_odds, book_label)
                book_cells.extend([line, odds])
            ws.append_row([
                today, f.get('game', ''), f.get('batting_team', ''), f.get('pitcher_name', ''),
                f.get('off_pctile', ''), f.get('off_bb_rate', ''),
                f.get('game_time', ''),
                *book_cells,
                '', '',  # actual_f5_runs / under_hit — filled in later by backfill_sheet_outcomes()
                f.get('game_id', ''),
            ], value_input_option='USER_ENTERED')
            existing_keys.add(key)
            rows_added += 1
        print(f"Google Sheets (OffenseQuality): {rows_added} rows added")
    except ImportError:
        print("gspread not installed — skipping Offense Quality sheet append")
    except Exception as e:
        print(f"Google Sheets (OffenseQuality) error: {e}")


JOINT_SHEET_TAB    = 'JointOffense'
JOINT_SHEET_HEADER = [
    'date', 'game', 'home_team', 'away_team', 'joint_off_pctile',
    'joint_home_off_pctile', 'joint_away_off_pctile', 'game_time',
    # game-level signal (not per-team), so only the books that actually post the joint/combined
    # market matter -- FanDuel (the gate book) and Pinnacle (thin coverage, ~15% of events, but
    # worth recording when present). Same Line/Odds pair pattern as PQ_SHEET_HEADER, but OVER
    # price/line, not UNDER -- this signal bets over.
    'fanduel_line', 'fanduel_odds', 'pinnacle_line', 'pinnacle_odds',
    'actual_joint_f5_runs', 'over_hit',  # auto-filled by backfill_sheet_outcomes() once F5 is final
    'game_id',                           # MLB gamePk -- lets backfill find the right boxscore
]


def _joint_book_line_odds(joint_odds, book_label):
    """Same as _book_line_odds, but reads the OVER price (this signal bets over, not under)."""
    info  = (joint_odds or {}).get(book_label) or {}
    point = info.get('point', '')
    price = info.get('over')
    odds  = '' if price is None else (f"+{price}" if price > 0 else str(price))
    return point, odds


def append_joint_to_sheet(flags):
    """
    Tracking-only log for the joint-offense signal (NOT a validated bet signal -- see
    JOINT_OFF_PCTILE_MAX comment block). Separate tab, same pattern as append_pq_to_sheet/
    append_off_to_sheet, but game-level rows (batting_team is the 'Joint Total' sentinel, not
    a real team) and OVER-side odds, not under.
    """
    if not SHEETS_CREDS or not flags:
        return
    try:
        gspread, sh = _open_sheet()
        ws = _get_or_create_ws(gspread, sh, JOINT_SHEET_TAB, JOINT_SHEET_HEADER)

        existing = ws.get_all_values()
        existing_keys = set()
        for row in existing[1:]:
            if len(row) >= 2:
                existing_keys.add(f"{row[0]}|{row[1]}")  # date|game -- no batting_team, one row/game
        today      = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
        rows_added = 0
        for f in flags:
            if not f.get('joint_signal'):
                continue
            key = f"{today}|{f.get('game','')}"
            if key in existing_keys:
                continue
            joint_odds = f.get('joint_odds_lines') or {}
            fd_line, fd_odds = _joint_book_line_odds(joint_odds, JOINT_GATE_BOOK)
            pn_line, pn_odds = _joint_book_line_odds(joint_odds, PINNACLE_BOOK_LABEL)
            ws.append_row([
                today, f.get('game', ''), f.get('home_team', ''), f.get('away_team', ''),
                f.get('joint_off_pctile', ''),
                f.get('joint_home_off_pctile', ''), f.get('joint_away_off_pctile', ''),
                f.get('game_time', ''),
                fd_line, fd_odds, pn_line, pn_odds,
                '', '',  # actual_joint_f5_runs / over_hit — filled in later by backfill
                f.get('game_id', ''),
            ], value_input_option='USER_ENTERED')
            existing_keys.add(key)
            rows_added += 1
        print(f"Google Sheets (JointOffense): {rows_added} rows added")
    except ImportError:
        print("gspread not installed — skipping Joint Offense sheet append")
    except Exception as e:
        print(f"Google Sheets (JointOffense) error: {e}")


FG_TT_SHEET_TAB    = 'FgTeamTotal'
# Own book list (SHEET_TRACKED_BOOKS + Kalshi) so the FG tabs get the Kalshi line-shop column
# without adding a blank Kalshi column to the F5-scoped PQ/OFF tabs (which Kalshi F5 isn't pulled for).
FG_TT_SHEET_BOOKS  = SHEET_TRACKED_BOOKS + ['Kalshi']
FG_TT_SHEET_HEADER = [
    'date', 'game', 'batting_team', 'fielding_team', 'fg_bp_score', 'fg_bp_pctile',
    'fg_bp_n_relievers', 'game_time',
    'pinnacle_line', 'pinnacle_odds', 'draftkings_line', 'draftkings_odds',
    'thescore_line', 'thescore_odds', 'kalshi_line', 'kalshi_odds',
    'actual_fg_runs', 'under_hit',  # auto-filled by backfill once the full game is final
    'game_id',
]


def append_fg_tt_to_sheet(flags):
    """
    Tracking-only log for the FG (full-game) bullpen composite team-total under signal (June 22,
    2026 -- see FG_BP_PCTILE_GATE comment block). Same pattern as append_pq_to_sheet, per-team
    rows, under-side odds (this signal bets under), separate tab since the FULL-GAME line means
    different actual-outcome columns than the F5-scoped sheets.
    """
    if not SHEETS_CREDS or not flags:
        return
    try:
        gspread, sh = _open_sheet()
        ws = _get_or_create_ws(gspread, sh, FG_TT_SHEET_TAB, FG_TT_SHEET_HEADER)

        existing = ws.get_all_values()
        existing_keys = set()
        for row in existing[1:]:
            if len(row) >= 3:
                existing_keys.add(f"{row[0]}|{row[1]}|{row[2]}")
        today      = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
        rows_added = 0
        for f in flags:
            if f.get('signal') != 'fg_tt_under':
                continue
            key = f"{today}|{f.get('game','')}|{f.get('batting_team','')}"
            if key in existing_keys:
                continue
            fg_odds = f.get('fg_odds_lines') or {}
            book_cells = []
            for book_label in FG_TT_SHEET_BOOKS:
                line, odds = _book_line_odds(fg_odds, book_label)
                book_cells.extend([line, odds])
            ws.append_row([
                today, f.get('game', ''), f.get('batting_team', ''), f.get('fielding_team', ''),
                f.get('fg_bp_score', ''), f.get('fg_bp_pctile', ''), f.get('fg_bp_n_relievers', ''),
                f.get('game_time', ''),
                *book_cells,
                '', '',  # actual_fg_runs / under_hit — filled in later by backfill
                f.get('game_id', ''),
            ], value_input_option='USER_ENTERED')
            existing_keys.add(key)
            rows_added += 1
        print(f"Google Sheets (FgTeamTotal): {rows_added} rows added")
    except ImportError:
        print("gspread not installed — skipping FG Team Total sheet append")
    except Exception as e:
        print(f"Google Sheets (FgTeamTotal) error: {e}")


FG_JOINT_SHEET_TAB    = 'FgJointTotal'
FG_JOINT_SHEET_HEADER = [
    'date', 'game', 'home_team', 'away_team', 'direction', 'tier', 'units',
    'combined_bp_pctile', 'home_bp_pctile', 'away_bp_pctile', 'game_time',
    # game-level signal; the directional price (under for an under flag, over for an over flag)
    # is recorded per book. Same Line/Odds pair pattern as the other tabs.
    'draftkings_line', 'draftkings_odds', 'thescore_line', 'thescore_odds',
    'fanduel_line', 'fanduel_odds', 'pinnacle_line', 'pinnacle_odds',
    'kalshi_line', 'kalshi_odds',
    'actual_fg_joint_runs', 'hit',  # manual / future backfill once the full game is final
    'game_id',
]
# Books whose FG game total is recorded as its own Line/Odds column pair. DraftKings + theScore
# Bet are the two usable books the signal was validated on; FanDuel and Pinnacle recorded for
# reference; Kalshi (event exchange) added June 22, 2026 as a bettable line-shopping source.
FG_JOINT_SHEET_BOOKS = ['DraftKings', 'theScore Bet', 'FanDuel', 'Pinnacle', 'Kalshi']


def _fg_joint_book_line_odds(joint_odds, book_label, direction):
    """(point, directional_odds) for one book — under price for an under flag, over for over."""
    info  = (joint_odds or {}).get(book_label) or {}
    point = info.get('point', '')
    price = info.get('over' if direction == 'over' else 'under')
    odds  = '' if price is None else (f"+{price}" if price > 0 else str(price))
    return point, odds


def append_fg_joint_to_sheet(flags):
    """
    Tracking-only log for the FG joint/combined-total bullpen signal (June 22, 2026 -- see the
    FG_JOINT_* constant block). Game-level rows (batting_team is a 'Joint Total Under/Over'
    sentinel, not a real team), directional odds (under or over depending on the flag's
    fg_joint_direction). Separate tab since the whole-game combined line means different
    actual-outcome columns than every other sheet. Outcome columns left for manual/future
    backfill (same as FgTeamTotal -- full-game-runs backfill isn't wired yet).
    """
    if not SHEETS_CREDS or not flags:
        return
    try:
        gspread, sh = _open_sheet()
        ws = _get_or_create_ws(gspread, sh, FG_JOINT_SHEET_TAB, FG_JOINT_SHEET_HEADER)

        existing = ws.get_all_values()
        existing_keys = set()
        for row in existing[1:]:
            if len(row) >= 3:
                existing_keys.add(f"{row[0]}|{row[1]}|{row[4] if len(row) > 4 else ''}")
        today      = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
        rows_added = 0
        for f in flags:
            if f.get('signal') != 'fg_joint_total':
                continue
            direction = f.get('fg_joint_direction', '')
            key = f"{today}|{f.get('game','')}|{direction}"
            if key in existing_keys:
                continue
            joint_odds = f.get('fgj_odds_lines') or {}
            book_cells = []
            for book_label in FG_JOINT_SHEET_BOOKS:
                line, odds = _fg_joint_book_line_odds(joint_odds, book_label, direction)
                book_cells.extend([line, odds])
            ws.append_row([
                today, f.get('game', ''), f.get('home_team', ''), f.get('away_team', ''),
                direction, f.get('fg_joint_tier', ''), f.get('fg_joint_units', ''),
                f.get('fgj_combined_pctile', ''), f.get('fgj_home_bp_pctile', ''),
                f.get('fgj_away_bp_pctile', ''), f.get('game_time', ''),
                *book_cells,
                '', '',  # actual_fg_joint_runs / hit — manual or future backfill
                f.get('game_id', ''),
            ], value_input_option='USER_ENTERED')
            existing_keys.add(key)
            rows_added += 1
        print(f"Google Sheets (FgJointTotal): {rows_added} rows added")
    except ImportError:
        print("gspread not installed — skipping FG Joint Total sheet append")
    except Exception as e:
        print(f"Google Sheets (FgJointTotal) error: {e}")


OFF_FADE_SHEET_TAB    = 'FgJointOffense'
OFF_FADE_SHEET_HEADER = [
    'date', 'game', 'home_team', 'away_team', 'direction',
    'combined_off', 'home_off', 'away_off', 'game_time',
    'draftkings_line', 'draftkings_odds', 'thescore_line', 'thescore_odds',
    'fanduel_line', 'fanduel_odds', 'pinnacle_line', 'pinnacle_odds', 'kalshi_line', 'kalshi_odds',
    'actual_fg_joint_runs', 'hit',   # auto-backfilled by get_fg_runs_for_game (full-game final)
    'game_id',
    # Lineup current-season PA behind each side's offense score (June 23, 2026). min_pa_avg is the
    # weaker-data side; log it to filter stale-prior rows out of the track record (WHERE min_pa_avg
    # >= 20). Appended at END so existing rows + outcome-backfill column positions are unaffected.
    # (This was also a Telegram stale gate until off-fade left Telegram June 28, 2026 -- Sheets only now.)
    'home_pa_avg', 'away_pa_avg', 'min_pa_avg',
]


def append_off_fade_to_sheet(flags):
    """
    Tracking log for the offense-overpriced FG joint UNDER lead (June 22, 2026 — the one clean
    leak-free lead from the post-bullpen search). Game-level, always UNDER. Same shape as the
    FgJointTotal tab; outcome columns auto-backfill via get_fg_runs_for_game.
    """
    if not SHEETS_CREDS or not flags:
        return
    try:
        gspread, sh = _open_sheet()
        ws = _get_or_create_ws(gspread, sh, OFF_FADE_SHEET_TAB, OFF_FADE_SHEET_HEADER)

        existing = ws.get_all_values()
        existing_keys = {f"{r[0]}|{r[1]}" for r in existing[1:] if len(r) >= 2}
        today      = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
        rows_added = 0
        for f in flags:
            if f.get('signal') != 'fg_joint_off_under':
                continue
            key = f"{today}|{f.get('game','')}"
            if key in existing_keys:
                continue
            odds = f.get('ofj_odds_lines') or {}
            book_cells = []
            for book_label in FG_JOINT_SHEET_BOOKS:
                line, o = _fg_joint_book_line_odds(odds, book_label, 'under')
                book_cells.extend([line, o])
            ws.append_row([
                today, f.get('game', ''), f.get('home_team', ''), f.get('away_team', ''), 'under',
                f.get('ofj_combined_off', ''), f.get('ofj_home_off', ''), f.get('ofj_away_off', ''),
                f.get('game_time', ''),
                *book_cells,
                '', '',  # actual_fg_joint_runs / hit — auto-backfill
                f.get('game_id', ''),
                f.get('ofj_home_pa_avg', ''), f.get('ofj_away_pa_avg', ''), f.get('ofj_min_pa_avg', ''),
            ], value_input_option='USER_ENTERED')
            existing_keys.add(key)
            rows_added += 1
        print(f"Google Sheets (FgJointOffense): {rows_added} rows added")
    except ImportError:
        print("gspread not installed — skipping FG Joint Offense sheet append")
    except Exception as e:
        print(f"Google Sheets (FgJointOffense) error: {e}")


F5_OVER_SHEET_TAB    = 'F5TtOver'
F5_OVER_SHEET_HEADER = [
    'date', 'game', 'batting_team', 'f5_over_flagged_team', 'joint_off_pctile',
    'joint_fanduel_point', 'game_time',
    'pinnacle_line', 'pinnacle_odds', 'draftkings_line', 'draftkings_odds',
    'thescore_line', 'thescore_odds',
    'actual_f5_runs', 'over_hit',  # auto-filled by backfill once F5 is final
    'game_id',
]


def _book_line_odds_over(team_odds, book_label):
    """Same as _book_line_odds, but reads the OVER price -- this derived signal bets over."""
    info  = (team_odds or {}).get(book_label) or {}
    point = info.get('point', '')
    price = info.get('over')
    odds  = '' if price is None else (f"+{price}" if price > 0 else str(price))
    return point, odds


def append_f5_over_to_sheet(flags):
    """
    Tracking-only log for the derived F5 TT over signal (June 22, 2026 -- Zach's "other side"
    hypothesis, see notes above the f5_tt_over flag construction). New and untested -- per-team
    rows like append_pq_to_sheet, but OVER-side odds off the F5 per-team market (team_lines /
    ODDS_F5_MARKET), since this signal bets the OTHER team's F5 total to go over.
    """
    if not SHEETS_CREDS or not flags:
        return
    try:
        gspread, sh = _open_sheet()
        ws = _get_or_create_ws(gspread, sh, F5_OVER_SHEET_TAB, F5_OVER_SHEET_HEADER)

        existing = ws.get_all_values()
        existing_keys = set()
        for row in existing[1:]:
            if len(row) >= 3:
                existing_keys.add(f"{row[0]}|{row[1]}|{row[2]}")
        today      = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
        rows_added = 0
        for f in flags:
            if f.get('signal') != 'f5_tt_over':
                continue
            key = f"{today}|{f.get('game','')}|{f.get('batting_team','')}"
            if key in existing_keys:
                continue
            # this flag's batting_team is the "other side" team -- f5_over_odds_lines is already
            # THAT team's own F5 odds (embedded at construction time, see get_tracking_only_flags).
            team_odds = f.get('f5_over_odds_lines') or {}
            book_cells = []
            for book_label in SHEET_TRACKED_BOOKS:
                line, odds = _book_line_odds_over(team_odds, book_label)
                book_cells.extend([line, odds])
            ws.append_row([
                today, f.get('game', ''), f.get('batting_team', ''),
                f.get('f5_over_flagged_team', ''),
                f.get('joint_off_pctile', ''), f.get('joint_fanduel_point', ''),
                f.get('game_time', ''),
                *book_cells,
                '', '',  # actual_f5_runs / over_hit — filled in later by backfill
                f.get('game_id', ''),
            ], value_input_option='USER_ENTERED')
            existing_keys.add(key)
            rows_added += 1
        print(f"Google Sheets (F5TtOver): {rows_added} rows added")
    except ImportError:
        print("gspread not installed — skipping F5 TT Over sheet append")
    except Exception as e:
        print(f"Google Sheets (F5TtOver) error: {e}")


# ── LINE-HISTORY CAPTURE for CLOSING-LINE VALUE (June 23, 2026) ────────────────────────
# The historical line data is a single near-close snapshot per game (no open->close pair), so
# CLV -- does an edge survive to the close, do we need to bet early -- can't be measured
# retroactively. This logs every flagged game's line on each /api/notify run, but writes a new
# row only when the line/price actually CHANGES vs what we last saw that day, so the result is a
# compact movement log: the first row per (game, market, book) is the "open" we observed, the
# last is the "close". Append-only to its own tab, fully isolated + try/except-guarded so it can
# never affect the live picks/notify flow. Day's change-state lives in ONE Redis key (1 GET + at
# most 1 SET per run) to avoid per-game Redis chatter. NOTE: live odds are Redis-cached
# ODDS_API_TTL (4h), so effective movement resolution is ~4h regardless of cron cadence -- enough
# to capture open vs near-close, not minute-by-minute. Analyze later by joining game_id to the
# per-signal sheets (which record which signal flagged the game).
LINEHIST_SHEET_TAB = 'LineHistory'
LINEHIST_HEADER = ['ts_utc', 'date', 'game_id', 'game', 'market', 'book', 'point', 'under', 'over', 'game_time']
# (odds-field name on the flag dict) -> market tag written to the sheet
LINEHIST_ODDS_FIELDS = [
    ('odds_lines',         'F5_TT'),         # sigma-under / pq_q4 / off_q3 -- F5 per-team total, under side
    ('fg_odds_lines',      'FG_TT'),         # fg_tt_under -- full-game per-team total
    ('joint_odds_lines',   'F5_JOINT'),      # joint_offense -- F5 combined total, over side
    ('f5_over_odds_lines', 'F5_TT_OVER'),    # f5_tt_over -- "other side" team's F5 total
    ('fgj_odds_lines',     'FG_JOINT'),      # fg_joint_total -- full-game combined total
    ('ofj_odds_lines',     'FG_JOINT_FADE'), # fg_joint_off_under -- offense-fade on FG combined total
]


def append_line_history(flags):
    """Best-effort line-movement log for CLV. Writes a row only when a (game, market, book)
    line/price changes vs the last value seen today. Never raises into the caller."""
    if not SHEETS_CREDS or not flags:
        return
    try:
        import gspread
        import json as _json
        from google.oauth2.service_account import Credentials
        today   = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
        now_utc = datetime.now(pytz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

        state_key = f"ozzie:lh_state:{today}"
        raw   = redis_get(state_key)
        state = _json.loads(raw) if raw else {}

        new_rows = []
        for f in flags:
            gid   = f.get('game_id', '')
            game  = f.get('game', '') or f"{f.get('away_team', '')} @ {f.get('home_team', '')}"
            gtime = f.get('game_time', '')
            for field, market in LINEHIST_ODDS_FIELDS:
                for book, info in (f.get(field) or {}).items():
                    pt, u, o = info.get('point'), info.get('under'), info.get('over')
                    if pt is None and u is None and o is None:
                        continue
                    k, val = f"{gid}|{market}|{book}", f"{pt}|{u}|{o}"
                    if state.get(k) == val:
                        continue
                    state[k] = val
                    new_rows.append([now_utc, today, gid, game, market, book,
                                     pt if pt is not None else '', u if u is not None else '',
                                     o if o is not None else '', gtime])
        if not new_rows:
            return

        creds_dict = _json.loads(SHEETS_CREDS)
        scopes = ['https://www.googleapis.com/auth/spreadsheets',
                  'https://www.googleapis.com/auth/drive']
        creds  = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc     = gspread.authorize(creds)
        sh     = gc.open_by_key(SHEETS_ID)
        try:
            ws = sh.worksheet(LINEHIST_SHEET_TAB)
            if ws.row_values(1) != LINEHIST_HEADER:
                ws.update('A1', [LINEHIST_HEADER])
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=LINEHIST_SHEET_TAB, rows=2000, cols=len(LINEHIST_HEADER))
            ws.append_row(LINEHIST_HEADER, value_input_option='USER_ENTERED')

        if hasattr(ws, 'append_rows'):
            ws.append_rows(new_rows, value_input_option='USER_ENTERED')
        else:  # very old gspread fallback
            for row in new_rows:
                ws.append_row(row, value_input_option='USER_ENTERED')
        redis_set(state_key, _json.dumps(state), ex=259200)  # 3 days
        print(f"Google Sheets (LineHistory): {len(new_rows)} rows added")
    except ImportError:
        print("gspread not installed — skipping LineHistory append")
    except Exception as e:
        print(f"Google Sheets (LineHistory) error: {e}")


# ── OUTCOME BACKFILL (June 19, 2026) ──────────────────────────────────────────────────
# Until now, actual_f5_runs/under_hit on both tracking sheets were 100% manual -- Zach had to
# look up each game's score and type it in. This fills them in automatically from MLB Stats
# API's linescore endpoint, keyed off the game_id column added the same session. Only grades
# rows where actual_f5_line is confirmed 1.5 (true for every row going forward now that the
# Pinnacle gate only lets 1.5-line flags through at all -- see PINNACLE GATE notes above
# get_odds_api_lines) -- older manually-entered rows without a game_id are left alone.

def get_f5_runs_for_game(game_id):
    """
    Returns {'home': int, 'away': int} runs scored through the first 5 innings for the given
    MLB gamePk, or None if 5 complete innings aren't recorded yet for both sides. Deliberately
    does NOT check overall game status -- the F5 score is locked in as soon as inning 5 is
    complete, regardless of whether the rest of the game has finished (or even started).
    """
    if not game_id:
        return None
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1/game/{game_id}/linescore", timeout=15)
        if not r.ok:
            return None
        data = r.json()
    except Exception as e:
        print(f"Linescore fetch error (game {game_id}): {e}")
        return None

    home_runs, away_runs, innings_seen = 0, 0, 0
    for inning in data.get('innings', []):
        if inning.get('num', 0) > 5:
            continue
        home, away = inning.get('home', {}), inning.get('away', {})
        if 'runs' not in home or 'runs' not in away:
            return None  # this inning is still in progress -- not safe to trust yet
        home_runs += home['runs']
        away_runs += away['runs']
        innings_seen += 1
    if innings_seen < 5:
        return None
    return {'home': home_runs, 'away': away_runs}


def get_fg_runs_for_game(game_id):
    """
    Returns {'home': int, 'away': int} FULL-GAME runs for a game that is FINAL, or None if the
    game isn't final yet (or on error). Unlike F5 (which locks once inning 5 is complete), a
    full-game total isn't safe to grade until the game is actually over (extras, rain-shortening),
    so this gates on status == 'Final'. The schedule endpoint carries BOTH the final score and the
    game status in one lightweight call, so no separate status fetch is needed. 'Final' also covers
    rain-shortened "Completed Early" games (a valid, gradeable final result).
    """
    if not game_id:
        return None
    try:
        r = requests.get("https://statsapi.mlb.com/api/v1/schedule",
                         params={'sportId': 1, 'gamePks': game_id}, timeout=15)
        if not r.ok:
            return None
        data = r.json()
    except Exception as e:
        print(f"Schedule fetch error (game {game_id}): {e}")
        return None
    for d in data.get('dates', []):
        for g in d.get('games', []):
            if str(g.get('gamePk')) != str(game_id):
                continue
            if (g.get('status') or {}).get('abstractGameState') != 'Final':
                return None
            teams = g.get('teams', {})
            home = teams.get('home', {}).get('score')
            away = teams.get('away', {}).get('score')
            if home is None or away is None:
                return None
            return {'home': int(home), 'away': int(away)}
    return None


def get_pitcher_ks_for_game(game_id, pitcher_name):
    """Actual strikeouts thrown by `pitcher_name` in FINAL game `game_id`, or None if the game
    isn't Final yet, the pitcher isn't in the boxscore, or on error. Gated on Final (like
    get_fg_runs_for_game) so a still-active starter never writes a PARTIAL K count -- once the
    actual_k cell is filled, backfill skips the row forever, so a half-game number would stick.
    Names matched via _kprop_norm_name (KProp stores 'Last, First'; the boxscore gives 'First Last')."""
    if not game_id or not pitcher_name:
        return None
    # 1) only grade finished games (schedule carries status in one light call)
    try:
        r = requests.get("https://statsapi.mlb.com/api/v1/schedule",
                         params={'sportId': 1, 'gamePks': game_id}, timeout=15)
        if not r.ok:
            return None
        sched = r.json()
    except Exception as e:
        print(f"K backfill schedule error (game {game_id}): {e}")
        return None
    is_final = any(str(g.get('gamePk')) == str(game_id)
                   and (g.get('status') or {}).get('abstractGameState') == 'Final'
                   for d in sched.get('dates', []) for g in d.get('games', []))
    if not is_final:
        return None
    # 2) find the pitcher in the boxscore and return his strikeouts
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore", timeout=15)
        if not r.ok:
            return None
        box = r.json()
    except Exception as e:
        print(f"K backfill boxscore error (game {game_id}): {e}")
        return None
    target = _kprop_norm_name(pitcher_name)
    for side in ('home', 'away'):
        players = ((box.get('teams', {}).get(side, {}) or {}).get('players', {})) or {}
        for p in players.values():
            full = (p.get('person') or {}).get('fullName', '')
            if _kprop_norm_name(full) != target:
                continue
            pitching = (p.get('stats') or {}).get('pitching') or {}
            so = pitching.get('strikeOuts')
            if so is None:          # name matched a non-pitcher entry -- keep looking
                continue
            return int(so)
    return None


def _backfill_tab(sh, gspread, tab, *, line_col, runs_col, hit_col, fetch, mode, direction_col=None):
    """
    Generic outcome backfill for one tracking tab. Fills runs_col + hit_col for any row that has a
    game_id and a recorded line but no outcome yet, skipping rows whose game isn't final/safe yet
    (fetch returns None). Returns count filled.
      fetch(game_id) -> {'home': int, 'away': int} or None (e.g. get_fg_runs_for_game for full-game,
        get_f5_runs_for_game for first-5).
      mode='team_under'/'team_over' -> per-team row (uses 'game' = "away@home" + 'batting_team' to
        pick the side), graded under/over vs line_col.
      mode='joint' -> game-level row, value = home+away, graded by the row's direction_col
        ('under'/'over') vs line_col.
    """
    try:
        ws = sh.worksheet(tab)
    except gspread.exceptions.WorksheetNotFound:
        return 0
    except Exception as e:
        print(f"Backfill ({tab}): worksheet lookup error: {e}")
        return 0
    try:
        rows = ws.get_all_values()
    except Exception as e:
        print(f"Backfill ({tab}): read error: {e}")
        return 0
    if len(rows) < 2:
        return 0
    hdr = rows[0]
    try:
        i_game   = hdr.index('game')
        i_line   = hdr.index(line_col)
        i_runs   = hdr.index(runs_col)
        i_hit    = hdr.index(hit_col)
        i_gameid = hdr.index('game_id')
        i_team   = hdr.index('batting_team') if mode in ('team_under', 'team_over') else None
        i_dir    = hdr.index(direction_col) if direction_col else None
    except ValueError:
        print(f"Backfill ({tab}): header missing expected columns (run an append first to migrate), skipping")
        return 0

    def cell(row, i):
        return row[i] if (i is not None and i < len(row)) else ''

    filled = 0
    for r_idx, row in enumerate(rows[1:], start=2):
        if cell(row, i_runs):                 # already filled
            continue
        line_str = cell(row, i_line)
        if not line_str:                      # no recorded line to grade against
            continue
        try:
            line_val = float(line_str)
        except ValueError:
            continue
        game_id = cell(row, i_gameid)
        if not game_id:
            continue
        res = fetch(game_id)
        if res is None:                       # not final / not safe to grade yet
            continue
        if mode == 'joint':
            value = res['home'] + res['away']
            direction = cell(row, i_dir)
            hit = ('Yes' if value < line_val else 'No') if direction == 'under' \
                  else ('Yes' if value > line_val else 'No')
        else:
            game_str = cell(row, i_game)
            if '@' not in game_str:
                continue
            away, home = game_str.split('@', 1)
            team_abb = NAME_TO_ABB.get(cell(row, i_team), cell(row, i_team))
            is_home = NAME_TO_ABB.get(home, home) == team_abb
            is_away = NAME_TO_ABB.get(away, away) == team_abb
            if not is_home and not is_away:
                continue
            value = res['home'] if is_home else res['away']
            hit = ('Yes' if value < line_val else 'No') if mode == 'team_under' \
                  else ('Yes' if value > line_val else 'No')
        try:
            ws.update_cell(r_idx, i_runs + 1, value)
            ws.update_cell(r_idx, i_hit + 1, hit)
            filled += 1
        except Exception as e:
            print(f"Backfill ({tab}): write error on row {r_idx}: {e}")
    print(f"Backfill ({tab}): {filled} row(s) filled")
    return filled


def backfill_sheet_outcomes():
    """
    Fills in actual_f5_runs/under_hit on PitcherQuality and OffenseQuality for any row with a
    known game_id, an actual_f5_line of 1.5, and no outcome yet. Safe to call repeatedly/on a
    schedule (e.g. cron-job.org, same pattern as /api/notify) -- rows already filled, or whose
    F5 score isn't final yet, are simply skipped each time. Returns a summary dict for logging.
    """
    summary = {'pq_filled': 0, 'off_filled': 0, 'joint_filled': 0,
               'fg_tt_filled': 0, 'fg_joint_filled': 0, 'f5_over_filled': 0, 'off_fade_filled': 0,
               'kprop_filled': 0}
    if not SHEETS_CREDS:
        return summary
    try:
        gspread, sh = _open_sheet()
    except ImportError:
        print("gspread not installed — skipping outcome backfill")
        return summary
    except Exception as e:
        print(f"Backfill: Sheets auth error: {e}")
        return summary

    for tab, summary_key in ((PQ_SHEET_TAB, 'pq_filled'), (OFF_SHEET_TAB, 'off_filled')):
        try:
            ws = sh.worksheet(tab)
        except gspread.exceptions.WorksheetNotFound:
            continue
        try:
            rows = ws.get_all_values()
        except Exception as e:
            print(f"Backfill ({tab}): read error: {e}")
            continue
        if len(rows) < 2:
            continue
        hdr = rows[0]
        try:
            i_game   = hdr.index('game')
            i_team   = hdr.index('batting_team')
            # pinnacle_line replaces the old combined 'actual_f5_line' column (June 20, 2026).
            # Grades against whatever line is actually recorded here, not a hardcoded 1.5 --
            # since the park gate exception (same day) lets flags through at 2.5+ in a
            # hitter-favorable park too, so a confirmed line can legitimately be e.g. 2.5.
            i_line   = hdr.index('pinnacle_line')
            i_runs   = hdr.index('actual_f5_runs')
            i_hit    = hdr.index('under_hit')
            i_gameid = hdr.index('game_id')
        except ValueError:
            print(f"Backfill ({tab}): header missing expected columns (run an append first to migrate it), skipping")
            continue

        def cell(row, i):
            return row[i] if i < len(row) else ''

        filled = 0
        for r_idx, row in enumerate(rows[1:], start=2):
            if cell(row, i_runs):            # already filled
                continue
            line_str = cell(row, i_line)
            if not line_str:                 # no confirmed Pinnacle line to grade against
                continue
            try:
                line_val = float(line_str)
            except ValueError:
                continue
            game_id = cell(row, i_gameid)
            if not game_id:
                continue
            game_str = cell(row, i_game)
            if '@' not in game_str:
                continue
            away, home = game_str.split('@', 1)
            team_abb = NAME_TO_ABB.get(cell(row, i_team), cell(row, i_team))
            is_home  = (NAME_TO_ABB.get(home, home) == team_abb)
            is_away  = (NAME_TO_ABB.get(away, away) == team_abb)
            if not is_home and not is_away:
                continue
            f5 = get_f5_runs_for_game(game_id)
            if f5 is None:
                continue
            runs      = f5['home'] if is_home else f5['away']
            under_hit = 'Yes' if runs < line_val else 'No'
            try:
                ws.update_cell(r_idx, i_runs + 1, runs)
                ws.update_cell(r_idx, i_hit + 1, under_hit)
                filled += 1
            except Exception as e:
                print(f"Backfill ({tab}): write error on row {r_idx}: {e}")
        summary[summary_key] = filled
        print(f"Backfill ({tab}): {filled} row(s) filled")

    # ── JointOffense backfill ── different shape from the loop above: one row per GAME, not
    # per team (no batting_team column), grades OVER not under, and runs = home + away summed
    # (get_f5_runs_for_game already returns both sides -- this is the only consumer that needs
    # the sum instead of picking one side).
    try:
        ws = sh.worksheet(JOINT_SHEET_TAB)
    except gspread.exceptions.WorksheetNotFound:
        ws = None
    except Exception as e:
        print(f"Backfill ({JOINT_SHEET_TAB}): worksheet lookup error: {e}")
        ws = None

    if ws is not None:
        try:
            rows = ws.get_all_values()
        except Exception as e:
            print(f"Backfill ({JOINT_SHEET_TAB}): read error: {e}")
            rows = []
        if len(rows) >= 2:
            hdr = rows[0]
            try:
                i_line   = hdr.index('fanduel_line')
                i_runs   = hdr.index('actual_joint_f5_runs')
                i_hit    = hdr.index('over_hit')
                i_gameid = hdr.index('game_id')

                def cell(row, i):
                    return row[i] if i < len(row) else ''

                filled = 0
                for r_idx, row in enumerate(rows[1:], start=2):
                    if cell(row, i_runs):                # already filled
                        continue
                    line_str = cell(row, i_line)
                    if not line_str:                     # no confirmed FanDuel line to grade against
                        continue
                    try:
                        line_val = float(line_str)
                    except ValueError:
                        continue
                    game_id = cell(row, i_gameid)
                    if not game_id:
                        continue
                    f5 = get_f5_runs_for_game(game_id)
                    if f5 is None:
                        continue
                    joint_runs = f5['home'] + f5['away']
                    over_hit   = 'Yes' if joint_runs > line_val else 'No'
                    try:
                        ws.update_cell(r_idx, i_runs + 1, joint_runs)
                        ws.update_cell(r_idx, i_hit + 1, over_hit)
                        filled += 1
                    except Exception as e:
                        print(f"Backfill ({JOINT_SHEET_TAB}): write error on row {r_idx}: {e}")
                summary['joint_filled'] = filled
                print(f"Backfill ({JOINT_SHEET_TAB}): {filled} row(s) filled")
            except ValueError:
                print(f"Backfill ({JOINT_SHEET_TAB}): header missing expected columns (run an "
                      f"append first to migrate it), skipping")

    # ── FULL-GAME tabs (June 22, 2026) ── now auto-backfilled via get_fg_runs_for_game (final
    # games only). FgTeamTotal = per-team full-game under (graded vs the gate's pinnacle_line);
    # FgJointTotal = game-level combined total, under OR over per the row's direction (graded vs
    # draftkings_line, the gate book). F5TtOver = per-team F5 over (graded vs pinnacle_line, F5 runs).
    summary['fg_tt_filled'] = _backfill_tab(
        sh, gspread, FG_TT_SHEET_TAB, line_col='pinnacle_line', runs_col='actual_fg_runs',
        hit_col='under_hit', fetch=get_fg_runs_for_game, mode='team_under')
    summary['fg_joint_filled'] = _backfill_tab(
        sh, gspread, FG_JOINT_SHEET_TAB, line_col='draftkings_line', runs_col='actual_fg_joint_runs',
        hit_col='hit', fetch=get_fg_runs_for_game, mode='joint', direction_col='direction')
    summary['f5_over_filled'] = _backfill_tab(
        sh, gspread, F5_OVER_SHEET_TAB, line_col='pinnacle_line', runs_col='actual_f5_runs',
        hit_col='over_hit', fetch=get_f5_runs_for_game, mode='team_over')
    summary['off_fade_filled'] = _backfill_tab(
        sh, gspread, OFF_FADE_SHEET_TAB, line_col='draftkings_line', runs_col='actual_fg_joint_runs',
        hit_col='hit', fetch=get_fg_runs_for_game, mode='joint', direction_col='direction')

    # ── KProp backfill (June 28, 2026) ── per-pitcher strikeout outcomes, the K-prop analog of the
    # team-total backfills above. Fills actual_k from the FINAL boxscore and k_hit = 1 if the over
    # won (actual_k > kprop_line) else 0. Self-maintaining on the same cron; rows already filled, or
    # whose game isn't Final yet, are skipped each pass. Grades every row (base and sharp tiers alike).
    try:
        ws = sh.worksheet(KPROP_SHEET_TAB)
    except gspread.exceptions.WorksheetNotFound:
        ws = None
    except Exception as e:
        print(f"Backfill ({KPROP_SHEET_TAB}): worksheet lookup error: {e}")
        ws = None
    if ws is not None:
        try:
            rows = ws.get_all_values()
        except Exception as e:
            print(f"Backfill ({KPROP_SHEET_TAB}): read error: {e}")
            rows = []
        if len(rows) >= 2:
            hdr = rows[0]
            try:
                i_pitcher = hdr.index('pitcher_name')
                i_line    = hdr.index('kprop_line')
                i_k       = hdr.index('actual_k')
                i_hit     = hdr.index('k_hit')
                i_gameid  = hdr.index('game_id')
            except ValueError:
                print(f"Backfill ({KPROP_SHEET_TAB}): header missing expected columns, skipping")
            else:
                def cell(row, i):
                    return row[i] if i < len(row) else ''
                filled = 0
                for r_idx, row in enumerate(rows[1:], start=2):
                    if cell(row, i_k):                       # already filled
                        continue
                    line_str = cell(row, i_line)
                    game_id  = cell(row, i_gameid)
                    pitcher  = cell(row, i_pitcher)
                    if not line_str or not game_id or not pitcher:
                        continue
                    try:
                        line_val = float(line_str)
                    except ValueError:
                        continue
                    ks = get_pitcher_ks_for_game(game_id, pitcher)
                    if ks is None:                           # game not Final / pitcher not found
                        continue
                    k_hit = 1 if ks > line_val else 0
                    try:
                        ws.update_cell(r_idx, i_k + 1, ks)
                        ws.update_cell(r_idx, i_hit + 1, k_hit)
                        filled += 1
                    except Exception as e:
                        print(f"Backfill ({KPROP_SHEET_TAB}): write error on row {r_idx}: {e}")
                summary['kprop_filled'] = filled
                print(f"Backfill ({KPROP_SHEET_TAB}): {filled} row(s) filled")

    return summary


@app.route('/api/backfill-outcomes')
def api_backfill_outcomes():
    secret = request.args.get('secret', '')
    if not NOTIFY_SECRET or secret != NOTIFY_SECRET:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        summary = backfill_sheet_outcomes()
        return jsonify({'status': 'ok', **summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/notify')
def api_notify():
    secret = request.args.get('secret', '')
    if not NOTIFY_SECRET or secret != NOTIFY_SECRET:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        # TRACKING-ONLY MODE (see comment above get_tracking_only_flags): Drive loading paused.
        today   = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
        games   = get_lineups_and_starters(today)
        flags   = get_tracking_only_flags(games)

        # PICKS CACHE / NOTIFY SYNC: refresh /api/picks' Redis cache with what was just computed
        # here, so the dashboard reflects a new flag immediately instead of waiting up to 30 min
        # for that route's own cache to expire on its own schedule (or, worse, never -- if the
        # live Pinnacle gate closes again before the next natural expiry, the dashboard would
        # otherwise never show a flag that did fire in Telegram). Best-effort: a failure here
        # should not block the Telegram send below.
        try:
            import json as _json
            picks_payload = build_picks_payload(today, games, flags)
            ttl = picks_cache_ttl()
            if ttl > 0 and flags:
                redis_set(f"ozzie:picks:{today}", _json.dumps(picks_payload), ex=ttl)
        except Exception as cache_err:
            print(f"Picks cache refresh from notify failed: {cache_err}")

        # VISIBILITY vs TRACKING, per Zach (June 22, 2026; updated June 28, 2026): every signal
        # below still gets logged to its own Sheet tab every cycle (see the unconditional
        # append_*_to_sheet calls further down) -- the "new_*" lists are NOT gated on whether they
        # appear in Telegram, only on the existing per-flag dedup (already_sent). The Telegram
        # MESSAGE is now K-PROP-FOCUSED: it includes kprop/pq/f5_over only. under (F5 TT U1.5) and
        # off-fade (joint offense fade) were dropped from the message June 28, 2026 -- joining off
        # and joint, which left June 22 -- all still computed and logged exactly as before (see
        # build_picks_payload
        # for the matching dashboard-side suppression).
        under_flags  = [f for f in flags if f.get('signal') == 'under']
        pq_flags     = [f for f in flags if f.get('pq_q4')]
        kprop_flags  = [f for f in flags if f.get('k_prop_flag')]
        off_flags    = [f for f in flags if f.get('off_q3_gate')]
        joint_flags  = [f for f in flags if f.get('joint_signal')]
        fg_tt_flags  = [f for f in flags if f.get('signal') == 'fg_tt_under']
        f5_over_flags = [f for f in flags if f.get('signal') == 'f5_tt_over']
        fg_joint_flags = [f for f in flags if f.get('signal') == 'fg_joint_total']
        off_fade_flags = [f for f in flags if f.get('signal') == 'fg_joint_off_under']
        if not (under_flags or pq_flags or kprop_flags or off_flags or joint_flags or fg_tt_flags
                or f5_over_flags or fg_joint_flags or off_fade_flags):
            return jsonify({'status': 'ok', 'new': 0, 'message': 'No flags today'})

        # CLV line-history capture (best-effort, see append_line_history). Runs on the FULL flag
        # set BEFORE the new-flag dedup below, so it still records the CLOSING line on later runs
        # when every flag is "already sent" and the route returns early at "No new flags".
        append_line_history(under_flags + pq_flags + off_flags + joint_flags + fg_tt_flags
                            + f5_over_flags + fg_joint_flags + off_fade_flags)

        redis_key    = f"ozzie:notified:{today}"
        existing_raw = redis_get(redis_key)
        already_sent = set(existing_raw.split(',')) if existing_raw else set()
        # Sharp K-prop pings use their OWN dedup namespace so that (a) a base sighting -- or a
        # different signal sharing this game|team flag_key -- never locks out a later base->sharp
        # upgrade, and (b) sharp still fires exactly once. Base picks stay Telegram-silent
        # (Sheet/app only) and are deliberately NOT tracked here; they never ping, so there is
        # nothing to dedup on them.
        ksharp_key   = f"ozzie:kprop_sharp_notified:{today}"
        ksharp_raw   = redis_get(ksharp_key)
        ksharp_sent  = set(ksharp_raw.split(',')) if ksharp_raw else set()
        if ksharp_raw is None:
            # First notify run since this namespace was introduced (deploy day): carry over any
            # sharp picks already announced today under the old shared dedup, so the switch-over
            # doesn't re-ping them once.
            ksharp_sent |= {flag_key(f) for f in kprop_flags
                            if f.get('k_prop_tier') == 'sharp' and flag_key(f) in already_sent}
        new_under   = [f for f in under_flags if flag_key(f) not in already_sent]
        new_pq      = [f for f in pq_flags if flag_key(f) not in already_sent]
        new_kprop   = [f for f in kprop_flags if flag_key(f) not in already_sent]
        new_off     = [f for f in off_flags if flag_key(f) not in already_sent]
        new_joint   = [f for f in joint_flags if flag_key(f) not in already_sent]
        new_fg_tt   = [f for f in fg_tt_flags if flag_key(f) not in already_sent]
        new_f5_over = [f for f in f5_over_flags if flag_key(f) not in already_sent]
        new_fg_joint = [f for f in fg_joint_flags if flag_key(f) not in already_sent]
        new_off_fade = [f for f in off_fade_flags if flag_key(f) not in already_sent]
        # Sharp K-prop upgrades dedup independently (ksharp_sent) -- computed off the FULL
        # kprop_flags list (not already_sent-filtered new_kprop) so a pick that opened base
        # (its flag_key already in already_sent, Telegram-silent) can still fire its first
        # sharp ping. Added to the new-flag guard so a base->sharp upgrade doesn't early-return.
        new_kprop_sharp = [f for f in kprop_flags
                           if f.get('k_prop_tier') == 'sharp' and flag_key(f) not in ksharp_sent]
        if not (new_under or new_pq or new_kprop or new_off or new_joint or new_fg_tt
                or new_f5_over or new_fg_joint or new_off_fade or new_kprop_sharp):
            return jsonify({'status': 'ok', 'new': 0, 'message': 'No new flags'})

        # ⚾ (not 🎯) on the top header so 🎯 is reserved exclusively for K-props (the focus) --
        # K-Prop section header, each K bet line (_kprop_tg_bet), and the 🎯K-PROP↑ cross-ref tag.
        lines = [f"⚾ <b>Ozzie — {today}</b>"]

        # ── CONSOLIDATED K-PROP SECTION (top) — SHARP-ONLY in Telegram as of June 29, 2026 per Zach.
        # The open/close + matchup analysis showed the band edge is a strong-pitching-matchup edge:
        # SHARP (opp F5 total <=1.5) carries it (~+18% 2026, both months), while base (opp F5 >=2.5)
        # band plays were net dead (positive May, negative June). So only SHARP fires here. Base picks
        # are NOT muted from tracking -- append_kprop_to_sheet(new_kprop) below still logs ALL of them
        # (sharp + base) to the KProp tab unfiltered, so the base forward-record keeps accumulating.
        # new_kprop_sharp is computed above (off kprop_flags via ksharp_sent) so base->sharp upgrades
        # survive the already_sent dedup that new_kprop is subject to.
        if new_kprop_sharp:
            lines.append("🎯 <b>K-Prop OVER · SHARP only — TRACKING ONLY (not yet a validated bet signal)</b>")
            lines.append(f"{len(new_kprop_sharp)} SHARP start(s) — over-favorite price + strong matchup "
                         "(DK over -120..-160, opp F5 total &lt;2.0, ≥7 prior starts). SHARP is where the edge "
                         "lives (~+18% 2026, both months); base plays log to the tracker only. Shop best over price.\n")
            for f in new_kprop_sharp:
                time = f" — {f['game_time']}" if f.get('game_time') else ''
                also = ' · also 📊Q4' if f.get('pq_q4') else ''
                lines.append(f"<b>{f['pitcher_name']}</b> (vs {f['batting_team']}){also} · "
                             f"{_kprop_tg_bet(f)}{time}")

        # NOTE: the F5 TT U1.5 UNDER block was removed from Telegram June 28, 2026 per Zach
        # (focusing notifications on K props) -- still computed above and still logged to its
        # Sheet tab below (append_to_sheet(new_under), unfiltered), just no longer pushed to
        # Telegram. Same VISIBILITY-vs-TRACKING split as off/joint (see note above new_under).

        # PQ_BF_STALE_THRESHOLD matches the dashboard's bfStale check (templates/index.html,
        # "Current-season BF: N -- mostly/entirely stale prior, do not trust"). Per Zach
        # (June 23, 2026, prompted by Shane Bieber's season debut firing a Q4 flag with zero
        # current-season innings behind it): the dashboard already shows that warning inline,
        # but Telegram had no equivalent check and sent a plain, unqualified flag. Rather than
        # add the warning text to Telegram too, suppress these from the NOTIFICATION entirely --
        # a flag that's "mostly/entirely stale prior" isn't actionable, and Zach is the only
        # consumer of these alerts (no audience-visibility reason to send it anyway). Still
        # computed/logged to the PitcherQuality Sheet tab below (new_pq, unfiltered) and still
        # visible on the dashboard (which already carries its own inline warning) -- only the
        # Telegram push is gated.
        PQ_BF_STALE_THRESHOLD = 20
        new_pq_for_telegram = [f for f in new_pq if (f.get('pq_current_bf') or 0) >= PQ_BF_STALE_THRESHOLD]
        new_pq_stale = [f for f in new_pq if f not in new_pq_for_telegram]

        if new_pq_for_telegram:
            if len(lines) > 1:
                lines.append("")
            lines.append(f"📊 <b>Pitcher Quality — TRACKING ONLY, not a bet signal</b>")
            lines.append(f"{len(new_pq_for_telegram)} Q4 pitcher(s) — only means something if <b>Pinnacle</b> below "
                         f"shows ✅1.5 (or 🏟️ = validated park exception, thinner sample — see pq_note)\n")
            for f in new_pq_for_telegram:
                pct  = f.get('pq_percentile')
                time = f" — {f['game_time']}" if f.get('game_time') else ''
                # ✅ confirms the actual gate: Pinnacle at 1.5. Park-exception flags carry 🏟️ on
                # the row instead (their Pinnacle line is the park point, not 1.5) -- see _pq_note.
                odds = format_odds_lines(f.get('odds_lines'), gate_book=PINNACLE_BOOK_LABEL, gate_point=1.5)
                tag  = ' 🏟️' if f.get('pinnacle_gate_reason') == 'park_exception' else ''
                kp   = f.get('projected_k')
                ok   = f.get('o_k_rate')
                lineup_label = (' · high-K lineup' if ok >= K_PROJ_LEAGUE_K_RATE else ' · contact lineup') if ok is not None else ''
                kp_s = f" — proj ~{kp} K{lineup_label}" if kp else ''
                # only cross-reference SHARP K plays -- those are the only ones shown in the K-Prop
                # section now (base K plays log to the tracker but don't fire), so a base tag here
                # would point at a bet that isn't in the message.
                kprop_tag = ' 🎯K-PROP↑' if (f.get('k_prop_flag') and f.get('k_prop_tier') == 'sharp') else ''
                lines.append(
                    f"📊 <b>{f['batting_team']}</b> vs {f['pitcher_name']}{tag}{kprop_tag} "
                    f"(pctile {pct:.0f}, K {f['pq_k_rate']*100:.1f}% / BB {f['pq_bb_rate']*100:.1f}% / "
                    f"HR {f['pq_hr_rate']*100:.1f}%){kp_s}{time}"
                )
                # K-prop bet (if this pitcher also fired it) is now shown in the consolidated
                # K-Prop section at the top; the 🎯K-PROP↑ tag above cross-references it.
                if odds:
                    lines.append(f"   {odds}")
        if new_pq_stale:
            print(f"PQ Telegram suppression: {len(new_pq_stale)} flag(s) held back for stale prior "
                  f"(current-season BF < {PQ_BF_STALE_THRESHOLD}): "
                  f"{[(f.get('pitcher_name'), f.get('pq_current_bf')) for f in new_pq_stale]}")

        # NOTE: the K-prop section was moved to the TOP of the message (consolidated, all K plays)
        # June 27, 2026 — see the new_kprop block right after the header. The old standalone-only
        # block here was removed so dual-signal (pq_q4) K plays aren't split across two sections.

        # NOTE: off (Offense Quality) and joint (Joint Offense / Combined F5 Total) are
        # deliberately NOT built into the Telegram message below, per Zach (June 22, 2026) --
        # they're still computed above and still logged to their Sheet tabs unconditionally
        # further down, just no longer surfaced in the notification itself. See VISIBILITY note
        # above new_under/new_pq/etc.

        if new_fg_tt:
            if len(lines) > 1:
                lines.append("")
            lines.append(f"🛢️ <b>FG Team Total Under (Bullpen) — TRACKING ONLY, not a bet signal</b>")
            lines.append(f"{len(new_fg_tt)} top-quartile opposing bullpen(s) — only means something "
                         f"if a line below shows ✅3.5/4.5/5.5 (see fg_note)\n")
            for f in new_fg_tt:
                pct  = f.get('fg_bp_pctile')
                time = f" — {f['game_time']}" if f.get('game_time') else ''
                odds = format_odds_lines(f.get('fg_odds_lines'))
                lines.append(
                    f"🛢️ <b>{f['batting_team']}</b> vs {f['fielding_team']} bullpen "
                    f"(pctile {pct:.0f}, {f.get('fg_bp_n_relievers','—')} relievers){time}"
                )
                if odds:
                    lines.append(f"   {odds}")

        if new_f5_over:
            if len(lines) > 1:
                lines.append("")
            lines.append(f"🔄 <b>F5 Team Total Over (Other Side) — NEW, UNTESTED, tracking only</b>")
            lines.append(f"{len(new_f5_over)} game(s): joint F5 over flagged, one team already "
                         f"F5-under flagged — surfacing the other side\n")
            for f in new_f5_over:
                time = f" — {f['game_time']}" if f.get('game_time') else ''
                odds = format_odds_lines(f.get('f5_over_odds_lines'))
                lines.append(
                    f"🔄 <b>{f['batting_team']}</b> (vs {f['f5_over_flagged_team']} flagged under) "
                    f"{f['away_team']} @ {f['home_team']}{time}"
                )
                if odds:
                    lines.append(f"   {odds}")

        if new_fg_joint:
            if len(lines) > 1:
                lines.append("")
            lines.append(f"🎰 <b>FG Joint/Combined Total (Bullpen) — TRACKING ONLY, not a bet signal</b>")
            lines.append(f"{len(new_fg_joint)} game(s) — combined (both teams') bullpen strength vs "
                         f"the whole-game total. UNDER tiered 🥇Gold 1.5u / 🥈Silver 1.0u / 🥉Bronze "
                         f"0.5u; OVER single 1.0u. ✅ = {FG_JOINT_GATE_BOOK} line in "
                         f"{FG_JOINT_LINE_MIN}-{FG_JOINT_LINE_MAX}\n")
            for f in new_fg_joint:
                direction = f.get('fg_joint_direction', '')
                tier  = f.get('fg_joint_tier', '')
                units = f.get('fg_joint_units', '')
                medal = {'Gold': '🥇', 'Silver': '🥈', 'Bronze': '🥉'}.get(tier, '')
                arrow = '🔻 UNDER' if direction == 'under' else '🔺 OVER'
                tier_prefix = f"{medal} {tier} " if direction == 'under' else ''
                gate_pt = f.get('fgj_gate_point')
                marker  = '✅' if (gate_pt is not None and
                                   FG_JOINT_LINE_MIN <= gate_pt <= FG_JOINT_LINE_MAX) else ''
                time = f" — {f['game_time']}" if f.get('game_time') else ''
                odds = format_fg_joint_odds_lines(f.get('fgj_odds_lines'), direction)
                lines.append(
                    f"🎰 <b>{f['away_team']} @ {f['home_team']}</b> — {tier_prefix}JOINT {arrow} "
                    f"({units}u) (combined bullpen pctile {f.get('fgj_combined_pctile')}; "
                    f"{f.get('fgj_away_bp_pctile')}/{f.get('fgj_home_bp_pctile')} away/home) "
                    f"[{FG_JOINT_GATE_BOOK} {gate_pt}{marker}]{time}"
                )
                if odds:
                    lines.append(f"   {odds}")

        # NOTE: the FG Joint Total OFFENSE FADE block was removed from Telegram June 28, 2026 per
        # Zach (focusing notifications on K props) -- still computed above and still logged to the
        # OffenseFade Sheet tab below (append_off_fade_to_sheet(new_off_fade), unfiltered) and still
        # captured in line-history, just no longer pushed to Telegram. The old OFF_FADE_PA_STALE
        # Telegram gate went away with the block (it only filtered the notification, never the log).

        # Only push if a real section rendered (len>1 = more than the bare header). With base K plays
        # now tracker-only, a base-K-only day would otherwise send a header with no body. The Sheet
        # appends + dedup below still run regardless, so base picks keep logging on a no-message day.
        if len(lines) > 1:
            send_telegram('\n'.join(lines))
        if new_under:
            append_to_sheet(new_under)
        if new_pq:
            append_pq_to_sheet(new_pq)
        if new_kprop:
            append_kprop_to_sheet(new_kprop)
        if new_off:
            append_off_to_sheet(new_off)
        if new_joint:
            append_joint_to_sheet(new_joint)
        if new_fg_tt:
            append_fg_tt_to_sheet(new_fg_tt)
        if new_f5_over:
            append_f5_over_to_sheet(new_f5_over)
        if new_fg_joint:
            append_fg_joint_to_sheet(new_fg_joint)
        if new_off_fade:
            append_off_fade_to_sheet(new_off_fade)
        all_sent = (already_sent | {flag_key(f) for f in new_under}
                    | {flag_key(f) for f in new_pq} | {flag_key(f) for f in new_kprop}
                    | {flag_key(f) for f in new_off}
                    | {flag_key(f) for f in new_joint} | {flag_key(f) for f in new_fg_tt}
                    | {flag_key(f) for f in new_f5_over} | {flag_key(f) for f in new_fg_joint}
                    | {flag_key(f) for f in new_off_fade})
        redis_set(redis_key, ','.join(all_sent), ex=86400)
        # Persist the sharp K-prop pings in their own namespace so each fires once and base->sharp
        # upgrades are remembered independently of the shared game|team flag_key set above.
        redis_set(ksharp_key, ','.join(ksharp_sent | {flag_key(f) for f in new_kprop_sharp}), ex=86400)
        return jsonify({'status': 'ok',
                        'new': (len(new_under) + len(new_pq) + len(new_kprop) + len(new_off)
                                + len(new_joint) + len(new_fg_tt) + len(new_f5_over)
                                + len(new_fg_joint) + len(new_off_fade)),
                        'flags': ([flag_key(f) for f in new_under] + [flag_key(f) for f in new_pq]
                                  + [flag_key(f) for f in new_kprop] + [flag_key(f) for f in new_off]
                                  + [flag_key(f) for f in new_joint] + [flag_key(f) for f in new_fg_tt]
                                  + [flag_key(f) for f in new_f5_over] + [flag_key(f) for f in new_fg_joint]
                                  + [flag_key(f) for f in new_off_fade])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


load_fg_profiles()
load_pitcher_quality_prior()
load_batter_offense_prior()
load_lineup_slot_weights()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
