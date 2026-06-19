import os
import csv
import math
import bisect
import pickle
import requests
import pandas as pd
import pytz
from datetime import datetime, date as _date
from collections import Counter
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import gdown
import tempfile

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
# CRITICAL — this ONLY works at the 1.5 F5 team-total line specifically. At 2.5 lines, the effect
# is flat-to-backward. This app does not fetch live odds/lines, so it cannot gate on line level —
# you must check the actual posted F5 line yourself before treating this as anything actionable.
# Offense-side K/BB/SLG composites (with and without platoon-matching) were tested extensively and
# showed ZERO signal anywhere — pitcher-only by design, this is not an oversight.

PQ_WEIGHTS       = {'k': 0.1186068959664584, 'bb': -0.156144018411192, 'hr': -0.1944839999359043}
PQ_PHANTOM_BF    = {'k_rate': 70, 'bb_rate': 170, 'hr_rate': 1320}
PQ_PRIOR_CSV     = 'pitcher_quality_prior_2026.csv'
PQ_Q4_PCTILE     = 75.0
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
    """Current-season K/BB/HR counts per pitcher via pybaseball (Baseball-Reference)."""
    try:
        import pybaseball as pb
        year = datetime.now().year
        ps = pb.pitching_stats_bref(year)
        ps = ps.sort_values('IP', ascending=False).drop_duplicates('Name', keep='first')
        ps = ps.dropna(subset=['mlbID'])
        season = {}
        for _, row in ps.iterrows():
            bf = row.get('BF', 0) or 0
            if bf <= 0:
                continue
            season[int(row['mlbID'])] = {
                'bf': bf, 'so': row.get('SO', 0) or 0,
                'bb': row.get('BB', 0) or 0, 'hr': row.get('HR', 0) or 0,
            }
        return season
    except Exception as e:
        print(f"Warning: pitcher quality current-season fetch failed: {e}")
        return {}


def get_pitcher_quality_population():
    """
    Returns dict: mlbID -> {score, percentile, quartile, k_rate, bb_rate, hr_rate}.
    Blends career prior with current-season-to-date (Marcel, leak-free as of right now).
    Cached PQ_SEASON_TTL seconds since current-season stats don't change minute to minute.
    """
    global _pq_population_cache, _pq_population_cache_time
    now = datetime.now().timestamp()
    if _pq_population_cache and _pq_population_cache_time and \
            (now - _pq_population_cache_time < PQ_SEASON_TTL):
        return _pq_population_cache

    load_pitcher_quality_prior()
    if not _pq_prior:
        return {}

    season = _pq_fetch_current_season()
    blended = {}
    for pid, prior in _pq_prior.items():
        s = season.get(pid, {'bf': 0, 'so': 0, 'bb': 0, 'hr': 0})
        bf = s['bf']
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

    if len(blended) < 10:
        return {}

    # Reference distribution (mean/std/percentile ranking) is computed from STARTERS ONLY —
    # the prior pool is ~60% relievers (different K/BB rate profile), and the validated research
    # quartile was defined among actual starting pitchers in 1.5-line games, not all of MLB.
    # Mixing relievers into the reference population measurably shifts the Q4 cutoff (verified
    # June 2026: ~1.7% of starters flip Q4 status between the two methods). Every pitcher still
    # gets scored/blended the same way — only the comparison population is restricted here.
    starter_ids = [pid for pid in blended if _pq_prior.get(pid, {}).get('is_starter')]
    ref_pool    = starter_ids if len(starter_ids) >= 10 else list(blended.keys())

    means  = {stat: sum(blended[pid][stat] for pid in ref_pool) / len(ref_pool) for stat in ('k_rate', 'bb_rate', 'hr_rate')}
    stds   = {stat: (sum((blended[pid][stat] - means[stat]) ** 2 for pid in ref_pool) / len(ref_pool)) ** 0.5
              for stat in ('k_rate', 'bb_rate', 'hr_rate')}

    def compute_score(b):
        z = {stat: (b[stat] - means[stat]) / stds[stat] if stds[stat] > 0 else 0.0
             for stat in ('k_rate', 'bb_rate', 'hr_rate')}
        return PQ_WEIGHTS['k'] * z['k_rate'] + PQ_WEIGHTS['bb'] * z['bb_rate'] + PQ_WEIGHTS['hr'] * z['hr_rate']

    ref_scores_sorted = sorted(compute_score(blended[pid]) for pid in ref_pool)
    n_ref = len(ref_scores_sorted)

    population = {}
    for pid, b in blended.items():
        score  = compute_score(b)
        pctile = 100.0 * bisect.bisect_left(ref_scores_sorted, score) / n_ref if n_ref > 1 else 50.0
        population[pid] = {
            'score':      round(score, 4),
            'percentile': round(pctile, 1),
            'quartile':   'Q4' if pctile >= PQ_Q4_PCTILE else ('Q1' if pctile < 25 else ('Q2' if pctile < 50 else 'Q3')),
            'k_rate':     round(blended[pid]['k_rate'], 4),
            'bb_rate':    round(blended[pid]['bb_rate'], 4),
            'hr_rate':    round(blended[pid]['hr_rate'], 4),
        }

    _pq_population_cache      = population
    _pq_population_cache_time = now
    return population


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

_off_prior            = {}
_off_prior_loaded      = False
_off_population_cache      = None
_off_population_cache_time = None
_lineup_slot_weights   = {}
_lineup_weights_loaded = False


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
    """Current-season PA/AB/K/BB/TB per batter via pybaseball (Baseball-Reference)."""
    try:
        import pybaseball as pb
        year = datetime.now().year
        bs = pb.batting_stats_bref(year)
        bs = bs.sort_values('PA', ascending=False).drop_duplicates('mlbID', keep='first')
        bs = bs.dropna(subset=['mlbID'])
        season = {}
        for _, row in bs.iterrows():
            pa = row.get('PA', 0) or 0
            if pa <= 0:
                continue
            ab = row.get('AB', 0) or 0
            tb = (row.get('H', 0) or 0) + (row.get('2B', 0) or 0) + 2 * (row.get('3B', 0) or 0) + 3 * (row.get('HR', 0) or 0)
            season[int(row['mlbID'])] = {
                'pa': pa, 'ab': ab, 'so': row.get('SO', 0) or 0,
                'bb': row.get('BB', 0) or 0, 'tb': tb,
            }
        return season
    except Exception as e:
        print(f"Warning: offense quality current-season fetch failed: {e}")
        return {}


def get_offense_quality_population():
    """
    Returns dict: mlbID -> {score, percentile, k_rate, bb_rate, slg}.
    Blends career prior with current-season-to-date (Marcel). Cached OFF_SEASON_TTL seconds.
    """
    global _off_population_cache, _off_population_cache_time
    now = datetime.now().timestamp()
    if _off_population_cache and _off_population_cache_time and \
            (now - _off_population_cache_time < OFF_SEASON_TTL):
        return _off_population_cache

    load_batter_offense_prior()
    if not _off_prior:
        return {}

    season = _off_fetch_current_season()
    blended = {}
    for bid, prior in _off_prior.items():
        s = season.get(bid, {'pa': 0, 'ab': 0, 'so': 0, 'bb': 0, 'tb': 0})
        pa, ab = s['pa'], s['ab']
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
        }

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
    wsum_pctile = wsum_bb = wtot = 0.0
    n_matched = 0
    for batter in lineup:
        info = population.get(batter.get('id'))
        slot = batter.get('batting_order')
        if info is None or slot not in _lineup_slot_weights:
            continue
        w = _lineup_slot_weights[slot]
        wsum_pctile += info['percentile'] * w
        wsum_bb     += info['bb_rate']    * w
        wtot        += w
        n_matched   += 1
    if n_matched < 5 or wtot <= 0:
        return None
    return {
        'off_pctile': round(wsum_pctile / wtot, 1),
        'o_bb_rate':  round(wsum_bb / wtot, 4),
    }


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
        try:
            model[name] = download_pkl(fid)
        except Exception as e:
            print(f"Warning: could not load {name}: {e}")
            model[name] = None
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
# books Zach can actually bet (Pinnacle is NOT used — he cannot access it from the US, see
# CLAUDE.md "IMPORTANT CORRECTION — VALIDATION BOOK"). Display-only: per Zach's choice, this
# does NOT filter/suppress any existing flag, it just attaches real line data to each one.
#
# Market key 'team_totals_1st_5_innings' requires the per-event odds endpoint (not the cheaper
# bulk endpoint), so this costs real API credits: cost = markets x regions per event, currently
# 1 market x 2 regions = 2 credits/game. regions=us covers DraftKings/Fanatics, regions=us2
# covers theScore Bet (formerly ESPN Bet — its bookmaker key may still be 'espnbet' on this API;
# TARGET_BOOKS below tries both so a rename doesn't silently break this). ~30-58 credits/day at
# a typical 15-29 game slate. Cached in Redis on ODDS_API_TTL so /api/picks and /api/notify don't
# double-spend credits when both run close together.

ODDS_API_KEY      = os.environ.get('ODDS_API_KEY', '')
ODDS_API_BASE     = 'https://api.the-odds-api.com/v4'
ODDS_F5_MARKET    = 'team_totals_1st_5_innings'
ODDS_REGIONS      = 'us,us2'
ODDS_API_TTL      = 14400  # 4h
TARGET_BOOKS = {
    'draftkings': 'DraftKings',
    'fanatics':   'Fanatics',
    'espnbet':    'theScore Bet',
    'thescorebet': 'theScore Bet',  # in case the API renames this key
}

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


def get_odds_api_lines(games):
    """
    Returns dict: team_abb -> {book_key: {'point': float, 'over': int, 'under': int,
    'over_profit': float, 'under_profit': float}}. Cached ODDS_API_TTL seconds in Redis
    (shared across /api/picks and /api/notify) since each call costs real API credits.
    """
    if not ODDS_API_KEY or not games:
        return {}

    import json as _json
    today     = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
    cache_key = f"ozzie:odds_lines:{today}"
    cached    = redis_get(cache_key)
    if cached:
        try:
            return _json.loads(cached)
        except Exception:
            pass

    try:
        ev_resp = requests.get(
            f"{ODDS_API_BASE}/sports/baseball_mlb/events",
            params={'apiKey': ODDS_API_KEY}, timeout=15)
        events = ev_resp.json() if ev_resp.ok else []
    except Exception as e:
        print(f"Odds API events fetch error: {e}")
        return {}

    wanted = {(g['home_team'], g['away_team']) for g in games}
    matched_events = []
    for ev in events:
        home_abb = NAME_TO_ABB.get(ev.get('home_team', ''))
        away_abb = NAME_TO_ABB.get(ev.get('away_team', ''))
        if (home_abb, away_abb) in wanted:
            matched_events.append((ev['id'], home_abb, away_abb))

    lines = {}
    for event_id, home_abb, away_abb in matched_events:
        try:
            r = requests.get(
                f"{ODDS_API_BASE}/sports/baseball_mlb/events/{event_id}/odds",
                params={'apiKey': ODDS_API_KEY, 'regions': ODDS_REGIONS,
                        'markets': ODDS_F5_MARKET, 'oddsFormat': 'american'},
                timeout=15)
            if not r.ok:
                continue
            data = r.json()
        except Exception as e:
            print(f"Odds API event {event_id} fetch error: {e}")
            continue

        for bm in data.get('bookmakers', []):
            book_key = bm.get('key', '')
            if book_key not in TARGET_BOOKS:
                continue
            book_label = TARGET_BOOKS[book_key]
            for market in bm.get('markets', []):
                if market.get('key') != ODDS_F5_MARKET:
                    continue
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
                    lines.setdefault(team_abb, {})[book_label] = {
                        'point':         over.get('point'),
                        'over':          int(over['price']),
                        'under':         int(under['price']),
                        'over_profit':   round(_odds_american_to_profit(over['price']), 3),
                        'under_profit':  round(_odds_american_to_profit(under['price']), 3),
                    }

    try:
        redis_set(cache_key, _json.dumps(lines), ex=ODDS_API_TTL)
    except Exception as e:
        print(f"Odds lines cache write error: {e}")
    return lines


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


def get_heatmap_flags(games, model):
    pitcher_scores  = model.get('all_pitcher_arch_scores', {})
    nb_bundle       = model.get('negbin_model_params')
    contact_rates   = model.get('pitcher_contact_rates', {})
    bb_gb_rates_all = model.get('pitcher_bb_gb_rates', {}) or {}
    flags           = []

    try:
        odds_lines = get_odds_api_lines(games)
    except Exception as e:
        print(f"Odds API lookup error: {e}")
        odds_lines = {}

    for game in games:
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

            # Pitcher quality Q4 surfaces even without any other signal — tracking only,
            # NOT a validated bet signal. Check the actual F5 line is 1.5 before acting on it.
            if signal is None and pq_q4:
                signal = 'pitcher_quality_only'

            # Offense quality Q3-band+low-BB gate, same tracking-only treatment. Independent of
            # pq_q4 — both describe different sides of the same matchup and can co-occur (visible
            # via the off_q3_gate / pq_q4 boolean fields below even when 'signal' picks one).
            if signal is None and off_q3_gate:
                signal = 'offense_quality_only'

            if signal is None:
                continue

            flag = {
                'game':             game_str,
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
                # Pitcher Quality Composite — tracking only, NOT validated for betting.
                # Only meaningful if today's actual F5 line for this matchup is 1.5 (check manually —
                # this app does not fetch live lines). 2026 OOS is weak/not significant overall;
                # see PQ_WEIGHTS comment block above for full validation status.
                'pq_score':         pq_info['score']      if pq_info else None,
                'pq_percentile':    pq_info['percentile'] if pq_info else None,
                'pq_quartile':      pq_info['quartile']   if pq_info else None,
                'pq_k_rate':        pq_info['k_rate']      if pq_info else None,
                'pq_bb_rate':       pq_info['bb_rate']     if pq_info else None,
                'pq_hr_rate':       pq_info['hr_rate']     if pq_info else None,
                'pq_q4':            pq_q4,
                'pq_note':          ('Top-quartile pitcher quality — IF the F5 line for this team is '
                                      '1.5, this historically favors UNDER (check the line yourself; '
                                      'not yet proven on 2026 — tracking signal only, do not bet)')
                                     if pq_q4 else None,
                # Offense Quality Composite — tracking only, NOT validated for betting.
                # Only meaningful if today's actual F5 line for THIS batting team is 1.5 (check
                # manually). See OFF_PHANTOM comment block above for full validation status.
                'off_pctile':       off_info['off_pctile'] if off_info else None,
                'off_bb_rate':      off_info['o_bb_rate']  if off_info else None,
                'off_q3_gate':      off_q3_gate,
                'off_note':         ('Mid-tier offense (Q3) with a low walk rate — IF the F5 line for '
                                      'this team is 1.5, this historically favors UNDER (check the line '
                                      'yourself; 2 years OOS-replicated but still tracking signal only, '
                                      'do not bet)')
                                     if off_q3_gate else None,
                # Live F5 odds (DraftKings/Fanatics/theScore Bet) — see get_odds_api_lines.
                # Display-only: does not filter pq_q4/off_q3_gate, just shows the real line/price
                # per book so you don't have to check manually. {} if ODDS_API_KEY isn't set or
                # this game/book combo wasn't found.
                'odds_lines':       odds_lines.get(batting_team, {}),
                'odds_has_1_5':     any(b.get('point') == 1.5 for b in odds_lines.get(batting_team, {}).values()),
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


@app.route('/api/picks')
def api_picks():
    if not session.get('authenticated'):
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        import json as _json
        et_now    = datetime.now(pytz.timezone('America/New_York'))
        today     = et_now.strftime('%Y-%m-%d')
        cache_key = f"ozzie:picks:{today}"

        cached = redis_get(cache_key)
        if cached:
            try:
                return jsonify(_json.loads(cached))
            except Exception:
                pass

        model     = load_model()
        games     = get_lineups_and_starters(today)
        picks     = get_hr_picks(games, model)
        complete  = sum(1 for g in games if g['home_lineup'] and g['away_lineup'])

        games_out = [{'away': g['away_team'], 'home': g['home_team'],
                      'complete': bool(g['home_lineup'] and g['away_lineup']),
                      'game_time': g.get('game_time')}
                     for g in games]

        heatmap_flags   = get_heatmap_flags(games, model)
        fg_under_flags  = [f for f in heatmap_flags if f.get('fg_under_signal')]
        over_info_flags = [f for f in heatmap_flags if f.get('signal') == 'over_info']
        pq_flags        = [f for f in heatmap_flags if f.get('pq_q4')]
        off_flags       = [f for f in heatmap_flags if f.get('off_q3_gate')]

        dfs_picks = {}
        try:
            dfs_picks = get_dfs_picks(games, model)
        except Exception as e:
            print(f"DFS picks error: {e}")

        payload = {
            'date':            today,
            'complete':        complete,
            'total':           len(games),
            'picks':           picks,
            'dfs_picks':       dfs_picks,
            'heatmap_flags':   heatmap_flags,
            'fg_under_flags':  fg_under_flags,
            'over_info_flags': over_info_flags,
            'pq_flags':        pq_flags,
            'off_flags':       off_flags,
            'fg_in_window':    is_fg_valid_window(),
            'games':           games_out,
        }

        now_et    = datetime.now(pytz.timezone('America/New_York'))
        expire_et = now_et.replace(hour=1, minute=0, second=0, microsecond=0)
        if now_et >= expire_et:
            expire_et = expire_et.replace(day=expire_et.day + 1)
        ttl = min(1800, int((expire_et - now_et).total_seconds()))
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


def format_odds_lines(odds_lines):
    """e.g. 'DraftKings 1.5✅ | Fanatics 2.5 | theScore Bet 1.5✅' — empty string if no data."""
    if not odds_lines:
        return ''
    parts = []
    for book, info in odds_lines.items():
        pt = info.get('point')
        marker = '✅' if pt == 1.5 else ''
        parts.append(f"{book} {pt}{marker}")
    return ' | '.join(parts)


def append_to_sheet(flags):
    if not SHEETS_CREDS or not flags:
        return
    try:
        import gspread
        import json as _json
        from google.oauth2.service_account import Credentials
        creds_dict = _json.loads(SHEETS_CREDS)
        scopes = ['https://www.googleapis.com/auth/spreadsheets',
                  'https://www.googleapis.com/auth/drive']
        creds  = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc     = gspread.authorize(creds)
        ws     = gc.open_by_key(SHEETS_ID).worksheet(SHEETS_TAB)
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
    'books_lines', 'actual_f5_line',  # books_lines + actual_f5_line auto-filled from live odds when available
    'actual_f5_runs', 'under_hit',    # still fill in by hand after the game — outcome isn't known yet
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
        import gspread
        import json as _json
        from google.oauth2.service_account import Credentials
        creds_dict = _json.loads(SHEETS_CREDS)
        scopes = ['https://www.googleapis.com/auth/spreadsheets',
                  'https://www.googleapis.com/auth/drive']
        creds  = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc     = gspread.authorize(creds)
        sh     = gc.open_by_key(SHEETS_ID)
        try:
            ws = sh.worksheet(PQ_SHEET_TAB)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=PQ_SHEET_TAB, rows=1000, cols=len(PQ_SHEET_HEADER))
            ws.append_row(PQ_SHEET_HEADER, value_input_option='USER_ENTERED')

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
            ws.append_row([
                today, f.get('game', ''), f.get('batting_team', ''), f.get('pitcher_name', ''),
                f.get('pq_score', ''), f.get('pq_percentile', ''),
                f.get('pq_k_rate', ''), f.get('pq_bb_rate', ''), f.get('pq_hr_rate', ''),
                f.get('game_time', ''),
                format_odds_lines(f.get('odds_lines')),
                1.5 if f.get('odds_has_1_5') else '',
                '', '',  # actual_f5_runs / under_hit — fill in by hand after the game
            ], value_input_option='USER_ENTERED')
            existing_keys.add(key)
            rows_added += 1
        print(f"Google Sheets (PitcherQuality): {rows_added} rows added")
    except ImportError:
        print("gspread not installed — skipping PQ sheet append")
    except Exception as e:
        print(f"Google Sheets (PitcherQuality) error: {e}")


OFF_SHEET_TAB    = 'OffenseQuality'
OFF_SHEET_HEADER = [
    'date', 'game', 'batting_team', 'pitcher_name', 'off_pctile', 'off_bb_rate',
    'game_time', 'books_lines', 'actual_f5_line',  # auto-filled from live odds when available
    'actual_f5_runs', 'under_hit',  # still fill in by hand after the game
]


def append_off_to_sheet(flags):
    """
    Tracking-only log for the Offense Quality Composite (NOT a validated bet signal — see
    OFF_PHANTOM comment block). Separate tab, same pattern as append_pq_to_sheet.
    """
    if not SHEETS_CREDS or not flags:
        return
    try:
        import gspread
        import json as _json
        from google.oauth2.service_account import Credentials
        creds_dict = _json.loads(SHEETS_CREDS)
        scopes = ['https://www.googleapis.com/auth/spreadsheets',
                  'https://www.googleapis.com/auth/drive']
        creds  = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc     = gspread.authorize(creds)
        sh     = gc.open_by_key(SHEETS_ID)
        try:
            ws = sh.worksheet(OFF_SHEET_TAB)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=OFF_SHEET_TAB, rows=1000, cols=len(OFF_SHEET_HEADER))
            ws.append_row(OFF_SHEET_HEADER, value_input_option='USER_ENTERED')

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
            ws.append_row([
                today, f.get('game', ''), f.get('batting_team', ''), f.get('pitcher_name', ''),
                f.get('off_pctile', ''), f.get('off_bb_rate', ''),
                f.get('game_time', ''),
                format_odds_lines(f.get('odds_lines')),
                1.5 if f.get('odds_has_1_5') else '',
                '', '',  # actual_f5_runs / under_hit — fill in by hand after the game
            ], value_input_option='USER_ENTERED')
            existing_keys.add(key)
            rows_added += 1
        print(f"Google Sheets (OffenseQuality): {rows_added} rows added")
    except ImportError:
        print("gspread not installed — skipping Offense Quality sheet append")
    except Exception as e:
        print(f"Google Sheets (OffenseQuality) error: {e}")


@app.route('/api/notify')
def api_notify():
    secret = request.args.get('secret', '')
    if not NOTIFY_SECRET or secret != NOTIFY_SECRET:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        model   = load_model()
        today   = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
        games   = get_lineups_and_starters(today)
        flags   = get_heatmap_flags(games, model)
        under_flags = [f for f in flags if f.get('signal') == 'under']
        pq_flags    = [f for f in flags if f.get('pq_q4')]
        off_flags   = [f for f in flags if f.get('off_q3_gate')]
        if not under_flags and not pq_flags and not off_flags:
            return jsonify({'status': 'ok', 'new': 0, 'message': 'No flags today'})

        redis_key    = f"ozzie:notified:{today}"
        existing_raw = redis_get(redis_key)
        already_sent = set(existing_raw.split(',')) if existing_raw else set()
        new_under = [f for f in under_flags if flag_key(f) not in already_sent]
        new_pq    = [f for f in pq_flags if flag_key(f) not in already_sent]
        new_off   = [f for f in off_flags if flag_key(f) not in already_sent]
        if not new_under and not new_pq and not new_off:
            return jsonify({'status': 'ok', 'new': 0, 'message': 'No new flags'})

        lines = [f"🎯 <b>Ozzie — {today}</b>"]

        if new_under:
            lines.append(f"{len(new_under)} new flag(s)\n")
            for f in new_under:
                label = f.get('confidence_label', '—')
                score = f.get('ozzie_score', '—')
                sigma = f.get('std_from_mean', 0)
                medal = {'Gold': '🥇', 'Silver': '🥈', 'Bronze': '🥉'}.get(label, '🥉')
                fg    = ' + FG ✅' if f.get('fg_under_signal') else ''
                comb  = ' + COMB ⚡' if f.get('combined_f5_signal') else ''
                time  = f" — {f['game_time']}" if f.get('game_time') else ''
                lines.append(
                    f"{medal} <b>{f['batting_team']}</b> vs {f['pitcher_name']} "
                    f"(UNDER #{score}, {sigma:+.2f}σ){fg}{comb}{time}"
                )
                if f.get('combined_f5_signal'):
                    lines.append("   ⚡ F5 Combined: U4.5 @ -200 | U5.5 @ -300")
                u25_str = f"{medal} U2.5 {f['min_u25']} / {f.get('unit_u25','')}" if f.get('min_u25') else "U2.5 No Bet"
                lines.append(f"   {u25_str}")

        if new_pq:
            if new_under:
                lines.append("")
            lines.append(f"📊 <b>Pitcher Quality — TRACKING ONLY, not a bet signal</b>")
            lines.append(f"{len(new_pq)} Q4 pitcher(s) — only means something if a line below shows ✅1.5\n")
            for f in new_pq:
                pct  = f.get('pq_percentile')
                time = f" — {f['game_time']}" if f.get('game_time') else ''
                odds = format_odds_lines(f.get('odds_lines'))
                lines.append(
                    f"📊 <b>{f['batting_team']}</b> vs {f['pitcher_name']} "
                    f"(pctile {pct:.0f}, K {f['pq_k_rate']*100:.1f}% / BB {f['pq_bb_rate']*100:.1f}% / "
                    f"HR {f['pq_hr_rate']*100:.1f}%){time}"
                )
                if odds:
                    lines.append(f"   {odds}")

        if new_off:
            if new_under or new_pq:
                lines.append("")
            lines.append(f"⚾ <b>Offense Quality — TRACKING ONLY, not a bet signal</b>")
            lines.append(f"{len(new_off)} mid-tier/low-BB offense(s) — only means something if a line below shows ✅1.5\n")
            for f in new_off:
                pct  = f.get('off_pctile')
                bb   = f.get('off_bb_rate')
                time = f" — {f['game_time']}" if f.get('game_time') else ''
                odds = format_odds_lines(f.get('odds_lines'))
                lines.append(
                    f"⚾ <b>{f['batting_team']}</b> vs {f['pitcher_name']} "
                    f"(off pctile {pct:.0f}, BB {bb*100:.1f}%){time}"
                )
                if odds:
                    lines.append(f"   {odds}")

        send_telegram('\n'.join(lines))
        if new_under:
            append_to_sheet(new_under)
        if new_pq:
            append_pq_to_sheet(new_pq)
        if new_off:
            append_off_to_sheet(new_off)
        all_sent = (already_sent | {flag_key(f) for f in new_under}
                    | {flag_key(f) for f in new_pq} | {flag_key(f) for f in new_off})
        redis_set(redis_key, ','.join(all_sent), ex=86400)
        return jsonify({'status': 'ok', 'new': len(new_under) + len(new_pq) + len(new_off),
                        'flags': [flag_key(f) for f in new_under] + [flag_key(f) for f in new_pq]
                                 + [flag_key(f) for f in new_off]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


load_fg_profiles()
load_pitcher_quality_prior()
load_batter_offense_prior()
load_lineup_slot_weights()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
