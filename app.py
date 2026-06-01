import os
import csv
import math
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
    # Heat map models
    'pitcher_tendency_map':     '1u328HojWnhcQWlgx0SW5DX-WrsThBxon',
    'batter_ev_profiles_lr':    '1dnFavwW5CPXoJy3IBZueAYOokb3oxXyE',
    'pitcher_contact_rates':    '1DXNF_rvjHBk31Ja6Tle3i8LrjwD5iXec',  # pitcher_contact_rates_2026.pkl (2022-2025 weighted 1:2:3:4)
    # NegBin expected runs model (Script S)
    'negbin_model_params':      '122sd0M7XFhb-JlU2qE7_wQey9iTntIv9',
}

LEAGUE_AVG      = 3.88
JUICY_THRESHOLD = 5.0
MIN_SENSOR_PA   = 20
STARTER_WEIGHT  = 0.80
BULLPEN_WEIGHT  = 0.20
ASSUMED_PAS     = 4
TB3_MULTIPLIER  = 1.3

# F5 heat map thresholds (validated on full MLB 2025, out-of-sample)
HEATMAP_MEAN            = 1.0133
HEATMAP_STD             = 0.0063
HEATMAP_OVER_THRESHOLD  = HEATMAP_MEAN + 1.5 * HEATMAP_STD
HEATMAP_UNDER_THRESHOLD = HEATMAP_MEAN - 1.0 * HEATMAP_STD  # Lowered from -2.0 (May 2026)

# F5 fair odds by pitcher category (from backtest)
F5_FAIR_ODDS = {
    'Very Juicy':  -126,
    'Juicy':       +118,
    'Average':     +133,
    'Safe':        +143,
    'Very Safe':   +191,
}

# ── FG Team Total Under — two-leg signal constants (revised May 2026) ─────────
# Filter: blend_70_30 z ≤ -0.5 AND opp_bp_weak_z 0.5→1.0
# off_z dropped — shown to be noise in full 2025 backtest (May 29 2026 session)
# Valid window: June 1 – July 31 ONLY (signal degrades in August, confirmed)
# Validated: n=33 (≤-0.5 threshold), 69.7% hit rate, +0.295 ROI vs Pinnacle
# Prior n=11 (≤-1.0): 90.9% hit, +0.675 ROI — threshold loosened for volume
# Static Apr-May profiles only — do NOT update mid-season
# Profiles: bullpen_profile_2026.csv, team_offense_baseline_2026.csv
FG_BLEND_MEAN        = 0.9984   # 2025 validated distribution mean
FG_BLEND_STD         = 0.0353   # 2025 validated distribution std
FG_STARTER_Z_THRESH  = -0.5     # blend_70_30 z-score threshold (loosened from -1.0)
FG_BP_BAND_LOW       =  0.5     # opposing BP weakness band — floor
FG_BP_BAND_HIGH      =  1.0     # opposing BP weakness band — ceiling (hard cap)
FG_VALID_START_MONTH =  6       # June
FG_VALID_START_DAY   =  1
FG_VALID_END_MONTH   =  7
FG_VALID_END_DAY     = 31       # Extended from July 15 — Jul signal confirmed clean

_model_cache      = None
_model_cache_time = None
MODEL_CACHE_TTL   = 3600

# ── FG profile cache (loaded once at startup) ─────────────────────────────────
_FG_OFF_Z     = {}
_FG_BP_WEAK_Z = {}
_FG_PROFILES_LOADED = False


def load_fg_profiles():
    """
    Load static Apr-May offense and bullpen profiles from CSV.
    Tries current year first, falls back to prior year.
    Populates module-level _FG_OFF_Z and _FG_BP_WEAK_Z dicts.
    Files must be in same directory as app.py.
    """
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

    print("Warning: FG profiles not found — FG three-leg signal disabled")


def is_fg_valid_window():
    """Returns True if today is within the June 1 – July 31 valid window."""
    today = _date.today()
    start = _date(today.year, FG_VALID_START_MONTH, FG_VALID_START_DAY)
    end   = _date(today.year, FG_VALID_END_MONTH,   FG_VALID_END_DAY)
    return start <= today <= end


def compute_fg_under_flag(batting_team, opp_team, blend_70_30_score):
    """
    Evaluate the two-leg FG team total under signal for one matchup.

    Signal: starter_z ≤ -0.5 AND opp_bp_weak_z in [0.5, 1.0]
    off_z is passed through for display only — not a signal gate (noise, May 2026).
    BP ceiling (1.0) is a hard cap — bp_z > 1.0 confirmed blowup risk.

    Parameters:
        batting_team      — short code e.g. 'LAD'
        opp_team          — short code e.g. 'SF' (the team pitching in innings 6-9)
        blend_70_30_score — raw blend_70_30 overlap score (NOT z-scored yet)

    Returns dict with:
        fg_under_signal   — True if both legs pass and in valid window
        starter_z         — z-scored blend_70_30
        opp_bp_weak_z     — opposing bullpen weakness z (from Apr-May profile)
        off_z             — batting team offense z (display only, not gating)
        bp_in_band        — True if opp_bp_weak_z is in [0.5, 1.0]
        in_valid_window   — True if today is June 1 – July 31
        reason            — human-readable explanation
    """
    in_window = is_fg_valid_window()

    starter_z = round((blend_70_30_score - FG_BLEND_MEAN) / FG_BLEND_STD, 3)

    off_z         = _FG_OFF_Z.get(batting_team)
    opp_bp_weak_z = _FG_BP_WEAK_Z.get(opp_team)

    if opp_bp_weak_z is None:
        return {
            'fg_under_signal': False,
            'starter_z':       starter_z,
            'opp_bp_weak_z':   None,
            'off_z':           off_z,
            'bp_in_band':      False,
            'in_valid_window': in_window,
            'reason':          'missing_profile_data',
        }

    starter_flag = starter_z <= FG_STARTER_Z_THRESH
    bp_band_flag = FG_BP_BAND_LOW <= opp_bp_weak_z <= FG_BP_BAND_HIGH
    signal       = in_window and starter_flag and bp_band_flag

    if signal:
        reason = 'all_conditions_met'
    elif not in_window:
        reason = 'outside_valid_window'
    elif not starter_flag:
        reason = f'starter_not_suppressed (z={starter_z:.2f}, need ≤{FG_STARTER_Z_THRESH})'
    else:
        reason = (f'bp_outside_band (z={opp_bp_weak_z:.2f}, '
                  f'need {FG_BP_BAND_LOW}–{FG_BP_BAND_HIGH})')

    return {
        'fg_under_signal': signal,
        'starter_z':       starter_z,
        'opp_bp_weak_z':   round(opp_bp_weak_z, 3),
        'off_z':           round(off_z, 3) if off_z is not None else None,
        'bp_in_band':      bp_band_flag,
        'in_valid_window': in_window,
        'reason':          reason,
    }


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
    _model_cache      = model
    _model_cache_time = now
    return model


def get_pitcher_avg_score(pitcher_id, pitcher_scores, archetypes):
    if pitcher_id not in pitcher_scores:
        return None, 'Unknown'
    pitcher    = pitcher_scores[pitcher_id]
    all_scores = [
        pitcher['archetypes'].get(ak, {}).get('shrunk_rate', LEAGUE_AVG)
        for ak in archetypes
    ]
    avg = sum(all_scores) / len(all_scores)
    if avg >= 5.5:
        category = 'Very Juicy'
    elif avg >= 4.5:
        category = 'Juicy'
    elif avg >= 3.5:
        category = 'Average'
    elif avg >= 3.0:
        category = 'Safe'
    else:
        category = 'Very Safe'
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
            gid  = g['gamePk']

            # Pregame lock — only score games not yet started
            game_status = g.get('status', {}).get('abstractGameState', '')
            if game_status not in ('Preview', ''):
                continue  # skip In Progress, Final, etc.

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
                    for pid in batting_order:
                        p = td.get('players', {}).get(f'ID{pid}', {})
                        lineup.append({
                            'id':   pid,
                            'name': p.get('person', {}).get('fullName', ''),
                            'hand': p.get('batSide', {}).get('code', 'R'),
                        })
            except Exception:
                pass
            # Parse game time — convert UTC to ET
            game_time = None
            try:
                raw_time = g.get('gameDate', '')
                if raw_time:
                    utc_dt   = datetime.strptime(raw_time, '%Y-%m-%dT%H:%M:%SZ')
                    utc_dt   = utc_dt.replace(tzinfo=pytz.utc)
                    et_dt    = utc_dt.astimezone(pytz.timezone('America/New_York'))
                    game_time = et_dt.strftime('%-I:%M %p ET').replace('AM', 'AM ET').replace('PM', 'PM ET')
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


# ── NegBin expected runs + fair odds (Script S) ───────────────────────────────

def get_expected_f5_runs(overlap_k_mod, avg_ip=None, model_bundle=None):
    """
    Predict bias-corrected expected F5 runs for one team-game matchup.
    Uses league avg defaults for lineup_walk_rate and bp_overlap_prob.
    Returns float or None if model unavailable.
    """
    if model_bundle is None:
        return None
    try:
        mb     = model_bundle
        coeffs = mb['coefficients']
        sp     = mb['scaler_params']
        feats  = [f for f in mb['feature_raw_names'][1:] if f in sp]

        defaults = {
            'overlap_k_mod':    overlap_k_mod,
            'lineup_walk_rate': sp['lineup_walk_rate']['mean']
                                if 'lineup_walk_rate' in sp else 0.09,
            'bp_overlap_prob':  sp['bp_overlap_prob']['mean']
                                if 'bp_overlap_prob' in sp else 1.00,
            'avg_ip':           avg_ip if avg_ip is not None
                                else mb.get('league_avg_ip', 5.43),
        }

        z_vals = [
            (defaults[f] - sp[f]['mean']) / sp[f]['std']
            for f in feats
        ]
        X      = [1.0] + z_vals
        log_mu = sum(c * x for c, x in zip(coeffs, X))
        raw    = math.exp(log_mu)
        return round(raw + mb.get('bias_correction', 0.0), 3)
    except Exception:
        return None


def poisson_cdf(k, lam):
    """P(X <= k) for Poisson(lam). Pure math, no scipy needed."""
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
    """Convert win probability to American odds string (no vig)."""
    if p is None or p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        odds = round(-100 * p / (1 - p))
    else:
        odds = round(100 * (1 - p) / p)
    return f'+{odds}' if odds > 0 else str(odds)


def get_f5_fair_odds(expected_runs):
    """
    Given expected F5 runs (NegBin point estimate), use Poisson CDF to
    compute fair implied odds for under 1.5 and under 2.5 F5 team totals.
    Returns dict of fair American odds (no vig) for both lines.
    """
    if expected_runs is None:
        return {
            'fair_under_1_5': None,
            'fair_over_1_5':  None,
            'fair_under_2_5': None,
            'fair_over_2_5':  None,
        }

    lam = expected_runs
    p_under_1_5 = poisson_cdf(1, lam)
    p_under_2_5 = poisson_cdf(2, lam)

    return {
        'fair_under_1_5': prob_to_american(p_under_1_5),
        'fair_over_1_5':  prob_to_american(1 - p_under_1_5),
        'fair_under_2_5': prob_to_american(p_under_2_5),
        'fair_over_2_5':  prob_to_american(1 - p_under_2_5),
    }


def get_confidence_tier(std_from_mean, is_away, line_point, k_rate):
    """
    Qualifier-based confidence tier for F5 under flags.
    Validated May 2026 on full 2025 Pinnacle fixed lines backtest (n=504).

    Qualifiers (0-4):
      q_sigma:    std_from_mean ≤ -1.5        (+1)
      q_away:     batting team is away         (+1)
      q_krate_lo: pitcher K rate ≥ 26%        (+1)
      q_krate_hi: pitcher K rate ≥ 28%        (+1)

    Bronze promotion: 0-2 qualifiers but K≥26% or line=2.5 → Bronze
    
    Outcomes:
      Gated:   no K≥26% and no line=2.5      (n=107, 67.3% hit)
      Bronze:  K≥26% or line=2.5 (0-2 quals) (n=141, 77.3% hit, +0.480 ROI)
      Silver:  3 qualifiers                   (n=157, 81.5% hit, +0.554 ROI)
      Gold:    4 qualifiers                   (n=99,  89.9% hit, +0.699 ROI)

    U2.5 overlay (manual): Bronze→Silver confidence, Silver→Gold confidence
    Minimum odds: U1.5 +100 | U2.5 -200 | U3.5 -350
    """
    q_sigma    = 1 if std_from_mean <= -1.5 else 0
    q_away     = 1 if is_away else 0
    q_krate_lo = 1 if (k_rate is not None and k_rate >= 0.26) else 0
    q_krate_hi = 1 if (k_rate is not None and k_rate >= 0.28) else 0
    q_line     = 1 if (line_point is not None and line_point <= 2.5) else 0

    total = q_sigma + q_away + q_krate_lo + q_krate_hi

    # Tier assignment
    if total == 4:
        label, color = 'Gold', 'gold'
    elif total == 3:
        label, color = 'Silver', 'silver'
    elif total <= 2 and (q_krate_lo or q_line):
        label, color = 'Bronze', 'bronze'
    else:
        return None  # gated — no K≥26% and no line=2.5

    # Unit sizing by tier
    unit_map = {
        'bronze': {'u15': '0.5u', 'u25': '1u'},
        'silver': {'u15': '1u',   'u25': '1.5u'},
        'gold':   {'u15': '1.5u', 'u25': '2u'},
    }
    units = unit_map.get(color, {'u15': '0.5u', 'u25': '1u'})

    return {
        'qualifier_count': total,
        'label':           label,
        'color':           color,
        'q_sigma':         bool(q_sigma),
        'q_away':          bool(q_away),
        'q_krate_lo':      bool(q_krate_lo),
        'q_krate_hi':      bool(q_krate_hi),
        'min_u15':         '+100',
        'min_u25':         '-200',
        'unit_u15':        units['u15'],
        'unit_u25':        units['u25'],
    }


def compute_run_edge(expected_f5_runs, f5_tt_line):
    """Positive = under value. Bet threshold: run_edge >= 0.40."""
    if expected_f5_runs is None or f5_tt_line is None:
        return None
    return round(f5_tt_line - expected_f5_runs, 3)


# ── Heat map scoring ──────────────────────────────────────────────────────────

def compute_batter_overlap(batter_id, pitcher_id, batter_hand, pitcher_hand, model):
    """
    Compute overlap score for a single batter vs pitcher.
    Returns float or None if insufficient data.
    Score > 1.0 = pitcher tends to throw to batter's preferred zones.
    Score < 1.0 = pitcher avoids batter's preferred zones.
    """
    batter_profiles    = model.get('batter_ev_profiles_lr', {})
    pitcher_tendencies = model.get('pitcher_tendency_map', {})
    archetypes         = model.get('archetypes', {})

    if batter_id not in batter_profiles: return None
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


def get_heatmap_flags(games, model):
    """
    For each game with a confirmed lineup, compute team overlap scores
    and flag games exceeding validated thresholds.

    F5 signal:
      Over:  score > HEATMAP_OVER_THRESHOLD  (+1.5 std)
      Under: score < HEATMAP_UNDER_THRESHOLD (-2.0 std)

    FG three-leg under signal (June 1 – July 15 only):
      blend_70_30 z ≤ -1.0
      opp_bp_weak_z: 0.5 → 1.0 (band)
      off_z ≥ 0.0

    Each flag includes NegBin expected F5 runs, fair odds, and FG signal.
    """
    pitcher_scores    = model.get('all_pitcher_arch_scores', {})
    nb_bundle         = model.get('negbin_model_params')
    contact_rates     = model.get('pitcher_contact_rates', {})  # {pitcher_id: {k_rate: float}}
    fg_in_window      = is_fg_valid_window()
    flags             = []

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

            # K rate for qualifier — from pitcher_contact_rates
            pitcher_k_rate = None
            if contact_rates and pitcher_id in contact_rates:
                pitcher_k_rate = contact_rates[pitcher_id].get('k_rate')

            scores         = []
            scored_batters = 0

            for batter in lineup:
                batter_id   = batter['id']
                batter_hand = batter.get('hand', 'R')
                s = compute_batter_overlap(
                    batter_id, pitcher_id,
                    batter_hand, pitcher_hand,
                    model
                )
                if s is not None:
                    scores.append(s)
                    scored_batters += 1

            if scored_batters < 3:
                continue

            team_score    = sum(scores) / len(scores)
            std_from_mean = (team_score - HEATMAP_MEAN) / HEATMAP_STD

            # NegBin expected runs + fair odds
            expected_f5 = get_expected_f5_runs(
                overlap_k_mod = team_score,
                model_bundle  = nb_bundle,
            )
            fair_odds = get_f5_fair_odds(expected_f5)

            # FG three-leg under signal
            fg_flag = compute_fg_under_flag(
                batting_team      = batting_team,
                opp_team          = fielding_team,
                blend_70_30_score = team_score,
            )

            if team_score >= HEATMAP_OVER_THRESHOLD:
                signal = 'over'
            elif team_score <= HEATMAP_UNDER_THRESHOLD:
                signal = 'under'
            else:
                # Not an F5 flag — but still compute FG signal for display
                # Only include in output if FG signal fires
                if fg_flag['fg_under_signal']:
                    signal = 'fg_under_only'
                else:
                    continue

            # Qualifier-based confidence tier (gates 0-1 qualifiers)
            confidence = None
            is_away    = (batting_team == game['away_team'])
            if signal == 'under':
                confidence = get_confidence_tier(
                    std_from_mean = std_from_mean,
                    is_away       = is_away,
                    line_point    = None,  # q_line retired — check U2.5 manually
                    k_rate        = pitcher_k_rate,
                )
                if confidence is None:
                    continue  # gated — 0-1 qualifiers

            flags.append({
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
                'fair_under_1_5':   fair_odds['fair_under_1_5'],
                'fair_over_1_5':    fair_odds['fair_over_1_5'],
                'fair_under_2_5':   fair_odds['fair_under_2_5'],
                'fair_over_2_5':    fair_odds['fair_over_2_5'],
                # Qualifier-based confidence tier
                'confidence_label':   confidence['label']           if confidence else None,
                'confidence_color':   confidence['color']           if confidence else None,
                'qualifier_count':    confidence['qualifier_count'] if confidence else 0,
                'q_sigma':            confidence['q_sigma']         if confidence else False,
                'q_away':             confidence['q_away']          if confidence else False,
                'q_krate_lo':         confidence['q_krate_lo']      if confidence else False,
                'q_krate_hi':         confidence['q_krate_hi']      if confidence else False,
                'min_u15':            confidence['min_u15']         if confidence else None,
                'min_u25':            confidence['min_u25']         if confidence else None,
                'unit_u15':           confidence['unit_u15']        if confidence else None,
                'unit_u25':           confidence['unit_u25']        if confidence else None,
                # FG two-leg under signal
                'fg_under_signal':  fg_flag['fg_under_signal'],
                'fg_starter_z':     fg_flag['starter_z'],
                'fg_opp_bp_weak_z': fg_flag['opp_bp_weak_z'],
                'fg_off_z':         fg_flag['off_z'],
                'fg_bp_in_band':    fg_flag['bp_in_band'],
                'fg_in_window':     fg_flag['in_valid_window'],
                'fg_reason':        fg_flag['reason'],
                'pitcher_k_rate':   round(pitcher_k_rate, 3) if pitcher_k_rate else None,

                'game_time':        game.get('game_time'),
            })

    flags.sort(key=lambda x: abs(x['std_from_mean']), reverse=True)

    # ── Combined F5 game total signal ─────────────────────────────────────────
    # Signal: either starter ≤ -2.0σ + away suppressed (≤-1.5σ)
    #         + home NOT suppressed (>-1.5σ) + home K rate < 28%
    # Validated: n=72, 69.4% U4.5 hit, be=-227
    # Min odds: U4.5 -200, U5.5 -300
    game_flags = {}
    for f in flags:
        if f['signal'] != 'under':
            continue
        g = f['game']
        if g not in game_flags:
            game_flags[g] = []
        game_flags[g].append(f)

    for f in flags:
        f['combined_f5_signal'] = False

    for game_str, gflags in game_flags.items():
        if len(gflags) < 2:
            continue
        away_flag = next((f for f in gflags if f['q_away']), None)
        home_flag = next((f for f in gflags if not f['q_away']), None)
        if not away_flag or not home_flag:
            continue

        away_std     = away_flag['std_from_mean']
        home_std     = home_flag['std_from_mean']
        home_k_rate  = home_flag.get('pitcher_k_rate') or 1.0

        either_sigma_20  = away_std <= -2.0 or home_std <= -2.0
        away_suppressed  = away_std <= -1.5
        home_not_sup     = home_std > -1.5
        home_low_k       = home_k_rate < 0.28

        if either_sigma_20 and away_suppressed and home_not_sup and home_low_k:
            away_flag['combined_f5_signal'] = True
            home_flag['combined_f5_signal'] = True

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
        model    = load_model()
        today    = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
        games    = get_lineups_and_starters(today)
        picks    = get_hr_picks(games, model)
        complete = sum(1 for g in games if g['home_lineup'] and g['away_lineup'])

        games_out = [{'away': g['away_team'], 'home': g['home_team'],
                      'complete': bool(g['home_lineup'] and g['away_lineup']),
                      'game_time': g.get('game_time')}
                     for g in games]

        heatmap_flags = get_heatmap_flags(games, model)

        # Separate FG signals for UI — flags where FG under fired
        fg_under_flags = [f for f in heatmap_flags if f.get('fg_under_signal')]

        return jsonify({
            'date':            today,
            'complete':        complete,
            'total':           len(games),
            'picks':           picks,
            'heatmap_flags':   heatmap_flags,
            'fg_under_flags':  fg_under_flags,
            'fg_in_window':    is_fg_valid_window(),
            'games':           games_out,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500




# ── Telegram + Redis notify route ─────────────────────────────────────────────

NOTIFY_SECRET       = os.environ.get('NOTIFY_SECRET', '')
TELEGRAM_BOT_TOKEN  = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID    = os.environ.get('TELEGRAM_CHAT_ID', '')
UPSTASH_URL         = os.environ.get('UPSTASH_REDIS_REST_URL', '')
UPSTASH_TOKEN       = os.environ.get('UPSTASH_REDIS_REST_TOKEN', '')




def redis_set(key, value, ex=86400):
    """Set a key in Upstash Redis with TTL in seconds (default 24h)."""
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
    """Fetch a key from Upstash Redis REST API. Returns value string or None."""
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
    """Send a message to the configured Telegram chat."""
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'},
            timeout=10
        )
    except Exception as e:
        print(f"Telegram send error: {e}")


def flag_key(flag):
    """Unique identifier for a flag — game + batting team."""
    return f"{flag['game']}|{flag['batting_team']}"


@app.route('/api/notify')
def api_notify():
    # Validate secret
    secret = request.args.get('secret', '')
    if not NOTIFY_SECRET or secret != NOTIFY_SECRET:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        # Score today's flags
        model   = load_model()
        today   = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
        games   = get_lineups_and_starters(today)
        flags   = get_heatmap_flags(games, model)

        # Only F5 under flags (what the app shows)
        under_flags = [f for f in flags if f.get('signal') == 'under']

        if not under_flags:
            return jsonify({'status': 'ok', 'new': 0, 'message': 'No flags today'})

        # Check Redis for already-notified flags today
        redis_key    = f"ozzie:notified:{today}"
        existing_raw = redis_get(redis_key)
        already_sent = set(existing_raw.split(',')) if existing_raw else set()

        # Find new flags
        new_flags = [f for f in under_flags if flag_key(f) not in already_sent]

        if not new_flags:
            return jsonify({'status': 'ok', 'new': 0, 'message': 'No new flags'})

        # Build Telegram message
        lines = [f"🎯 <b>Ozzie — {today}</b>", f"{len(new_flags)} new flag(s)\n"]
        for f in new_flags:
            tier  = f.get('confidence_label', '—')
            sigma = f.get('std_from_mean', 0)
            medal = {'Gold': '🥇', 'Silver': '🥈', 'Bronze': '🥉'}.get(tier, '🥉')
            fg    = ' + FG ✅' if f.get('fg_under_signal') else ''
            comb  = ' + COMB ⚡' if f.get('combined_f5_signal') else ''
            time  = f" — {f['game_time']}" if f.get('game_time') else ''
            lines.append(
                f"{medal} <b>{f['batting_team']}</b> vs {f['pitcher_name']} "
                f"({tier}, {sigma:+.2f}σ){fg}{comb}{time}"
            )
            if f.get('combined_f5_signal'):
                lines.append(f"   ⚡ F5 Combined: U4.5 @ -200 | U5.5 @ -300")
            u15 = f.get('min_u15','—')
            u25 = f.get('min_u25','—')
            uu15 = f.get('unit_u15','')
            uu25 = f.get('unit_u25','')
            lines.append(
                f"   U1.5 {u15} ({uu15}) | U2.5 {u25} ({uu25})"
            )

        send_telegram('\n'.join(lines))

        # Update Redis — add new flag keys, keep TTL 24h
        all_sent = already_sent | {flag_key(f) for f in new_flags}
        redis_set(redis_key, ','.join(all_sent), ex=86400)

        return jsonify({'status': 'ok', 'new': len(new_flags),
                        'flags': [flag_key(f) for f in new_flags]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Load FG profiles at startup
load_fg_profiles()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
