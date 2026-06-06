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
HEATMAP_OVER_THRESHOLD  = HEATMAP_MEAN + 2.0 * HEATMAP_STD
HEATMAP_UNDER_THRESHOLD = HEATMAP_MEAN - 1.0 * HEATMAP_STD

OVER_OFF_Z_MIN = -1.0

F5_FAIR_ODDS = {
    'Very Juicy':  -126,
    'Juicy':       +118,
    'Average':     +133,
    'Safe':        +143,
    'Very Safe':   +191,
}

# ── OZZIE SCORING SYSTEM v2 ───────────────────────────────────────────────
# Validated: 2024 OOS + 2025 OOS + 2026 YTD, n=1034 games
# Gate: std_from_mean <= -1.0 AND >= 2 original qualifiers
#
# POINT SYSTEM (max 10):
#   Park good    +3  (TEX,BAL,MIL,CIN,DET,COL,LAD,NYY,CWS)
#   Park neutral +1
#   Park bad      0  (ATH,CHC,MIA,NYM,SF,HOU,TB,STL,BOS,SEA,PIT,MIN,CLE)
#   BB rate <6%  +2
#   GB rate 42-48% +1
#   K rate >=28% +1
#   Away team    +1
#   Sigma <=-2.0 +1

PARK_GOOD = {'TEX', 'BAL', 'MIL', 'CIN', 'DET', 'COL', 'LAD', 'NYY', 'CWS'}
PARK_BAD  = {'ATH', 'CHC', 'MIA', 'NYM', 'SF', 'HOU', 'TB', 'STL', 'BOS', 'SEA', 'PIT', 'MIN', 'CLE'}

# Per-score betting rules
# u25_min / u15_min: None = No Bet
# u15_caution: show warning symbol
SCORE_RULES = {
    9: {'color': 'diamond', 'u25_min': '-500', 'u25_units': '2u',   'u15_min': '-150', 'u15_units': '0.5u', 'u15_caution': False},
    8: {'color': 'diamond', 'u25_min': '-500', 'u25_units': '3u',   'u15_min': '-150', 'u15_units': '1.5u', 'u15_caution': False},
    7: {'color': 'gold',    'u25_min': '-300', 'u25_units': '2u',   'u15_min': '-150', 'u15_units': '0.5u', 'u15_caution': True},
    6: {'color': 'gold',    'u25_min': '-300', 'u25_units': '2u',   'u15_min': '-150', 'u15_units': '1u',   'u15_caution': False},
    5: {'color': 'silver',  'u25_min': '-300', 'u25_units': '1.5u', 'u15_min': '-150', 'u15_units': '1u',   'u15_caution': False},
    4: {'color': 'bronze',  'u25_min': '-210', 'u25_units': '0.5u', 'u15_min': None,   'u15_units': None,   'u15_caution': False},
    3: {'color': 'bronze',  'u25_min': None,   'u25_units': None,   'u15_min': None,   'u15_units': None,   'u15_caution': False},
}

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
        return {'fair_under_1_5': None, 'fair_over_1_5': None,
                'fair_under_2_5': None, 'fair_over_2_5': None}
    lam = expected_runs
    p_under_1_5 = poisson_cdf(1, lam)
    p_under_2_5 = poisson_cdf(2, lam)
    return {
        'fair_under_1_5': prob_to_american(p_under_1_5),
        'fair_over_1_5':  prob_to_american(1 - p_under_1_5),
        'fair_under_2_5': prob_to_american(p_under_2_5),
        'fair_over_2_5':  prob_to_american(1 - p_under_2_5),
    }


def get_ozzie_score(std_from_mean, is_away, k_rate, park, bb_gb_rates):
    """
    Ozzie Scoring System v2 — June 2026
    Gate: std_from_mean <= -1.0 AND >= 2 of (away, k>=26%, k>=28%)
    Returns None if game does not meet minimum requirements.
    """
    q_away     = 1 if is_away else 0
    q_krate_lo = 1 if (k_rate is not None and k_rate >= 0.26) else 0
    q_krate_hi = 1 if (k_rate is not None and k_rate >= 0.28) else 0

    if std_from_mean > -1.0:
        return None
    if (q_away + q_krate_lo + q_krate_hi) < 2:
        return None

    # Park points
    if park in PARK_GOOD:
        park_pts, park_tier = 3, 'good'
    elif park in PARK_BAD:
        park_pts, park_tier = 0, 'bad'
    else:
        park_pts, park_tier = 1, 'neutral'

    bb_rate  = bb_gb_rates.get('bb_rate') if bb_gb_rates else None
    q_bb_vlo = 1 if (bb_rate is not None and bb_rate < 0.06) else 0

    gb_rate  = bb_gb_rates.get('gb_rate') if bb_gb_rates else None
    q_gb_mid = 1 if (gb_rate is not None and 0.42 <= gb_rate < 0.48) else 0

    q_sigma_20 = 1 if std_from_mean <= -2.0 else 0

    total_score = park_pts + (q_bb_vlo * 2) + q_gb_mid + q_krate_hi + q_away + q_sigma_20

    # Gate out scores 1-2 and 3 with no u25
    if total_score < 3:
        return None

    rules = SCORE_RULES.get(total_score, SCORE_RULES[3])
    color = rules['color']

    # Label for display
    label_map = {'diamond': 'Diamond', 'gold': 'Gold', 'silver': 'Silver', 'bronze': 'Bronze'}
    label = label_map.get(color, 'Bronze')

    return {
        'ozzie_score':      total_score,
        'confidence_label': label,
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
        'min_u25':          rules['u25_min'],
        'unit_u25':         rules['u25_units'],
        'min_u15':          rules['u15_min'],
        'unit_u15':         rules['u15_units'],
        'u15_caution':      rules['u15_caution'],
    }


def compute_run_edge(expected_f5_runs, f5_tt_line):
    if expected_f5_runs is None or f5_tt_line is None:
        return None
    return round(f5_tt_line - expected_f5_runs, 3)


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
            batting_off_z = _FG_OFF_Z.get(batting_team)
            over_off_z_ok = (batting_off_z is None or batting_off_z > OVER_OFF_Z_MIN)
            over_away_ok  = is_away

            if team_score >= HEATMAP_OVER_THRESHOLD and over_off_z_ok and over_away_ok:
                signal = 'over'
            elif team_score >= HEATMAP_OVER_THRESHOLD and (not over_off_z_ok or not over_away_ok):
                signal = 'over_suppressed'
            elif team_score <= HEATMAP_UNDER_THRESHOLD:
                signal = 'under'
            else:
                if fg_flag['fg_under_signal']:
                    signal = 'fg_under_only'
                else:
                    continue

            if signal == 'over_suppressed':
                continue

            confidence = None
            if signal == 'under':
                confidence = get_ozzie_score(
                    std_from_mean=std_from_mean,
                    is_away=is_away,
                    k_rate=pitcher_k_rate,
                    park=fielding_team,
                    bb_gb_rates=bb_gb_rates)
                if confidence is None:
                    continue

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
                'over_off_z':       round(batting_off_z, 3) if batting_off_z is not None else None,
                'ozzie_score':          confidence['ozzie_score']       if confidence else None,
                'confidence_label':     confidence['confidence_label']  if confidence else None,
                'confidence_color':     confidence['confidence_color']  if confidence else None,
                'park_tier':            confidence['park_tier']         if confidence else None,
                'q_away':               confidence['q_away']            if confidence else False,
                'q_krate_lo':           confidence['q_krate_lo']        if confidence else False,
                'q_krate_hi':           confidence['q_krate_hi']        if confidence else False,
                'q_bb_vlo':             confidence['q_bb_vlo']          if confidence else False,
                'q_gb_mid':             confidence['q_gb_mid']          if confidence else False,
                'q_sigma_20':           confidence['q_sigma_20']        if confidence else False,
                'bb_rate':              confidence['bb_rate']           if confidence else None,
                'gb_rate':              confidence['gb_rate']           if confidence else None,
                'pitcher_k_rate':       round(pitcher_k_rate, 3)        if pitcher_k_rate else None,
                'min_u15':              confidence['min_u15']           if confidence else None,
                'min_u25':              confidence['min_u25']           if confidence else None,
                'unit_u15':             confidence['unit_u15']          if confidence else None,
                'unit_u25':             confidence['unit_u25']          if confidence else None,
                'u15_caution':          confidence['u15_caution']       if confidence else False,
                'fg_under_signal':      fg_flag['fg_under_signal'],
                'fg_starter_z':         fg_flag['starter_z'],
                'fg_opp_bp_weak_z':     fg_flag['opp_bp_weak_z'],
                'fg_off_z':             fg_flag['off_z'],
                'fg_bp_in_band':        fg_flag['bp_in_band'],
                'fg_in_window':         fg_flag['in_valid_window'],
                'fg_reason':            fg_flag['reason'],
                'game_time':            game.get('game_time'),
            })

    flags.sort(key=lambda x: (x.get('ozzie_score') or 0), reverse=True)

    # Combined F5 signal
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
        away_std    = away_flag['std_from_mean']
        home_std    = home_flag['std_from_mean']
        home_k_rate = home_flag.get('pitcher_k_rate') or 1.0
        if (away_std <= -2.0 or home_std <= -2.0) and \
           away_std <= -1.5 and home_std > -1.5 and home_k_rate < 0.28:
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

        heatmap_flags  = get_heatmap_flags(games, model)
        fg_under_flags = [f for f in heatmap_flags if f.get('fg_under_signal')]
        over_flags     = [f for f in heatmap_flags if f.get('signal') == 'over']

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
            'over_flags':      over_flags,
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


# ── Telegram + Redis + Sheets ─────────────────────────────────────────────────

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
            key = f"{today}|{f.get('game','')}|{f.get('batting_team','')}"
            if key in existing_keys:
                continue
            r = len(existing) + rows_added + 1
            ws.append_row([
                today,
                f.get('game', ''),
                f.get('batting_team', ''),
                f.get('pitcher_name', ''),
                f.get('confidence_label', ''),
                f.get('ozzie_score', ''),
                f.get('std_from_mean', ''),
                f.get('park_tier', ''),
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
                f'=IF(AND(Q{r}<>"",W{r}<>""),IF(W{r}<Q{r},"Yes","No"),"")'.format(r=r),
                f'=IF(AND(T{r}<>"",W{r}<>""),IF(W{r}<T{r},"Yes","No"),"")'.format(r=r),
                f'=IF(W{r}="","",IF(R{r}<>"",IF(X{r}="Yes",IF(Q{r}<0,R{r}*(100/ABS(Q{r})),R{r}*(Q{r}/100)),-R{r}),0)+IF(U{r}<>"",IF(Y{r}="Yes",IF(T{r}<0,U{r}*(100/ABS(T{r})),U{r}*(T{r}/100)),-U{r}),0))'.format(r=r),
            ], value_input_option='USER_ENTERED')
            existing_keys.add(key)
            rows_added += 1
        print(f"Google Sheets: {rows_added} rows added")
    except ImportError:
        print("gspread not installed — skipping sheet append")
    except Exception as e:
        print(f"Google Sheets error: {e}")


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
        over_flags  = [f for f in flags if f.get('signal') == 'over']
        if not under_flags and not over_flags:
            return jsonify({'status': 'ok', 'new': 0, 'message': 'No flags today'})
        redis_key    = f"ozzie:notified:{today}"
        existing_raw = redis_get(redis_key)
        already_sent = set(existing_raw.split(',')) if existing_raw else set()
        new_under = [f for f in under_flags if flag_key(f) not in already_sent]
        new_over  = [f for f in over_flags  if flag_key(f) not in already_sent]
        new_flags = new_under + new_over
        if not new_flags:
            return jsonify({'status': 'ok', 'new': 0, 'message': 'No new flags'})
        lines = [f"🎯 <b>Ozzie — {today}</b>", f"{len(new_flags)} new flag(s)\n"]
        for f in new_under:
            label   = f.get('confidence_label', '—')
            score   = f.get('ozzie_score', '—')
            sigma   = f.get('std_from_mean', 0)
            park    = f.get('park_tier', '')
            medal   = {'Diamond': '💎', 'Gold': '🥇', 'Silver': '🥈', 'Bronze': '🥉'}.get(label, '🥉')
            fg      = ' + FG ✅' if f.get('fg_under_signal') else ''
            comb    = ' + COMB ⚡' if f.get('combined_f5_signal') else ''
            time    = f" — {f['game_time']}" if f.get('game_time') else ''
            lines.append(
                f"{medal} <b>{f['batting_team']}</b> vs {f['pitcher_name']} "
                f"(#{score}, {sigma:+.2f}σ, {park}){fg}{comb}{time}"
            )
            if f.get('combined_f5_signal'):
                lines.append("   ⚡ F5 Combined: U4.5 @ -200 | U5.5 @ -300")
            # U2.5 line
            if f.get('min_u25'):
                u25_str = f"U2.5 {f['min_u25']} / {f.get('unit_u25','')}"
            else:
                u25_str = "U2.5 No Bet"
            # U1.5 line
            if f.get('min_u15'):
                caution = ' ⚠️' if f.get('u15_caution') else ''
                u15_str = f"U1.5 {f['min_u15']} / {f.get('unit_u15','')}{caution}"
            else:
                u15_str = "U1.5 No Bet"
            lines.append(f"   {u25_str} | {u15_str}")
        for f in new_over:
            sigma   = f.get('std_from_mean', 0)
            off_z   = f.get('over_off_z')
            off_str = f" off_z={off_z:+.2f}" if off_z is not None else ''
            time    = f" — {f['game_time']}" if f.get('game_time') else ''
            lines.append(
                f"📈 <b>{f['batting_team']}</b> vs {f['pitcher_name']} "
                f"(OVER, {sigma:+.2f}σ{off_str}){time}"
            )
            lines.append("   O2.5 target: better than -275 | 73.3% hit (away + off_z>-1.0, 2025 OOS)")
        send_telegram('\n'.join(lines))
        append_to_sheet(new_flags)
        all_sent = already_sent | {flag_key(f) for f in new_flags}
        redis_set(redis_key, ','.join(all_sent), ex=86400)
        return jsonify({'status': 'ok', 'new': len(new_flags),
                        'flags': [flag_key(f) for f in new_flags]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


load_fg_profiles()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
