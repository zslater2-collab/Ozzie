import os
import pickle
import requests
import pandas as pd
import pytz
from datetime import datetime
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import gdown
import tempfile

try:
    from scipy.stats import nbinom
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'zach-picks-secret-2026')
APP_PASSWORD    = os.environ.get('APP_PASSWORD', 'picks2026')

FILE_IDS = {
    'all_pitcher_arch_scores':  '1qA9iSXEv1ONRlXvm0O5TzLSQTUKFVQGn',
    'arch_hitter_map_combined': '15wE3HfjaR4g68YnwcSCQV76_t0F4I7YG',
    'archetypes_combined':      '1WNBjJ98Q8n20oDjF5AMVFeeUIJiqkz4h',
    'all_parks':                '1Wch81FIHxpoJXboFOV3p3C_06PFnUNB7',
    'team_bullpen_scores':      '1-W_hgheeGdMeSDA6enW2EJgvXLXtzqlP',
    'batter_power_map':         '1ZHGuGMmkjd-uW2sbKq1wMMq3p4zSMJbg',
    # ── Heatmap / team total models ──────────────────────────────────────
    'pitcher_tendency_map':     '1u328HojWnhcQWlgx0SW5DX-WrsThBxon',
    'batter_ev_profiles_lr':    '1dnFavwW5CPXoJy3IBZueAYOokb3oxXyE',
    'pitcher_contact_rates':    '1OzWCA_NviSwLLzh1ZtgLildwFQwRk6ly',
    'batter_walk_rates':        '1gQNL-3v7txA9V87a9VsxzZmLZCCsgJ3m',
    # ── Full game innings gate ────────────────────────────────────────────
    # Upload pitcher_innings_share_2025.pkl to Drive and paste ID here
    'pitcher_innings_share':    'REPLACE_WITH_DRIVE_ID',
}

LEAGUE_AVG      = 3.88
JUICY_THRESHOLD = 5.0
MIN_SENSOR_PA   = 20
STARTER_WEIGHT  = 0.80
BULLPEN_WEIGHT  = 0.20
ASSUMED_PAS     = 4
TB3_MULTIPLIER  = 1.3

# ── F5 heatmap thresholds (validated, out-of-sample 2025) ────────────────────
OVERLAP_POP_MEAN    = 0.9984
OVERLAP_POP_STD     = 0.0598
WALK_POP_MEAN       = 0.0834
WALK_POP_STD        = 0.0104
WALK_BLEND_WEIGHT   = 0.30
HEATMAP_MEAN        = 0.9984
HEATMAP_STD         = 0.0455
HEATMAP_UNDER_THRESHOLD = HEATMAP_MEAN - 2.0 * HEATMAP_STD   # -2.0σ
LEAGUE_K_RATE           = 0.217

# ── Full game thresholds (starter-only, validated Path A backtest) ────────────
# overlap_k_mod distribution: mean=0.998363, std=0.059773 (from backtest)
# -1.0σ threshold; IP gate applied separately
FG_BLEND_MEAN       = 0.998363
FG_BLEND_STD        = 0.059773
FG_UNDER_THRESHOLD  = FG_BLEND_MEAN - 1.0 * FG_BLEND_STD     # 0.9386
MIN_STARTER_IP      = 5.0   # avg IP gate — filters short starters (validated Q2)

# ── Game total thresholds (both-suppressed filter, validated game total backtest) ─
# Both teams must independently score below zero Z (both_under = True)
# Combined Z-score of sum must clear -1.5σ
# Backtest: Pinnacle +0.145 ROI / DK +0.127 ROI at n=114
# Population distribution: sum_z mean≈0, std≈1.422 (near √2 — teams independent)
GT_SUM_Z_MEAN   = 0.0204    # from game_total_signal_check
GT_SUM_Z_STD    = 1.4218
GT_UNDER_SIGMA  = 1.5       # -1.5σ threshold on combined Z
GT_UNDER_THR    = GT_SUM_Z_MEAN - GT_UNDER_SIGMA * GT_SUM_Z_STD  # ≈ -2.112

# Per-team Z uses the same FG_BLEND distribution
# both_under: both teams' overlap_k_mod_z < 0 (independently suppressed)

# ── F5 Combined Total thresholds (Kalshi: totals_1st_5_innings) ───────────────
# Both starters must independently score below their F5 mean (both z < 0)
# Combined Z of the pair must clear -1.5σ
# Backtest (Pinnacle): +0.099 ROI at n=123 — conservative on sharp book
# Kalshi expected stronger: lines set more mechanically (~5.0 vs Pinnacle 4.55)
# Signal check: flagged games avg 4.36 F5 combined vs population 4.96
# F5 walk-blended signal — same as per-team F5 under (+0.640 ROI)
F5C_SUM_Z_MEAN  = 0.0221    # from f5_combined_signal_check
F5C_SUM_Z_STD   = 1.3390
F5C_UNDER_SIGMA = 1.5       # -1.5σ threshold on combined Z
F5C_UNDER_THR   = F5C_SUM_Z_MEAN - F5C_UNDER_SIGMA * F5C_SUM_Z_STD  # ≈ -1.987

# ── Fair odds calibration — Negative Binomial fit on fg_runs (Script R) ──────
# E[fg_runs] = FG_BETA0 + FG_BETA1 × overlap_score
# NB dispersion: r=3.181 p=r/(r+μ) per score
# Fit validated: deltas < 0.5% across full run distribution
FG_BETA0 = -1.005553
FG_BETA1 =  5.448378
FG_NB_R  =  3.1811
FG_LINES = [3.5, 4.5, 5.5]

# Empirical fallback table if scipy unavailable
# Keyed by overlap score bucket upper bound → {line: american_odds}
_FAIR_ODDS_FALLBACK = {
    0.88: {3.5: -130, 4.5: -231, 5.5: -489},
    0.91: {3.5: -152, 4.5: -250, 5.5: -473},
    0.94: {3.5: -101, 4.5: -180, 5.5: -300},
    0.97: {3.5: +116, 4.5: -145, 5.5: -232},
    1.00: {3.5: +113, 4.5: -145, 5.5: -227},
    1.03: {3.5: +127, 4.5: -137, 5.5: -207},
    1.06: {3.5: +138, 4.5: -122, 5.5: -184},
    1.09: {3.5: +145, 4.5: -124, 5.5: -217},
    9.99: {3.5: +159, 4.5: -102, 5.5: -168},
}

_model_cache      = None
_model_cache_time = None
MODEL_CACHE_TTL   = 3600


# ── Fair odds ─────────────────────────────────────────────────────────────────

def get_fair_odds(overlap_score, lines=FG_LINES):
    """
    Return fair American odds for under at each line given overlap_k_mod score.
    Uses Negative Binomial(r=3.181, p=r/(r+μ)) where μ = β0 + β1 × score.
    Falls back to empirical lookup table if scipy unavailable.
    Note: calibrated on full game runs. F5 odds are approximate.
    """
    lam = max(FG_BETA0 + FG_BETA1 * overlap_score, 0.5)

    def _fmt(odds):
        return f'+{odds}' if odds > 0 else str(odds)

    if SCIPY_AVAILABLE:
        p_nb = FG_NB_R / (FG_NB_R + lam)   # recompute p for this score's mean
        result = {}
        for line in lines:
            k    = int(line - 0.5)           # 4.5 → 4, 3.5 → 3
            prob = nbinom.cdf(k, FG_NB_R, p_nb)
            prob = min(max(prob, 0.001), 0.999)
            odds = round(-(prob / (1 - prob)) * 100) if prob >= 0.5 \
                   else round(((1 - prob) / prob) * 100)
            result[line] = _fmt(odds)
        return result
    else:
        # Empirical fallback
        for cutoff in sorted(_FAIR_ODDS_FALLBACK.keys()):
            if overlap_score <= cutoff:
                bucket = _FAIR_ODDS_FALLBACK[cutoff]
                return {line: _fmt(bucket.get(line, 0)) for line in lines}
        bucket = _FAIR_ODDS_FALLBACK[9.99]
        return {line: _fmt(bucket.get(line, 0)) for line in lines}


def get_f5c_fair_odds(sum_z, lines=[3.5, 4.5, 5.5, 6.5]):
    """
    Fair odds for F5 combined total (both teams through 5 innings).
    E[F5_combined] = 4.96 + 0.28 × sum_z
    NegBin: r=3.97 calibrated from F5 combined distribution (mean=4.96, std=3.34)
    Lines: 4.5 / 5.5 / 6.5 (population mean = 4.96)
    """
    F5C_NB_R  = 3.97
    lam       = max(4.96 + 0.28 * sum_z, 0.5)
    p_nb      = F5C_NB_R / (F5C_NB_R + lam)

    def _fmt(o): return f'+{o}' if o > 0 else str(o)

    if SCIPY_AVAILABLE:
        result = {}
        for line in lines:
            k    = int(line - 0.5)
            prob = min(max(nbinom.cdf(k, F5C_NB_R, p_nb), 0.001), 0.999)
            odds = round(-(prob/(1-prob))*100) if prob >= 0.5 \
                   else round(((1-prob)/prob)*100)
            result[line] = _fmt(odds)
        return result
    else:
        # Empirical fallback at -1.5σ combined
        for line, p in zip([4.5, 5.5, 6.5], [0.612, 0.701, 0.806]):
            pass
        return {line: _fmt(round(-(p/(1-p))*100))
                for line, p in zip([3.5, 4.5, 5.5, 6.5], [0.376, 0.612, 0.701, 0.806])}


def get_gt_fair_odds(sum_z, lines=[7.5, 8.5, 9.5]):
    """
    Fair odds for game total (both teams, full game).
    E[game_total] = 8.88 + 0.38 × sum_z
    NegBin: r=6.57 calibrated from game total distribution (mean=8.88, std=4.57)
    Lines: 7.5 / 8.5 / 9.5 (population mean = 8.88)
    """
    GT_NB_R = 6.57
    lam     = max(8.88 + 0.38 * sum_z, 0.5)
    p_nb    = GT_NB_R / (GT_NB_R + lam)

    def _fmt(o): return f'+{o}' if o > 0 else str(o)

    if SCIPY_AVAILABLE:
        result = {}
        for line in lines:
            k    = int(line - 0.5)
            prob = min(max(nbinom.cdf(k, GT_NB_R, p_nb), 0.001), 0.999)
            odds = round(-(prob/(1-prob))*100) if prob >= 0.5 \
                   else round(((1-prob)/prob)*100)
            result[line] = _fmt(odds)
        return result
    else:
        # Empirical fallback at -1.5σ combined
        return {line: _fmt(round(-(p/(1-p))*100))
                for line, p in zip([7.5, 8.5, 9.5], [0.514, 0.579, 0.686])}


# ── Model loading ─────────────────────────────────────────────────────────────

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
        if fid == 'REPLACE_WITH_DRIVE_ID':
            model[name] = {}
            continue
        try:
            model[name] = download_pkl(fid)
        except Exception:
            model[name] = {}
    model['arch_hitter_map'] = model.get('arch_hitter_map_combined', {})
    model['archetypes']      = model.get('archetypes_combined', {})
    _model_cache      = model
    _model_cache_time = now
    return model


# ── Lineup fetch ──────────────────────────────────────────────────────────────

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
            home = g['teams']['home']['team'].get('abbreviation') or \
                   g['teams']['home']['team'].get('name', 'HOME')
            away = g['teams']['away']['team'].get('abbreviation') or \
                   g['teams']['away']['team'].get('name', 'AWAY')
            hp   = g['teams']['home'].get('probablePitcher', {})
            ap   = g['teams']['away'].get('probablePitcher', {})
            home_lineup, away_lineup = [], []
            try:
                bs = requests.get(
                    f"https://statsapi.mlb.com/api/v1/game/{gid}/boxscore",
                    timeout=15).json()
                for team_key, lineup in [('home', home_lineup), ('away', away_lineup)]:
                    td            = bs.get('teams', {}).get(team_key, {})
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
            })
    return games


# ── Shared scoring helpers ────────────────────────────────────────────────────

def compute_batter_overlap(batter_id, pitcher_id, batter_hand, pitcher_hand, model):
    batter_profiles    = model.get('batter_ev_profiles_lr', {})
    pitcher_tendencies = model.get('pitcher_tendency_map', {})
    archetypes         = model.get('archetypes', {})
    if batter_id not in batter_profiles:    return None
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

def _compute_overlap_k_mod(lineup, pitcher_id, pitcher_hand, model):
    """
    Shared steps 1+2: raw overlap → K-rate modified score.
    Returns (overlap_k_mod, walk_vals) or (None, []) if insufficient data.
    """
    contact_rates     = model.get('pitcher_contact_rates', {})
    batter_walk_rates = model.get('batter_walk_rates', {})
    raw_scores, walk_vals = [], []

    for batter in lineup:
        batter_id   = batter['id']
        batter_hand = batter.get('hand', 'R')
        s = compute_batter_overlap(batter_id, pitcher_id, batter_hand, pitcher_hand, model)
        if s is not None:
            raw_scores.append(s)
        w = batter_walk_rates.get(batter_id, {}).get('walk_rate')
        if w is not None:
            walk_vals.append(w)

    if len(raw_scores) < 3:
        return None, []

    p_stats  = contact_rates.get(pitcher_id, {})
    k_rate   = p_stats.get('k_rate', LEAGUE_K_RATE)
    if not isinstance(k_rate, float) or k_rate != k_rate:
        k_rate = LEAGUE_K_RATE
    k_mod         = (1 - k_rate) / (1 - LEAGUE_K_RATE)
    overlap_k_mod = (sum(raw_scores) / len(raw_scores)) * k_mod
    return overlap_k_mod, walk_vals

def pa_rate_to_game_odds(pa_rate_pct):
    pa_rate   = pa_rate_pct / 100
    game_prob = 1 - (1 - pa_rate) ** ASSUMED_PAS
    if game_prob <= 0: return '+9999'
    fair = round((1 / game_prob - 1) * 100)
    return f'+{fair}' if fair > 0 else str(fair)

def get_bullpen_score(team, arch_key, model):
    bs = model.get('team_bullpen_scores', {})
    if team in bs and arch_key in bs[team]:
        return bs[team][arch_key].get('shrunk_rate', LEAGUE_AVG)
    return LEAGUE_AVG

def get_park_factor(fielding_team, batter_hand, model):
    all_parks = model.get('all_parks', {})
    if fielding_team not in all_parks: return 1.0
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


# ── F5 Team Total Under flags ─────────────────────────────────────────────────

def get_heatmap_flags(games, model):
    """
    F5 team total under flags.
    Signal: overlap_k_mod + 70/30 walk blend, threshold -2.0σ.
    Adds fair_odds (approximate — calibrated on full game runs) and avg_ip.
    """
    pitcher_scores   = model.get('all_pitcher_arch_scores', {})
    innings_share    = model.get('pitcher_innings_share', {})
    flags = []

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

            overlap_k_mod, walk_vals = _compute_overlap_k_mod(
                lineup, pitcher_id, pitcher_hand, model
            )
            if overlap_k_mod is None:
                continue

            # 70/30 walk blend
            if len(walk_vals) >= 3:
                lineup_walk_rate = sum(walk_vals) / len(walk_vals)
                ov_z    = (overlap_k_mod    - OVERLAP_POP_MEAN) / OVERLAP_POP_STD
                wk_z    = (lineup_walk_rate - WALK_POP_MEAN)    / WALK_POP_STD
                blend_z = ov_z * (1 - WALK_BLEND_WEIGHT) + wk_z * WALK_BLEND_WEIGHT
                team_score = blend_z * OVERLAP_POP_STD + OVERLAP_POP_MEAN
            else:
                team_score = overlap_k_mod

            if team_score > HEATMAP_UNDER_THRESHOLD:
                continue

            std_from_mean = (team_score - HEATMAP_MEAN) / HEATMAP_STD

            # Starter avg IP (informational for F5 — no gate applied)
            ip_data = innings_share.get(int(pitcher_id)) if pitcher_id else None
            avg_ip  = round(ip_data['avg_ip'], 1) if ip_data else None

            flags.append({
                'game':            game_str,
                'batting_team':    batting_team,
                'fielding_team':   fielding_team,
                'pitcher_name':    pitcher_name,
                'pitcher_hand':    pitcher_hand,
                'pitcher_id':      pitcher_id,
                'overlap_score':   round(team_score, 4),
                'std_from_mean':   round(std_from_mean, 2),
                'signal':          'under',
                'batters_scored':  sum(1 for b in lineup if compute_batter_overlap(
                                       b['id'], pitcher_id, b.get('hand','R'),
                                       pitcher_hand, model) is not None),
                'lineup_complete': len(lineup) >= 8,
                'avg_ip':          avg_ip,
                'fair_odds':       get_fair_odds(team_score),
            })

    flags.sort(key=lambda x: abs(x['std_from_mean']), reverse=True)
    return flags


# ── Full Game Team Total Under flags ─────────────────────────────────────────

def get_fullgame_flags(games, model):
    """
    Full game team total under flags.
    Signal: overlap_k_mod only (starter, no walk blend — matches backtest).
    Gate:   avg_ip >= MIN_STARTER_IP (5.0) — validated Q2, +0.134 ROI lift.
    Threshold: -1.0σ from FG_BLEND_MEAN/STD.
    Adds fair_odds (NegBin calibrated) and avg_ip.
    """
    pitcher_scores = model.get('all_pitcher_arch_scores', {})
    innings_share  = model.get('pitcher_innings_share', {})
    flags = []

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

            # ── IP gate — skip short starters ────────────────────────────
            ip_data = innings_share.get(int(pitcher_id)) if pitcher_id else None
            avg_ip  = ip_data['avg_ip'] if ip_data else None
            if innings_share and (avg_ip is None or avg_ip < MIN_STARTER_IP):
                # innings_share loaded but pitcher below threshold — skip
                # if innings_share empty (Drive ID not set), gate is bypassed
                if innings_share:
                    continue

            # ── Starter signal (overlap_k_mod, no walk blend) ─────────────
            overlap_k_mod, _ = _compute_overlap_k_mod(
                lineup, pitcher_id, pitcher_hand, model
            )
            if overlap_k_mod is None:
                continue

            std_from_mean = (overlap_k_mod - FG_BLEND_MEAN) / FG_BLEND_STD

            if overlap_k_mod > FG_UNDER_THRESHOLD:
                continue

            flags.append({
                'game':            game_str,
                'batting_team':    batting_team,
                'fielding_team':   fielding_team,
                'pitcher_name':    pitcher_name,
                'pitcher_hand':    pitcher_hand,
                'pitcher_id':      pitcher_id,
                'overlap_score':   round(overlap_k_mod, 4),
                'std_from_mean':   round(std_from_mean, 2),
                'signal':          'under',
                'market':          'full_game',
                'batters_scored':  sum(1 for b in lineup if compute_batter_overlap(
                                       b['id'], pitcher_id, b.get('hand','R'),
                                       pitcher_hand, model) is not None),
                'lineup_complete': len(lineup) >= 8,
                'avg_ip':          round(avg_ip, 1) if avg_ip else None,
                'fair_odds':       get_fair_odds(overlap_k_mod),
            })

    flags.sort(key=lambda x: abs(x['std_from_mean']), reverse=True)
    return flags


# ── HR Watchlist ──────────────────────────────────────────────────────────────

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
                    arch_data    = pitcher['archetypes'].get(arch_key, {})
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


# ── F5 Combined Total Under flags (Kalshi market) ────────────────────────────

def get_f5combined_flags(games, model):
    """
    F5 combined total under flags — both starters suppressing both lineups.
    Uses walk-blended F5 signal (same as per-team F5 under, +0.640 ROI).
    Gate: both teams independently z < 0 AND combined Z ≤ -1.5σ.
    Backtest: Pinnacle +0.099 ROI. Kalshi expected stronger (mechanical lines).
    Market: totals_1st_5_innings — available on Kalshi/Robinhood, no limits.
    Flagged games avg 4.36 F5 combined runs vs population mean 4.96.
    """
    pitcher_scores = model.get('all_pitcher_arch_scores', {})
    flags = []

    for game in games:
        game_str = f"{game['away_team']}@{game['home_team']}"
        matchups  = [
            (game['away_lineup'], game['home_pitcher_id'],
             game['home_pitcher_name'], game['away_team'], game['home_team']),
            (game['home_lineup'], game['away_pitcher_id'],
             game['away_pitcher_name'], game['home_team'], game['away_team']),
        ]

        team_scores = []
        for lineup, pitcher_id, pitcher_name, batting_team, fielding_team in matchups:
            if not lineup or not pitcher_id:
                continue

            pitcher_hand = 'R'
            if pitcher_id in pitcher_scores:
                pitcher_hand = pitcher_scores[pitcher_id].get('p_throws', 'R')

            overlap_k_mod, walk_vals = _compute_overlap_k_mod(
                lineup, pitcher_id, pitcher_hand, model
            )
            if overlap_k_mod is None:
                continue

            # F5 walk-blended score (same as per-team F5 signal)
            if len(walk_vals) >= 3:
                lineup_walk_rate = sum(walk_vals) / len(walk_vals)
                ov_z    = (overlap_k_mod    - OVERLAP_POP_MEAN) / OVERLAP_POP_STD
                wk_z    = (lineup_walk_rate - WALK_POP_MEAN)    / WALK_POP_STD
                blend_z = ov_z * (1 - WALK_BLEND_WEIGHT) + wk_z * WALK_BLEND_WEIGHT
                f5_score = blend_z * OVERLAP_POP_STD + OVERLAP_POP_MEAN
            else:
                f5_score = overlap_k_mod

            # Per-team Z using F5/heatmap distribution
            f5_z = (f5_score - HEATMAP_MEAN) / HEATMAP_STD

            team_scores.append({
                'batting_team':   batting_team,
                'pitcher_name':   pitcher_name,
                'pitcher_hand':   pitcher_hand,
                'pitcher_id':     pitcher_id,
                'f5_score':       f5_score,
                'f5_z':           f5_z,
                'lineup_complete': len(lineup) >= 8,
                'batters_scored': sum(
                    1 for b in lineup
                    if compute_batter_overlap(b['id'], pitcher_id,
                                              b.get('hand','R'),
                                              pitcher_hand, model) is not None
                ),
            })

        if len(team_scores) != 2:
            continue

        z_scores = [ts['f5_z'] for ts in team_scores]

        # Gate 1: both teams independently suppressed
        if not all(z < 0 for z in z_scores):
            continue

        # Gate 2: combined Z clears -1.5σ threshold
        sum_z      = sum(z_scores)
        combined_z = (sum_z - F5C_SUM_Z_MEAN) / F5C_SUM_Z_STD
        if combined_z > -F5C_UNDER_SIGMA:
            continue

        flags.append({
            'game':            game_str,
            'market':          'f5_combined',
            'signal':          'under',
            'combined_z':      round(combined_z, 2),
            'sum_z':           round(sum_z, 3),
            # Team A (away batting vs home pitcher)
            'team_a':          team_scores[0]['batting_team'],
            'pitcher_a':       team_scores[0]['pitcher_name'],
            'pitcher_a_hand':  team_scores[0]['pitcher_hand'],
            'score_a':         round(team_scores[0]['f5_score'], 4),
            'z_a':             round(z_scores[0], 2),
            # Team B (home batting vs away pitcher)
            'team_b':          team_scores[1]['batting_team'],
            'pitcher_b':       team_scores[1]['pitcher_name'],
            'pitcher_b_hand':  team_scores[1]['pitcher_hand'],
            'score_b':         round(team_scores[1]['f5_score'], 4),
            'z_b':             round(z_scores[1], 2),
            'lineup_complete': all(ts['lineup_complete'] for ts in team_scores),
            'fair_odds':       get_f5c_fair_odds(sum_z),
        })

    flags.sort(key=lambda x: x['combined_z'])
    return flags


# ── Game Total Under flags ────────────────────────────────────────────────────

def get_gametotal_flags(games, model):
    """
    Game total under flags — combined suppression signal.
    Both starters must independently score below their mean (both_under).
    Combined Z-score of the pair must clear -1.5σ.
    Signal: overlap_k_mod per team, no walk blend (matches backtest).
    Threshold: GT_UNDER_THR = -1.5σ on combined sum_z distribution.
    Backtest: Pinnacle +0.145 ROI / DK +0.127 ROI at n=114 (2025).
    """
    pitcher_scores = model.get('all_pitcher_arch_scores', {})
    innings_share  = model.get('pitcher_innings_share', {})
    flags = []

    for game in games:
        # Score both matchups in this game
        game_str   = f"{game['away_team']}@{game['home_team']}"
        matchups   = [
            (game['away_lineup'], game['home_pitcher_id'],
             game['home_pitcher_name'], game['away_team'], game['home_team']),
            (game['home_lineup'], game['away_pitcher_id'],
             game['away_pitcher_name'], game['home_team'], game['away_team']),
        ]

        team_scores = []   # [(batting_team, overlap_k_mod, pitcher_name, hand, avg_ip)]
        for lineup, pitcher_id, pitcher_name, batting_team, fielding_team in matchups:
            if not lineup or not pitcher_id:
                continue

            pitcher_hand = 'R'
            if pitcher_id in pitcher_scores:
                pitcher_hand = pitcher_scores[pitcher_id].get('p_throws', 'R')

            overlap_k_mod, _ = _compute_overlap_k_mod(
                lineup, pitcher_id, pitcher_hand, model
            )
            if overlap_k_mod is None:
                continue

            ip_data = innings_share.get(int(pitcher_id)) if pitcher_id else None
            avg_ip  = round(ip_data['avg_ip'], 1) if ip_data else None

            team_scores.append({
                'batting_team': batting_team,
                'pitcher_name': pitcher_name,
                'pitcher_hand': pitcher_hand,
                'pitcher_id':   pitcher_id,
                'overlap_k_mod': overlap_k_mod,
                'avg_ip':        avg_ip,
                'batters_scored': sum(
                    1 for b in lineup
                    if compute_batter_overlap(b['id'], pitcher_id,
                                              b.get('hand','R'),
                                              pitcher_hand, model) is not None
                ),
                'lineup_complete': len(lineup) >= 8,
            })

        # Need both teams scored
        if len(team_scores) != 2:
            continue

        # Per-team Z-scores (using FG distribution)
        z_scores = [
            (ts['overlap_k_mod'] - FG_BLEND_MEAN) / FG_BLEND_STD
            for ts in team_scores
        ]

        # Gate 1: both teams independently suppressed (both z < 0)
        if not all(z < 0 for z in z_scores):
            continue

        # Gate 2: combined sum Z clears threshold
        sum_z     = sum(z_scores)
        combined_z = (sum_z - GT_SUM_Z_MEAN) / GT_SUM_Z_STD
        if combined_z > -GT_UNDER_SIGMA:
            continue

        flags.append({
            'game':           game_str,
            'market':         'game_total',
            'signal':         'under',
            'combined_z':     round(combined_z, 2),
            'sum_z':          round(sum_z, 3),
            # Team A (away batting)
            'team_a':         team_scores[0]['batting_team'],
            'pitcher_a':      team_scores[0]['pitcher_name'],
            'pitcher_a_hand': team_scores[0]['pitcher_hand'],
            'score_a':        round(team_scores[0]['overlap_k_mod'], 4),
            'z_a':            round(z_scores[0], 2),
            'avg_ip_a':       team_scores[0]['avg_ip'],
            # Team B (home batting)
            'team_b':         team_scores[1]['batting_team'],
            'pitcher_b':      team_scores[1]['pitcher_name'],
            'pitcher_b_hand': team_scores[1]['pitcher_hand'],
            'score_b':        round(team_scores[1]['overlap_k_mod'], 4),
            'z_b':            round(z_scores[1], 2),
            'avg_ip_b':       team_scores[1]['avg_ip'],
            # Lineup quality
            'lineup_complete': all(ts['lineup_complete'] for ts in team_scores),
            'fair_odds':       get_gt_fair_odds(sum_z),
        })

    flags.sort(key=lambda x: x['combined_z'])   # most suppressed first
    return flags


# ── Routes ────────────────────────────────────────────────────────────────────

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
                      'complete': bool(g['home_lineup'] and g['away_lineup'])}
                     for g in games]

        heatmap_flags    = get_heatmap_flags(games, model)
        fullgame_flags   = get_fullgame_flags(games, model)
        gametotal_flags  = get_gametotal_flags(games, model)
        f5combined_flags = get_f5combined_flags(games, model)

        return jsonify({
            'date':             today,
            'complete':         complete,
            'total':            len(games),
            'picks':            picks,
            'heatmap_flags':    heatmap_flags,
            'fullgame_flags':   fullgame_flags,
            'gametotal_flags':  gametotal_flags,
            'f5combined_flags': f5combined_flags,
            'games':            games_out,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
