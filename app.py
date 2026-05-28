import os
import pickle
import requests
import pandas as pd
import pytz
from datetime import datetime
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
    # ── Heat map models ──────────────────────────────────────────────────
    'pitcher_tendency_map':     '1u328HojWnhcQWlgx0SW5DX-WrsThBxon',
    'batter_ev_profiles_lr':    '1dnFavwW5CPXoJy3IBZueAYOokb3oxXyE',
    'negbin_model_params':      '122sd0M7XFhb-JlU2qE7_wQey9iTntIv9',
}

LEAGUE_AVG      = 3.88
JUICY_THRESHOLD = 5.0
MIN_SENSOR_PA   = 20
STARTER_WEIGHT  = 0.80
BULLPEN_WEIGHT  = 0.20
ASSUMED_PAS     = 4
TB3_MULTIPLIER  = 1.3

# ── Heat map thresholds (validated on full MLB 2025, out-of-sample) ───────────
HEATMAP_MEAN            = 1.0133
HEATMAP_STD             = 0.0063
HEATMAP_OVER_THRESHOLD  = HEATMAP_MEAN + 1.5 * HEATMAP_STD   # 1.0228
HEATMAP_UNDER_THRESHOLD = HEATMAP_MEAN - 1.5 * HEATMAP_STD   # 1.0038

# F5 fair odds by pitcher category (from backtest)
F5_FAIR_ODDS = {
    'Very Juicy':  -126,
    'Juicy':       +118,
    'Average':     +133,
    'Safe':        +143,
    'Very Safe':   +191,
}

_model_cache      = None
_model_cache_time = None
MODEL_CACHE_TTL   = 3600

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
        model[name] = download_pkl(fid)
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

# ── Heat map scoring ──────────────────────────────────────────────────────────

def compute_batter_overlap(batter_id, pitcher_id, batter_hand, pitcher_hand, model):
    """
    Compute overlap score for a single batter vs pitcher.
    Returns float or None if insufficient data.
    Score > 1.0 = pitcher tends to throw to batter's preferred zones.
    Score < 1.0 = pitcher avoids batter's preferred zones.
    """
    batter_profiles = model.get('batter_ev_profiles_lr', {})
    pitcher_tendencies = model.get('pitcher_tendency_map', {})
    archetypes = model.get('archetypes', {})

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

    Over:  score > HEATMAP_OVER_THRESHOLD  (+1.5 std, p=0.023)
    Under: score < HEATMAP_UNDER_THRESHOLD (-1.5 std, p=0.067)
    """
    pitcher_scores = model.get('all_pitcher_arch_scores', {})
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

            # Get pitcher handedness from existing pitcher scores
            pitcher_hand = 'R'  # default
            if pitcher_id in pitcher_scores:
                pitcher_hand = pitcher_scores[pitcher_id].get('p_throws', 'R')

            scores = []
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

            team_score = sum(scores) / len(scores)
            std_from_mean = (team_score - HEATMAP_MEAN) / HEATMAP_STD

            if team_score >= HEATMAP_OVER_THRESHOLD:
                signal = 'over'
            elif team_score <= HEATMAP_UNDER_THRESHOLD:
                signal = 'under'
            else:
                continue  # no flag — middle of distribution

            flags.append({
                'game':           game_str,
                'batting_team':   batting_team,
                'fielding_team':  fielding_team,
                'pitcher_name':   pitcher_name,
                'pitcher_hand':   pitcher_hand,
                'pitcher_id':     pitcher_id,
                'overlap_score':  round(team_score, 4),
                'std_from_mean':  round(std_from_mean, 2),
                'signal':         signal,
                'batters_scored': scored_batters,
                'lineup_complete': len(lineup) >= 8,
            })

    # Sort: strongest signals first (furthest from mean)
    flags.sort(key=lambda x: abs(x['std_from_mean']), reverse=True)
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
                        'player_name':  batter['name'],
                        'player_id':    batter_id,
                        'pitcher_name': pitcher_name,
                        'pitcher_id':   pitcher_id,
                        'game':         game_str,
                        'batting_team': batting_team,
                        'fielding_team': fielding_team,
                        'combined':     round(combined, 2),
                        'hr_fair':      hr_odds_str,
                        'hr_odds_num':  hr_odds_num,
                        'tb3_fair':     pa_rate_to_game_odds(adjusted * TB3_MULTIPLIER),
                        'arch_name':    archetypes[arch_key]['name'],
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
                      'complete': bool(g['home_lineup'] and g['away_lineup'])}
                     for g in games]

        # ── Heat map TT flags (primary signal) ───────────────────────────
        heatmap_flags = get_heatmap_flags(games, model)

        return jsonify({
            'date':           today,
            'complete':       complete,
            'total':          len(games),
            'picks':          picks,
            'heatmap_flags':  heatmap_flags,
            'games':          games_out,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
