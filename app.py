import os
import pickle
import requests
import pandas as pd
from datetime import datetime
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
}

LEAGUE_AVG      = 3.88
JUICY_THRESHOLD = 4.5
MIN_SENSOR_PA   = 20
STARTER_WEIGHT  = 0.80
BULLPEN_WEIGHT  = 0.20
ASSUMED_PAS     = 4
TB3_MULTIPLIER  = 1.3

_model_cache      = None
_model_cache_time = None
MODEL_CACHE_TTL   = 3600

# ── Model loading ──────────────────────────────────────────────────────────────

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

# ── MLB API ────────────────────────────────────────────────────────────────────

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
            gid   = g['gamePk']
            home  = g['teams']['home']['team']['abbreviation']
            away  = g['teams']['away']['team']['abbreviation']
            hp    = g['teams']['home'].get('probablePitcher', {})
            ap    = g['teams']['away'].get('probablePitcher', {})

            home_lineup, away_lineup = [], []
            try:
                bs = requests.get(
                    f"https://statsapi.mlb.com/api/v1/game/{gid}/boxscore",
                    timeout=15).json()
                for team_key, lineup in [('home', home_lineup), ('away', away_lineup)]:
                    td = bs.get('teams', {}).get(team_key, {})
                    for pid in td.get('battingOrder', []):
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

# ── Scoring helpers ────────────────────────────────────────────────────────────

def pa_rate_to_game_odds(pa_rate_pct):
    pa_rate  = pa_rate_pct / 100
    game_prob = 1 - (1 - pa_rate) ** ASSUMED_PAS
    if game_prob <= 0:
        return '+9999'
    fair = round((1 / game_prob - 1) * 100)
    return f'+{fair}' if fair > 0 else str(fair)

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
        dims  = all_parks[fielding_team].get('dimensions', {})
        valid = [p for p in all_parks.values() if 'dimensions' in p]
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

# ── Picks engine ───────────────────────────────────────────────────────────────

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
             f"{game['away_team']}@{game['home_team']}", game['home_pitcher_name']),
            (game['home_lineup'], game['away_pitcher_id'], game['away_team'],
             f"{game['away_team']}@{game['home_team']}", game['away_pitcher_name']),
        ]
        for lineup, pitcher_id, fielding_team, game_str, pitcher_name in matchups:
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
                    starter_rate  = arch_data.get('shrunk_rate', LEAGUE_AVG)
                    bullpen_rate  = get_bullpen_score(fielding_team, arch_key, model)
                    combined      = STARTER_WEIGHT * starter_rate + BULLPEN_WEIGHT * bullpen_rate
                    if combined < JUICY_THRESHOLD:
                        continue
                    pf       = get_park_factor(fielding_team, batter_hand, model)
                    adjusted = combined * pf
                    picks.append({
                        'player_name':  batter['name'],
                        'player_id':    batter_id,
                        'pitcher_name': pitcher_name,
                        'game':         game_str,
                        'combined':     round(combined, 2),
                        'hr_fair':      pa_rate_to_game_odds(adjusted),
                        'tb3_fair':     pa_rate_to_game_odds(adjusted * TB3_MULTIPLIER),
                        'arch_name':    archetypes[arch_key]['name'],
                    })

    seen = {}
    for p in picks:
        pid = p['player_id']
        if pid not in seen or p['combined'] > seen[pid]['combined']:
            seen[pid] = p
    return sorted(seen.values(), key=lambda x: x['combined'], reverse=True)

# ── Routes ─────────────────────────────────────────────────────────────────────

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
        model     = load_model()
        today     = datetime.now().strftime('%Y-%m-%d')
        games     = get_lineups_and_starters(today)
        picks     = get_hr_picks(games, model)
        complete  = sum(1 for g in games if g['home_lineup'] and g['away_lineup'])
        games_out = [{'away': g['away_team'], 'home': g['home_team'],
                      'complete': bool(g['home_lineup'] and g['away_lineup'])}
                     for g in games]
        return jsonify({
            'date':     today,
            'complete': complete,
            'total':    len(games),
            'picks':    picks,
            'games':    games_out,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
