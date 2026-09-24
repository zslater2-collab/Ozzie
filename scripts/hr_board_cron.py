"""
HR/DFS board — cloud cron (runs in GitHub Actions inside the Ozzie repo).
Self-contained: no ozzie_data dependency. Maintains a compact PA cache
(data/hr_pa_2026.csv.gz), pulls only new days via pybaseball, rebuilds
season-to-date (point-in-time) stats, builds today's board from live MLB
lineups, grades the archive, and writes the artifacts the app reads:
  hr_board_latest.json, hr_board_perf.json, hr_board_archive.csv   (repo root)

Signal (validated leak-free): hitter power (season HR-rate, shrunk)
  x pitcher HR-rate-allowed (shrunk) / league   [log5]   x park x weather
  x non-suppression (pitcher xwoba allowed vs league).  DFS ranking tool.
"""
import os, sys, json, math, glob, unicodedata
from datetime import datetime, timedelta
import pandas as pd, numpy as np, requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data')
PA_CACHE = os.path.join(DATA, 'hr_pa_2026.csv.gz')
ARCH_CSV = os.path.join(REPO, 'hr_board_archive.csv')
LATEST   = os.path.join(REPO, 'hr_board_latest.json')
PERF     = os.path.join(REPO, 'hr_board_perf.json')
GRADED   = os.path.join(REPO, 'hr_board_graded.csv')   # row-level prediction+edges+outcome ledger
RAIN     = os.path.join(REPO, 'rain_flags_latest.json')   # K-prop "rain around gametime" stay-away

K_HIT, K_PIT = 120, 150
MIN_HITTER_PA, MIN_PITCHER_BF = 80, 100
# Eligibility is a live POWER floor (shrunk season HR/PA %), replacing the old frozen-archetype gate
# which benched emerging sluggers (see hr-board archetype analysis 2026-08). The board still ranks by
# model prob and the display shows the top of the board, so this floor mainly keeps the archive/JSON
# lean; ~league-average HR/PA is ~3%, so 2.0 keeps genuine HR threats and trims slap hitters.
HR_RATE_FLOOR = 2.0
# log5 (hitter_rate x pitcher_rate / league) OVERSTATES when BOTH sides are above league -- the
# model was calibrated in 3 of 4 hitter/pitcher cells but ran +5pp hot in the both-high cell (which
# is exactly the top of the board). Two compounding causes: regression-to-mean gets multiplied, and
# the true interaction is sub-multiplicative. Fix = dampen ONLY the joint excess: multiply pa_hr by
# exp(-K * relu(log hitter_ratio) * relu(log pitcher_ratio)) so the penalty is zero unless BOTH ratios
# exceed league. K=4.5 (grid-searched on the graded archive: halves calibration error 1.92->0.92,
# both-high gap +5.1->~+1, residual recal slope 0.48->0.77). Mechanistic, not a blunt global haircut.
KHR_INTERACTION_DAMP = 4.5
AVG_PA_VS_GAME = 4.1
TEAM_PA_PER_GAME = 38.0   # slot expected PA = share * this (lineup_slot_pa_weights.csv)
SLOT_PA_SHARE = {1:0.1242,2:0.1210,3:0.1181,4:0.1175,5:0.1129,
                 6:0.1089,7:0.1036,8:0.0995,9:0.0943}
PA_COLS = ['game_date','game_pk','at_bat_number','batter','pitcher','events',
           'stand','estimated_woba_using_speedangle','launch_speed']

import pickle
arch  = pickle.load(open(os.path.join(DATA,'archetypes_combined.pkl'),'rb'))
hmap  = pickle.load(open(os.path.join(DATA,'arch_hitter_map_combined.pkl'),'rb'))
parks = pickle.load(open(os.path.join(DATA,'all_parks.pkl'),'rb'))
b2a = {}
for ak, ids in hmap.items():
    for i in ids: b2a.setdefault(int(i), []).append(ak)

# empirical HR-specific park factor (built offline, leak-free 2021-2025; ATH=2025 Sacramento only;
# see ozzie_data/build_hr_park_factor.py). Hand-specific HR multiplier centered on 1.0 -- captures
# altitude/air the fence-geometry factor misses (Coors scores <1.0 on geometry alone). Fail-open.
try:
    HRPF = pickle.load(open(os.path.join(DATA,'hr_park_factor.pkl'),'rb'))
except Exception as e:
    print(f'HR park factor load failed ({e}); board falls back to geometry park factor.'); HRPF = {}
# statsapi/board team codes -> parks.pkl / HRPF keys (statsapi uses AZ/KC/TB; our data uses ARI/KCR/TBR).
# Without this, AZ/KC/TB home games silently got a neutral 1.0 park AND weather (bug found 2026-09-08).
TEAM_CODE = {'AZ':'ARI','KC':'KCR','TB':'TBR'}
def _pk(team): return TEAM_CODE.get(team, team)
def hr_park_factor(team, hand):
    r = HRPF.get(_pk(team))
    if not r: return None
    return r.get(f'pf_{hand}', r.get('pf'))

# ---------------- PA cache: load + incremental pull ----------------
def refresh_pa_cache():
    pa = pd.read_csv(PA_CACHE)
    pa['game_date'] = pa['game_date'].astype(str)
    last = pa['game_date'].max()
    start = (datetime.strptime(last, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    end   = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')   # through yesterday
    if start > end:
        print(f'PA cache current through {last}; no new days.')
        return pa
    try:
        from pybaseball import statcast
        print(f'Pulling statcast {start}..{end} ...')
        new = statcast(start_dt=start, end_dt=end)
        if new is not None and len(new):
            for c in PA_COLS:
                if c not in new.columns: new[c] = np.nan
            new = new[PA_COLS].copy()
            new['game_date'] = pd.to_datetime(new['game_date']).dt.strftime('%Y-%m-%d')
            new = new.dropna(subset=['events']).drop_duplicates(['game_pk','at_bat_number'])
            pa = pd.concat([pa, new], ignore_index=True).drop_duplicates(['game_pk','at_bat_number'])
            pa.to_csv(PA_CACHE, index=False, compression='gzip')
            print(f'Appended {len(new)} PA; cache now through {pa["game_date"].max()}.')
    except Exception as e:
        print(f'statcast pull failed ({e}); using existing cache through {last}.')
    return pa

# ---------------- stats (season-to-date, shrunk) ----------------
def build_stats(pa):
    pa = pa.copy()
    pa['is_hr'] = (pa['events'] == 'home_run').astype(int)
    lg_hr = pa['is_hr'].mean(); lg_xw = pa['estimated_woba_using_speedangle'].mean()
    sh = lambda hr,n,k: (hr + lg_hr*k)/(n+k)
    H = {int(b): dict(pa=len(g), hr_rate=100*sh(g['is_hr'].sum(),len(g),K_HIT),
                      stand=(g['stand'].mode().iat[0] if not g['stand'].mode().empty else 'R'))
         for b,g in pa.groupby('batter')}
    P = {int(p): dict(bf=len(g), hr_rate=100*sh(g['is_hr'].sum(),len(g),K_PIT),
                      xwoba=g['estimated_woba_using_speedangle'].mean())
         for p,g in pa.groupby('pitcher')}
    meta = dict(lg_hr_rate=round(100*lg_hr,3), lg_xwoba=round(lg_xw,4),
                asof=str(pa['game_date'].max()))
    return H, P, meta

# ---------------- park / weather / lineups (ported) ----------------
def get_park_factor(team, hand, arch_key):
    team = _pk(team)
    if team not in parks: return 1.0
    dims = parks[team]['dimensions']
    avg = {k: sum(p['dimensions'][k] for p in parks.values())/len(parks)
           for k in ['left_field','left_center','center','right_center','right_field']}
    if hand == 'L':
        pull,pull_c,oppo = dims['right_field'],dims['right_center'],dims['left_field']
        a_pull,a_pull_c,a_oppo = avg['right_field'],avg['right_center'],avg['left_field']
    else:
        pull,pull_c,oppo = dims['left_field'],dims['left_center'],dims['right_field']
        a_pull,a_pull_c,a_oppo = avg['left_field'],avg['left_center'],avg['right_field']
    cf=dims['center']; a_cf=avg['center']
    # arch_key None -> unclassified hitter: use hand-based geometry with a neutral (most-common
    # power) pull profile so these bats still get a real park factor instead of a flat 1.0.
    base=arch_key.replace('_L','') if arch_key else 'neutral'
    if base in ('middle_ff','middle_sl','neutral'):
        rel=0.5*pull+0.3*pull_c+0.2*cf; av=0.5*a_pull+0.3*a_pull_c+0.2*a_cf
    elif base=='oppo_ff':
        rel=0.5*oppo+0.3*dims['right_center']+0.2*cf; av=0.5*a_oppo+0.3*avg['right_center']+0.2*a_cf
    elif base=='inside_br':
        rel=0.7*pull+0.2*pull_c+0.1*cf; av=0.7*a_pull+0.2*a_pull_c+0.1*a_cf
    elif base=='high_ff':
        rel=0.3*pull+0.3*cf+0.4*oppo; av=0.3*a_pull+0.3*a_cf+0.4*a_oppo
    else: return 1.0
    return max(0.85, min(1.15, round(av/rel,3)))

def get_weather_factor(team):
    team = _pk(team)
    if team not in parks or parks[team].get('roof', False): return 1.0
    p = parks[team]
    try:
        url=(f'https://api.open-meteo.com/v1/forecast?latitude={p["lat"]}&longitude={p["lon"]}'
             f'&current=temperature_2m,wind_speed_10m,wind_direction_10m'
             f'&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=auto')
        c=requests.get(url,timeout=10).json()['current']
        fo={'NYY':225,'BOS':95,'BAL':45,'CLE':135,'CWS':135,'DET':170,'KCR':0,
            'MIN':135,'LAA':225,'OAK':225,'SEA':180}
        tf=max(0.95,min(1.05,1.0+((c['temperature_2m']-70)/10)*0.01))
        wc=math.cos(math.radians((c['wind_direction_10m']-fo.get(team,180))%360))
        wf=max(0.92,min(1.08,1.0+(wc*c['wind_speed_10m']*0.003)))
        return max(0.90,min(1.10,round(tf*wf,3)))
    except Exception:
        return 1.0

def get_lineups(game_date):
    url=(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={game_date}'
         f'&hydrate=lineups,probablePitcher')
    data=requests.get(url,timeout=15).json()
    tmap={t['id']:t.get('abbreviation','???').upper()
          for t in requests.get('https://statsapi.mlb.com/api/v1/teams?sportId=1',timeout=15).json().get('teams',[])}
    games=[]
    for d in data.get('dates',[]):
        for g in d.get('games',[]):
            hid,aid=g['teams']['home']['team']['id'],g['teams']['away']['team']['id']
            def plist(key):
                return [{'id':p['id'],'pos':p.get('primaryPosition',{}).get('abbreviation','')}
                        for p in g.get('lineups',{}).get(key,[]) if p.get('id')]
            hl=plist('homePlayers'); al=plist('awayPlayers')
            hp=g['teams']['home'].get('probablePitcher',{}); ap=g['teams']['away'].get('probablePitcher',{})
            games.append(dict(home=tmap.get(hid,str(hid)),away=tmap.get(aid,str(aid)),
                home_id=hid,away_id=aid,home_lineup=hl,away_lineup=al,
                home_starter=hp.get('id'), away_starter=ap.get('id'),
                start=g.get('gameDate'),   # ISO UTC first-pitch
                state=g.get('status',{}).get('abstractGameState','')))
    # pitchHand isn't in the probablePitcher hydrate -> batch-fetch it from /people (one call, fail-open)
    ids=[str(x) for g in games for x in (g['home_starter'],g['away_starter']) if x]
    hands={}
    if ids:
        try:
            r=requests.get('https://statsapi.mlb.com/api/v1/people',params={'personIds':','.join(sorted(set(ids)))},timeout=15)
            hands={p['id']:(p.get('pitchHand',{}) or {}).get('code') for p in r.json().get('people',[]) if p.get('id')}
        except Exception: pass
    for g in games:
        g['home_starter_hand']=hands.get(g['home_starter']); g['away_starter_hand']=hands.get(g['away_starter'])
    return games

_ROSTER_CACHE = {}
def team_proj_batters(team_id):
    """Non-pitcher hitters on a team's active roster — the projected pool when the official
    lineup isn't posted yet (the power floor is applied later in build_board). Cached per run."""
    if team_id in _ROSTER_CACHE: return _ROSTER_CACHE[team_id]
    out=[]
    try:
        url=f'https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active'
        for p in requests.get(url,timeout=15).json().get('roster',[]):
            pid=p.get('person',{}).get('id')
            if pid and p.get('position',{}).get('type')!='Pitcher':   # power floor applied in build_board
                out.append({'id':int(pid),'pos':p.get('position',{}).get('abbreviation',''),'proj':True})
    except Exception as e:
        print(f'roster fetch failed for {team_id}: {e}')
    _ROSTER_CACHE[team_id]=out
    return out

def game_start_et(iso):
    """'2026-07-30T23:05:00Z' -> ('7:05p', datetime UTC) or ('', None)."""
    if not iso: return '', None
    try:
        dt = datetime.fromisoformat(iso.replace('Z','+00:00'))
        try:
            et = pd.Timestamp(dt).tz_convert('America/New_York')
            h = et.hour%12 or 12
            return f"{h}:{et.minute:02d}{'a' if et.hour<12 else 'p'}", dt
        except Exception:
            return dt.strftime('%H:%MZ'), dt
    except Exception:
        return '', None

def prob_to_american(gp):
    gp = min(max(gp/100.0, 0.005), 0.95)
    return round(-(gp/(1-gp))*100) if gp>=0.5 else round(((1-gp)/gp)*100)

def hr_prob(pa_hr, avg_pa=AVG_PA_VS_GAME):
    gp = 1-(1-pa_hr/100)**avg_pa
    return round(gp*100,1), prob_to_american(gp*100)

# ---------------- market edge layer (anytime-HR odds vs model) ----------------
# The board ranks by model probability; that answers "who is most likely to homer", NOT "where is
# the bet". A hitter can top the board and still be a bad wager if the book prices him accordingly.
# So we pull the real anytime-HR price (batter_home_runs Over 0.5), de-vig it to a market probability,
# and compute EDGE = recalibrated-model% - market%. Ranking by edge only bets a top-prob guy when his
# price hasn't already swallowed the value. Fail-open: no key / API error -> board is unchanged.
ODDS_KEY = os.environ.get('ODDS_API_KEY', '')
HR_ODDS_REGIONS = os.environ.get('HR_ODDS_REGIONS', 'us,us2')   # us2 adds the soft books; ~2 credits/event
# BETTABLE = books Zach has accounts at -> the "best odds"/best-price shopping comes ONLY from these.
# (DraftKings/FanDuel/BetMGM/Fanatics don't expose batter_home_runs on this feed, so in practice only
#  Caesars/theScore/Bally return a bettable HR price -- confirmed 2026-08-27.)
BETTABLE_BOOKS = {'DraftKings','FanDuel','BetMGM','Caesars','theScore Bet','Fanatics','Bally Bet'}
# TRACK-ONLY soft books: included ONLY in the market-average / de-vig consensus, never in "best odds".
TRACK_ONLY_BOOKS = {'Fliff','Hard Rock Bet','BetRivers','BetPARX'}
ALL_BOOKS = BETTABLE_BOOKS | TRACK_ONLY_BOOKS
BOOK_LABELS = {'draftkings':'DraftKings','fanduel':'FanDuel','betmgm':'BetMGM','caesars':'Caesars',
               'williamhill_us':'Caesars','espnbet':'theScore Bet','thescorebet':'theScore Bet',
               'fanatics':'Fanatics','ballybet':'Bally Bet',
               'fliff':'Fliff','hardrockbet':'Hard Rock Bet','betrivers':'BetRivers','betparx':'BetPARX'}

def _norm(s):
    if not isinstance(s,str): return ''
    s=unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower()
    s=s.replace('.','').replace("'",'').replace('-',' ').strip()
    return ' '.join(s.split())

def _implied(a):
    a=float(a); return (-a)/(-a+100.0) if a<0 else 100.0/(a+100.0)
def _payout(a):
    a=float(a); return (a/100.0) if a>0 else (100.0/-a)

def fetch_hr_odds(game_date):
    """{norm_player: {over, book, mkt_prob(%), n_books}} for anytime-HR (batter_home_runs Over 0.5),
    playable books only, best over price + de-vigged consensus prob. Fail-open -> {}."""
    if not ODDS_KEY:
        print('HR odds: no ODDS_API_KEY -> edge layer skipped (board ranks by probability as before).')
        return {}
    base='https://api.the-odds-api.com/v4/sports/baseball_mlb'
    try:
        evs=requests.get(f'{base}/events',params={'apiKey':ODDS_KEY},timeout=15).json()
    except Exception as e:
        print(f'HR odds: events fetch failed ({e}); edge layer skipped.'); return {}
    if not isinstance(evs,list):
        print('HR odds: unexpected events response; edge layer skipped.'); return {}
    from datetime import timezone
    try:
        from zoneinfo import ZoneInfo; ET=ZoneInfo('America/New_York')
    except Exception:
        ET=timezone(timedelta(hours=-4))   # MLB season is EDT; fallback if zoneinfo unavailable
    now=datetime.now(timezone.utc)
    acc={}   # norm -> book -> {'over':price,'under':price}
    n_ev=0
    for ev in evs:
        ct=ev.get('commence_time') or ''
        try:
            cdt=datetime.fromisoformat(ct.replace('Z','+00:00'))
        except Exception:
            continue
        # match on the ET SLATE date, not the raw UTC prefix: a 8pm+ ET game has a NEXT-DAY UTC
        # commence, so ct[:10] dropped every late/west-coast game even though its HR props post all
        # day. (Bug found 2026-08-27 -- late games "never showed odds".)
        if cdt.astimezone(ET).strftime('%Y-%m-%d')!=game_date: continue
        # only price UPCOMING games -- a started game's odds are gone/stale and we'd never bet them;
        # skipping them also saves credits (we grade edge off the pre-game price we captured).
        if cdt<=now: continue
        try:
            r=requests.get(f"{base}/events/{ev['id']}/odds",
                params={'apiKey':ODDS_KEY,'regions':HR_ODDS_REGIONS,
                        'markets':'batter_home_runs','oddsFormat':'american'},timeout=15)
            if r.status_code!=200: continue
            data=r.json(); n_ev+=1
        except Exception:
            continue
        for bm in data.get('bookmakers',[]):
            lbl=BOOK_LABELS.get(bm.get('key'))
            if lbl not in ALL_BOOKS: continue   # bettable + track-only (soft) books both accumulate
            for mk in bm.get('markets',[]):
                if mk.get('key')!='batter_home_runs': continue
                for o in mk.get('outcomes',[]):
                    if o.get('point') not in (0.5, None): continue   # anytime-HR line
                    nm=_norm(o.get('description','')); side=(o.get('name') or '').lower()
                    if nm and side in ('over','under') and o.get('price') is not None:
                        acc.setdefault(nm,{}).setdefault(lbl,{})[side]=o['price']
    out={}
    for nm,bks in acc.items():
        overs=[(b,v['over']) for b,v in bks.items() if v.get('over') is not None]
        if not overs: continue
        # BEST price = only books Zach can actually bet; None if only soft books priced it (tracking-only)
        overs_bet=[(b,p) for b,p in overs if b in BETTABLE_BOOKS]
        if overs_bet:
            best_book,best_over=max(overs_bet,key=lambda x:_payout(x[1]))
        else:
            best_book,best_over=None,None
        # de-vig consensus + AVERAGE across ALL tracked books (bettable + soft) = the broader "market"
        novigs=[_implied(v['over'])/(_implied(v['over'])+_implied(v['under']))
                for v in bks.values() if v.get('over') is not None and v.get('under') is not None]
        if not novigs: novigs=[_implied(p) for _,p in overs]   # one-sided quotes -> raw implied
        avg_am=prob_to_american(100*float(np.mean([_implied(p) for _,p in overs])))   # market-avg over price
        # per-book list for the Explorer expander: mark bettable-best + which are track-only/soft
        bp=sorted([{'book':b,'price':int(v['over']),
                    'best':(best_over is not None and int(v['over'])==int(best_over) and b in BETTABLE_BOOKS),
                    'bettable':b in BETTABLE_BOOKS}
                   for b,v in bks.items() if v.get('over') is not None], key=lambda x:-_payout(x['price']))
        out[nm]={'over':(int(best_over) if best_over is not None else None),'book':best_book,'book_prices':bp,
                 'mkt_prob':round(100*float(np.median(novigs)),1),'mkt_avg':avg_am,
                 'n_books':len(overs),'n_bettable':len(overs_bet)}
    print(f'HR odds: {n_ev} events priced, {len(out)} players with anytime-HR props '
          f'(regions={HR_ODDS_REGIONS}).')
    return out

# ---------------- DraftKings salary (DFS leverage) ----------------
# Unofficial public DK JSON (same feed the draft screen uses). We use ONLY the salary NUMBER; who is
# actually playing comes from statsapi lineups in build_board, so DK's (pre-lock, stale) probable-pitcher
# flags don't matter. Accent-safe name join (the Rodon/Sanchez landmine). Fail-open -> {} (no salary col).
DK_UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
# DK intermittently blocks GitHub-runner (datacenter) IPs and returns an HTML challenge page instead of
# JSON, which makes .json() raise. Retry with backoff + a rotating real browser UA so a transient block
# on one attempt recovers within the same run instead of blanking every salary.
DK_UAS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0',
]
_DK_SESSION = None
def _dk_session():
    """A requests.Session primed with DK's Akamai bot cookies (bm_*/ak_bmsc). api.draftkings.com sits
    behind an Akamai WAF that 403s cookieless/datacenter requests; warming the session on the public site
    first collects the cookies the API call then needs. Best-effort -- returns a session either way."""
    global _DK_SESSION
    if _DK_SESSION is not None:
        return _DK_SESSION
    s = requests.Session()
    s.headers.update({'User-Agent': DK_UAS[0], 'Accept': 'text/html,application/xhtml+xml,application/json,*/*',
                      'Accept-Language': 'en-US,en;q=0.9'})
    for warm in ('https://www.draftkings.com/', 'https://www.draftkings.com/lobby'):
        try: s.get(warm, timeout=15)   # sets .draftkings.com Akamai cookies (carry to the api subdomain)
        except Exception: pass
    _DK_SESSION = s
    return s
def _dk_get_json(url, timeout=20, tries=4):
    """GET url and parse JSON via the primed session, retrying with backoff + rotating UA. On a WAF 403
    (returns HTML -> .json() raises) a later try re-warms the cookies. Returns JSON or None (never raises)."""
    import time, random
    s = _dk_session()
    last = None
    for i in range(tries):
        try:
            r = s.get(url, headers={'User-Agent': DK_UAS[i % len(DK_UAS)],
                                    'Accept': 'application/json, text/plain, */*',
                                    'Referer': 'https://www.draftkings.com/'}, timeout=timeout)
            return r.json()
        except Exception as e:
            last = e
            if i < tries - 1:
                time.sleep(1.5 * (i + 1) + random.random())   # 1.5s, 3s, 4.5s (+jitter) backoff
                if i == 0:                                     # first failure: re-prime cookies once
                    globals()['_DK_SESSION'] = None; s = _dk_session()
    print(f'DK salary: gave up after {tries} tries ({last}).')
    return None
def _dk_team(t):   # DK/statsapi codes -> our ARI/KCR/TBR space so both sides of the join agree
    return {'AZ':'ARI','KC':'KCR','TB':'TBR'}.get(str(t).upper(), str(t).upper())
def _dk_players(dgid):
    """List of (playerId, norm_name, salary, team) for a Classic draft group. Uses the www
    'getavailableplayers' feed the draft screen itself calls -- NOT api.draftkings.com, whose Akamai WAF
    hard-403s datacenter/runner IPs. Team resolved from htabbr/atabbr via tid. Fail-open -> []."""
    j = _dk_get_json(f'https://www.draftkings.com/lineup/getavailableplayers?draftGroupId={dgid}')
    if not j: return []
    rows = []
    for p in (j.get('playerList') or []):
        nm = _norm(f"{p.get('fn','')} {p.get('ln','')}"); sal = p.get('s')
        tm = _dk_team(p.get('htabbr') if p.get('tid') == p.get('htid') else p.get('atabbr'))
        if nm and sal: rows.append((p.get('pid'), nm, int(sal), tm))
    return rows
def fetch_dk_salaries(date):
    """{norm_name: [(team, salary), ...]} for the date's largest MLB Classic slate. Fail-open -> {}."""
    lob = _dk_get_json('https://www.draftkings.com/lobby/getcontests?sport=MLB', timeout=15)
    if not lob:
        print('DK salary: lobby fetch failed; skipped.'); return {}
    cand = []
    for dgp in (lob.get('DraftGroups') or []):
        sd = (dgp.get('StartDate') or '')[:10]
        gc = dgp.get('GameCount') or 0
        # GameTypeId==2 = Classic (the salary format we draft). Other IDs are Showdown/Tiers/Snake/
        # "Home Runs" variants whose draftables are empty or differently priced -- exclude them.
        if sd == date and gc and dgp.get('GameTypeId') == 2:
            cand.append((gc, dgp.get('DraftGroupId')))
    if not cand:
        print(f'DK salary: no Classic draft group dated {date}; skipped.'); return {}
    dgid = max(cand)[1]   # most games = the main Classic slate
    main = _dk_players(dgid)
    if not main:
        print(f'DK salary: draftables fetch failed for {dgid}; skipped.'); return {}
    out, seen = {}, set()
    for pid, nm, sal, tm in main:
        if pid in seen: continue
        seen.add(pid)
        out.setdefault(nm, []).append((tm, sal))
    print(f'DK salary: slate {dgid} ({max(cand)[0]} games) -> {len(out)} players priced.')
    # Backfill ONLY players the main slate never priced (e.g. early ~6pm-ET games that sit before the
    # Main slate's first-pitch cutoff). We do NOT touch anyone already priced above -- other Classic
    # groups (early/all-day) just fill the gaps. seen (playerId) prevents re-adding main-slate players.
    added = 0
    for gc, dgid2 in sorted(cand, reverse=True):
        if dgid2 == dgid: continue
        extra = _dk_players(dgid2)
        if not extra:
            print(f'DK salary: backfill slate {dgid2} fetch failed; skipped.'); continue
        for pid, nm, sal, tm in extra:
            if pid in seen: continue
            seen.add(pid)
            if nm not in out:   # name-level guard: never overwrite a main-slate price
                out.setdefault(nm, []).append((tm, sal)); added += 1
    if added:
        print(f'DK salary: backfilled {added} players from {len(cand)-1} other Classic slate(s).')
    return out
def _dk_salary(dk, name, team):
    lst = dk.get(_norm(name))
    if not lst: return np.nan
    tm = _dk_team(team)
    for t, s in lst:
        if t == tm: return s
    return max(s for _, s in lst)   # name match, team unknown -> highest-salary entry (the starter)

# ---------------- board ----------------
def build_board(game_date, H, P, meta):
    from datetime import timezone
    now = datetime.now(timezone.utc)
    lg, lg_xw = meta['lg_hr_rate'], meta['lg_xwoba']
    rows=[]
    for g in get_lineups(game_date):
        gtime, gstart = game_start_et(g.get('start'))
        gstart_ms = int(gstart.timestamp()*1000) if gstart is not None else None
        # upcoming = first pitch still in the future (and not Live/Final). Full slate is
        # kept for grading; only the display filters to upcoming (see main()).
        upcoming = not (g.get('state') in ('Live','Final') or (gstart is not None and gstart <= now))
        wx=get_weather_factor(g['home'])
        for starter, sthand, lineup, bat, bat_id, field in [
            (g['home_starter'],g.get('home_starter_hand'),g['away_lineup'],g['away'],g['away_id'],g['home']),
            (g['away_starter'],g.get('away_starter_hand'),g['home_lineup'],g['home'],g['home_id'],g['away'])]:
            if not starter: continue
            pit=P.get(int(starter))
            # UNVERIFIED starters (thin/no statcast) are no longer dropped -- they're kept and flagged
            # so a small slate isn't starved of DFS options. A thin pitcher's shrunk hr_rate already
            # regresses hard to league (K_PIT=150); a true no-data debut is scored league-NEUTRAL
            # (pit_hr=lg -> log5 term collapses to the hitter's own rate; supp=1). Leak-safe: neutral,
            # no future info. Frontend hides these behind a "Show unverified" toggle by default.
            unverified = (not pit) or (pit['bf'] < MIN_PITCHER_BF)
            if not pit:
                pit_hr_val, supp, pit_bf = lg, 1.0, 0          # debut / no statcast -> neutral matchup
            else:
                pit_hr_val = pit['hr_rate']
                supp = float(np.clip(pit['xwoba']/lg_xw,0.90,1.12))
                pit_bf = pit['bf']
            # official lineup if posted; else projected pool (upcoming games only)
            proj = not lineup
            if proj:
                if not upcoming: continue
                lineup = team_proj_batters(bat_id)
            for slot, pl in enumerate(lineup, start=1):
                bid=int(pl['id']); pos=pl.get('pos','')
                h=H.get(bid)
                if not h or h['pa']<MIN_HITTER_PA: continue
                if h['hr_rate']<HR_RATE_FLOOR: continue          # live power gate (was: bid in b2a)
                if bid in b2a:                                    # archetyped -> hand from archetype + display label
                    ak=b2a[bid][0]; hand='L' if ak.endswith('_L') else 'R'; arch_name=arch[ak]['name']
                else:                                            # unclassified power hitter -> hand from batting side, no label
                    ak=None; hand=h.get('stand','R'); arch_name=''
                # empirical HR park factor (hand-specific, leak-free 2021-2025); fence geometry is the fallback
                park=hr_park_factor(g['home'],hand)
                if park is None:
                    park=get_park_factor(g['home'],hand,ak)
                # dampen the log5 joint-extreme overstatement (see KHR_INTERACTION_DAMP)
                _damp=np.exp(-KHR_INTERACTION_DAMP*max(0.0,np.log(h['hr_rate']/lg))*max(0.0,np.log(pit_hr_val/lg)))
                pa_hr=(h['hr_rate']*pit_hr_val/lg)*park*wx*supp*_damp
                # slot-weighted PA when the lineup is official; neutral PA when projected
                exp_pa=(AVG_PA_VS_GAME if proj else SLOT_PA_SHARE.get(slot,0.10)*TEAM_PA_PER_GAME)
                prob,amer=hr_prob(pa_hr, exp_pa)
                rows.append(dict(batter=bid,pitcher=int(starter),game=f'{g["away"]}@{g["home"]}',team=bat,
                    gtime=gtime,gstart_ms=gstart_ms,upcoming=upcoming,proj=proj,
                    unverified=unverified,pit_bf=pit_bf,
                    slot=(None if proj else slot),pos=pos,arch=arch_name,bat_hand=hand,pit_hand=sthand,
                    hit_hr=round(h['hr_rate'],2),pit_hr=round(pit_hr_val,2),
                    park=round(park,3),wx=round(wx,3),supp=round(supp,3),
                    pa_hr=round(pa_hr,3),hr_prob=prob,fair=('+%d'%amer if amer>0 else str(amer))))
    if not rows: return pd.DataFrame()
    df=pd.DataFrame(rows).sort_values('hr_prob',ascending=False).reset_index(drop=True)
    try:
        from pybaseball import playerid_reverse_lookup
        ids=list(set(df['batter'])|set(df['pitcher']))
        nm=playerid_reverse_lookup(ids,key_type='mlbam')
        nm['n']=(nm['name_first'].str.title()+' '+nm['name_last'].str.title())
        m=dict(zip(nm['key_mlbam'],nm['n']))
        df['Batter']=df['batter'].map(m); df['Pitcher']=df['pitcher'].map(m)
    except Exception as e:
        print(f'name lookup failed: {e}')
        df['Batter']=df['batter']; df['Pitcher']=df['pitcher']
    return df

# ---------------- grade ----------------
def grade(archive, pa):
    hr=pa.copy(); hr['is_hr']=(hr['events']=='home_run').astype(int)
    # per batter-day: max -> had_hr (0/1), sum -> hr_count (for 2+ HR / multi-HR tracking)
    out=hr.groupby(['game_date','batter'])['is_hr'].agg(had_hr='max',hr_count='sum').reset_index()
    out.columns=['date','batter','had_hr','hr_count']
    df=archive.merge(out,on=['date','batter'],how='left')
    g=df[df['had_hr'].notna()].copy()
    if g.empty: return {'graded_dates':0}
    g['had_hr']=g['had_hr'].astype(int)
    g['hr_count']=g['hr_count'].fillna(0).astype(int)
    g['multi_hr']=(g['hr_count']>=2).astype(int)
    g['rank']=g.groupby('date')['hr_prob'].rank(ascending=False,method='first')
    g['bucket']=g['rank'].apply(lambda r:'top10' if r<=10 else('top25' if r<=25 else 'rest'))
    base=float(g['had_hr'].mean())
    perf={'graded_dates':int(g['date'].nunique()),'graded_picks':int(len(g)),
          'base_hr_rate':round(100*base,2),'buckets':{},'calibration':[]}
    for b in ['top10','top25','rest']:
        s=g[g['bucket']==b]
        if len(s): perf['buckets'][b]={'n':int(len(s)),'hit_rate':round(100*s['had_hr'].mean(),2),
            'lift_vs_base':round(100*(s['had_hr'].mean()-base),2)}
    # ---- 2+ HR (multi-HR) tracking ----------------------------------------------------------
    # Two rates that answer different questions: per-pick 2+HR rate (unconditional -- what a 2+HR
    # bet actually hits, with the fair-breakeven American price) and multi-among-winners (given the
    # bat homered, how often it went 2+ -- the "fat tail" the board's top picks carry vs the ~6%
    # MLB baseline). Sliced by board rank so the top-of-board concentration is visible.
    mlb_hit=out[out['hr_count']>=1]
    def _be(p): return None if p<=0 else int(round((1-p)/p*100))
    perf['multi_hr']={'graded':int(g['multi_hr'].sum()),
        'mlb_multi_among_hr_pct':round(100*(mlb_hit['hr_count']>=2).mean(),2),
        'tiers':{}}
    for lab,mask in [('top3',g['rank']<=3),('top5',g['rank']<=5),('top10',g['rank']<=10),
                     ('top25',g['rank']<=25),('all',g['rank']>=1)]:
        s=g[mask]; w=s[s['had_hr']==1]
        if len(s):
            p=float(s['multi_hr'].mean())
            perf['multi_hr']['tiers'][lab]={'n':int(len(s)),'multi':int(s['multi_hr'].sum()),
                'rate_per_pick':round(100*p,2),'breakeven_odds':_be(p),
                'multi_among_winners':round(100*w['multi_hr'].mean(),2) if len(w) else None}
    try:
        g['q']=pd.qcut(g['hr_prob'],5,labels=False,duplicates='drop')
        for q,s in g.groupby('q'):
            perf['calibration'].append({'q':int(q),'pred':round(s['hr_prob'].mean(),1),
                'actual':round(100*s['had_hr'].mean(),1),'n':int(len(s))})
    except Exception: pass
    # self-updating linear recalibration: regress realized HR (0/1)*100 on predicted %.
    # identity until enough graded picks so a thin sample can't distort the board.
    slope, intercept = 1.0, 0.0
    if len(g) >= 400:
        x = g['hr_prob'].to_numpy(); y = 100.0*g['had_hr'].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        slope = float(np.clip(slope, 0.2, 1.0))
    perf['calib'] = {'slope': round(slope,4), 'intercept': round(float(intercept),3), 'n': int(len(g))}

    # ---- recalibrated model + the full edge family (per row) -> persisted graded ledger ----
    # NOTE: forward grading shows model-vs-market "edge" is INVERTED (pos_edge underperforms
    # base in every prob band -- it measures where our log5 overrates weak hitters vs an accurate
    # market, not a bet). Kept + graded so the ledger documents the fade; NOT a bet signal.
    g['model_recal'] = (slope*g['hr_prob'] + intercept).round(2)
    _imp = lambda a: (round(_implied(a)*100, 2) if pd.notna(a) else np.nan)   # american -> implied %
    for c in ('mkt_prob','mkt_over','mkt_avg'):
        if c not in g.columns: g[c] = np.nan
    g['best_impl'] = g['mkt_over'].apply(_imp)
    g['avg_impl']  = g['mkt_avg'].apply(_imp)
    g['edge']      = (g['model_recal'] - g['mkt_prob']).round(2)    # model - de-vigged consensus
    g['edge_best'] = (g['model_recal'] - g['best_impl']).round(2)   # model - best bettable price
    g['edge_shop'] = (g['avg_impl']    - g['best_impl']).round(2)   # best price beats market-avg (line shop)
    g['roi'] = np.where(g['had_hr']==1, g['mkt_over'].apply(lambda a: _payout(a) if pd.notna(a) else np.nan), -1.0)
    g.loc[g['mkt_over'].isna(), 'roi'] = np.nan

    # ---- ⭐ prop+power overlap (forward-track) ------------------------------------------------
    # prop-over (edge -6..-3, market underprices) AND a raw-power bat (top HR% of the day).
    # Overlap grades stronger than either filter alone on both single-HR ROI and the 2+HR tail --
    # the prop filter selects mispricing, the power filter re-introduces the raw pop that carries
    # the multi-HR upside. POWER BAR = daily percentile, NOT a fixed count: a data-chosen top-15%
    # of that day's HR% (grade_power_bar.py). Beats a fixed top-N and an absolute HR% floor at
    # matched sample AND is positive both graded months, where fixed-count flipped +93%/-23%.
    PP_PCTILE = 0.85   # top 15% of the slate's HR% -- adapts to slate size (4-game day vs 15-game day)
    g['hr_pctile'] = g.groupby('date')['hr_prob'].rank(pct=True)   # 1.0 = highest HR% that day
    g['is_prop']    = ((g['edge']>=-6) & (g['edge']<-3)).astype('Int64')
    g['prop_power'] = (((g['edge']>=-6) & (g['edge']<-3)) & (g['hr_pctile']>=PP_PCTILE)).astype(int)
    pp = g[(g['prop_power']==1) & g['mkt_over'].notna()].copy()
    if len(pp):
        m = pp['mkt_over'].apply(_payout)
        wj = pp[pp['had_hr']==1]
        perf['prop_power'] = {'power_bar':f'top {int(round((1-PP_PCTILE)*100))}% HR% (daily pctile)','n':int(len(pp)),
            'hit_rate':round(100*pp['had_hr'].mean(),2),
            'roi_raw':round(100*float(np.where(pp['had_hr']==1, m, -1.0).mean()),2),
            'roi_boost25':round(100*float(np.where(pp['had_hr']==1, m*1.25, -1.0).mean()),2),
            'avg_price':int(round(100*m.mean())),
            'winners':int(len(wj)),'multi':int(wj['multi_hr'].sum()),
            'multi_among_winners':round(100*wj['multi_hr'].mean(),2) if len(wj) else None}

    # ---- 💣 smash flag (2+ HR nibble screen, forward-track) -----------------------------------
    # DIFFERENT job than ⭐ prop+power. That flag optimizes the single-HR bet; this one screens the
    # spots where an *unconditional* 2+HR nibble is defensible. Data-chosen: the day's very top HR%
    # bats concentrate the 2+HR tail better than any absolute prob floor OR the prop overlap (the
    # prop band strips out mashers -- they're correctly priced). Bar = top ~2% of the day's HR%
    # (typically the 1-2 highest-projected bats). VERY THIN (single-digit multi events) -> graded to
    # forward-track only; the breakeven odds tell you the price a 2+HR bet must beat.
    SMASH_PCTILE = 0.99   # top ~1% of the day's HR% -- the 1-2 absolute-top bats
    g['smash'] = (g['hr_pctile'] >= SMASH_PCTILE).astype(int)
    sm = g[g['smash']==1].copy()
    if len(sm):
        p2 = float(sm['multi_hr'].mean())                        # unconditional 2+HR rate per pick
        be2 = int(round((1-p2)/p2*100)) if p2>0 else None        # fair 2+HR breakeven (American)
        perf['smash'] = {'bar':f'top {int(round((1-SMASH_PCTILE)*100))}% HR% (daily pctile)',
            'n':int(len(sm)),'multi_2plus':int(sm['multi_hr'].sum()),
            'rate_2plus':round(100*p2,2),'breakeven_odds':be2,
            'had_hr_rate':round(100*sm['had_hr'].mean(),2),
            'note':'thin sample -- forward-track, do not size'}

    led_cols = [c for c in ['date','batter','Batter','Pitcher','game','team','slot','pos','bat_hand',
                'hit_hr','pit_hr','park','wx','supp','hr_prob','model_recal','mkt_prob','mkt_over','mkt_avg',
                'salary','leverage','edge','edge_best','edge_shop','is_prop','prop_power','smash','had_hr','hr_count','multi_hr','roi'] if c in g.columns]
    try:
        g.sort_values(['date','hr_prob'], ascending=[True,False])[led_cols].to_csv(GRADED, index=False)
    except Exception as e:
        print(f'graded ledger write failed: {e}')

    # EDGE grading (picks with a captured market price) -- edge + edge_best (both invert), plus a
    # Shop CLV stat (avg pts the best bettable price beats the market average -- a real, positive lever).
    ge = g[g['mkt_prob'].notna() & g['mkt_over'].notna()].copy()
    if len(ge):
        eb={}
        for lab,mask in [('pos_edge', ge['edge']>0), ('neg_edge', ge['edge']<=0), ('edge_ge3', ge['edge']>=3),
                         ('best_pos', ge['edge_best']>0), ('best_neg', ge['edge_best']<=0)]:
            s=ge[mask]
            if len(s): eb[lab]={'n':int(len(s)),'hit_rate':round(100*s['had_hr'].mean(),2),
                                'roi':round(100*s['roi'].mean(),2)}
        perf['edge_buckets']=eb
        perf['edge_graded_picks']=int(len(ge))
        perf['shop']={'n':int(ge['edge_shop'].notna().sum()),
                      'avg_pts':round(float(ge['edge_shop'].mean()),2) if ge['edge_shop'].notna().any() else None}
        perf['edge_blind']={'n':int(len(ge)),'roi':round(100*ge['roi'].mean(),2),
                            'actual_hr':round(100*ge['had_hr'].mean(),2),
                            'mkt_implied':round(float(ge['mkt_prob'].mean()),2)}
    # ---- watchable readout each cron run (logs to the Actions output; also persisted in perf.json) ----
    print(f"CALIBRATION: actual% = {perf['calib']['intercept']} + {perf['calib']['slope']}*pred%  "
          f"(slope<1 = runs hot; n={perf['calib']['n']})")
    eb = perf.get('edge_buckets') or {}
    if eb:
        gp = perf.get('edge_graded_picks', 0)
        parts = [f"{k} n={v['n']} hit={v['hit_rate']}% roi={v['roi']:+}%" for k,v in eb.items()]
        print(f"EDGE CALIBRATION ({gp} picks w/ odds): " + " | ".join(parts))
        b = perf.get('edge_blind') or {}
        if b: print(f"  blind CLV: roi={b['roi']:+}%  actual {b['actual_hr']}% vs market {b['mkt_implied']}%  "
                    f"[need ~200+ picks-with-odds before trusting pos_edge]")
    else:
        print("EDGE CALIBRATION: 0 picks with captured odds yet (coverage grows with the tz/carry-forward fixes)")
    return perf

# ---------------- K-prop rain stay-away flags ----------------
# Simple, glanceable "don't take a pitcher's OVER when there's clearly considerable rain around
# gametime" flag (a 30+ min delay usually ends the starter's night early). Open-air parks only,
# forecast window = first pitch -1h .. +3h (pregame + early innings). Thresholds tuned to CLEARLY
# wet, not marginal. Written as a small JSON the app READS (no live weather in the request path --
# same safe pattern as the HR board; the /api/notify weather incident is why).
_RAIN_MODELS = ['gfs_hrrr', 'ecmwf_ifs025', 'icon_seamless']   # US hi-res + ECMWF + ICON
def _rain_forecast(lat, lon, fp_utc):
    """(max precip_mm, max pop) over [first_pitch-1h, +3h], taking the MAX across several models.
    A SINGLE model badly under-forecasts fast storm lines: on 8/7 the default (GFS) showed 19-42% POP
    for 4 parks that all went to rain delay, while ECMWF had them at 69-92%. For a STAY-AWAY veto,
    catching a real system matters more than a false skip, so we trust whichever model sees the rain.
    None only if EVERY model errored (a genuinely dry day returns (0, 0), not None)."""
    got = False
    precip_max, pop_max = 0.0, 0.0
    start = fp_utc - timedelta(hours=1); end = fp_utc + timedelta(hours=3)
    for mdl in _RAIN_MODELS:
        try:
            r = requests.get('https://api.open-meteo.com/v1/forecast', params={
                'latitude': lat, 'longitude': lon, 'hourly': 'precipitation,precipitation_probability',
                'forecast_days': 3, 'timezone': 'UTC', 'models': mdl}, timeout=15)
            r.raise_for_status(); h = r.json().get('hourly', {})
            if 'time' not in h:
                continue
            pr, po = 0.0, 0.0
            for t, p, pp in zip(h['time'], h.get('precipitation', []), h.get('precipitation_probability', [])):
                tt = datetime.fromisoformat(t).replace(tzinfo=fp_utc.tzinfo)
                if start <= tt < end:
                    pr += (p or 0); po = max(po, pp or 0)
            precip_max = max(precip_max, pr); pop_max = max(pop_max, po); got = True
        except Exception:
            continue
    return (round(precip_max, 2), pop_max) if got else None

def build_rain_flags(date):
    """Write rain_flags_latest.json: open-air games with considerable rain around gametime."""
    from datetime import timezone
    flagged = []
    try:
        lineups = get_lineups(date)
    except Exception as e:
        print(f'rain flags: lineup fetch failed ({e})'); lineups = []
    for g in lineups:
        home = g['home']; p = parks.get(home)
        if not p or p.get('roof'):
            continue                                   # unknown or roofed -> immune
        _, fp = game_start_et(g.get('start'))
        if fp is None:
            continue
        if fp.tzinfo is None:
            fp = fp.replace(tzinfo=timezone.utc)
        fc = _rain_forecast(p['lat'], p['lon'], fp)
        if fc is None:
            continue
        precip, pop = fc
        # OR logic, not AND (fixed 8/7): a confident ACCUMULATION or a high PROBABILITY each counts on
        # its own. Requiring both missed PIT on 8/7 -- 6mm of forecast rain but the model hedged POP at
        # 19%, so the game went to a rain delay unflagged. POP alone is unreliable for fast storm lines.
        risk = ('high' if (pop >= 60 or precip >= 4) else
                'moderate' if (pop >= 50 or precip >= 2) else None)
        if not risk:
            continue
        flagged.append({'home': home, 'away': g['away'], 'game': f"{g['away']}@{home}",
                        'park': p.get('name', home), 'precip_mm': precip, 'pop_max': pop,
                        'risk': risk, 'first_pitch': g.get('start')})
    json.dump({'date': date, 'generated_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'),
               'games': flagged}, open(RAIN, 'w'), indent=2)
    print(f'Rain flags {date}: {len(flagged)} open-air game(s) with considerable rain around gametime'
          + (': ' + ', '.join(x['game'] for x in flagged) if flagged else ''))


def main():
    date = sys.argv[1] if len(sys.argv)>1 else datetime.utcnow().strftime('%Y-%m-%d')
    pa = refresh_pa_cache()
    H,P,meta = build_stats(pa)
    df = build_board(date, H, P, meta)

    # anytime-HR market odds (once per run) -> attach best price + de-vigged prob to every row so the
    # archive can grade edge forward, and the display can rank by it. Fail-open -> empty dict.
    hr_odds = fetch_hr_odds(date) if not df.empty else {}

    if not df.empty:
        keep=['batter','pitcher','game','team','gtime','gstart_ms','upcoming','proj','unverified','pit_bf','slot','pos','arch','bat_hand','pit_hand',
              'hit_hr','pit_hr','park','wx','supp','pa_hr','hr_prob','fair','Batter','Pitcher']
        day=df[[c for c in keep if c in df.columns]].copy(); day.insert(0,'date',date)
        if hr_odds:
            _k=day['Batter'].map(_norm)
            day['mkt_over']=_k.map(lambda n:hr_odds.get(n,{}).get('over'))
            day['mkt_book']=_k.map(lambda n:hr_odds.get(n,{}).get('book'))
            day['mkt_prob']=_k.map(lambda n:hr_odds.get(n,{}).get('mkt_prob'))
            day['mkt_avg']=_k.map(lambda n:hr_odds.get(n,{}).get('mkt_avg'))          # market-avg over price (all books)
            day['mkt_n_books']=_k.map(lambda n:hr_odds.get(n,{}).get('n_books'))
            day['mkt_n_bettable']=_k.map(lambda n:hr_odds.get(n,{}).get('n_bettable'))
            day['book_prices']=_k.map(lambda n:hr_odds.get(n,{}).get('book_prices') or [])
        # DK salary (DFS leverage). Accent-safe name+team join to the players the board already has;
        # report coverage so a broken/stale pull is visible. Fail-open -> no salary column.
        dk_sal = fetch_dk_salaries(date)
        if dk_sal and 'Batter' in day.columns:
            teams = day['team'] if 'team' in day.columns else pd.Series(['']*len(day), index=day.index)
            day['salary'] = [ _dk_salary(dk_sal, nm, tm) for nm, tm in zip(day['Batter'], teams) ]
            cov = day['salary'].notna().mean()
            print(f'DK salary join: {int(day["salary"].notna().sum())}/{len(day)} board players matched ({cov:.0%} coverage).')
        # archive only OFFICIAL-lineup rows (projected picks are speculative -> excluded
        # from the forward-track so grading stays honest); keeps RAW hr_prob for calib.
        # ALSO exclude UNVERIFIED rows (thin/no-data starter, scored league-neutral): they're a
        # different, lower-confidence population, so grading/calibrating on them would distort the
        # tracked hit-rate + the calib slope for the confident board. Displayed (behind the toggle),
        # not graded. Drop the display-only book_prices list column so it doesn't bloat the CSV.
        _amask = (day['proj']==False) if 'proj' in day.columns else pd.Series(True, index=day.index)
        if 'unverified' in day.columns: _amask &= (day['unverified']==False)
        official=day[_amask].drop(columns=['book_prices'], errors='ignore')
        # CARRY FORWARD odds: HR props post late, so a game's price is often captured only by a later
        # run -- but once that game STARTS, fetch_hr_odds skips it (and an empty fetch adds no columns),
        # so without this each subsequent run would blank a price we already had. Ensure the odds
        # columns exist, then backfill any missing value from the prior archive for the same
        # (date, batter). The current run's fetch always wins where it has a value.
        ODDC=['mkt_over','mkt_book','mkt_prob','mkt_avg','mkt_n_books','mkt_n_bettable']
        for c in ODDC:
            if c not in official.columns: official[c]=np.nan
        if os.path.exists(ARCH_CSV):
            old=pd.read_csv(ARCH_CSV); old['date']=old['date'].astype(str)
            od=old[old['date']==date]
            if not od.empty:
                prior=od.drop_duplicates('batter').set_index('batter')
                for c in ODDC:
                    if c in prior.columns:
                        fill=official['batter'].map(prior[c])
                        official[c]=official[c].where(official[c].notna(), fill)
            arch_df=pd.concat([old[old['date']!=date],official],ignore_index=True)
        else:
            arch_df=official
        arch_df.to_csv(ARCH_CSV,index=False)
    else:
        print(f'No board rows for {date} (lineups not posted?). Grading only.')

    # grade the archive -> perf + self-updating calibration
    perf = {}
    if os.path.exists(ARCH_CSV):
        arch_df=pd.read_csv(ARCH_CSV); arch_df['date']=arch_df['date'].astype(str)
        perf=grade(arch_df,pa); json.dump(perf,open(PERF,'w'),indent=2)
        print(f'Graded {perf.get("graded_dates",0)} dates; top25 '
              f'{perf.get("buckets",{}).get("top25",{}).get("hit_rate","-")}% '
              f'vs base {perf.get("base_hr_rate","-")}%')

    build_rain_flags(date)   # K-prop rain stay-away flags (independent of the HR board)

    # write display artifact: only UPCOMING games (first pitch still ahead),
    # recalibrated probability + DFS tier tags. Archive kept the full slate above.
    if not df.empty:
        cal = perf.get('calib', {'slope':1.0,'intercept':0.0})
        d = day[day['upcoming']==True].copy() if 'upcoming' in day.columns else day.copy()
        if d.empty:
            print(f'No upcoming games left for {date} (slate already started/done).')
            return
        d['hr_prob_raw'] = d['hr_prob']
        d['hr_prob'] = (cal['intercept'] + cal['slope']*d['hr_prob_raw']).clip(0.5, 60).round(1)
        d['fair'] = d['hr_prob'].apply(lambda p: ('+%d'%a if (a:=prob_to_american(p))>0 else str(a)))
        # EDGE = recalibrated model% - de-vigged market%. Forward grading shows this edge is INVERTED
        # (pos_edge underperforms base in every prob band -- it's a fade, not a bet; see edge_buckets).
        # So it is NO LONGER used to rank -- kept as a context column. Rank by the validated signal:
        # recalibrated model probability. (The Explorer re-sorts client-side; this sets the archive order
        # + the hidden compact board.) DFS value now comes from salary/leverage, not model-vs-market edge.
        if 'mkt_prob' in d.columns and d['mkt_prob'].notna().any():
            d['edge'] = (d['hr_prob'] - d['mkt_prob']).round(1)
        d = d.sort_values('hr_prob', ascending=False).reset_index(drop=True)
        ranked_by = 'prob'
        # DFS tiers by rank: Chalk (top 10), Value (11-25, the reliable band), Deep (26+).
        d['tier'] = ['chalk' if i<10 else ('value' if i<25 else 'deep') for i in range(len(d))]
        nproj = int(d['proj'].sum()) if 'proj' in d.columns else 0
        # to_json converts NaN->null (valid JSON); plain json.dump would emit bare NaN,
        # which Python tolerates but browser JSON.parse rejects -> blank board.
        rows_json = json.loads(d.to_json(orient='records'))
        json.dump({'date':date,'asof':meta['asof'],'calib':cal,'ranked_by':ranked_by,
                   'rows':rows_json}, open(LATEST,'w'), indent=2)
        print(f'Board {date}: {len(day)} rows -> {len(d)} upcoming shown, ranked_by={ranked_by} '
              f'({nproj} projected, recal slope {cal.get("slope")}, asof {meta["asof"]}).')

if __name__=='__main__':
    main()
