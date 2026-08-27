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
RAIN     = os.path.join(REPO, 'rain_flags_latest.json')   # K-prop "rain around gametime" stay-away

K_HIT, K_PIT = 120, 150
MIN_HITTER_PA, MIN_PITCHER_BF = 80, 100
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
    H = {int(b): dict(pa=len(g), hr_rate=100*sh(g['is_hr'].sum(),len(g),K_HIT))
         for b,g in pa.groupby('batter')}
    P = {int(p): dict(bf=len(g), hr_rate=100*sh(g['is_hr'].sum(),len(g),K_PIT),
                      xwoba=g['estimated_woba_using_speedangle'].mean())
         for p,g in pa.groupby('pitcher')}
    meta = dict(lg_hr_rate=round(100*lg_hr,3), lg_xwoba=round(lg_xw,4),
                asof=str(pa['game_date'].max()))
    return H, P, meta

# ---------------- park / weather / lineups (ported) ----------------
def get_park_factor(team, hand, arch_key):
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
    cf=dims['center']; a_cf=avg['center']; base=arch_key.replace('_L','')
    if base in ('middle_ff','middle_sl'):
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
    """Archetype hitters on a team's active roster — the projected pool when the
    official lineup isn't posted yet. Cached per run."""
    if team_id in _ROSTER_CACHE: return _ROSTER_CACHE[team_id]
    out=[]
    try:
        url=f'https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active'
        for p in requests.get(url,timeout=15).json().get('roster',[]):
            pid=p.get('person',{}).get('id')
            if pid and int(pid) in b2a and p.get('position',{}).get('type')!='Pitcher':
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
HR_ODDS_REGIONS = os.environ.get('HR_ODDS_REGIONS', 'us')   # 'us' keeps credit cost ~1/event
PLAYABLE_BOOKS = {'DraftKings','FanDuel','BetMGM','Caesars','theScore Bet','Fanatics','Bally Bet'}
BOOK_LABELS = {'draftkings':'DraftKings','fanduel':'FanDuel','betmgm':'BetMGM','caesars':'Caesars',
               'williamhill_us':'Caesars','espnbet':'theScore Bet','thescorebet':'theScore Bet',
               'fanatics':'Fanatics','ballybet':'Bally Bet'}

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
            if lbl not in PLAYABLE_BOOKS: continue
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
        best_book,best_over=max(overs,key=lambda x:_payout(x[1]))
        novigs=[_implied(v['over'])/(_implied(v['over'])+_implied(v['under']))
                for v in bks.values() if v.get('over') is not None and v.get('under') is not None]
        if not novigs: novigs=[_implied(best_over)]   # one-sided quote -> raw implied (slightly rich)
        # full per-book list for the Explorer's odds-by-book expander (best flagged, ties all get it)
        bp=sorted([{'book':b,'price':int(v['over']),'best':int(v['over'])==int(best_over)}
                   for b,v in bks.items() if v.get('over') is not None], key=lambda x:-_payout(x['price']))
        out[nm]={'over':int(best_over),'book':best_book,'book_prices':bp,
                 'mkt_prob':round(100*float(np.median(novigs)),1),'n_books':len(overs)}
    print(f'HR odds: {n_ev} events priced, {len(out)} players with anytime-HR props '
          f'(regions={HR_ODDS_REGIONS}).')
    return out

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
            if not pit or pit['bf']<MIN_PITCHER_BF: continue
            supp=float(np.clip(pit['xwoba']/lg_xw,0.90,1.12))
            # official lineup if posted; else projected pool (upcoming games only)
            proj = not lineup
            if proj:
                if not upcoming: continue
                lineup = team_proj_batters(bat_id)
            for slot, pl in enumerate(lineup, start=1):
                bid=int(pl['id']); pos=pl.get('pos','')
                if bid not in b2a: continue
                h=H.get(bid)
                if not h or h['pa']<MIN_HITTER_PA: continue
                ak=b2a[bid][0]; hand='L' if ak.endswith('_L') else 'R'
                park=get_park_factor(g['home'],hand,ak)
                pa_hr=(h['hr_rate']*pit['hr_rate']/lg)*park*wx*supp
                # slot-weighted PA when the lineup is official; neutral PA when projected
                exp_pa=(AVG_PA_VS_GAME if proj else SLOT_PA_SHARE.get(slot,0.10)*TEAM_PA_PER_GAME)
                prob,amer=hr_prob(pa_hr, exp_pa)
                rows.append(dict(batter=bid,pitcher=int(starter),game=f'{g["away"]}@{g["home"]}',team=bat,
                    gtime=gtime,gstart_ms=gstart_ms,upcoming=upcoming,proj=proj,
                    slot=(None if proj else slot),pos=pos,arch=arch[ak]['name'],bat_hand=hand,pit_hand=sthand,
                    hit_hr=round(h['hr_rate'],2),pit_hr=round(pit['hr_rate'],2),
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
    out=hr.groupby(['game_date','batter'])['is_hr'].max().reset_index()
    out.columns=['date','batter','had_hr']
    df=archive.merge(out,on=['date','batter'],how='left')
    g=df[df['had_hr'].notna()].copy()
    if g.empty: return {'graded_dates':0}
    g['had_hr']=g['had_hr'].astype(int)
    g['rank']=g.groupby('date')['hr_prob'].rank(ascending=False,method='first')
    g['bucket']=g['rank'].apply(lambda r:'top10' if r<=10 else('top25' if r<=25 else 'rest'))
    base=float(g['had_hr'].mean())
    perf={'graded_dates':int(g['date'].nunique()),'graded_picks':int(len(g)),
          'base_hr_rate':round(100*base,2),'buckets':{},'calibration':[]}
    for b in ['top10','top25','rest']:
        s=g[g['bucket']==b]
        if len(s): perf['buckets'][b]={'n':int(len(s)),'hit_rate':round(100*s['had_hr'].mean(),2),
            'lift_vs_base':round(100*(s['had_hr'].mean()-base),2)}
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

    # EDGE grading (only where the archive captured a market price): does recal-model-vs-market edge
    # actually predict? Report hit rate AND realized ROI at the taken price for positive vs negative
    # edge, plus by edge size. Guards against the model manufacturing fake edge where it runs hot.
    if 'mkt_prob' in g.columns and 'mkt_over' in g.columns:
        ge = g[g['mkt_prob'].notna() & g['mkt_over'].notna()].copy()
        if len(ge):
            ge['edge'] = (slope*ge['hr_prob'] + intercept) - ge['mkt_prob']   # recal model% - market%
            ge['roi']  = np.where(ge['had_hr']==1, ge['mkt_over'].apply(_payout), -1.0)
            eb={}
            for lab,mask in [('pos_edge', ge['edge']>0), ('neg_edge', ge['edge']<=0),
                             ('edge_ge3', ge['edge']>=3)]:
                s=ge[mask]
                if len(s): eb[lab]={'n':int(len(s)),'hit_rate':round(100*s['had_hr'].mean(),2),
                                    'roi':round(100*s['roi'].mean(),2)}
            perf['edge_buckets']=eb
            perf['edge_graded_picks']=int(len(ge))
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
        keep=['batter','pitcher','game','team','gtime','gstart_ms','upcoming','proj','slot','pos','arch','bat_hand','pit_hand',
              'hit_hr','pit_hr','park','wx','supp','pa_hr','hr_prob','fair','Batter','Pitcher']
        day=df[[c for c in keep if c in df.columns]].copy(); day.insert(0,'date',date)
        if hr_odds:
            _k=day['Batter'].map(_norm)
            day['mkt_over']=_k.map(lambda n:hr_odds.get(n,{}).get('over'))
            day['mkt_book']=_k.map(lambda n:hr_odds.get(n,{}).get('book'))
            day['mkt_prob']=_k.map(lambda n:hr_odds.get(n,{}).get('mkt_prob'))
            day['mkt_n_books']=_k.map(lambda n:hr_odds.get(n,{}).get('n_books'))
            day['book_prices']=_k.map(lambda n:hr_odds.get(n,{}).get('book_prices') or [])
        # archive only OFFICIAL-lineup rows (projected picks are speculative -> excluded
        # from the forward-track so grading stays honest); keeps RAW hr_prob for calib.
        # Drop the display-only book_prices list column so it doesn't bloat/round-trip in the CSV.
        official=(day[day['proj']==False] if 'proj' in day.columns else day).drop(columns=['book_prices'], errors='ignore')
        # CARRY FORWARD odds: HR props post late, so a game's price is often captured only by a later
        # run -- but once that game STARTS, fetch_hr_odds skips it (and an empty fetch adds no columns),
        # so without this each subsequent run would blank a price we already had. Ensure the odds
        # columns exist, then backfill any missing value from the prior archive for the same
        # (date, batter). The current run's fetch always wins where it has a value.
        ODDC=['mkt_over','mkt_book','mkt_prob','mkt_n_books']
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
        # EDGE = recalibrated model% - de-vigged market% (present only where a book priced the player).
        # Ranking by edge is the fix for "why skip the top-prob guys": a chalk hitter stays on top
        # UNLESS his price already ate the value; a mid-prob hitter the book underrates rises.
        if hr_odds and 'mkt_prob' in d.columns and d['mkt_prob'].notna().any():
            d['edge'] = (d['hr_prob'] - d['mkt_prob']).round(1)
            ranked_by = 'edge'
            # priced rows first, ranked by edge desc; unpriced fall below, by probability
            d['_priced'] = d['mkt_prob'].notna()
            d = d.sort_values(['_priced','edge','hr_prob'], ascending=[False,False,False]).drop(columns='_priced').reset_index(drop=True)
            def _etier(r):
                e=r['edge']
                if pd.isna(e): return 'noodds'          # no market -> can't call value
                if e>=3: return 'value'                 # >=3 pts model over market = the play
                if e>=0: return 'lean'                  # small positive edge
                return 'fade'                           # market prices it richer than the model
            d['tier']=d.apply(_etier,axis=1)
        else:
            d = d.sort_values('hr_prob', ascending=False).reset_index(drop=True)
            ranked_by = 'prob'
            # legacy DFS tiers by rank: Chalk (top 10, model runs hot), Value (11-25), Deep (26+).
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
