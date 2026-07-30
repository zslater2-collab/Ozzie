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
import os, sys, json, math, glob
from datetime import datetime, timedelta
import pandas as pd, numpy as np, requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data')
PA_CACHE = os.path.join(DATA, 'hr_pa_2026.csv.gz')
ARCH_CSV = os.path.join(REPO, 'hr_board_archive.csv')
LATEST   = os.path.join(REPO, 'hr_board_latest.json')
PERF     = os.path.join(REPO, 'hr_board_perf.json')

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
            hl=[p['id'] for p in g.get('lineups',{}).get('homePlayers',[]) if p.get('id')]
            al=[p['id'] for p in g.get('lineups',{}).get('awayPlayers',[]) if p.get('id')]
            games.append(dict(home=tmap.get(hid,str(hid)),away=tmap.get(aid,str(aid)),
                home_lineup=hl,away_lineup=al,
                home_starter=g['teams']['home'].get('probablePitcher',{}).get('id'),
                away_starter=g['teams']['away'].get('probablePitcher',{}).get('id')))
    return games

def prob_to_american(gp):
    gp = min(max(gp/100.0, 0.005), 0.95)
    return round(-(gp/(1-gp))*100) if gp>=0.5 else round(((1-gp)/gp)*100)

def hr_prob(pa_hr, avg_pa=AVG_PA_VS_GAME):
    gp = 1-(1-pa_hr/100)**avg_pa
    return round(gp*100,1), prob_to_american(gp*100)

# ---------------- board ----------------
def build_board(game_date, H, P, meta):
    lg, lg_xw = meta['lg_hr_rate'], meta['lg_xwoba']
    rows=[]
    for g in get_lineups(game_date):
        wx=get_weather_factor(g['home'])
        for starter, lineup, bat, field in [
            (g['home_starter'],g['away_lineup'],g['away'],g['home']),
            (g['away_starter'],g['home_lineup'],g['home'],g['away'])]:
            if not starter or not lineup: continue
            pit=P.get(int(starter))
            if not pit or pit['bf']<MIN_PITCHER_BF: continue
            supp=float(np.clip(pit['xwoba']/lg_xw,0.90,1.12))
            for slot, bid in enumerate(lineup, start=1):
                bid=int(bid)
                if bid not in b2a: continue
                h=H.get(bid)
                if not h or h['pa']<MIN_HITTER_PA: continue
                ak=b2a[bid][0]; hand='L' if ak.endswith('_L') else 'R'
                park=get_park_factor(g['home'],hand,ak)
                pa_hr=(h['hr_rate']*pit['hr_rate']/lg)*park*wx*supp
                exp_pa=SLOT_PA_SHARE.get(slot,0.10)*TEAM_PA_PER_GAME   # slot-weighted PA
                prob,amer=hr_prob(pa_hr, exp_pa)
                rows.append(dict(batter=bid,pitcher=int(starter),game=f'{g["away"]}@{g["home"]}',
                    slot=slot,arch=arch[ak]['name'],hit_hr=round(h['hr_rate'],2),pit_hr=round(pit['hr_rate'],2),
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
    return perf

def main():
    date = sys.argv[1] if len(sys.argv)>1 else datetime.utcnow().strftime('%Y-%m-%d')
    pa = refresh_pa_cache()
    H,P,meta = build_stats(pa)
    df = build_board(date, H, P, meta)

    if not df.empty:
        keep=['batter','pitcher','game','slot','arch','hit_hr','pit_hr','park','wx','supp',
              'pa_hr','hr_prob','fair','Batter','Pitcher']
        day=df[[c for c in keep if c in df.columns]].copy(); day.insert(0,'date',date)
        if os.path.exists(ARCH_CSV):
            old=pd.read_csv(ARCH_CSV); old['date']=old['date'].astype(str)
            arch_df=pd.concat([old[old['date']!=date],day],ignore_index=True)
        else:
            arch_df=day
        arch_df.to_csv(ARCH_CSV,index=False)   # archive keeps RAW hr_prob (stable calib base)
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

    # write display artifact: recalibrated probability + DFS tier tags
    if not df.empty:
        cal = perf.get('calib', {'slope':1.0,'intercept':0.0})
        d = day.copy()
        d['hr_prob_raw'] = d['hr_prob']
        d['hr_prob'] = (cal['intercept'] + cal['slope']*d['hr_prob_raw']).clip(0.5, 60).round(1)
        d = d.sort_values('hr_prob', ascending=False).reset_index(drop=True)
        d['fair'] = d['hr_prob'].apply(lambda p: ('+%d'%a if (a:=prob_to_american(p))>0 else str(a)))
        # tier by rank: top 10 = Chalk (model runs hot here, priciest in DFS),
        # 11-25 = Value (the reliable band, one tier down), rest dropped from the board.
        d['tier'] = ['chalk' if i<10 else ('value' if i<25 else 'deep') for i in range(len(d))]
        top = d[d['tier']!='deep']
        json.dump({'date':date,'asof':meta['asof'],'calib':cal,
                   'rows':top.to_dict(orient='records')}, open(LATEST,'w'), indent=2)
        print(f'Board {date}: {len(day)} rows -> {len(top)} shown '
              f'(recal slope {cal.get("slope")}, asof {meta["asof"]}).')

if __name__=='__main__':
    main()
