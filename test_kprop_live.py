"""
LOCAL confirmation for the bundled K-prop fetch — does NOT deploy or touch production.
Mirrors exactly what get_odds_api_lines now does (5 markets in one per-event call) so you can
verify, before pushing:
  1. pitcher_strikeouts parses live (and the F5 total still comes back in the same call)
  2. the Odds API pitcher names match statcast ("Last, First") after normalization
  3. which starters would FIRE the over-favorite signal today (DK over -120..-160)

RUN (PowerShell):
    $env:ODDS_API_KEY = "your_key_here"
    python test_kprop_live.py
Cost: 1 events-list call + (MAX_EVENTS) per-event calls. Keep MAX_EVENTS small to limit credits.
"""
import os, requests
import pandas as pd

API_KEY    = os.environ.get('ODDS_API_KEY', '')
BASE       = 'https://api.the-odds-api.com/v4'
REGIONS    = 'us,us2,eu'
F5_MARKET  = 'team_totals_1st_5_innings'
KPROP      = 'pitcher_strikeouts'
MARKETS    = f'{F5_MARKET},{KPROP}'          # subset of the app's 5 — enough to confirm
MAX_EVENTS = 4                                # keep credit spend low
FAV_LO, FAV_HI = -160, -120

TARGET_BOOKS = {'draftkings':'DraftKings','fanatics':'Fanatics','espnbet':'theScore Bet',
                'thescorebet':'theScore Bet','fanduel':'FanDuel','betmgm':'BetMGM','caesars':'Caesars'}

if not API_KEY:
    raise SystemExit("Set ODDS_API_KEY first:  $env:ODDS_API_KEY = 'your_key'")

def norm(n):
    if not n: return ''
    n = n.strip()
    if ',' in n:
        p = [x.strip() for x in n.split(',',1)]
        n = f"{p[1]} {p[0]}" if len(p)==2 else p[0]
    return ' '.join(n.lower().replace('.','').split())

# 1) events list
ev = requests.get(f"{BASE}/sports/baseball_mlb/events", params={'apiKey':API_KEY}, timeout=15)
print(f"events call: status={ev.status_code}, remaining credits={ev.headers.get('x-requests-remaining','?')}")
events = ev.json() if ev.ok else []
print(f"{len(events)} upcoming events; sampling {min(MAX_EVENTS,len(events))}\n")

# statcast starter names for match check
try:
    sc = pd.read_pickle(r'C:\Users\zslat\Documents\statcast_2026_cache.pkl')
    sc_names = {norm(x) for x in sc[sc['inning']==1]['player_name'].dropna().unique()}
    print(f"loaded {len(sc_names)} statcast starter names for match check\n")
except Exception as e:
    sc_names = set(); print(f"(couldn't load statcast names: {e})\n")

odds_names, fired = set(), []
for e in events[:MAX_EVENTS]:
    r = requests.get(f"{BASE}/sports/baseball_mlb/events/{e['id']}/odds",
                     params={'apiKey':API_KEY,'regions':REGIONS,'markets':MARKETS,'oddsFormat':'american'},
                     timeout=15)
    if not r.ok:
        print(f"  {e.get('away_team')} @ {e.get('home_team')}: per-event call FAILED {r.status_code}"); continue
    data = r.json()
    has_f5 = has_kp = False
    kp_by_pitcher = {}
    for bm in data.get('bookmakers', []):
        label = TARGET_BOOKS.get(bm.get('key',''))
        if not label: continue
        for m in bm.get('markets', []):
            if m['key']==F5_MARKET: has_f5 = True
            if m['key']==KPROP:
                has_kp = True
                by = {}
                for o in m.get('outcomes', []):
                    by.setdefault(norm(o.get('description','')),{})[o.get('name','').lower()] = o
                for nm, sides in by.items():
                    if sides.get('over'):
                        odds_names.add(nm)
                        kp_by_pitcher.setdefault(nm,{})[label] = int(sides['over']['price'])
    print(f"  {e.get('away_team')} @ {e.get('home_team')}: F5={'OK' if has_f5 else 'MISSING'}  Kprop={'OK' if has_kp else 'MISSING'}  pitchers={len(kp_by_pitcher)}")
    for nm, books in kp_by_pitcher.items():
        dk = books.get('DraftKings')
        best_book = max(books, key=books.get); best = books[best_book]
        match = 'matches statcast' if nm in sc_names else 'NO statcast match'
        sig = ' <== WOULD FIRE (DK over in band)' if (dk is not None and FAV_LO<=dk<=FAV_HI) else ''
        print(f"      {nm:<22} DK over {dk}  best {best:+d}({best_book})  [{match}]{sig}")
        if dk is not None and FAV_LO<=dk<=FAV_HI: fired.append(nm)

# 3) name-match summary
if sc_names and odds_names:
    matched = odds_names & sc_names
    print(f"\nNAME MATCH: {len(matched)}/{len(odds_names)} odds-api pitchers matched statcast "
          f"({len(matched)/len(odds_names)*100:.0f}%)")
    unmatched = sorted(odds_names - sc_names)
    if unmatched: print(f"  unmatched (would be missed): {unmatched}")
print(f"\nWould fire over-favorite signal on {len(fired)} starter(s) in this sample: {fired}")
print("\nIf F5=OK everywhere (bundling didn't break totals), Kprop=OK, and match rate is high — safe to push.")
