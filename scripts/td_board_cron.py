"""
NFL Anytime-TD board -- cloud cron (GitHub Actions, inside the Ozzie repo). Self-contained: no
dependency on the standalone nfl_td project. Pulls nflverse (nflreadpy) + live anytime-TD odds
(The Odds API), builds the calibrated role-based board the app reads, and runs the 2026-H1
forward-tracker for the two research watchlist leans. Writes to repo root:
  td_board_latest.json         (the app's /api/td_board reads this)
  nfl_watchlist_log.csv        (forward-test log: ATD longshots + receptions high-line unders)

Research basis (see nfl_td/GAMEPLAN.md): market is efficient -> this is a decision/shopping aid,
not an edge finder. model_p is Platt-calibrated on 2024+2025 (A,B below). Best price = shop DK+FD.
The two logged leans replicated across 2024+2025 but are THIN/H1-only -> forward-track, don't size.
"""
import os, sys, json, math, time, unicodedata, datetime as dt
import requests, polars as pl, nflreadpy as nfl
from scipy.stats import poisson
import numpy as np

REPO   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD  = os.path.join(REPO, "td_board_latest.json")
LOG    = os.path.join(REPO, "nfl_watchlist_log.csv")
SEASON = int(os.environ.get("NFL_SEASON", dt.date.today().year))
KEY    = os.environ.get("ODDS_API_KEY", "")
B, S   = "https://api.the-odds-api.com/v4", "americanfootball_nfl"
MA_BOOKS = "draftkings,fanduel,betmgm,williamhill_us,fanatics,espnbet"
TD_PER_POINT = 0.108
CAL_A, CAL_B = -0.421, 0.551          # Platt calibration fit on 2024+2025 (calibrate_model.py)

# ── helpers ─────────────────────────────────────────────────────────────────
def norm(s):
    if s is None: return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode().lower().replace(".","").replace("'","").replace("-"," ")
    return " ".join(p for p in s.split() if p not in ("jr","sr","ii","iii","iv","v"))
def a2p(a): return None if a is None else ((-a)/(-a+100) if a<0 else 100/(a+100))
def win_prof(a): return a/100 if a>0 else 100/(-a)
def bucket(pos): return "QB" if pos=="QB" else ("RB" if pos in ("RB","FB") else ("WR/TE" if pos in ("WR","TE") else "OTHER"))
def calibrate(p):
    p = min(max(p,1e-6),1-1e-6); return 1/(1+math.exp(-(CAL_A+CAL_B*math.log(p/(1-p)))))
def _get(url, params):
    for _ in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code==200: return r.json()
            if r.status_code==429: time.sleep(2); continue
        except Exception: time.sleep(1)
    return None

def current_week():
    sched = nfl.load_schedules([SEASON])
    played = sched.filter(pl.col("home_score").is_not_null())
    return (int(played["week"].max())+1) if played.height else 1

# ── weather (Open-Meteo, free) -- soft fade-context for pass-catcher longshots ───────
# Backtest (2024-25): bad weather = LOWER-SCORING game (fades all TDs, esp. pass-catchers),
# NOT a run/pass reallocation (RB rush-TD share barely moves). So this is fade-context only.
STADIUMS = {
 "ARI":(33.5276,-112.2626,"retract"),"ATL":(33.7554,-84.4008,"retract"),"BAL":(39.2780,-76.6227,"out"),
 "BUF":(42.7738,-78.7870,"out"),"CAR":(35.2258,-80.8528,"out"),"CHI":(41.8623,-87.6167,"out"),
 "CIN":(39.0955,-84.5161,"out"),"CLE":(41.5061,-81.6995,"out"),"DAL":(32.7473,-97.0945,"retract"),
 "DEN":(39.7439,-105.0201,"out"),"DET":(42.3400,-83.0456,"dome"),"GB":(44.5013,-88.0622,"out"),
 "HOU":(29.6847,-95.4107,"retract"),"IND":(39.7601,-86.1639,"retract"),"JAX":(30.3239,-81.6373,"out"),
 "KC":(39.0489,-94.4839,"out"),"LV":(36.0909,-115.1833,"dome"),"LAC":(33.9535,-118.3392,"fixed"),
 "LAR":(33.9535,-118.3392,"fixed"),"MIA":(25.9580,-80.2389,"out"),"MIN":(44.9736,-93.2578,"dome"),
 "NE":(42.0909,-71.2643,"out"),"NO":(29.9511,-90.0812,"dome"),"NYG":(40.8135,-74.0745,"out"),
 "NYJ":(40.8135,-74.0745,"out"),"PHI":(39.9008,-75.1675,"out"),"PIT":(40.4468,-80.0158,"out"),
 "SF":(37.4032,-121.9698,"out"),"SEA":(47.5952,-122.3316,"out"),"TB":(27.9759,-82.5033,"out"),
 "TEN":(36.1665,-86.7713,"out"),"WAS":(38.9077,-76.8645,"out"),
}
def wx_for(home, commence):
    st = STADIUMS.get(home)
    if not st: return {"tag":"?","precip_mm":None,"wind_mph":None}
    lat,lon,roof = st
    if roof != "out": return {"tag":"indoor","precip_mm":0.0,"wind_mph":0.0}
    try:
        h = requests.get("https://api.open-meteo.com/v1/forecast",
            {"latitude":lat,"longitude":lon,"hourly":"precipitation,wind_speed_10m","timezone":"UTC",
             "wind_speed_unit":"mph","precipitation_unit":"mm","forecast_days":8}, timeout=15).json()["hourly"]
        tgt = dt.datetime.fromisoformat(commence.replace("Z","+00:00")).replace(tzinfo=None)
        i = min(range(len(h["time"])), key=lambda k: abs((dt.datetime.fromisoformat(h["time"][k])-tgt).total_seconds()))
        pr,wd = h["precipitation"][i], h["wind_speed_10m"][i]
        tags=[];
        if pr>=0.8: tags.append("wet")
        if wd>=15: tags.append("windy")
        return {"tag":"+".join(tags) if tags else "clear","precip_mm":round(pr,2),"wind_mph":round(wd,1)}
    except Exception:
        return {"tag":"?","precip_mm":None,"wind_mph":None}

# ── leak-free role priors (trailing thru latest completed week + prior-yr shrink) ────
def priors_current():
    def load_opp(seasons):
        o = nfl.load_ff_opportunity(seasons, stat_type="weekly")
        for c in ("rush_touchdown_exp","rec_touchdown_exp","rush_touchdown_exp_team","rec_touchdown_exp_team"):
            if c in o.columns: o = o.with_columns(pl.col(c).cast(pl.Float64, strict=False))
        o = o.with_columns(
            (pl.col("rush_touchdown_exp").fill_null(0)+pl.col("rec_touchdown_exp").fill_null(0)).alias("scorer_exp"),
            (pl.col("rush_touchdown_exp_team").fill_null(0)+pl.col("rec_touchdown_exp_team").fill_null(0)).alias("team_exp"))
        return o.with_columns(
            pl.col("position").map_elements(bucket, return_dtype=pl.Utf8).alias("bkt"),
            (pl.when(pl.col("team_exp")>0).then(pl.col("scorer_exp")/pl.col("team_exp")).otherwise(None)).alias("share"))
    cur = load_opp([SEASON])
    py  = (load_opp([SEASON-1]).group_by("player_id").agg(
              pl.col("scorer_exp").mean().alias("py_exp"), pl.col("share").mean().alias("py_share")))
    g = (cur.group_by("player_id").agg(
            pl.col("full_name").last().alias("full_name"), pl.col("bkt").last().alias("bkt"),
            pl.col("posteam").last().alias("posteam"),
            pl.col("scorer_exp").mean().alias("ytd_exp"), pl.col("share").mean().alias("ytd_share"),
            pl.len().alias("gp"))
         .join(py, on="player_id", how="left"))
    K = 4.0
    g = g.with_columns(pl.col("py_share").is_not_null().alias("has_py"),
                       pl.col("full_name").map_elements(norm, return_dtype=pl.Utf8).alias("nkey"),
                       pl.col("ytd_share").fill_null(pl.col("py_share")).alias("ytd_share"))
    g = g.with_columns(((pl.col("ytd_share")*pl.col("gp")+pl.col("py_share").fill_null(0)*K)/(pl.col("gp")+K)).alias("share"))
    g = g.filter(pl.col("player_id").is_not_null() & pl.col("full_name").is_not_null() & (pl.col("gp")<=25))
    # snap-share role gate
    try:
        sc = nfl.load_snap_counts([SEASON]).filter(pl.col("offense_pct").is_not_null())
        snap = (sc.with_columns(pl.col("player").map_elements(norm, return_dtype=pl.Utf8).alias("nkey"))
                  .group_by("nkey").agg(pl.col("offense_pct").mean().alias("snap")))
        g = g.join(snap, on="nkey", how="left")
    except Exception:
        g = g.with_columns(pl.lit(None, dtype=pl.Float64).alias("snap"))
    return g.select(["nkey","full_name","bkt","posteam","share","has_py","snap","gp"])

# ── live odds ────────────────────────────────────────────────────────────────
def slate_events():
    evs = _get(f"{B}/sports/{S}/events", {"apiKey": KEY}) or []
    horizon = dt.datetime.now(dt.timezone.utc)+dt.timedelta(days=8)
    return [e for e in evs if dt.datetime.fromisoformat(e["commence_time"].replace("Z","+00:00"))<horizon]

def pull_atd(events, tmap):
    gl, atd = [], []
    for e in events:
        d = _get(f"{B}/sports/{S}/events/{e['id']}/odds",
                 {"apiKey":KEY,"regions":"us,us2","markets":"player_anytime_td,totals,spreads",
                  "oddsFormat":"american","bookmakers":MA_BOOKS})
        if not d: continue
        home, away = e["home_team"], e["away_team"]
        for bk in d.get("bookmakers", []):
            total=spread=None
            for mk in bk.get("markets", []):
                if mk["key"]=="totals":
                    for o in mk["outcomes"]:
                        if o.get("name")=="Over": total=o.get("point")
                elif mk["key"]=="spreads":
                    for o in mk["outcomes"]:
                        if o.get("name")==home: spread=o.get("point")
                elif mk["key"]=="player_anytime_td":
                    for o in mk["outcomes"]:
                        if o.get("name")=="Yes":
                            atd.append({"nkey":norm(o.get("description")),"player":o.get("description"),
                                        "book":bk["key"],"price":o.get("price")})
            if total is not None and spread is not None:
                gl.append({"home":tmap.get(home),"away":tmap.get(away),
                           "home_imp":total/2-spread/2,"away_imp":total/2+spread/2})
        time.sleep(0.25)
    return gl, atd

def implied_by_team(gl):
    rows=[]
    for r in gl:
        rows.append({"team":r["home"],"imp":r["home_imp"],"opp":r["away"]})
        rows.append({"team":r["away"],"imp":r["away_imp"],"opp":r["home"]})
    if not rows: return {}
    agg=pl.DataFrame(rows).group_by("team").agg(pl.col("imp").median().alias("imp"),pl.col("opp").first().alias("opp"))
    return {r["team"]:r for r in agg.iter_rows(named=True)}

# ── build board ───────────────────────────────────────────────────────────────
def build_board(events, tmap):
    pri = priors_current()
    gl, atd = pull_atd(events, tmap)
    imp = implied_by_team(gl)
    # team abbr -> {commence ISO, game label AWAY@HOME} for start-time / game filters
    team_ev = {}
    for e in events:
        a, h = tmap.get(e["away_team"]), tmap.get(e["home_team"])
        info = {"commence": e.get("commence_time"), "game": f"{a}@{h}"}
        if a: team_ev[a] = info
        if h: team_ev[h] = info
    mkt = None
    if atd:
        a = pl.DataFrame(atd).with_columns(pl.col("price").map_elements(a2p, return_dtype=pl.Float64).alias("imp_p"))
        mkt = a.group_by("nkey").agg(
            pl.col("price").sort_by("imp_p").first().alias("best_price"),
            pl.col("book").sort_by("imp_p").first().alias("best_book"),
            pl.col("imp_p").median().alias("mkt_p"), pl.col("book").n_unique().alias("n_books"))
    rows=[]
    for r in pri.iter_rows(named=True):
        ti = imp.get(r["posteam"])
        if ti is None or r["share"] is None or r["share"]<=0: continue
        lam = r["share"]*ti["imp"]*TD_PER_POINT
        cal = calibrate(1-math.exp(-lam))
        role_ok = (r["snap"] is not None and r["snap"]>=0.5)
        thin = (not r["has_py"]) and (not role_ok) and (r["gp"]<2)
        te = team_ev.get(r["posteam"], {})
        rows.append({"player":r["full_name"],"nkey":r["nkey"],"pos":r["bkt"],"team":r["posteam"],
                     "opp":ti["opp"],"game":te.get("game"),"commence":te.get("commence"),
                     "implied_total":round(ti["imp"],2),"td_share":round(r["share"],3),
                     "model_p_cal":round(cal,3),"thin":bool(thin),"snap_share":round(r["snap"],2) if r["snap"] is not None else None})
    board = pl.DataFrame(rows)
    if mkt is not None:
        board = board.join(mkt, on="nkey", how="left")
    else:
        board = board.with_columns([pl.lit(None).alias(c) for c in ("best_price","best_book","mkt_p","n_books")])
    board = board.with_columns([
        pl.col("mkt_p").alias("mkt_p_consensus"),
        (pl.col("model_p_cal")-pl.col("mkt_p")).alias("edge"),
        # 🎯 focus slice = the ONE thing that replicated 2024+2025 (see nfl_td/GAMEPLAN): WR/TE
        # established-role deep-longshots (<7% implied) in season H1 hit ~11-14% vs ~5% priced
        # (+6-8pp). RB/QB longshots are priced right -> excluded. Filter to these, shop best price.
        pl.when(pl.col("mkt_p").is_null()).then(pl.lit("no price"))
          .when((~pl.col("thin"))&(pl.col("pos")=="WR/TE")&(pl.col("mkt_p")<0.07)).then(pl.lit("H1-longshot-watch"))
          .otherwise(pl.lit("")).alias("flag")]).sort("edge", descending=True, nulls_last=True)
    # weather: one lookup per unique game (home stadium + kickoff), attached to rows as context.
    wxc = {}
    for g, c in {(r["game"], r["commence"]) for r in board.iter_rows(named=True) if r["game"] and r["commence"]}:
        wxc[(g, c)] = wx_for(g.split("@")[1], c)
    board = board.with_columns(
        pl.struct(["game","commence"]).map_elements(
            lambda s: (wxc.get((s["game"], s["commence"])) or {}).get("tag"), return_dtype=pl.Utf8).alias("wx_tag"),
        pl.struct(["game","commence"]).map_elements(
            lambda s: (wxc.get((s["game"], s["commence"])) or {}).get("precip_mm"), return_dtype=pl.Float64).alias("wx_precip"),
        pl.struct(["game","commence"]).map_elements(
            lambda s: (wxc.get((s["game"], s["commence"])) or {}).get("wind_mph"), return_dtype=pl.Float64).alias("wx_wind"))
    meta={"generated":dt.datetime.now().isoformat(timespec="minutes"),"season":SEASON,
          "best_book_rule":"shop DK+FD; Caesars runs rich",
          "note":"role x market implied total, Platt-calibrated. Efficient market -> decision aid, not edge."}
    json.dump({"meta":meta,"rows":board.to_dicts()}, open(BOARD,"w"), indent=1)
    # COVERAGE GUARD: every team with a posted total (on the slate) must land >=1 player row.
    # A gap = an abbr join mismatch like LAR/LA silently dropping a whole team. (N<32 is normal:
    # games already kicked off drop from the pre-game odds feed; byes start ~wk5.)
    missing=sorted(set(imp.keys())-set(board["team"].to_list()))
    if missing: print(f"[board] !! WARN slate teams with ZERO player rows (abbr mismatch?): {missing}")
    else: print(f"[board] coverage OK: all {len(imp)} slate teams have >=1 player row")
    priced=board.filter(pl.col("mkt_p").is_not_null()&(~pl.col("thin")))
    print(f"[board] wrote {BOARD}: {board.height} rows, {priced.height} priced trusted")
    return board

# ── forward tracker (both leans) ────────────────────────────────────────────
def recv_current():
    cur=(nfl.load_player_stats([SEASON]).with_columns(pl.col("week").cast(pl.Int64))
         .filter(pl.col("position").is_in(["WR","TE","RB","FB"]))
         .with_columns(pl.col("receptions").fill_null(0).cast(pl.Float64),
                       pl.col("player_display_name").map_elements(norm,return_dtype=pl.Utf8).alias("nkey")))
    py=(nfl.load_player_stats([SEASON-1]).filter(pl.col("week")<=18)
        .with_columns(pl.col("receptions").fill_null(0).cast(pl.Float64),
                      pl.col("player_display_name").map_elements(norm,return_dtype=pl.Utf8).alias("nkey"))
        .group_by("nkey").agg(pl.col("receptions").mean().alias("py_rec")))
    g=cur.group_by("nkey").agg(pl.col("receptions").mean().alias("ytd"),pl.len().alias("g")).join(py,on="nkey",how="left")
    K=4.0
    return g.with_columns(pl.col("ytd").fill_null(pl.col("py_rec")).alias("ytd")).with_columns(
        ((pl.col("ytd")*pl.col("g")+pl.col("py_rec").fill_null(0)*K)/(pl.col("g")+K)).alias("proj")).select(["nkey","proj"])

def pull_recv(events):
    rows=[]
    for e in events:
        d=_get(f"{B}/sports/{S}/events/{e['id']}/odds",
               {"apiKey":KEY,"regions":"us,us2","markets":"player_receptions","oddsFormat":"american","bookmakers":MA_BOOKS})
        if not d: continue
        for bk in d.get("bookmakers",[]):
            for mk in bk.get("markets",[]):
                if mk["key"]!="player_receptions": continue
                for o in mk["outcomes"]:
                    rows.append({"book":bk["key"],"player":o.get("description"),"side":o.get("name"),
                                 "line":o.get("point"),"price":o.get("price")})
        time.sleep(0.25)
    return pl.DataFrame(rows) if rows else pl.DataFrame()

def read_log(): return pl.read_csv(LOG) if os.path.exists(LOG) else None

def tracker_grade():
    log=read_log()
    if log is None: return
    if not log.filter(pl.col("won").is_null()).height: print("[grade] nothing ungraded"); return
    ps=nfl.load_player_stats(log["season"].unique().to_list()).with_columns(pl.col("week").cast(pl.Int64),
        pl.col("player_display_name").map_elements(norm,return_dtype=pl.Utf8).alias("nkey"),
        (pl.col("rushing_tds").fill_null(0)+pl.col("receiving_tds").fill_null(0)).alias("td"),
        pl.col("receptions").fill_null(0).alias("rec"))
    done=set((r["season"],r["week"]) for r in ps.select(["season","week"]).unique().iter_rows(named=True))
    rows=log.to_dicts(); n=0
    for r in rows:
        if r["won"] is not None or (r["season"],r["week"]) not in done: continue
        st=ps.filter((pl.col("season")==r["season"])&(pl.col("week")==r["week"])&(pl.col("nkey")==norm(r["player"])))
        if not st.height: continue
        s=st.row(0,named=True)
        won=(1 if s["td"]>=1 else 0) if r["market"]=="ATD_longshot" else (1 if s["rec"]<r["line"] else 0)
        r["won"]=won; r["profit"]=win_prof(int(r["price"])) if won else -1.0; n+=1
    out=pl.DataFrame(rows); out.write_csv(LOG)
    print(f"[grade] settled +{n}")
    for r in out.filter(pl.col("won").is_not_null()).group_by("market").agg(
            pl.len().alias("n"),pl.col("won").mean().alias("hit"),pl.col("profit").mean().alias("roi")).iter_rows(named=True):
        print(f"   {r['market']:<18} n={r['n']:<4} hit={r['hit']:.1%} ROI={r['roi']:+.1%}")

def tracker_log(board, events, wk):
    picks=[]
    for r in board.filter(pl.col("flag")=="H1-longshot-watch").iter_rows(named=True):
        if r["best_price"] is None: continue
        picks.append({"season":SEASON,"week":wk,"market":"ATD_longshot","side":"Yes","player":r["player"],
                      "line":0.5,"price":r["best_price"],"book":r["best_book"],
                      "model_p":r["model_p_cal"],"mkt_p":r["mkt_p_consensus"]})
    od=pull_recv(events)
    if od.height:
        od=od.with_columns(pl.col("player").map_elements(norm,return_dtype=pl.Utf8).alias("nkey"),
                           pl.col("price").map_elements(a2p,return_dtype=pl.Float64).alias("imp_p"))
        ln=od.group_by("nkey").agg(pl.col("line").median().alias("line"))
        un=(od.filter(pl.col("side")=="Under").join(ln,on="nkey").group_by(["nkey","line"]).agg(
              pl.col("imp_p").median().alias("mkt_p"),pl.col("price").sort_by("imp_p").first().alias("best_price"),
              pl.col("book").sort_by("imp_p").first().alias("best_book"),pl.col("player").first().alias("player"))
            .join(recv_current(),on="nkey",how="inner"))
        pu=poisson.cdf(np.floor(un["line"].to_numpy()),un["proj"].to_numpy())
        un=un.with_columns(pl.Series("model_p",pu)).with_columns((pl.col("model_p")-pl.col("mkt_p")).alias("edge"))
        for r in un.filter((pl.col("line")>=4.5)&(pl.col("edge")>0.03)).iter_rows(named=True):
            picks.append({"season":SEASON,"week":wk,"market":"RECV_under_combo","side":"Under","player":r["player"],
                          "line":r["line"],"price":r["best_price"],"book":r["best_book"],
                          "model_p":round(r["model_p"],3),"mkt_p":round(r["mkt_p"],3)})
    if not picks: print(f"[log] wk{wk}: no picks"); return
    new=pl.DataFrame(picks).with_columns(pl.lit(None,dtype=pl.Int64).alias("won"),
            pl.lit(None,dtype=pl.Float64).alias("profit"),pl.lit(dt.date.today().isoformat()).alias("logged"))
    old=read_log()
    if old is not None:
        new=new.join(old.select(["season","week","market","player"]).unique(),on=["season","week","market","player"],how="anti")
        out=pl.concat([old,new],how="diagonal_relaxed")
    else: out=new
    out.write_csv(LOG); print(f"[log] wk{wk}: +{new.height} picks -> {LOG}")

def main():
    if not KEY: print("no ODDS_API_KEY -- board will be role-only / unpriced")
    # nflverse disagrees with itself: load_teams() calls the Rams "LAR" but the priors/pbp feed
    # (posteam) codes them "LA" -> imp.get("LA") missed and silently dropped EVERY Rams player.
    # Canonicalize to the priors spelling. Chargers safe (both=LAC). Same class as accent-drops.
    TEAM_ABBR_FIX={"LAR":"LA"}
    tmap={r["team_name"]:TEAM_ABBR_FIX.get(r["team_abbr"],r["team_abbr"])
          for r in nfl.load_teams().select(["team_name","team_abbr"]).to_dicts()}
    events=slate_events()
    print(f"[cron] season {SEASON}, {len(events)} slate events")
    wk=current_week()
    tracker_grade()                       # settle last week first
    board=build_board(events, tmap)
    try: tracker_log(board, events, wk)   # fail-open: board is the priority artifact
    except Exception as ex: print(f"[log] skipped: {ex}")

if __name__=="__main__":
    main()
