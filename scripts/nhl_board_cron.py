"""
NHL Anytime Goal-Scorer board -- cloud cron (GitHub Actions, inside the Ozzie repo).

Self-contained: no dependency on the standalone nhl_goals research project. Pulls the free NHL
stats REST API (point-in-time skater rates) + live anytime-goal odds (The Odds API), and writes
  nhl_board_latest.json    (the app's /api/nhl_board reads this)

MODEL
  lambda = role_share (point-in-time data) x team_exp_goals (from the market)
  P(anytime goal) = 1 - exp(-lambda)
  role_share = exp_goals / sum over the team's projected top-18 dressed skaters
  exp_goals  = g60_ev * toi_ev/3600 + g60_pp * toi_pp/3600
Role from data, magnitude from the line.

LEAK DISCIPLINE (Ozzie Rule #1): season-to-date aggregates are requested with an explicit
gameDate <= YESTERDAY window, so nothing from today's games can enter a feature for today's
games. That is the whole reason we do not call the plain season-summary endpoint, which would
silently include tonight's results on a re-run.

RESEARCH BASIS (nhl_goals/GAMEPLAN.md) -- this is a DECISION/SHOPPING AID, not an edge finder:
  * market is efficient: every price band lands within 0.4-3.6pp of implied, all negative
  * an 823-cell intersection search produced one candidate (PP defensemen) that FLIPPED sign
    out-of-sample (+1.49pp in 2025-26 -> -2.42pp in 2024-25). No selection edge is shipped.
  * the ONE durable effect is LINE SHOPPING: best-of-4 books -7.2% vs worst-of-4 -17.5%
    (+10.3pp). DK best on 45.8% of quotes, FD 43.4%, BetMGM 10.7%, Caesars 0.2% (rich, avoid).
  * model_p was verified already calibrated (pred .1518 vs actual .1548); the Platt constants
    below are a near-identity refit kept for consistency with calibrate_model.py.
"""
import os, sys, json, math, time, unicodedata, datetime as dt
import requests

REPO  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD = os.path.join(REPO, "nhl_board_latest.json")
# Shots-on-goal LINE CAPTURE -- a separate artifact, deliberately not folded into the board
# payload (the board is served to the browser on every page load; ~1k SOG quotes would bloat it
# for no UI benefit). Committed by the same workflow, so git history gives the price panel the
# same way it did for the HR board.
SOG   = os.path.join(REPO, "nhl_sog_latest.json")

KEY = os.environ.get("ODDS_API_KEY", "")
OB, SPORT = "https://api.the-odds-api.com/v4", "icehockey_nhl"
NB = "https://api.nhle.com/stats/rest/en/skater"
MA_BOOKS = "draftkings,fanduel,betmgm,williamhill_us,fanatics,espnbet,ballybet"
REGIONS = "us,us2"
SOFT_BOOKS = {"draftkings", "fanduel"}      # the validated price leaders
# Books Zach can actually bet (MA-legal). Others are priced for reference//shopping context only.
BETTABLE_BOOKS = {"draftkings", "fanduel", "betmgm", "williamhill_us",
                  "espnbet", "fanatics", "ballybet"}

PRIOR_H = {"ev": 5.0, "pp": 1.2}            # Gamma-Poisson prior strength, in TOI hours
TOI_K   = 6.0                               # TOI shrinkage, in games
DRESSED = 18
MIN_GP  = 3
CAL_A, CAL_B = -0.0852, 0.9292              # near-identity; model verified already calibrated
LG = {"ev": 0.55, "pp": 1.6}                # league goals/60 fallbacks
LG_TOI = {"ev": 850.0, "pp": 79.0}          # league TOI/game fallbacks (seconds)

TEAM_ABBR = {
 "Anaheim Ducks": "ANA", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF",
 "Calgary Flames": "CGY", "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI",
 "Colorado Avalanche": "COL", "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL",
 "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM", "Florida Panthers": "FLA",
 "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN", "Montreal Canadiens": "MTL",
 "Montréal Canadiens": "MTL", "Nashville Predators": "NSH", "New Jersey Devils": "NJD",
 "New York Islanders": "NYI", "New York Rangers": "NYR", "Ottawa Senators": "OTT",
 "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJS",
 "Seattle Kraken": "SEA", "St Louis Blues": "STL", "St. Louis Blues": "STL",
 "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR", "Utah Mammoth": "UTA",
 "Utah Hockey Club": "UTA", "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK",
 "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
}


# -- helpers -----------------------------------------------------------------
def norm(s):
    """Accent/punct-insensitive key. The NHL is full of accents (Pettersson, Necas, Stutzle)
    and an unnormalized join silently drops them -- the Ozzie accent landmine."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("-", " ").replace("’", "")
    s = s.replace("'", "")
    return " ".join(p for p in s.split() if p not in ("jr", "sr", "ii", "iii", "iv"))


def a2p(a):
    if a is None:
        return None
    a = float(a)
    return (-a) / (-a + 100.0) if a < 0 else 100.0 / (a + 100.0)


def calibrate(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return 1 / (1 + math.exp(-(CAL_A + CAL_B * math.log(p / (1 - p)))))


def season_id(today=None):
    d = today or dt.date.today()
    y = d.year if d.month >= 8 else d.year - 1
    return "%d%d" % (y, y + 1)


def _nhl(report, cayenne, tries=6):
    """Paged NHL stats REST call. limit caps at 100; backs off on 429."""
    rows, start, delay = [], 0, 2.0
    while True:
        try:
            r = requests.get(NB + "/" + report, timeout=60, params={
                "isAggregate": "true", "isGame": "true", "start": start, "limit": 100,
                "cayenneExp": cayenne})
        except Exception:
            r = None
        if r is None or r.status_code != 200:
            if tries <= 0:
                raise RuntimeError("NHL %s failed" % report)
            tries -= 1
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        d = r.json().get("data", [])
        rows += d
        if len(d) < 100:
            return rows
        start += 100
        if start >= 10000:
            return rows
        time.sleep(0.3)


def skater_state(season, upto):
    """Cumulative-through-`upto` per-skater totals. `upto` is YESTERDAY -> strictly point-in-time
    for today's slate; this is why we do not use the plain season-summary endpoint."""
    cay = 'gameTypeId=2 and seasonId=%s and gameDate<="%s"' % (season, upto)
    summ = dict((r["playerId"], r) for r in _nhl("summary", cay))
    toi = dict((r["playerId"], r) for r in _nhl("timeonice", cay))
    out = {}
    for pid, s in summ.items():
        t = toi.get(pid, {})
        g_pp = s.get("ppGoals") or 0
        g_sh = s.get("shGoals") or 0
        out[pid] = {
            "pid": pid, "name": s.get("skaterFullName"), "pos": s.get("positionCode"),
            "teams": (s.get("teamAbbrevs") or ""), "gp": s.get("gamesPlayed") or 0,
            "g_ev": max((s.get("goals") or 0) - g_pp - g_sh, 0), "g_pp": g_pp,
            "toi_ev": t.get("evTimeOnIce") or 0, "toi_pp": t.get("ppTimeOnIce") or 0,
        }
    return out


def rosters(season):
    """playerId -> CURRENT team, from the free per-team roster endpoint.

    Necessary, not cosmetic: at season start every skater's only stats row is last season's, so
    teamAbbrevs would assign every offseason-traded player to his OLD club -- wrong opponent and
    wrong team total, silently. In-season it also fixes trades faster than the stats feed."""
    out = {}
    for ab in sorted(set(TEAM_ABBR.values())):
        try:
            js = requests.get("https://api-web.nhle.com/v1/roster/%s/%s" % (ab, season),
                              timeout=25).json()
        except Exception:
            continue
        for grp in ("forwards", "defensemen"):
            for p in js.get(grp, []):
                if p.get("id"):
                    out[p["id"]] = ab
        time.sleep(0.05)
    return out


def rates(cur, anchor, team_of):
    """Gamma-Poisson shrunk goal rates + shrunk projected TOI -> expected goals.

    Universe is cur UNION anchor: a skater with no games yet this season still belongs on the
    board, carried entirely by his shrunk prior-season rate (weight_this_season = gp/(gp+K) = 0).
    Iterating only over `cur` emptied the whole board before opening night."""
    out = {}
    blank = {"gp": 0, "g_ev": 0, "g_pp": 0, "toi_ev": 0, "toi_pp": 0}
    for pid in set(cur) | set(anchor):
        c = cur.get(pid)
        if c is None:
            a0 = anchor[pid]
            c = dict(blank, pid=pid, name=a0["name"], pos=a0["pos"], teams=a0["teams"])
        a = anchor.get(pid)
        eg = {}
        for s in ("ev", "pp"):
            b = PRIOR_H[s] * 3600.0
            if a and a["gp"] >= 10 and a["toi_" + s] > 0:
                anch = (a["g_" + s] + LG[s] * b / 3600.0) / ((a["toi_" + s] + b) / 3600.0)
                anch_toi = a["toi_" + s] / a["gp"]
            else:
                anch, anch_toi = LG[s], LG_TOI[s]
            g60 = (c["g_" + s] + anch * b / 3600.0) / ((c["toi_" + s] + b) / 3600.0)
            w = c["gp"] / float(c["gp"] + TOI_K)
            trail = (c["toi_" + s] / c["gp"]) if c["gp"] else anch_toi
            toi = w * trail + (1 - w) * anch_toi
            eg[s] = g60 * toi / 3600.0
            eg["toi_" + s] = toi
        team = team_of.get(pid) or (c["teams"].split(",")[-1].strip()
                                    if c["teams"] else None)
        out[pid] = dict(c, exp_goals=eg["ev"] + eg["pp"],
                        pp_min=round(eg["toi_pp"] / 60, 1), team=team)
    return out


# -- odds --------------------------------------------------------------------
def _odds(url, params):
    try:
        r = requests.get(url, params=params, timeout=30)
    except Exception:
        return None
    if r.status_code != 200:
        print("  ! odds %s %s" % (r.status_code, r.text[:140]))
        return None
    _odds.rem = r.headers.get("x-requests-remaining", getattr(_odds, "rem", "?"))
    return r.json()


_odds.rem = "?"


def devig2(pa, pb):
    if pa is None or pb is None or (pa + pb) <= 0:
        return None
    return pa / (pa + pb)


def poisson_mean_from_over(line, p_over):
    k = int(math.floor(line))

    def f(mu):
        c = math.exp(-mu)
        s = c
        for i in range(1, k + 1):
            c *= mu / i
            s += c
        return 1.0 - s

    lo, hi = 1.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) < p_over:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _pwin(ma, mb, n=15):
    def pmf(mu):
        c = math.exp(-mu)
        o = [c]
        for i in range(1, n + 1):
            c *= mu / i
            o.append(c)
        return o

    pa, pb = pmf(ma), pmf(mb)
    w = t = 0.0
    for i, ai in enumerate(pa):
        for j, bj in enumerate(pb):
            if i > j:
                w += ai * bj
            elif i == j:
                t += ai * bj
    return w + 0.5 * t


def split_total(total, p_home):
    lo, hi = -3.0, 3.0
    for _ in range(40):
        m = (lo + hi) / 2
        if _pwin((total + m) / 2, (total - m) / 2) < p_home:
            lo = m
        else:
            hi = m
    m = (lo + hi) / 2
    return (total + m) / 2, (total - m) / 2


def slate(hours=30):
    js = _odds(OB + "/sports/" + SPORT + "/events", {"apiKey": KEY}) or []
    cut = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=hours)
    out = []
    for e in js:
        c = dt.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        if c <= cut:
            out.append(e)
    return out


def _med(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def game_lines(ev):
    """-> {abbr: {implied, opp, commence, src}} using POSTED team totals where available."""
    d = _odds(OB + "/sports/" + SPORT + "/events/" + ev["id"] + "/odds",
              {"apiKey": KEY, "regions": REGIONS, "markets": "totals,h2h,team_totals",
               "oddsFormat": "american", "bookmakers": MA_BOOKS})
    if not d:
        return {}
    home, away = ev["home_team"], ev["away_team"]
    tt = {home: [], away: []}
    derived = {home: [], away: []}
    for b in d.get("bookmakers", []):
        m = dict((x["key"], x) for x in b.get("markets", []))
        if "team_totals" in m:
            by = {}
            for o in m["team_totals"]["outcomes"]:
                by.setdefault(o.get("description"), {})[o["name"]] = o
            for team, sides in by.items():
                if "Over" in sides and "Under" in sides and team in tt:
                    fair = devig2(a2p(sides["Over"]["price"]), a2p(sides["Under"]["price"]))
                    if fair:
                        tt[team].append(poisson_mean_from_over(sides["Over"]["point"], fair))
        total = h_ml = a_ml = None
        if "totals" in m:
            for o in m["totals"]["outcomes"]:
                if o["name"] == "Over":
                    total = o["point"]
        if "h2h" in m:
            for o in m["h2h"]["outcomes"]:
                if o["name"] == home:
                    h_ml = o["price"]
                elif o["name"] == away:
                    a_ml = o["price"]
        if total and h_ml and a_ml:
            ph = devig2(a2p(h_ml), a2p(a_ml))
            if ph:
                h, a = split_total(total, ph)
                derived[home].append(h)
                derived[away].append(a)

    out = {}
    for team, opp in ((home, away), (away, home)):
        val, src = _med(tt[team]), "posted"
        if val is None:
            val, src = _med(derived[team]), "derived"
        ab = TEAM_ABBR.get(team)
        if ab and val:
            out[ab] = {"implied": val, "opp": TEAM_ABBR.get(opp), "src": src,
                       "commence": ev["commence_time"],
                       "game": "%s@%s" % (TEAM_ABBR.get(away), TEAM_ABBR.get(home))}
    return out


def props(ev):
    d = _odds(OB + "/sports/" + SPORT + "/events/" + ev["id"] + "/odds",
              {"apiKey": KEY, "regions": REGIONS,
               "markets": "player_goal_scorer_anytime",
               "oddsFormat": "american", "bookmakers": MA_BOOKS})
    out = {}
    if not d:
        return out
    for b in d.get("bookmakers", []):
        for mk in b.get("markets", []):
            if mk["key"] != "player_goal_scorer_anytime":
                continue
            for o in mk["outcomes"]:
                if o.get("name") != "Yes":
                    continue
                k = norm(o.get("description"))
                out.setdefault(k, {"player": o.get("description"), "q": []})
                out[k]["q"].append((o.get("price"), b["key"]))
    return out


# Shots on goal is a TWO-WAY market: measured median vig 6.51%, against 17-29% on first-scorer
# and 5-8% on anytime-goal. It also sits on an outcome our role signal predicts far better than
# goals (model_p vs SOG r=0.412 / pp_min r=0.370, against 0.246 / 0.202 for goals), because
# goals = shots x shooting% and game-level shooting% is mostly noise.
# We have ZERO history of these lines, so nothing can be backtested until a record exists --
# that is the only purpose of this capture. No model, no flag, no claim: just the prices.
# Cost: ~1 credit per event-market-region, so 2 markets x 2 regions = ~4/event (~52 a run on a
# 13-game slate). Set NHL_SOG=0 to switch the capture off if that ever matters.
SOG_MARKETS = "player_shots_on_goal,player_shots_on_goal_alternate"


def sog_props(ev):
    """Long-form SOG quotes for one event. All books, not just MA-legal: dispersion across books
    is part of what we want to study, and cost is per market-region, not per book."""
    if os.environ.get("NHL_SOG", "1") != "1":
        return []
    d = _odds(OB + "/sports/" + SPORT + "/events/" + ev["id"] + "/odds",
              {"apiKey": KEY, "regions": REGIONS, "markets": SOG_MARKETS,
               "oddsFormat": "american"})
    if not d:
        return []
    game = "%s@%s" % (TEAM_ABBR.get(d.get("away_team"), d.get("away_team")),
                      TEAM_ABBR.get(d.get("home_team"), d.get("home_team")))
    out = []
    for b in d.get("bookmakers", []):
        for mk in b.get("markets", []):
            if not mk["key"].startswith("player_shots_on_goal"):
                continue
            for o in mk.get("outcomes", []):
                if o.get("description") is None or o.get("point") is None:
                    continue
                out.append({"game": game, "commence": ev.get("commence_time"),
                            "book": b["key"], "market": mk["key"],
                            "player": o.get("description"), "pn": norm(o.get("description")),
                            "line": o.get("point"), "side": str(o.get("name", "")).lower(),
                            "price": o.get("price"),
                            "bettable": b["key"] in BETTABLE_BOOKS})
    return out


# -- build -------------------------------------------------------------------
def main():
    if not KEY:
        print("no ODDS_API_KEY -- cannot price the board")
        sys.exit(0)
    today = dt.date.today()
    yday = (today - dt.timedelta(days=1)).isoformat()
    sid = season_id(today)
    prev = "%d%d" % (int(sid[:4]) - 1, int(sid[4:]) - 1)
    print("[cron] season %s (anchor %s); features through %s" % (sid, prev, yday))

    cur = skater_state(sid, yday)
    anchor = skater_state(prev, "%d-08-01" % int(sid[:4]))
    team_of = rosters(sid)
    st = rates(cur, anchor, team_of)
    print("[cron] %d skaters (%d with games this season, %d anchored, %d on rosters)"
          % (len(st), len(cur), len(anchor), len(team_of)))

    # ambiguous normalized names (VAN dressed two Elias Petterssons) -> cannot attribute a price
    byname = {}
    for p in st.values():
        byname.setdefault((norm(p["name"]), p["team"]), []).append(p)
    ambiguous = set(k for k, v in byname.items() if len(v) > 1)
    if ambiguous:
        print("[cron] %d ambiguous name/team keys excluded: %s"
              % (len(ambiguous), sorted(n for n, _ in ambiguous)))

    events = slate(30)
    print("[cron] %d events in window" % len(events))
    imp, pr, sog = {}, {}, []
    for e in events:
        imp.update(game_lines(e))
        for k, v in props(e).items():
            pr[k] = v
        sog.extend(sog_props(e))
    print("[cron] %d teams priced, %d skaters quoted, %d SOG quotes (credits %s)"
          % (len(imp), len(pr), len(sog), _odds.rem))

    # SOG line capture -- written even when empty, so a run that returned nothing is visible in
    # git rather than looking like the job never ran.
    try:
        with open(SOG, "w") as f:
            json.dump({"meta": {
                "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ"),
                "n_events": len(events), "n_quotes": len(sog),
                "books": sorted({q["book"] for q in sog}),
                "credits_left": _odds.rem,
                "note": "line capture only -- no model, no flag, no claimed edge"},
                "quotes": sog}, f, indent=1)
        print("[cron] wrote %s: %d quotes across %d books"
              % (SOG, len(sog), len({q["book"] for q in sog})))
    except Exception as e:
        print("[cron] SOG write failed (non-fatal): %s" % e)

    # role-share denominator = the team's projected top-18 by exp_goals, INDEPENDENT of which
    # players a book happened to post (books post partial rosters; dividing by the posted subset
    # inflates every share on thin teams and manufactures fake edge -- caught in research).
    # Early season nobody clears MIN_GP, so the gate is applied only once a team actually has
    # players who clear it; otherwise the roster carries the denominator on prior-season rates.
    byteam = {}
    for p in st.values():
        if p["team"]:
            byteam.setdefault(p["team"], []).append(p)
    for t, v in list(byteam.items()):
        seasoned = [p for p in v if p["gp"] >= MIN_GP]
        if len(seasoned) >= DRESSED:
            byteam[t] = seasoned
    denom = {}
    for t, v in byteam.items():
        top = sorted(v, key=lambda z: -z["exp_goals"])[:DRESSED]
        denom[t] = sum(x["exp_goals"] for x in top)

    rows, unmatched = [], []
    for key, m in pr.items():
        cands = [p for p in st.values()
                 if norm(p["name"]) == key and p["team"] in imp]
        if len(cands) != 1:
            unmatched.append(m["player"])
            continue
        p = cands[0]
        if (norm(p["name"]), p["team"]) in ambiguous:
            unmatched.append(m["player"] + " (ambiguous name)")
            continue
        ti = imp[p["team"]]
        dn = denom.get(p["team"])
        if not dn:
            unmatched.append(m["player"] + " (no team prior)")
            continue
        share = p["exp_goals"] / dn
        model_p = calibrate(1 - math.exp(-(share * ti["implied"])))
        prices = [(x, bk) for x, bk in m["q"] if x is not None]
        if not prices:
            continue
        best_price, best_book = max(prices, key=lambda t: t[0])
        worst = min(x for x, _ in prices)
        probs = sorted(a2p(x) for x, _ in prices)
        n = len(probs)
        mkt_p = probs[n // 2] if n % 2 else (probs[n // 2 - 1] + probs[n // 2]) / 2
        # per-book prices, so the app can filter to ONE book and judge where a profit boost is
        # best spent: a boost is worth most where that book is generous RELATIVE to the market,
        # not where the overall best price happens to live (which may be a different book).
        book_prices = sorted(
            [{"book": bk, "price": x, "best": x == best_price,
              "bettable": bk in BETTABLE_BOOKS} for x, bk in prices],
            key=lambda d: -d["price"])
        rows.append({
            "book_prices": book_prices,
            "player": p["name"], "pos": p["pos"], "team": p["team"], "opp": ti["opp"],
            "game": ti["game"], "commence": ti["commence"],
            "gp": p["gp"], "pp_min": p["pp_min"],
            "implied_total": round(ti["implied"], 2), "tt_source": ti["src"],
            "model_p": round(model_p, 4), "mkt_p": round(mkt_p, 4),
            "edge": round(model_p - mkt_p, 4),
            "best_price": best_price, "best_book": best_book,
            "worst_price": worst, "n_books": len(prices),
            "shop_gain": round(a2p(worst) - a2p(best_price), 4),
            "soft_best": best_book in SOFT_BOOKS,
            "thin": p["gp"] < 10,
        })
    rows.sort(key=lambda r: -r["model_p"])
    payload = {
        "meta": {
            "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ"),
            "season": sid, "features_through": yday,
            "n_rows": len(rows), "n_teams": len(imp), "n_games": len(events),
            "unmatched": unmatched[:30], "credits_left": _odds.rem,
        },
        "rows": rows,
    }
    with open(BOARD, "w") as f:
        json.dump(payload, f, indent=1)
    print("[cron] wrote %s: %d skaters, %d unmatched" % (BOARD, len(rows), len(unmatched)))
    for r in rows[:10]:
        print("   %-24s %s v%s  pp%4.1fm  model %.3f  mkt %.3f  %+5d %s"
              % (r["player"][:22], r["team"], r["opp"], r["pp_min"],
                 r["model_p"], r["mkt_p"], r["best_price"], r["best_book"]))


if __name__ == "__main__":
    main()
