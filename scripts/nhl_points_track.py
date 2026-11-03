"""NHL 1+ POINTS forward tracker -- the price panel the points model does not yet have.

WHY THIS EXISTS
  The 1+ points market is the one actually bet, and until 2026-11-02 its capture was written by
  the board cron and then silently dropped by the workflow, so NO price history exists. Every
  number in the 5-week study of this market is a BASE RATE, never an ROI: a selection that hits
  48% is worthless if the book charges 50%. This job accumulates the (price, projection, outcome)
  triples that turn base rates into a graded ROI. Nothing here is a signal yet; it is a recorder.

WHAT IT RECORDS
  nhl_points_log.csv    one row per (skater, game date) -- UPSERTED to the LATEST snapshot taken
                        strictly BEFORE puck drop, which is the price actually bettable. Per-run
                        price movement stays recoverable from nhl_points_latest.json's git
                        history, so the log keeps closing state rather than every poll.
  nhl_points_perf.json  graded performance by selection cell, plus the breakeven profit boost
                        at every leg count from 1 to 8.

LEAK DISCIPLINE (Ozzie Rule #1)
  * a row is logged only while pulled_at < commence; a started game is never recorded
  * selection flags (D rank, team implied, favourite) are FROZEN at capture time, so a cell can
    never be re-cut later using something that was not known pregame
  * outcomes are written by a separate pass, keyed to rows already on disk, into y_* columns that
    are never read back as inputs
"""
import os
import csv
import json
import time
import unicodedata
import datetime as dt
from collections import defaultdict

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD = os.path.join(REPO, "nhl_board_latest.json")
LOG = os.path.join(REPO, "nhl_points_log.csv")
PERF = os.path.join(REPO, "nhl_points_perf.json")

FIELDS = ["gameDate", "player", "pn", "pos", "team", "opp", "game", "commence", "pulled_at",
          "hours_before", "model_pt", "model_p", "pt_price", "pt_book", "pt_mkt_p", "pt_books",
          "pp_min", "implied_total", "game_total", "fav", "d_rank_pt", "d_rank_goal", "gp",
          "y_point", "y_goal", "graded_at"]


def norm(s):
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s if c.isalnum() or c == " ").strip()


def etdate(iso):
    """The game's EASTERN calendar date -- the key the NHL's own feeds use. A 02:10Z puck drop is
    the previous evening's game. Flat EST is safe for DATE bucketing: EST and EDT disagree only
    for starts in the 04:00-05:00Z hour, which the NHL does not schedule."""
    try:
        c = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except Exception:
        return None
    try:
        from zoneinfo import ZoneInfo
        return c.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    except Exception:
        return c.astimezone(dt.timezone(dt.timedelta(hours=-5))).date().isoformat()


def a2p(price):
    return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)


def payout(price):
    return price / 100.0 if price > 0 else 100.0 / -price


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def load_log():
    if not os.path.exists(LOG):
        return {}
    out = {}
    with open(LOG, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["pn"], r["gameDate"])] = r
    return out


def write_log(rows):
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(rows, key=lambda z: (rows[z]["gameDate"], rows[z]["player"])):
            w.writerow({c: rows[k].get(c, "") for c in FIELDS})


def capture(log):
    """Append tonight's pregame points quotes, with every selection flag frozen as of now."""
    board = json.load(open(BOARD, encoding="utf-8"))
    rows = board.get("rows") or []
    gen = (board.get("meta") or {}).get("generated")
    try:
        pulled = dt.datetime.strptime(gen, "%Y-%m-%d %H:%MZ").replace(tzinfo=dt.timezone.utc)
    except Exception:
        print("[pts] board has no usable `generated` stamp; nothing captured")
        return log, 0

    # game total + favourite from BOTH sides' posted team totals, frozen pregame
    sides = defaultdict(dict)
    for r in rows:
        if r.get("implied_total") is not None:
            sides[(r.get("game"), r.get("commence"))][r.get("team")] = r["implied_total"]
    gtot, fav = {}, {}
    for k, d in sides.items():
        if len(d) == 2:
            tot = sum(d.values())
            gtot[k] = round(tot, 2)
            for tm, v in d.items():
                fav[(k, tm)] = v > tot / 2.0

    # within-team defenseman ranks, by the points projection and by the goal projection, so the
    # graded report can say which flag actually selects better once prices are in evidence
    drk_pt, drk_goal = {}, {}
    byteam = defaultdict(list)
    for r in rows:
        if r.get("pos") == "D" and r.get("commence"):
            byteam[(etdate(r["commence"]), r.get("team"))].append(r)
    for v in byteam.values():
        for key, dest in (("model_pt", drk_pt), ("model_p", drk_goal)):
            ok = [x for x in v if x.get(key) is not None]
            ok.sort(key=lambda x: -x[key])
            for i, x in enumerate(ok):
                dest[(norm(x["player"]), etdate(x["commence"]))] = i + 1

    added = 0
    for r in rows:
        c = r.get("commence")
        if not c or r.get("model_pt") is None or r.get("pt_price") is None:
            continue
        try:
            cdt = dt.datetime.fromisoformat(c.replace("Z", "+00:00"))
        except Exception:
            continue
        if pulled >= cdt:
            continue                       # under way -> not bettable, never logged
        gd = etdate(c)
        pn = norm(r["player"])
        prev = log.get((pn, gd))
        if prev and prev.get("pulled_at", "") >= gen:
            continue                       # already hold an equal-or-later pregame snapshot
        key = (r.get("game"), c)
        log[(pn, gd)] = {
            "gameDate": gd, "player": r["player"], "pn": pn, "pos": r.get("pos"),
            "team": r.get("team"), "opp": r.get("opp"), "game": r.get("game"), "commence": c,
            "pulled_at": gen,
            "hours_before": round((cdt - pulled).total_seconds() / 3600.0, 2),
            "model_pt": r["model_pt"], "model_p": r.get("model_p"),
            "pt_price": r["pt_price"], "pt_book": r.get("pt_book"),
            "pt_mkt_p": r.get("pt_mkt_p"), "pt_books": r.get("pt_books"),
            "pp_min": r.get("pp_min"), "implied_total": r.get("implied_total"),
            "game_total": gtot.get(key), "fav": fav.get((key, r.get("team"))),
            "d_rank_pt": drk_pt.get((pn, gd)), "d_rank_goal": drk_goal.get((pn, gd)),
            "gp": r.get("gp"),
            "y_point": (prev or {}).get("y_point", ""),
            "y_goal": (prev or {}).get("y_goal", ""),
            "graded_at": (prev or {}).get("graded_at", ""),
        }
        added += 1
    print("[pts] captured %d pregame rows (log now %d)" % (added, len(log)))
    return log, added


def outcomes(d0, d1):
    """Per-skater per-game results for a closed date range. Free NHL stats REST, with backoff."""
    out, start = {}, 0
    cay = "gameDate>='%s' and gameDate<='%s' and gameTypeId=2" % (d0, d1)
    for _ in range(120):
        got = None
        for att in range(6):
            try:
                r = requests.get("https://api.nhle.com/stats/rest/en/skater/summary", timeout=60,
                                 params={"isAggregate": "false", "isGame": "true", "limit": 100,
                                         "start": start, "cayenneExp": cay})
                if r.status_code == 200:
                    got = (r.json() or {}).get("data", []) or []
                    break
                if r.status_code == 429:
                    time.sleep(5 * (att + 1))
                    continue
                print("[pts]   ! outcomes status %s" % r.status_code)
                return out
            except Exception as e:
                print("[pts]   ! %s" % e)
                time.sleep(4 * (att + 1))
        if not got:
            return out
        for x in got:
            out[(norm(x.get("skaterFullName")), x.get("gameDate"))] = x
        start += 100
        if len(got) < 100:
            break
        time.sleep(0.8)
    return out


def grade(log):
    """Fill y_point / y_goal for rows whose game date has closed. Ungraded rows stay ungraded:
    a skater who never dressed voids the bet and must not be counted as a loss."""
    today = dt.date.today().isoformat()
    todo = sorted({r["gameDate"] for r in log.values()
                   if not str(r.get("y_point", "")).strip() and r["gameDate"] < today})
    if not todo:
        print("[pts] nothing to grade")
        return log, 0
    print("[pts] grading %d date(s): %s .. %s" % (len(todo), todo[0], todo[-1]))
    o = outcomes(todo[0], todo[-1])
    print("[pts] pulled %d player-game outcomes" % len(o))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    n = 0
    for r in log.values():
        if str(r.get("y_point", "")).strip() or r["gameDate"] >= today:
            continue
        x = o.get((r["pn"], r["gameDate"]))
        if x is None:
            continue
        r["y_point"] = 1 if (x.get("points") or 0) > 0 else 0
        r["y_goal"] = 1 if (x.get("goals") or 0) > 0 else 0
        r["graded_at"] = stamp
        n += 1
    print("[pts] graded %d rows" % n)
    return log, n


def need_boost(p, m, legs):
    """Breakeven PROFIT boost for `legs` independent legs, each hitting p with realised return
    multiplier m (= 1 + ROI). d = m/p pins the representative price to the measured result, so
    the hit rate and the payout can never be pooled apart from each other.

      parlay pays 1 + (D-1)(1+b) on a win, D = d**legs, P = p**legs
      breakeven b* = (1/P - 1)/(D - 1) - 1
    """
    if p <= 0.0 or m <= 0.0:
        return None
    d = m / p
    P, D = p ** legs, d ** legs
    if D <= 1.0 or P <= 0.0:
        return None
    return (1.0 / P - 1.0) / (D - 1.0) - 1.0


def report(log):
    graded = [r for r in log.values() if str(r.get("y_point", "")).strip()]
    out = {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ"),
        "n_logged": len(log), "n_graded": len(graded),
        "note": ("1+ points, best bettable book price, pregame only. Base rates for these cells "
                 "were already measured as stable over 2026-10-01..11-02; the open question "
                 "these rows answer is whether the PRICE gives any of it back. Treat every ROI "
                 "as provisional until a cell clears a few hundred graded legs."),
        "cells": {},
    }

    def cell(rows, label):
        rows = [r for r in rows if fnum(r.get("pt_price")) is not None]
        if len(rows) < 10:
            out["cells"][label] = {"n": len(rows), "status": "too thin to report"}
            return
        n = len(rows)
        wins = sum(int(r["y_point"]) for r in rows)
        ret = sum((payout(fnum(r["pt_price"])) if int(r["y_point"]) else -1.0) for r in rows)
        m = ret / n + 1.0
        p = wins / float(n)
        out["cells"][label] = {
            "n": n, "wins": wins, "hit": round(p, 4),
            "projected": round(sum(fnum(r["model_pt"]) for r in rows) / n, 4),
            "priced": round(sum(a2p(fnum(r["pt_price"])) for r in rows) / n, 4),
            "roi": round(100.0 * (m - 1.0), 2),
            "breakeven_boost_pct": {
                str(L): (None if need_boost(p, m, L) is None
                         else round(100.0 * need_boost(p, m, L), 1))
                for L in (1, 2, 3, 4, 5, 6, 7, 8)},
        }

    D = [r for r in graded if r["pos"] == "D"]
    r1 = lambda r: str(r.get("d_rank_pt")) == "1"
    r2 = lambda r: str(r.get("d_rank_pt")) in ("1", "2")
    cell(graded, "all logged skaters")
    cell([r for r in graded if r["pos"] != "D"], "forwards")
    cell(D, "defensemen, all")
    cell([r for r in D if r1(r)], "D: top-1 by points projection")
    cell([r for r in D if r2(r)], "D: top-2 by points projection")
    cell([r for r in D if str(r.get("d_rank_goal")) in ("1", "2")],
         "D: top-2 by GOAL projection (the old flag)")
    cell([r for r in D if r2(r) and str(r.get("fav")) == "True"],
         "D: top-2 points proj, on favourite")
    cell([r for r in D if r2(r) and (fnum(r.get("implied_total")) or 0) >= 3.4],
         "D: top-2 points proj, team implied >= 3.4")
    cell([r for r in D if r2(r) and str(r.get("fav")) == "True"
          and (fnum(r.get("game_total")) or 0) >= 6.5],
         "D: top-2 points proj, fav + total >= 6.5")

    json.dump(out, open(PERF, "w"), indent=1)
    print("[pts] wrote %s" % PERF)
    for k, v in out["cells"].items():
        if v.get("status"):
            print("   %-46s n=%4d  %s" % (k, v["n"], v["status"]))
        else:
            print("   %-46s n=%4d hit %.3f proj %.3f priced %.3f  ROI %+6.1f%%  5-leg %s%%"
                  % (k, v["n"], v["hit"], v["projected"], v["priced"], v["roi"],
                     v["breakeven_boost_pct"]["5"]))


def main():
    log = load_log()
    log, _ = capture(log)
    log, _ = grade(log)
    write_log(log)
    print("[pts] wrote %s (%d rows)" % (LOG, len(log)))
    report(log)


if __name__ == "__main__":
    main()
