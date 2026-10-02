"""
UFC / MMA + Boxing moneyline board -- cloud cron (GitHub Actions, inside the Ozzie repo).

Writes nhl-style artifact:  mma_board_latest.json   (the app's /api/mma_board reads this)

WHAT THIS IS, EXACTLY
---------------------
A LINE-SHOPPING board. Moneyline only, no model, no fight predictions, no claimed edge --
Zach asked for exactly this ("I'd only be interested in Moneyline prices... value to be found
in line shopping... enjoyable to bet even without any type of real edge"). Nothing here tries
to forecast a fight.

WHY MONEYLINE IS A BETTER SHOPPING TARGET THAN THE PROP BOARDS
-------------------------------------------------------------
It is a TWO-WAY market, so both sides are posted. That buys two things the anytime-scorer
boards cannot have (they only ever get the vig-inflated "Yes" side):
  1. a real NO-VIG fair probability, de-vigged per book then medianed across books
  2. a real HOLD measurement -- per book, and for the best-of-all-books pair, which is where
     the shopping value shows up as a number
Measured on the 2026-10-04 card: single-book hold ran +4.2% to +5.9%, while the best-of-4
combined hold fell to +2.9% to +3.7%. Shopping roughly HALVES the vig. That is the product.

HONEST LABELLING
----------------
`ev_best` is fair-from-the-books vs the best book's price. It therefore measures how much
better the best price is than the market consensus -- i.e. the SHOPPING gain. It is NOT an
independent edge and must never be presented as one; we have no fight model at all.
A negative ev_best is normal and expected: it is the vig you still pay after shopping.

BOOK COVERAGE (audited 2026-10-02, 44 upcoming events)
------------------------------------------------------
MA-legal books posting MMA h2h: DraftKings (29 events), FanDuel (29), Caesars (18),
Bally Bet (17). BetMGM, theScore Bet and Fanatics post NOTHING for MMA in this feed -- so this
is a FOUR-book shop, not seven. Do not quietly imply otherwise on the board.

COST
----
h2h is a FEATURED market, so the whole-sport /odds endpoint returns every event in one call:
~1-2 credits per run for the entire board, versus the per-event calls the prop boards need.
"""
import os, sys, json, math, datetime as dt
import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD = os.path.join(REPO, "mma_board_latest.json")

KEY = os.environ.get("ODDS_API_KEY", "")
B = "https://api.the-odds-api.com/v4"
# Both combat sports share one board and one tab. Boxing is included because it is the
# WIDEST-disagreeing market measured (DK-vs-FanDuel median 1.47pp, max 4.94pp -- the largest of
# any sport scanned), so its cheap-hold screen is the most productive. But Zach says he would
# rarely bet it, so the app DEFAULTS to MMA and boxing is opt-in via the sport filter.
SPORTS = [("mma_mixed_martial_arts", "MMA"), ("boxing_boxing", "Boxing")]
# MA-legal only: the board exists to tell Zach where to actually place a bet.
MA_BOOKS = "draftkings,fanduel,betmgm,williamhill_us,espnbet,fanatics,ballybet"
BETTABLE = {"draftkings", "fanduel", "betmgm", "williamhill_us",
            "espnbet", "fanatics", "ballybet"}
REGIONS = "us,us2"
HORIZON_DAYS = 21          # fight weeks are what matter; far-future headliners are noise


def a2p(a):
    if a is None:
        return None
    a = float(a)
    return (-a) / (-a + 100.0) if a < 0 else 100.0 / (a + 100.0)


def profit(a):
    """Profit per 1 unit staked."""
    if a is None:
        return None
    a = float(a)
    return a / 100.0 if a > 0 else 100.0 / (-a)


def med(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def fetch_one(sport):
    try:
        r = requests.get("%s/sports/%s/odds" % (B, sport), timeout=40, params={
            "apiKey": KEY, "regions": REGIONS, "markets": "h2h",
            "oddsFormat": "american", "bookmakers": MA_BOOKS})
    except Exception as e:
        print("odds fetch failed: %s" % e)
        return [], "?"
    if r.status_code != 200:
        print("odds %s %s" % (r.status_code, r.text[:160]))
        return [], "?"
    return r.json(), r.headers.get("x-requests-remaining", "?")


def fetch():
    """One call per combat sport (h2h is featured, so each is ~1-2 credits for its whole slate)."""
    out, rem = [], "?"
    for key, label in SPORTS:
        evs, rem = fetch_one(key)
        for e in evs:
            e["_sport"] = label
        out += evs
        print("[cron]   %-26s %d events" % (key, len(evs)))
    return out, rem


def main():
    if not KEY:
        print("no ODDS_API_KEY -- cannot build the MMA board")
        sys.exit(0)
    events, rem = fetch()
    now = dt.datetime.now(dt.timezone.utc)
    cut = now + dt.timedelta(days=HORIZON_DAYS)
    print("[cron] %d combat-sport events returned (credits left %s)" % (len(events), rem))

    # card = all fights sharing a date. No promotion field exists in this feed, so we cannot
    # label "UFC" vs PFL/Bellator with any confidence -- we group by date and name the card
    # after its MAIN EVENT (the latest commence time on that date), which is how cards read.
    cards = {}
    for e in events:
        c = dt.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        if c < now or c > cut:
            continue
        cards.setdefault((e.get("_sport", "MMA"), c.date().isoformat()), []).append((c, e))

    rows = []
    for (sport, day), fights in sorted(cards.items()):
        main_ev = max(fights, key=lambda t: t[0])[1]
        card_name = "%s - %s vs %s" % (day, main_ev["away_team"], main_ev["home_team"])
        for c, e in sorted(fights, key=lambda t: t[0]):
            a, h = e["away_team"], e["home_team"]
            quotes = {a: [], h: []}
            per_book_fair = {a: [], h: []}
            for bk in e.get("bookmakers", []):
                o = {}
                for m in bk.get("markets", []):
                    if m.get("key") != "h2h":
                        continue
                    for x in m.get("outcomes", []):
                        if x.get("price") is not None:
                            o[x["name"]] = x["price"]
                if a in o:
                    quotes[a].append((o[a], bk["key"]))
                if h in o:
                    quotes[h].append((o[h], bk["key"]))
                # a book with BOTH sides gives a clean two-way de-vig
                if a in o and h in o:
                    pa, ph = a2p(o[a]), a2p(o[h])
                    tot = pa + ph
                    if tot > 0:
                        per_book_fair[a].append(pa / tot)
                        per_book_fair[h].append(ph / tot)
            if not quotes[a] or not quotes[h]:
                continue
            # combined hold at the BEST price on each side -- the shopping number. Below 0 = arb.
            best = {}
            for f in (a, h):
                p, bkk = max(quotes[f], key=lambda t: t[0])
                best[f] = (p, bkk)
            combined_hold = (a2p(best[a][0]) + a2p(best[h][0])) - 1.0
            for f, opp in ((a, h), (h, a)):
                qs = quotes[f]
                bp, bb = best[f]
                worst = min(x for x, _ in qs)
                fair = med(per_book_fair[f])
                ev = None
                if fair is not None:
                    ev = round(100 * (fair * profit(bp) - (1 - fair)), 2)
                rows.append({
                    "sport": sport,
                    "fighter": f, "opponent": opp, "card": card_name, "card_date": day,
                    "commence": e["commence_time"],
                    "is_main": e["id"] == main_ev["id"],
                    "best_price": bp, "best_book": bb,
                    "worst_price": worst, "n_books": len(qs),
                    "fair_p": round(fair, 4) if fair is not None else None,
                    "mkt_p": round(a2p(bp), 4),
                    "ev_best": ev,
                    "shop_gain": round(a2p(worst) - a2p(bp), 4),
                    # the hedge leg: best price available on the OTHER side, so the app can work
                    # out whether a boost on this side locks a profit against it
                    "hedge_price": best[opp][0], "hedge_book": best[opp][1],
                    "combined_hold": round(combined_hold, 4),
                    "low_hold": combined_hold < 0.02,
                    "arb": combined_hold < 0,
                    "book_prices": sorted(
                        [{"book": k, "price": p, "best": p == bp, "bettable": k in BETTABLE}
                         for p, k in qs], key=lambda d: -d["price"]),
                })
    rows.sort(key=lambda r: (r["commence"], -(r["fair_p"] or 0)))
    books = sorted({q["book"] for r in rows for q in r["book_prices"]})
    payload = {
        "meta": {
            "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ"),
            "n_rows": len(rows), "n_fights": len(rows) // 2,
            "n_cards": len(cards), "books": books, "credits_left": rem,
            "sports": sorted({r["sport"] for r in rows}),
            "note": ("moneyline line-shopping board; no fight model and no claimed edge. "
                     "ev_best = best book's price vs the books' own no-vig consensus, i.e. the "
                     "SHOPPING gain, not an independent edge."),
        },
        "rows": rows,
    }
    with open(BOARD, "w") as f:
        json.dump(payload, f, indent=1)
    bysport = {}
    for r in rows:
        bysport[r["sport"]] = bysport.get(r["sport"], 0) + 1
    print("[cron] wrote %s: %d fighters across %d fights, %d cards; books=%s; by sport=%s"
          % (BOARD, len(rows), len(rows) // 2, len(cards), books, bysport))
    arbs = [r for r in rows if r["arb"]]
    if arbs:
        print("[cron] !! %d sides in a negative-hold (arb) fight" % len(arbs))
    for r in rows[:12]:
        print("   %-22s vs %-20s %+6d %-14s fair %5.1f%%  EV %+6.2f%%  hold %+5.2f%%"
              % (r["fighter"][:22], r["opponent"][:20], r["best_price"], r["best_book"],
                 100 * (r["fair_p"] or 0), r["ev_best"] or 0, 100 * r["combined_hold"]))


if __name__ == "__main__":
    main()
