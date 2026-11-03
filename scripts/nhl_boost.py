"""Parlay / boost calculator for the NHL 1+ POINTS board.

Leg count is an INPUT, never an assumption: the breakeven boost rises steeply with legs, so the
same boost that rescues a 3-leg can be nowhere near enough for a 6-leg.

  stake 1, decimal D = prod(d_i), win prob P = prod(p_i)
  a PROFIT boost b pays  1 + (D-1)(1+b)  on a win
  EV(b)        = P * (1 + (D-1)(1+b)) - 1
  breakeven b* = (1/P - 1)/(D - 1) - 1

Two probabilities are reported for every basket and they mean different things:

  MARKET  each leg's price de-vigged by the measured 1+ points overround (3.43%/side, flat across
          every price band). This is the STRUCTURAL number -- it assumes no edge whatsoever, only
          that you pay the posted juice. Trust it.
  MODEL   each leg's model_pt from the board. The points model is calibrated on 5 weeks of
          outcomes (defensemen: predicted .2789 vs actual .2799) but has NO graded price history,
          so model EV is a PROJECTION, not a measured edge. See nhl_points_perf.json, which fills
          in as nhl_points_track.py accumulates graded legs.

Independence: legs are assumed independent, which holds while they are in DIFFERENT games. Two
skaters in the same game (or worse, on the same power-play unit) are strongly positively
correlated -- the parlay then wins more often than P suggests but pays the same, so P understates
it. The tool refuses to silently pretend otherwise and warns instead.

Usage
  python scripts/nhl_boost.py --top-d 5 --boost 30
  python scripts/nhl_boost.py --players "Quinn Hughes,Adam Fox,Erik Karlsson" --boost 50
  python scripts/nhl_boost.py --prices -250,-172,-210 --probs 0.686,0.628,0.624 --boost 25
  python scripts/nhl_boost.py --top-d 8 --grid
"""
import os
import json
import argparse
import unicodedata
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD = os.path.join(REPO, "nhl_board_latest.json")
OVERROUND = 0.0343          # measured 1+ points per-side vig; flat across price bands


def norm(s):
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s if c.isalnum() or c == " ").strip()


def a2p(price):
    return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)


def dec(price):
    return 1.0 + (price / 100.0 if price > 0 else 100.0 / -price)


def ev(P, D, b):
    return P * (1.0 + (D - 1.0) * (1.0 + b)) - 1.0


def breakeven(P, D):
    if D <= 1.0 or P <= 0.0:
        return None
    return (1.0 / P - 1.0) / (D - 1.0) - 1.0


def load_legs(args):
    """-> list of {name, price, p_model, p_mkt, game, team}"""
    if args.prices:
        prices = [float(x) for x in args.prices.split(",")]
        probs = [float(x) for x in args.probs.split(",")] if args.probs else [None] * len(prices)
        if len(probs) != len(prices):
            raise SystemExit("--probs must have the same count as --prices")
        return [{"name": "leg%d" % (i + 1), "price": pr, "p_model": pm,
                 "p_mkt": a2p(pr) / (1.0 + OVERROUND), "game": None, "team": None}
                for i, (pr, pm) in enumerate(zip(prices, probs))]

    board = json.load(open(BOARD, encoding="utf-8"))
    rows = [r for r in board.get("rows") or []
            if r.get("pt_price") is not None and r.get("model_pt") is not None]
    if not rows:
        raise SystemExit("board carries no 1+ points prices yet -- run nhl_board_cron.py first")

    def mk(r):
        return {"name": r["player"], "price": float(r["pt_price"]), "p_model": float(r["model_pt"]),
                "p_mkt": a2p(float(r["pt_price"])) / (1.0 + OVERROUND),
                "game": r.get("game"), "team": r.get("team")}

    if args.players:
        want = [norm(x) for x in args.players.split(",") if x.strip()]
        have = {norm(r["player"]): r for r in rows}
        legs, missing = [], []
        for w in want:
            if w in have:
                legs.append(mk(have[w]))
            else:
                missing.append(w)
        if missing:
            print("! not on tonight's board with a points price: %s" % ", ".join(missing))
        return legs

    n = args.top_d or args.top_f or 5
    pool = [r for r in rows if (r.get("pos") == "D") == bool(args.top_d)]
    # Price band matters more than it looks. A profit boost is worth b*(1-p) on a fairly priced
    # leg, so it pays far more on a long price than a short one: at -250 (p=.69) a 30% boost
    # recovers ~9pp, at +150 (p=.40) it recovers ~18pp. The top projected D are the SHORTEST
    # points prices, which is where a boost does the least work -- hence this filter.
    if args.min_price is not None:
        pool = [r for r in pool if float(r["pt_price"]) >= args.min_price]
    if args.max_price is not None:
        pool = [r for r in pool if float(r["pt_price"]) <= args.max_price]
    if not pool:
        raise SystemExit("no legs in that price band on tonight's board")
    # one per team: the within-team top projection, so a basket is not two partners on one unit
    best = {}
    for r in pool:
        t = r.get("team")
        if t not in best or r["model_pt"] > best[t]["model_pt"]:
            best[t] = r
    rank = sorted(best.values(), key=lambda r: -r["model_pt"])[:n]
    return [mk(r) for r in rank]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-d", type=int, help="auto-pick the N best defensemen (one per team)")
    ap.add_argument("--top-f", type=int, help="auto-pick the N best forwards (one per team)")
    ap.add_argument("--players", help="comma-separated names from tonight's board")
    ap.add_argument("--prices", help="comma-separated american prices, for a manual basket")
    ap.add_argument("--probs", help="comma-separated model probabilities to match --prices")
    ap.add_argument("--boost", type=float, default=0.0, help="profit boost in percent, e.g. 30")
    ap.add_argument("--grid", action="store_true", help="show the leg-count x boost grid")
    ap.add_argument("--min-price", type=float, help="only legs at or above this american price")
    ap.add_argument("--max-price", type=float, help="only legs at or below this american price")
    args = ap.parse_args()

    legs = load_legs(args)
    if not legs:
        raise SystemExit("no legs selected")
    b = args.boost / 100.0

    print("BASKET (%d legs, 1+ points)" % len(legs))
    print("  %-26s %8s %9s %9s  %s" % ("leg", "price", "model", "mkt(dev)", "game"))
    for L in legs:
        print("  %-26s %+8d %9s %9.3f  %s"
              % (L["name"][:24], int(L["price"]),
                 ("%.3f" % L["p_model"]) if L["p_model"] is not None else "-",
                 L["p_mkt"], L["game"] or ""))

    games = [L["game"] for L in legs if L["game"]]
    dup = {g for g in games if games.count(g) > 1}
    if dup:
        print("\n! %d leg(s) share a game (%s). Independence does NOT hold; the true win"
              % (sum(games.count(g) for g in dup), ", ".join(sorted(dup))))
        print("  probability is HIGHER than P below while the payout is unchanged, so every EV")
        print("  here is understated. Correlated legs are a real reason to prefer such a parlay,")
        print("  but this tool cannot size the effect -- treat the numbers as a floor.")

    D = 1.0
    for L in legs:
        D *= dec(L["price"])
    P_mkt = 1.0
    for L in legs:
        P_mkt *= L["p_mkt"]
    have_model = all(L["p_model"] is not None for L in legs)
    P_mod = 1.0
    if have_model:
        for L in legs:
            P_mod *= L["p_model"]

    print("\n  parlay decimal        %.2f  (%+d american)"
          % (D, round((D - 1.0) * 100) if D >= 2 else round(-100.0 / (D - 1.0))))
    print("  win prob (market)     %.4f   -> fair decimal %.2f" % (P_mkt, 1.0 / P_mkt))
    if have_model:
        print("  win prob (model)      %.4f   -> fair decimal %.2f" % (P_mod, 1.0 / P_mod))

    be_m = breakeven(P_mkt, D)
    print("\n  BREAKEVEN BOOST (market, no edge assumed):  %+.1f%%" % (100.0 * be_m))
    if have_model:
        be_p = breakeven(P_mod, D)
        print("  BREAKEVEN BOOST (model projection):         %+.1f%%" % (100.0 * be_p))
    if args.boost:
        print("\n  AT YOUR %.0f%% BOOST:" % args.boost)
        print("    EV (market, structural)  %+.2f%% of stake" % (100.0 * ev(P_mkt, D, b)))
        if have_model:
            print("    EV (model projection)    %+.2f%% of stake" % (100.0 * ev(P_mod, D, b)))
        verdict = "CLEARS" if args.boost / 100.0 > be_m else "does NOT clear"
        print("    your boost %s the market breakeven of %+.1f%%" % (verdict, 100.0 * be_m))

    if args.grid:
        print("\nBREAKEVEN BOOST by leg count (market probabilities, best legs first):")
        print("  %5s %10s %12s %12s" % ("legs", "decimal", "win prob", "breakeven"))
        D2, P2 = 1.0, 1.0
        for i, L in enumerate(legs, 1):
            D2 *= dec(L["price"])
            P2 *= L["p_mkt"]
            be = breakeven(P2, D2)
            print("  %5d %10.2f %12.4f %11s" % (i, D2, P2,
                                                "n/a" if be is None else "%+.1f%%" % (100.0 * be)))
        print("\n  Read it as: the longer the parlay, the bigger the boost it needs just to break")
        print("  even. With a fixed boost, fewer legs is the safer side of this trade.")


if __name__ == "__main__":
    main()
