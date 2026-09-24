#!/usr/bin/env python3
"""
hr_2plus_ev.py -- 2+ HR (multi-HR) bet EV / boost calculator.

Answers one question: given a bat's board HR prob and a book's 2+HR price, what
profit-boost % do you need for the bet to be EV-positive?

The true 2+HR rate is estimated EMPIRICALLY from the graded ledger (multi_hr ~ hr_prob,
logistic fit) so it self-updates every time the cron regrades. Do NOT derive the tail
from the single-HR model prob -- that prob "runs hot" (calibration slope ~0.58) and the
2+HR event is far rarer than any Poisson-from-the-inflated-prob would imply. The ledger
is the ground truth.

Usage:
  python scripts/hr_2plus_ev.py --prob 24.7 --odds 3500
  python scripts/hr_2plus_ev.py --prob 24.7 --odds 3500 --boost 25
  python scripts/hr_2plus_ev.py --prob 18                       # just the fair 2+HR rate/odds
"""
import os, argparse
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRADED = os.path.join(REPO, 'hr_board_graded.csv')


def american_to_mult(a):
    """American odds -> profit multiple per $1 staked (win pays stake*mult)."""
    a = float(a)
    return a / 100.0 if a > 0 else 100.0 / abs(a)


def mult_to_american(m):
    return f'+{int(round(m*100))}' if m >= 1 else f'-{int(round(100/m))}'


def implied(a):
    a = float(a)
    return 100.0 / (a + 100.0) if a > 0 else abs(a) / (abs(a) + 100.0)


def fit_true_rate(graded_path=GRADED):
    """Logistic fit of multi_hr ~ hr_prob on the graded ledger. Returns a callable
    hr_prob(%) -> estimated per-pick 2+HR rate, plus the empirical band table for context."""
    g = pd.read_csv(graded_path)
    g = g[g['had_hr'].notna() & g['hr_prob'].notna()].copy()
    if 'multi_hr' not in g.columns:
        raise SystemExit("graded ledger has no multi_hr column -- run the cron grade() once to backfill it.")
    g['multi_hr'] = g['multi_hr'].fillna(0).astype(int)
    from sklearn.linear_model import LogisticRegression
    X = g[['hr_prob']].to_numpy(); y = g['multi_hr'].to_numpy()
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
    def rate(prob):
        return float(lr.predict_proba([[prob]])[0, 1])
    # empirical bands for a sanity column
    bands = []
    for lo, hi in [(0,10),(10,15),(15,20),(20,25),(25,100)]:
        s = g[(g['hr_prob'] >= lo) & (g['hr_prob'] < hi)]
        if len(s):
            bands.append((lo, hi, len(s), int(s['multi_hr'].sum()), 100*s['multi_hr'].mean()))
    return rate, bands, len(g), int(g['multi_hr'].sum())


def boost_needed(p, odds):
    """Min profit-boost fraction so EV>=0 at the given base American odds.
    EV=0  <=>  boosted_mult = (1-p)/p ; boosted_mult = raw_mult*(1+b)."""
    raw = american_to_mult(odds)
    fair = (1 - p) / p  # multiple that makes it a fair bet
    return max(0.0, fair / raw - 1.0)


def ev_roi(p, odds):
    m = american_to_mult(odds)
    return p * m - (1 - p)


def boosted_american(odds, boost_pct):
    return american_to_mult(odds) * (1 + boost_pct/100.0) * 100.0  # -> +American number


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prob', type=float, required=True, help="bat's board single-HR prob %% (hr_prob column)")
    ap.add_argument('--odds', type=float, help='book 2+HR price, American (e.g. 3500). omit for fair-value only')
    ap.add_argument('--boost', type=float, default=0.0, help='profit boost %% already applied (e.g. 25)')
    a = ap.parse_args()

    rate, bands, n, nmulti = fit_true_rate()
    p = rate(a.prob)
    fair_mult = (1 - p) / p

    print(f"\n  ledger: {n} graded picks, {nmulti} multi-HR events")
    print("  empirical 2+HR rate by hr_prob band:")
    for lo, hi, bn, bk, br in bands:
        print(f"    {lo:2d}-{hi:<3d}  n={bn:4d}  2+HR={bk:2d}  {br:4.2f}%")
    print(f"\n  bat hr_prob = {a.prob:.1f}%")
    print(f"  est true 2+HR rate = {100*p:.2f}%   (fair odds {mult_to_american(fair_mult)})")

    if a.odds is None:
        print()
        return

    # effective price after any boost already applied
    eff = boosted_american(a.odds, a.boost) if a.boost else a.odds
    print(f"\n  book price      : {mult_to_american(american_to_mult(a.odds))}  (implied {100*implied(a.odds):.2f}%)")
    if a.boost:
        print(f"  after +{a.boost:.0f}% boost: {mult_to_american(american_to_mult(eff))}  (implied {100*implied(eff):.2f}%)")
    print(f"  EV at that price: {100*ev_roi(p, eff):+.1f}% ROI")

    need = boost_needed(p, a.odds)   # boost vs the BASE price
    if need <= 0:
        print(f"\n  >> EV+ already at the base price (no boost needed).")
    else:
        need_odds = boosted_american(a.odds, need*100)
        print(f"\n  >> BOOST NEEDED for EV+ : +{100*need:.0f}%   (base {mult_to_american(american_to_mult(a.odds))} -> {mult_to_american(american_to_mult(need_odds))})")
        if a.boost:
            verdict = "clears it" if a.boost/100 >= need else "NOT enough"
            print(f"     your +{a.boost:.0f}% boost {verdict}.")
    print()


if __name__ == '__main__':
    main()
