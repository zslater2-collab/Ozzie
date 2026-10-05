"""
board_health.py -- watchdog for the board artifacts.

WHY THIS EXISTS
---------------
Two silent failures hit the boards in two days (2026-10-03/04), and in both the board simply
stopped updating while every number on it still looked perfectly reasonable:

  1. GitHub executes this repo's scheduled runs 3-6 HOURS late, so the Sunday NFL refresh was
     landing after the 1pm kickoffs, and td-board (then 3x/week) sat 3 days old.
  2. The four board bots raced each other's pushes; the loser built fine and then threw the
     artifact away on a rejected push.

Neither could be caught from inside a board workflow, because the failure is the run not
happening (or not landing) at all. So this runs OUTSIDE them, checks how old each artifact
actually is, and re-dispatches whatever is stale.

HOW AGE IS MEASURED
-------------------
Preferred: the pull timestamp inside the payload. Fallback: the git commit time of the file,
which is when the artifact last successfully landed -- deliberately NOT file mtime, which in CI
is just the checkout time and would make everything look fresh.

OUTPUT
------
board_health.json  -- per-board age/status, committed so the app can show a banner
stdout + GITHUB_OUTPUT: stale=<comma-separated workflow files> for the watchdog to dispatch.
`consecutive_stale` carries forward, so a board that re-dispatch does NOT fix shows a climbing
count instead of being retried forever in silence.
"""
import json, os, subprocess, sys, datetime as dt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEALTH = os.path.join(REPO, "board_health.json")

# max_age_h budgets the cron cadence PLUS ~6h of observed drift, so a late-but-working run does
# not trip the alarm. Tighten only if the drift situation improves.
BOARDS = [
    {"name": "MLB HR",   "file": "hr_board_latest.json",  "wf": "hr_board.yml",
     "stamp": ["odds_at", "generated_at"],          "max_age_h": 10,
     "sport": "baseball_mlb",            "idle_window_h": 30},
    {"name": "NFL TD",   "file": "td_board_latest.json",  "wf": "td_board.yml",
     "stamp": ["meta.generated"],                   "max_age_h": 22,
     "sport": "americanfootball_nfl",     "idle_window_h": 120},
    {"name": "NHL goals", "file": "nhl_board_latest.json", "wf": "nhl_board.yml",
     "stamp": ["meta.generated"],                   "max_age_h": 16,
     "sport": "icehockey_nhl",            "idle_window_h": 30},
    {"name": "Combat",   "file": "mma_board_latest.json", "wf": "mma_board.yml",
     "stamp": ["meta.generated"],                   "max_age_h": 16,
     "sport": "mma_mixed_martial_arts",   "idle_window_h": 72},
]


def has_upcoming(sport, window_h):
    """Does this sport have an event inside its window? The /events endpoint costs ZERO credits
    (verified), so this is free season-awareness.

    Without it, every offseason looks like an outage: the board legitimately has nothing to do,
    its artifact stops changing, and the watchdog re-dispatches the cron every 2h for months and
    escalates forever -- the fastest way to make a watchdog worth ignoring. Returns None if the
    check could not be made, so an API problem never silences a genuine staleness alarm."""
    key = os.environ.get("ODDS_API_KEY")
    if not key or not sport:
        return None
    try:
        import urllib.request
        url = "https://api.the-odds-api.com/v4/sports/%s/events?apiKey=%s" % (sport, key)
        with urllib.request.urlopen(url, timeout=25) as r:
            evs = json.loads(r.read().decode())
    except Exception as e:
        print("  (events check failed for %s: %s)" % (sport, str(e)[:70]))
        return None
    if not isinstance(evs, list):
        return None
    cut = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=window_h)
    now = dt.datetime.now(dt.timezone.utc)
    for e in evs:
        try:
            c = dt.datetime.fromisoformat(str(e.get("commence_time")).replace("Z", "+00:00"))
        except Exception:
            continue
        if now - dt.timedelta(hours=6) < c < cut:
            return True
    return False


def dig(obj, path):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def parse_stamp(s):
    """Boards stamp time in several shapes: '2026-10-03 14:05Z', '2026-10-04T15:51' (naive,
    treated as UTC), '2026-10-02T19:54Z'. Accept all of them."""
    if not s:
        return None
    s = str(s).strip().replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        d = dt.datetime.fromisoformat(s)
    except Exception:
        return None
    return d.replace(tzinfo=dt.timezone.utc) if d.tzinfo is None else d


def git_commit_time(path):
    """When the artifact last LANDED. Not mtime -- in CI that is just the checkout time."""
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cI", "--", path],
                             cwd=REPO, capture_output=True, text=True, timeout=30)
        return parse_stamp((out.stdout or "").strip())
    except Exception:
        return None


def main():
    prev = {}
    if os.path.exists(HEALTH):
        try:
            prev = {b["file"]: b for b in json.load(open(HEALTH)).get("boards", [])}
        except Exception:
            prev = {}

    now = dt.datetime.now(dt.timezone.utc)
    out, stale_wfs = [], []
    for b in BOARDS:
        path = os.path.join(REPO, b["file"])
        rec = {"name": b["name"], "file": b["file"], "workflow": b["wf"],
               "max_age_h": b["max_age_h"]}
        if not os.path.exists(path):
            rec.update(status="missing", age_h=None, source=None, rows=0)
            stale_wfs.append(b["wf"])
        else:
            try:
                js = json.load(open(path))
            except Exception as e:
                js = None
                rec["parse_error"] = str(e)[:120]
            ts, src = None, None
            if js is not None:
                for key in b["stamp"]:
                    ts = parse_stamp(dig(js, key))
                    if ts:
                        src = key
                        break
            if ts is None:
                ts, src = git_commit_time(b["file"]), "git-commit"
            rows = len(((js or {}).get("rows")) or []) if js is not None else 0
            age = (now - ts).total_seconds() / 3600.0 if ts else None
            rec.update(age_h=(round(age, 2) if age is not None else None),
                       source=src, rows=rows,
                       stamp=(ts.strftime("%Y-%m-%d %H:%MZ") if ts else None))
            if js is None:
                rec["status"] = "unreadable"
                stale_wfs.append(b["wf"])
            elif age is None:
                rec["status"] = "unknown"
            elif age > b["max_age_h"]:
                # Season-aware: no games in the window means the board is idle by design, not
                # broken. Only a board that is stale WITH games to price gets re-dispatched.
                up = has_upcoming(b.get("sport"), b.get("idle_window_h", 36))
                if up is False:
                    rec["status"] = "idle"
                    rec["note"] = "no upcoming events -- offseason/empty slate, not an outage"
                else:
                    rec["status"] = "stale"
                    if up is None:
                        rec["note"] = "events check unavailable; treating as stale"
                    stale_wfs.append(b["wf"])
            else:
                rec["status"] = "ok"
        p = prev.get(b["file"], {})
        rec["consecutive_stale"] = (p.get("consecutive_stale", 0) + 1) \
            if rec["status"] != "ok" else 0
        if rec["consecutive_stale"] >= 3:
            # re-dispatch is not fixing it -> say so loudly rather than retrying in silence
            rec["escalate"] = True
        out.append(rec)

    payload = {"checked_at": now.strftime("%Y-%m-%d %H:%MZ"),
               "n_stale": sum(1 for r in out if r["status"] not in ("ok", "idle")),
               "boards": out}
    with open(HEALTH, "w") as f:
        json.dump(payload, f, indent=1)

    for r in out:
        flag = {"ok": "OK  ", "idle": "IDLE"}.get(r["status"], "FAIL")
        age = ("%.1fh" % r["age_h"]) if r["age_h"] is not None else "?"
        print("%s %-10s age=%-7s limit=%sh rows=%s via=%s%s"
              % (flag, r["name"], age, r["max_age_h"], r["rows"], r["source"],
                 ("  consecutive=%d" % r["consecutive_stale"]) if r["consecutive_stale"] else ""))

    uniq = sorted(set(stale_wfs))
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as f:
            f.write("stale=%s\n" % ",".join(uniq))
            f.write("n_stale=%d\n" % len(uniq))
            f.write("escalate=%s\n" % ("1" if any(r.get("escalate") for r in out) else ""))
    print("\nstale workflows to re-dispatch: %s" % (", ".join(uniq) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
