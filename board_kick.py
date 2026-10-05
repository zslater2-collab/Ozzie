"""Re-dispatch a stale board workflow from OUTSIDE GitHub's scheduler.

WHY THIS EXISTS
---------------
board_watchdog.yml already notices stale boards and re-dispatches them. It cannot be trusted
alone, because it is itself a GitHub scheduled job sitting in the same starved queue as the
boards it polices. On 2026-10-05 GitHub silently dropped the HR board's 14:00 and 15:00 UTC
crons AND 10 of the watchdog's own 12 daily runs: the board sat at its 09:25Z stamp until
21:26Z, so CWS@CLE (5:00p ET) was last priced 11.5h before first pitch and the next refresh
landed 26 minutes after it started. The watchdog's one run in that window (15:18Z) saw age
5.89h, under the 10h limit, called it ok, and never ran again. When GitHub drops runs, the
detector goes down with the patient.

So the trigger moves off GitHub's scheduler entirely. Two callers, one engine:

  A. app.py board routes call kick_stale_boards() in a daemon thread on page load. Fires
     exactly when Zach is looking at a board, which is exactly when freshness matters.
  B. /api/kick_boards, hit by a free external cron (cron-job.org), so it also self-heals
     when nobody is looking.

GitHub Actions stays as the primary schedule; this is the backstop for when it doesn't run.

SETUP (Render env vars)
-----------------------
  GH_DISPATCH_TOKEN  fine-grained PAT, repo zslater2-collab/Ozzie, Actions: read+write. This
                     is the ONLY thing that makes the module live; without it every stale
                     board is reported and skipped with "GH_DISPATCH_TOKEN unset", which is
                     also the safe default if the token is ever revoked.
  KICK_SECRET        shared secret for /api/kick_boards (falls back to NOTIFY_SECRET).
  ODDS_API_KEY       already set; used ONLY for the free /events season check (0 credits).
Then point a free scheduler at  https://<host>/api/kick_boards?secret=<KICK_SECRET>  every
20-30 min. Credit cost is bounded by COOLDOWN_S, not by how often it is called.

SAFETY
------
  * Reuses scripts/board_health.py's BOARDS spec and season-awareness, so thresholds and the
    offseason rule can never drift between the watchdog and this module.
  * Season-aware: a board with no upcoming events is idle by design and is NEVER dispatched.
    An unavailable events check does NOT dispatch here (unlike the watchdog, which fail-safes
    to "stale" because its only cost is one extra run -- this path can fire on every page load,
    so it fail-CLOSES instead and lets the watchdog be the loud one).
  * The cooldown is written BEFORE the dispatch, never after. An exception between the two then
    costs one missed kick instead of a dispatch loop -- the /api/notify Telegram-spam lesson
    (a slow step between the action and the mark is how you re-fire forever).
  * Read-only on disk, no artifact writes: Render's filesystem is ephemeral and
    board_health.json belongs to the watchdog.
"""

import os
import sys
import json
import datetime as dt

import requests

BASE = os.path.dirname(os.path.abspath(__file__))

# Single source of truth for thresholds / sport keys / idle windows: the watchdog's own spec.
sys.path.insert(0, os.path.join(BASE, "scripts"))
try:
    from board_health import BOARDS, dig, parse_stamp, has_upcoming
    _SPEC_OK = True
except Exception as _e:                                    # pragma: no cover
    print("[KICK] cannot import board_health spec: %s" % _e)
    BOARDS, _SPEC_OK = [], False

GH_REPO    = os.environ.get("GH_REPO", "zslater2-collab/Ozzie")
GH_TOKEN   = os.environ.get("GH_DISPATCH_TOKEN", "")
GH_REF     = os.environ.get("GH_DISPATCH_REF", "main")
# One kick per workflow per 30 min. A board run takes ~1-6 min and Render needs a redeploy on
# top, so anything shorter just stacks duplicate runs and burns odds credits.
COOLDOWN_S = int(os.environ.get("BOARD_KICK_COOLDOWN_S", "1800"))


def _spec_for(file_name):
    for b in BOARDS:
        if b["file"] == file_name:
            return b
    return {}


def board_status(only_file=None):
    """Age + staleness for each board, read from the SAME stamps the watchdog uses.

    Deliberately no git-commit fallback (the watchdog's last resort): on Render every file's
    commit time is the deploy time, which would read as fresh forever and mask a dead board.
    A board with no readable stamp is reported unknown and never dispatched from here."""
    now, out = dt.datetime.now(dt.timezone.utc), []
    for b in BOARDS:
        if only_file and b["file"] != only_file:
            continue
        rec = {"name": b["name"], "file": b["file"], "workflow": b["wf"],
               "max_age_h": b["max_age_h"], "age_h": None, "stamp": None, "source": None}
        path = os.path.join(BASE, b["file"])
        if not os.path.exists(path):
            rec["status"] = "missing"
            out.append(rec)
            continue
        try:
            js = json.load(open(path))
        except Exception as e:
            rec.update(status="unreadable", error=str(e)[:120])
            out.append(rec)
            continue
        ts = None
        for key in b["stamp"]:
            ts = parse_stamp(dig(js, key))
            if ts:
                rec["source"] = key
                break
        if ts is None:
            rec["status"] = "unknown"
        else:
            age = (now - ts).total_seconds() / 3600.0
            rec["age_h"] = round(age, 2)
            rec["stamp"] = ts.strftime("%Y-%m-%d %H:%MZ")
            rec["status"] = "stale" if age > b["max_age_h"] else "ok"
        rec["rows"] = len((js.get("rows") or [])) if isinstance(js, dict) else 0
        out.append(rec)
    return out


def _dispatch(wf):
    r = requests.post(
        "https://api.github.com/repos/%s/actions/workflows/%s/dispatches" % (GH_REPO, wf),
        headers={"Authorization": "Bearer %s" % GH_TOKEN,
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28"},
        json={"ref": GH_REF}, timeout=12)
    # 204 = accepted. Anything else is worth seeing in the Render log.
    if r.status_code != 204:
        print("[KICK] dispatch %s -> %s %s" % (wf, r.status_code, r.text[:200]))
    return r.status_code == 204


def kick_stale_boards(reason="", only_file=None, redis_get=None, redis_set=None,
                      dry_run=False):
    """Dispatch the workflow of every stale board that actually has games to price.

    only_file limits it to one board (the page-load caller passes the board being viewed, so
    looking at the HR board never fires the NHL cron). redis_get/redis_set are injected from
    app.py so the cooldown is shared across Render instances; without them there is no
    cooldown, so DO pass them in production."""
    res = {"reason": reason, "checked": [], "kicked": [], "skipped": [],
           "configured": bool(GH_TOKEN and _SPEC_OK)}
    if not _SPEC_OK:
        res["error"] = "board spec unavailable"
        return res

    for rec in board_status(only_file=only_file):
        res["checked"].append({k: rec.get(k) for k in ("name", "status", "age_h", "stamp")})
        if rec["status"] not in ("stale", "missing", "unreadable"):
            continue
        wf = rec["workflow"]
        spec = _spec_for(rec["file"])

        # Idle-by-design beats stale: no events in the window means nothing to refresh. Unlike
        # the watchdog this fail-CLOSES on an unavailable check -- see module docstring.
        up = has_upcoming(spec.get("sport"), spec.get("idle_window_h", 36))
        if up is not True:
            res["skipped"].append({"workflow": wf,
                                   "why": "idle (no upcoming events)" if up is False
                                          else "events check unavailable"})
            continue

        key = "ozzie:boardkick:%s" % wf
        if redis_get and redis_get(key):
            res["skipped"].append({"workflow": wf,
                                   "why": "cooling down (<%ss)" % COOLDOWN_S})
            continue
        if not GH_TOKEN:
            res["skipped"].append({"workflow": wf, "why": "GH_DISPATCH_TOKEN unset"})
            continue
        if dry_run:
            res["skipped"].append({"workflow": wf, "why": "dry_run",
                                   "age_h": rec["age_h"]})
            continue

        # Cooldown FIRST, then dispatch. Never the other way round.
        if redis_set:
            redis_set(key, dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
                      ex=COOLDOWN_S)
        ok = _dispatch(wf)
        if ok:
            res["kicked"].append({"workflow": wf, "age_h": rec["age_h"]})
        else:
            res["skipped"].append({"workflow": wf, "why": "dispatch failed (see log)"})
        print("[KICK] %s age=%sh status=%s reason=%s -> %s"
              % (wf, rec["age_h"], rec["status"], reason,
                 "dispatched" if ok else "FAILED"))
    return res


if __name__ == "__main__":
    # Local check: python board_kick.py         -> report only, dispatches nothing
    #              python board_kick.py --kick  -> really dispatch (needs GH_DISPATCH_TOKEN)
    live = "--kick" in sys.argv
    print(json.dumps(kick_stale_boards(reason="cli", dry_run=not live), indent=1))
