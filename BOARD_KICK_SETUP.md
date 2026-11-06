# Board self-heal — one-time setup

The code (`board_kick.py` + four hooks in `app.py`, commit `a2513dd`) is already on main and
**inert until the steps below are done**. Two websites, about 10 minutes.

## Why this exists

On 2026-10-05 the HR board sat 12 hours stale straight through a 5:00p first pitch. GitHub
silently *dropped* `hr_board.yml`'s 14:00 and 15:00 UTC crons — not late, never ran — so the
09:25Z stamp stood until 21:26Z, 26 minutes after CWS@CLE started. That game's HR props had
last been priced 11.5h out, before lineups posted.

`board_watchdog.yml` exists to catch exactly that and didn't, because it is a GitHub scheduled
job policing other GitHub scheduled jobs. Set to 12 checks a day, it ran 2. **When GitHub drops
runs, the detector goes down with the patient** — nothing written inside Actions can fix a
failure mode of "the run does not happen." So the trigger moves off that scheduler:

- **A — the app itself.** Open a board, and if that artifact is stale and the sport has games,
  the app asks GitHub to rebuild it. Fires exactly when you're looking, which is when
  freshness matters.
- **B — a free external cron.** Hits `/api/kick_boards` every 20 min, so boards also heal
  when nobody is looking.

GitHub Actions stays the primary schedule. This is the backstop for when it doesn't run.

---

## Part 1 — Make the GitHub token

A password that lets the app tell GitHub "rebuild the board now."

1. Go to <https://github.com/settings/personal-access-tokens>
2. Click the green **Generate new token** (top right).
3. Fill in:
   - **Token name:** `ozzie-board-kick`
   - **Expiration:** **1 year** (see *If the token expires* below)
   - **Resource owner:** your own account (`zslater2-collab`)
   - **Repository access:** **Only select repositories** → pick **Ozzie**
4. Scroll to **Permissions** → expand **Repository permissions**.
5. Find **Actions** → change its dropdown from "No access" to **Read and write**.
   That is the only permission needed. GitHub auto-adds "Metadata: Read-only" by itself —
   normal, leave it.
6. Bottom of the page → **Generate token**.
7. Copy the `github_pat_...` string **now**. GitHub shows it once and never again.

## Part 2 — Put it into Render

1. <https://dashboard.render.com> → click the Ozzie web service.
2. **Grab two things off this page first:**
   - The app URL at the top, like `https://ozzie-xxxx.onrender.com` — needed in Part 3.
   - Left sidebar → **Environment** → find `NOTIFY_SECRET`, click the reveal icon, copy the
     value — also needed in Part 3.
3. Still on **Environment** → **Add Environment Variable**
   - **Key:** `GH_DISPATCH_TOKEN`
   - **Value:** the `github_pat_...` token from Part 1
4. **Save, rebuild, and deploy.** Wait for green, ~2-3 min.

**Trigger A is live at the end of Part 2.** That alone would have fixed 2026-10-05.

## Part 3 — The external cron (heals when you are *not* looking)

1. <https://cron-job.org> → **Sign up** (free, email + password), confirm the email.
2. **Create cronjob.**
3. **Title:** `Ozzie board refresh`
4. **URL** — fill in your two values from Part 2 step 2, keep `?secret=` exactly as written:
   ```
   https://YOUR-APP.onrender.com/api/kick_boards?secret=YOUR-NOTIFY-SECRET
   ```
5. **Schedule:** **Every 10 minutes** (preset, or Custom with minutes `*/10`).

   > **Not 20.** A Render free web service spins down after ~15 minutes with no traffic, and a
   > cold start was measured at **32.5 seconds** (0.09s once warm). A 20-minute interval is
   > therefore *longer than the sleep timer*, so every single call arrives at a sleeping instance
   > and waits on a full boot — past cron-job.org's 30s request timeout, which it reports as
   > **503 Service Unavailable** and emails you about. 10 minutes keeps the instance warm, so this
   > job and the `/api/notify` one both land on a live service. The ping is free: `/api/kick_boards`
   > only reads artifacts and calls the Odds API's `/events`, which costs zero credits.
   >
   > The trade-off: an always-warm service uses roughly 730 of Render free's 750 instance-hours a
   > month. That is fine for one service and will NOT fit two, so if you add another free web
   > service, move this back out and accept the cold starts instead.
6. **Create.**
7. Click into the job → **Test run**. You want **200**. A **401** means the secret in the URL
   doesn't match `NOTIFY_SECRET` — recheck step 4.

## Part 4 — Confirm it actually dispatches

Open `https://YOUR-APP.onrender.com/api/kick_boards?secret=YOUR-NOTIFY-SECRET` in a browser.
The JSON tells you the state of all four boards:

- `"configured": true` — the token is wired up correctly. **This is the thing to check.**
- `"kicked": [...]` — it just asked GitHub to rebuild those. Confirm with
  `gh run list --limit 5`: you should see a run with event `workflow_dispatch`.
- `"skipped": [{"why": "GH_DISPATCH_TOKEN unset"}]` — Part 2 didn't take.
- `"skipped": [{"why": "cooling down (<1800s)"}]` — normal, it rebuilt within the last 30 min.
- `"skipped": [{"why": "idle (no upcoming events)"}]` — normal out of season.
- All four `"status": "ok"` and empty `kicked`/`skipped` — normal and correct; nothing is
  stale, so there is nothing to do. **This cannot prove a real dispatch works.** To see one,
  wait until a board is genuinely stale, or temporarily lower that board's `max_age_h` in
  `scripts/board_health.py`.

---

## Things to know

**Cost.** Each rebuild spends Odds API credits. A 30-minute per-board cooldown is baked in
(`COOLDOWN_S`, shared via Upstash so it holds across Render instances), so no matter how often
the cron or your page loads fire, a board cannot rebuild more than twice an hour. The quota is
100,000 credits/**month** and all four boards together burn on the order of a few hundred a
day, so this is not cost-bound.

**If the token expires** (1 year out), the self-heal quietly stops and you are back to relying
on GitHub's crons — no error, no alarm, the exact silent failure this was built to fix. It is
*reported* rather than ignored: `/api/kick_boards` will say `GH_DISPATCH_TOKEN unset` for any
stale board. Worth re-reading that endpoint next October. "No expiration" avoids the cliff —
slightly worse if the token ever leaks, but it can only rebuild a board on one repo.

**Open question, not yet decided:** MLB HR's `max_age_h` is 10h, which is loose for an
afternoon first pitch — a 5:25am stamp only trips at 3:25pm ET. Enough to have rescued
2026-10-05, but tightening it buys more margin at the cost of more rebuilds.

## Part 5 — Seasonal: which cron jobs to pause

`/api/notify` is **MLB only** — it reads lineups and starters and flags K-props. Through the MLB
offseason it has nothing to do, so leaving it scheduled just wakes the service to no purpose and
turns every cold start into a "Failed (503)" email. Observed 2026-11-06 18:51 GMT.

| cron-job.org job | endpoint | season | offseason action |
|---|---|---|---|
| Ozzie board refresh | `/api/kick_boards` | all year — NHL, NFL and MMA run now | leave running, every 10 min |
| K-prop notify | `/api/notify` | MLB only (~late Mar to Oct) | **pause until spring** |

While it is paused the board refresh keeps the service warm on its own, so nothing else changes.

A 503 from these jobs is a cold start, not an app fault: the instance was asleep and the caller
gave up before boot finished. Confirm by requesting the URL twice — a slow first response followed
by a fast second one is a cold start, whereas a persistent 5xx is a real outage, and a 401 is a
secret mismatch. Raising the job's timeout to 30s and enabling its retry setting helps at the
margin but cannot cover a 32s boot; keeping the service warm is what actually fixes it.
