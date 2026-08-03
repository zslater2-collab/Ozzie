"""
weather_veto.py -- STAY-AWAY filter on K-prop OVERS when iffy weather is set to hit early.

Not a bet, a veto. Mechanism (validated in ozzie_data/06_tests/weather_kprop_study.py on 2025-26
actual precip): meaningful rain in innings ~1-5 at an OPEN-AIR park pulls the starter early ->
fewer batters faced -> OVER busts. Wet-early open-air overs hit 34% / -38% ROI vs 51% / -3% dry;
broad across 12+ parks, both years. Roofed parks are immune.

Live version uses the Open-Meteo FORECAST api (free, no key) at pick time -- point-in-time honest,
no leakage. Returns a flag + the numbers so thresholds stay tunable.

  from weather_veto import weather_veto
  v = weather_veto("PHI", commence_utc)   # commence_utc: tz-aware UTC datetime
  if v["veto"]: ...skip / de-emphasize the over...
"""
import time
import pandas as pd
import requests

# thresholds from the study's payoff bucket (early-window rain). Tunable.
WINDOW_H = 3          # innings ~1-5
PRECIP_MM = 2.0       # expected total precip in window
POP_PCT = 60          # max hourly rain probability in window


def _load_local_parks():
    """CLI/standalone fallback only. In the app, pass parks= from the loaded model (the app
    runs on Render where no local pickle exists)."""
    try:
        return pd.read_pickle(r"C:\Users\zslat\Documents\Ozzie\data\all_parks.pkl")
    except Exception:
        return {}


def weather_veto(park_abbr, commence_utc, parks=None, window_h=WINDOW_H,
                 precip_mm=PRECIP_MM, pop_pct=POP_PCT, timeout=20):
    """park_abbr like 'PHI'; commence_utc tz-aware (or ISO 'Z') UTC. parks = the all_parks dict
    (each val has lat/lon/roof); falls back to the local pickle for CLI use.
    Fail-open (veto=False) on any error -- weather must never break the pick pipeline."""
    if parks is None:
        parks = _load_local_parks()
    roofed = park_abbr in parks and parks[park_abbr].get("roof")
    out = {"veto": False, "roofed": bool(roofed), "park": park_abbr,
           "precip_mm": None, "pop_max": None, "reason": ""}
    if roofed:
        out["reason"] = "roofed_immune"
        return out
    if park_abbr not in parks:
        out["reason"] = "unknown_park"
        return out
    try:
        c = pd.Timestamp(commence_utc)
        c = c.tz_localize("UTC") if c.tzinfo is None else c.tz_convert("UTC")
        p = parks[park_abbr]
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": p["lat"], "longitude": p["lon"],
            "hourly": "precipitation,precipitation_probability",
            "forecast_days": 3, "timezone": "UTC"}, timeout=timeout)
        r.raise_for_status()
        h = r.json()["hourly"]
        w = pd.DataFrame(h)
        w["t"] = pd.to_datetime(w.time, utc=True)
        win = w[(w.t >= c) & (w.t < c + pd.Timedelta(hours=window_h))]
        if win.empty:
            out["reason"] = "forecast_window_missing"
            return out
        pm = float(win.precipitation.sum())
        pop = float(win.precipitation_probability.max())
        out["precip_mm"], out["pop_max"] = round(pm, 2), pop
        if pm >= precip_mm and pop >= pop_pct:
            out["veto"] = True
            out["reason"] = f"early_rain precip={pm:.1f}mm pop={pop:.0f}%"
        else:
            out["reason"] = f"clear enough precip={pm:.1f}mm pop={pop:.0f}%"
        return out
    except Exception as e:
        out["reason"] = f"error_failopen:{e}"
        return out


_CACHE = {}          # (park, first_pitch_iso) -> (result, fetched_at)
_TTL_S = 3 * 3600     # forecasts update slowly; one pull per game per few hours is plenty


def weather_veto_cached(park_abbr, commence_utc, parks=None, ttl_s=_TTL_S, **kw):
    """Same as weather_veto but memoized per (park, start) so the app's frequent rebuilds
    hit Open-Meteo once per game, not once per pitcher per cycle. Safe to call in a hot loop."""
    key = (park_abbr, str(commence_utc))
    hit = _CACHE.get(key)
    if hit is not None and (time.time() - hit[1]) < ttl_s:
        return hit[0]
    res = weather_veto(park_abbr, commence_utc, parks=parks, **kw)
    # don't cache transient errors (let the next cycle retry); do cache clean/veto answers
    if not res["reason"].startswith("error_failopen") and res["reason"] != "forecast_window_missing":
        _CACHE[key] = (res, time.time())
    return res


if __name__ == "__main__":
    import datetime as dt
    parks = _load_local_parks()
    # demo: same window at every open-air park ~today, so Zach can see what it flags live
    c = pd.Timestamp(dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0),
                     tz="UTC") + pd.Timedelta(hours=6)
    print(f"demo window start (UTC): {c}")
    rows = []
    for ab in sorted(parks):
        v = weather_veto(ab, c, parks=parks)
        rows.append(v)
    d = pd.DataFrame(rows)
    print(d[["park", "roofed", "precip_mm", "pop_max", "veto", "reason"]].to_string(index=False))
    print(f"\nwould VETO overs at: {sorted(d[d.veto].park)}")
