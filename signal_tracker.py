"""
signal_tracker.py — Performance tracking add-on for VISH_SCAN

Bolts onto your EXISTING scanner. Does not touch your signal logic.

What it does:
  1. Records every qualifying signal (symbol, date, price, signal type)
  2. On every run, checks tracked stocks against milestone levels
  3. Fires a Telegram-ready alert the FIRST time a stock crosses
     +5%, +10%, +15%, +20%, +25%
  4. Warns once if a signal falls to -8%
  5. Closes each signal at 30 days with a final summary
  6. Maintains a running track record

State lives in signal_tracker_state.json (separate from your scanner_state.json
so nothing collides).
"""

import os
import json
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))
STATE_FILE = "signal_tracker_state.json"

# ============================================================
# SETTINGS — tune these
# ============================================================

# Which of your signal types are worth tracking.
# These strings MUST match your scanner's signal names exactly.
# Leave as None to track every signal your scanner emits.
TRACK_SIGNALS = {
    "Golden Cross",
    "Buy Now",
    "Aggressive Accumulation",
    "Accumulate",              # plain accumulate
    "Accumulate + Support",
    "Support Zone",
    # Not tracked by default — these are hold/early signals, not entries.
    # Uncomment either one if you want them tracked too.
    # "Wait Zone",
    # "Watchlist",
}

MILESTONES = [5, 10, 15, 20, 25]   # % gains that trigger an alert
DRAWDOWN_ALERT = -8                # % loss that triggers one warning
MAX_TRACK_DAYS = 30                # close the signal after this many days
COOLDOWN_DAYS = 10                 # after a signal closes, wait this long before
                                   # the same stock can be tracked again
STALE_GRACE_DAYS = 15              # if a tracked stock has no price for this long
                                   # past its close date, drop it (e.g. removed
                                   # from your watchlist)
HISTORY_KEEP_DAYS = 365            # prune closed signals older than this


# ============================================================
# STATE
# ============================================================
def _load():
    if not os.path.exists(STATE_FILE):
        return {"tracking": [], "closed": []}
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
        s.setdefault("tracking", [])
        s.setdefault("closed", [])
        return s
    except Exception as e:
        print(f"[tracker] state unreadable ({e}) — starting fresh")
        return {"tracking": [], "closed": []}


def _save(state):
    cutoff = (datetime.now(IST) - timedelta(days=HISTORY_KEEP_DAYS)).strftime("%Y-%m-%d")
    state["closed"] = [c for c in state["closed"] if c.get("close_date", "") >= cutoff]
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"[tracker] saved: {len(state['tracking'])} tracking, "
          f"{len(state['closed'])} closed")


def _age(date_str, today):
    d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=IST)
    return (today.date() - d.date()).days


# ============================================================
# CORE
# ============================================================
def _record(state, alerts, today):
    """Add new signals to tracking. Returns list of newly tracked."""
    today_str = today.strftime("%Y-%m-%d")
    added = []
    skipped_types = {}

    for sym, info in alerts.items():
        sig = info.get("signal", "")
        price = info.get("price")
        if price is None or price <= 0:
            continue
        if TRACK_SIGNALS is not None and sig not in TRACK_SIGNALS:
            skipped_types[sig] = skipped_types.get(sig, 0) + 1
            continue

        # already being tracked
        if any(t["symbol"] == sym for t in state["tracking"]):
            continue

        # closed recently — cooldown measured from CLOSE date.
        # Stops an ongoing condition from immediately re-entering,
        # while still allowing a genuinely new signal later.
        recent = [c for c in state["closed"]
                  if c["symbol"] == sym
                  and _age(c.get("close_date", c["signal_date"]), today) < COOLDOWN_DAYS]
        if recent:
            continue

        entry = {
            "symbol": sym,
            "signal": sig,
            "signal_date": today_str,
            "entry_price": round(float(price), 2),
            "milestones_hit": [],
            "drawdown_alerted": False,
            "peak_pct": 0.0,
            "trough_pct": 0.0,
        }
        state["tracking"].append(entry)
        added.append(entry)

    # Self-diagnostic: surface signal names that fired but aren't tracked.
    # If a name you EXPECT to track shows up here, it is a spelling mismatch
    # between your scanner and TRACK_SIGNALS.
    for sig, n in skipped_types.items():
        print(f"[tracker] NOT TRACKED: '{sig}' fired for {n} stock(s) "
              f"— not in TRACK_SIGNALS")

    return added


def _check(state, prices, today):
    """
    Update every tracked signal against current price.
    Returns (milestone_events, drawdown_events, closed_events, dropped).
    """
    milestone_events, drawdown_events, closed_events = [], [], []
    dropped = []
    still_tracking = []

    for t in state["tracking"]:
        now_price = prices.get(t["symbol"])
        age = _age(t["signal_date"], today)

        if now_price is None or now_price <= 0:
            # No price this run — normally just retry next run.
            # But if a stock is removed from the watchlist or yfinance
            # keeps failing, it would sit here forever. Drop it after a grace period.
            if age > MAX_TRACK_DAYS + STALE_GRACE_DAYS:
                dropped.append(t["symbol"])
                print(f"[tracker] DROPPED {t['symbol']} — no price for "
                      f"{age} days (removed from watchlist?)")
            else:
                still_tracking.append(t)
            continue

        pct = (now_price - t["entry_price"]) / t["entry_price"] * 100
        t["peak_pct"] = max(t["peak_pct"], round(pct, 2))
        t["trough_pct"] = min(t["trough_pct"], round(pct, 2))

        closing_now = age >= MAX_TRACK_DAYS

        # --- milestone crossings (highest new one only) ---
        crossed = [m for m in MILESTONES
                   if pct >= m and m not in t["milestones_hit"]]
        if crossed:
            t["milestones_hit"].extend(crossed)
            # Suppress the milestone alert if this signal is closing in the
            # same message — the CLOSED summary already reports the final number.
            if not closing_now:
                milestone_events.append({
                    **t,
                    "milestone": max(crossed),
                    "current_price": round(now_price, 2),
                    "return_pct": round(pct, 2),
                    "days": age,
                })

        # --- drawdown warning (once) ---
        if pct <= DRAWDOWN_ALERT and not t["drawdown_alerted"]:
            t["drawdown_alerted"] = True
            if not closing_now:
                drawdown_events.append({
                    **t,
                    "current_price": round(now_price, 2),
                    "return_pct": round(pct, 2),
                    "days": age,
                })

        # --- close out at max age ---
        if closing_now:
            rec = dict(t)
            rec.update({
                "close_date": today.strftime("%Y-%m-%d"),
                "close_price": round(now_price, 2),
                "final_pct": round(pct, 2),
                "days_held": age,
            })
            state["closed"].append(rec)
            closed_events.append(rec)
        else:
            still_tracking.append(t)

    state["tracking"] = still_tracking
    return milestone_events, drawdown_events, closed_events, dropped


def _record_stats(state):
    closed = state["closed"]
    if not closed:
        return None
    finals = [c["final_pct"] for c in closed]
    peaks = [c.get("peak_pct", 0) for c in closed]
    wins = [x for x in finals if x > 0]
    hit5 = sum(1 for p in peaks if p >= 5)
    hit10 = sum(1 for p in peaks if p >= 10)
    return {
        "n": len(closed),
        "win_rate": len(wins) / len(closed) * 100,
        "avg_final": sum(finals) / len(finals),
        "avg_peak": sum(peaks) / len(peaks),
        "hit5_pct": hit5 / len(closed) * 100,
        "hit10_pct": hit10 / len(closed) * 100,
        "best": max(finals),
        "worst": min(finals),
        "since": min(c["signal_date"] for c in closed),
    }


# ============================================================
# TELEGRAM TEXT
# ============================================================
def _fmt_date(s):
    return datetime.strptime(s, "%Y-%m-%d").strftime("%d %b")


def _build_text(new, milestones, drawdowns, closed, stats, state):
    L = []

    if milestones:
        L += ["", "🎯 <b>MILESTONE HIT</b>", "━━━━━━━━━━━━━━"]
        for m in sorted(milestones, key=lambda x: x["return_pct"], reverse=True):
            name = m["symbol"].replace(".NS", "")
            L.append(f"<b>{name}</b> · +{m['return_pct']:.1f}% 🚀")
            L.append(f"Signalled {_fmt_date(m['signal_date'])} @ ₹{m['entry_price']:,.0f}"
                     f" ({m['signal']})")
            L.append(f"Now ₹{m['current_price']:,.0f} · {m['days']} days")
            if m["peak_pct"] > m["return_pct"] + 0.5:
                L.append(f"<i>Peak so far +{m['peak_pct']:.1f}%</i>")
            L.append("")

    if drawdowns:
        L += ["⚠️ <b>UNDER PRESSURE</b>", "━━━━━━━━━━━━━━"]
        for d in drawdowns:
            name = d["symbol"].replace(".NS", "")
            L.append(f"<b>{name}</b> · {d['return_pct']:.1f}%")
            L.append(f"Signalled {_fmt_date(d['signal_date'])} @ ₹{d['entry_price']:,.0f}"
                     f" → ₹{d['current_price']:,.0f}")
            L.append("")

    if closed:
        L += [f"📕 <b>CLOSED ({MAX_TRACK_DAYS}-DAY)</b>", "━━━━━━━━━━━━━━"]
        for c in sorted(closed, key=lambda x: x["final_pct"], reverse=True):
            name = c["symbol"].replace(".NS", "")
            mark = "✅" if c["final_pct"] > 0 else "❌"
            sign = "+" if c["final_pct"] > 0 else ""
            detail = f"peak +{c['peak_pct']:.1f}%"
            if c.get("trough_pct", 0) < -2:
                detail += f", low {c['trough_pct']:.1f}%"
            L.append(f"<b>{name}</b> {sign}{c['final_pct']:.1f}% {mark}"
                     f"  <i>({detail})</i>")
        L.append("")

    if stats:
        s = stats
        since = datetime.strptime(s["since"], "%Y-%m-%d").strftime("%d %b %Y")
        L += ["📋 <b>TRACK RECORD</b>", f"<i>Since {since}</i>", "━━━━━━━━━━━━━━",
              f"{s['n']} signals closed · {s['win_rate']:.0f}% positive",
              f"Touched +5% at some point: {s['hit5_pct']:.0f}%",
              f"Touched +10% at some point: {s['hit10_pct']:.0f}%",
              f"Avg final {s['avg_final']:+.1f}% · avg peak {s['avg_peak']:+.1f}%",
              f"Best {s['best']:+.1f}% · Worst {s['worst']:+.1f}%",
              ""]

    if new:
        names = ", ".join(n["symbol"].replace(".NS", "") for n in new)
        L.append(f"🆕 Now tracking: {names}")

    n_track = len(state["tracking"])
    if n_track:
        L.append(f"👁 Tracking {n_track} live signal(s)")

    if L:
        L.append("")
        L.append("<i>Tracking is a record of past signals, not a recommendation "
                 "or a claim about future performance.</i>")

    return "\n".join(L)


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================
def track_and_report(alerts: dict, prices: dict, now: datetime = None) -> str:
    """
    alerts : {"SYM.NS": {"signal": "Buy Now", "price": 200.0}, ...}
             Signals fired by your scanner THIS run.
    prices : {"SYM.NS": 220.0, ...}
             Latest price for every stock you scanned.

    Returns Telegram-ready HTML text to append to your message.
    Returns "" if there is nothing to report.
    """
    today = now or datetime.now(IST)
    state = _load()

    # Check existing BEFORE recording, so a stock closed today
    # can be re-recorded today if it signals again.
    milestones, drawdowns, closed, dropped = _check(state, prices, today)
    new = _record(state, alerts, today)
    stats = _record_stats(state)

    text = _build_text(new, milestones, drawdowns, closed, stats, state)
    _save(state)

    print(f"[tracker] new={len(new)} milestones={len(milestones)} "
          f"drawdowns={len(drawdowns)} closed={len(closed)} dropped={len(dropped)}")
    return text
