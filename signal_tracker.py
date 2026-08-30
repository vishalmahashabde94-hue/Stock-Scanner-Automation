"""
signal_tracker.py — 30-Day Full Cycle tracker for VISH_SCAN

Bolts onto your EXISTING scanner. Does not touch your signal logic.

v3 CHANGES (from 15-day version):
  - Full 30-day cycle: reports at day 15 (checkpoint), closes at day 30 (final)
  - Day 15: Shows current %, peak so far, trough so far
  - Day 30: Final %, peak ever reached, lowest point touched
  - Track record now shows BOTH final % AND peak % to measure 
    "opportunity captured vs opportunity available"
  - Only tracks Buy Now, Aggressive Accumulation, Accumulate

State lives in signal_tracker_state.json (separate from your
scanner_state.json so nothing collides).
"""

import os
import json
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))
STATE_FILE = "signal_tracker_state.json"

# ============================================================
# SETTINGS
# ============================================================

TRACK_SIGNALS = {
    "Buy Now",
    "Aggressive Accumulation",
    "Accumulate",
}

CHECKPOINT_DAY = 15         # report progress on this day
TRACK_DAYS = 30             # close and final report on this day
COOLDOWN_DAYS = 10          # after a signal closes, wait before same stock can be tracked again
STALE_GRACE_DAYS = 15       # if no price for this long past close date, drop it
HISTORY_KEEP_DAYS = 365     # prune closed signals older than this


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
    """Add new signals to tracking. Silent."""
    today_str = today.strftime("%Y-%m-%d")
    added = []
    skipped_types = {}

    for sym, info in alerts.items():
        sig = info.get("signal", "")
        price = info.get("price")
        if price is None or price <= 0:
            continue
        if sig not in TRACK_SIGNALS:
            skipped_types[sig] = skipped_types.get(sig, 0) + 1
            continue

        if any(t["symbol"] == sym for t in state["tracking"]):
            continue

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
            "peak_pct": 0.0,
            "trough_pct": 0.0,
            "checkpoint_reported": False,
        }
        state["tracking"].append(entry)
        added.append(entry)

    for sig, n in skipped_types.items():
        print(f"[tracker] NOT TRACKED: '{sig}' fired for {n} stock(s) "
              f"— not in TRACK_SIGNALS")

    return added


def _check(state, prices, today):
    """
    Update tracked signals. Reports at day 15 (checkpoint) and day 30 (final).
    Returns (checkpoint_events, closed_events, dropped).
    """
    checkpoint_events = []
    closed_events = []
    dropped = []
    still_tracking = []

    for t in state["tracking"]:
        now_price = prices.get(t["symbol"])
        age = _age(t["signal_date"], today)

        if now_price is None or now_price <= 0:
            if age > TRACK_DAYS + STALE_GRACE_DAYS:
                dropped.append(t["symbol"])
                print(f"[tracker] DROPPED {t['symbol']} — no price for "
                      f"{age} days (removed from watchlist?)")
            else:
                still_tracking.append(t)
            continue

        pct = (now_price - t["entry_price"]) / t["entry_price"] * 100
        t["peak_pct"] = max(t["peak_pct"], round(pct, 2))
        t["trough_pct"] = min(t["trough_pct"], round(pct, 2))

        # --- Day 15 checkpoint ---
        if age == CHECKPOINT_DAY and not t["checkpoint_reported"]:
            t["checkpoint_reported"] = True
            checkpoint_events.append({
                **t,
                "current_price": round(now_price, 2),
                "return_pct": round(pct, 2),
                "days": age,
            })

        # --- Day 30 close ---
        if age >= TRACK_DAYS:
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
    return checkpoint_events, closed_events, dropped


def _record_stats(state):
    closed = state["closed"]
    if not closed:
        return None
    finals = [c["final_pct"] for c in closed]
    peaks = [c.get("peak_pct", 0) for c in closed]
    wins = [x for x in finals if x > 0]
    troughs = [c.get("trough_pct", 0) for c in closed]
    
    return {
        "n": len(closed),
        "win_rate": len(wins) / len(closed) * 100,
        "avg_final": sum(finals) / len(finals),
        "avg_peak": sum(peaks) / len(peaks),
        "best_final": max(finals),
        "worst_final": min(finals),
        "best_peak": max(peaks),
        "worst_trough": min(troughs),
        "since": min(c["signal_date"] for c in closed),
    }


# ============================================================
# TELEGRAM TEXT
# ============================================================
def _fmt_date(s):
    return datetime.strptime(s, "%Y-%m-%d").strftime("%d %b")


def _build_text(checkpoints, closed, stats):
    L = []

    if checkpoints:
        L += ["", "⏱ <b>15-DAY CHECKPOINT</b>", "━━━━━━━━━━━━━━"]
        for c in sorted(checkpoints, key=lambda x: x["return_pct"], reverse=True):
            name = c["symbol"].replace(".NS", "")
            mark = "📈" if c["return_pct"] > 0 else "📉"
            sign = "+" if c["return_pct"] > 0 else ""
            L.append(
                f"<b>{name}</b> ({c['signal']})  {mark}\n"
                f"Signalled {_fmt_date(c['signal_date'])} @ ₹{c['entry_price']:,.0f}\n"
                f"Now ₹{c['current_price']:,.0f}  {sign}{c['return_pct']:.1f}%\n"
                f"Peak so far +{c['peak_pct']:.1f}% · Low {c['trough_pct']:.1f}%"
            )
            L.append("")

    if closed:
        L += ["", "📆 <b>30-DAY FINAL</b>", "━━━━━━━━━━━━━━"]
        for c in sorted(closed, key=lambda x: x["final_pct"], reverse=True):
            name = c["symbol"].replace(".NS", "")
            mark = "✅" if c["final_pct"] > 0 else "❌"
            sign = "+" if c["final_pct"] > 0 else ""
            
            # Show opportunity captured vs available
            opportunity = ""
            if c.get("peak_pct", 0) > c["final_pct"] + 0.5:
                opportunity = f" (peaked +{c['peak_pct']:.1f}%, left {c['peak_pct'] - c['final_pct']:.1f}%)"
            if c.get("trough_pct", 0) < c["final_pct"] - 0.5:
                opportunity += f" (recovered from {c['trough_pct']:.1f}%)"
            
            L.append(
                f"<b>{name}</b> ({c['signal']})\n"
                f"{_fmt_date(c['signal_date'])} @ ₹{c['entry_price']:,.0f} "
                f"→ ₹{c['close_price']:,.0f}\n"
                f"{sign}{c['final_pct']:.1f}% {mark}{opportunity}"
            )
            L.append("")

    if stats:
        s = stats
        since = datetime.strptime(s["since"], "%Y-%m-%d").strftime("%d %b %Y")
        L += [
            "",
            "📋 <b>TRACK RECORD (30-day cycles)</b>",
            f"<i>Since {since}</i>",
            "━━━━━━━━━━━━━━",
            f"{s['n']} signals closed · {s['win_rate']:.0f}% positive",
            f"Avg final {s['avg_final']:+.1f}% · Avg peak {s['avg_peak']:+.1f}%",
            f"Best final {s['best_final']:+.1f}% · Best peak +{s['best_peak']:.1f}%",
            f"Worst final {s['worst_final']:.1f}% · Worst correction {s['worst_trough']:.1f}%",
            ""
        ]

    if L:
        L.append("<i>Journey tracking is a record of past signals, not a "
                 "recommendation or a claim about future performance.</i>")

    return "\n".join(L)


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================
def track_and_report(alerts: dict, prices: dict, now: datetime = None) -> str:
    """
    alerts : {"SYM.NS": {"signal": "Buy Now", "price": 200.0}, ...}
    prices : {"SYM.NS": 220.0, ...}

    Returns Telegram-ready HTML text. Returns "" if nothing to report.
    """
    today = now or datetime.now(IST)
    state = _load()

    checkpoints, closed, dropped = _check(state, prices, today)
    new = _record(state, alerts, today)
    stats = _record_stats(state)

    text = _build_text(checkpoints, closed, stats)
    _save(state)

    print(f"[tracker] new={len(new)} checkpoint={len(checkpoints)} closed={len(closed)} dropped={len(dropped)}")
    return text
