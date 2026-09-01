"""
signal_tracker.py — backward-compatible 30-day cycle tracker for VISH_SCAN.

This module preserves legacy tracking records, reports one checkpoint on or
after day 15, and closes a signal on or after day 30.  State is stored in
signal_tracker_state.json alongside this file.

Schema v3.1 fixes the v2 -> v3 migration where legacy records did not contain
``checkpoint_reported`` and therefore raised KeyError before Telegram delivery.
"""

from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


IST = timezone(timedelta(hours=5, minutes=30))
STATE_FILE = Path(__file__).resolve().with_name("signal_tracker_state.json")
STATE_SCHEMA_VERSION = 3

# ============================================================
# SETTINGS
# ============================================================

TRACK_SIGNALS = {
    "Buy Now",
    "Aggressive Accumulation",
    "Accumulate",
}

CHECKPOINT_DAY = 15
TRACK_DAYS = 30
COOLDOWN_DAYS = 10
STALE_GRACE_DAYS = 15
HISTORY_KEEP_DAYS = 365


# ============================================================
# STATE VALIDATION / MIGRATION
# ============================================================

def _empty_state() -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "tracking": [],
        "closed": [],
    }


def _state_path() -> Path:
    """Return STATE_FILE as a Path, including when tests monkey-patch it."""
    return Path(STATE_FILE)


def _valid_date(value: Any, field: str, symbol: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{symbol}: {field} must be YYYY-MM-DD")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{symbol}: invalid {field}={value!r}") from exc
    return value


def _positive_price(value: Any, field: str, symbol: str) -> float:
    try:
        price = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{symbol}: invalid {field}={value!r}") from exc
    if price <= 0:
        raise ValueError(f"{symbol}: {field} must be positive")
    return round(price, 2)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return default


def _migrate_state(raw: Any) -> tuple[dict[str, Any], int]:
    """
    Validate state and add fields introduced by newer tracker versions.

    Unknown legacy fields are deliberately retained, so historical details
    such as milestones_hit and drawdown_alerted are not destroyed.
    """
    if not isinstance(raw, dict):
        raise ValueError("tracker state root must be a JSON object")

    state = dict(raw)
    tracking = state.get("tracking", [])
    closed = state.get("closed", [])
    if not isinstance(tracking, list) or not isinstance(closed, list):
        raise ValueError("tracker state 'tracking' and 'closed' must be lists")

    migrations = 0
    migrated_tracking: list[dict[str, Any]] = []

    for index, original in enumerate(tracking):
        if not isinstance(original, dict):
            raise ValueError(f"tracking[{index}] must be a JSON object")

        item = dict(original)
        symbol = item.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"tracking[{index}] has no valid symbol")

        signal = item.get("signal")
        if not isinstance(signal, str) or not signal.strip():
            raise ValueError(f"{symbol}: missing signal")

        item["signal_date"] = _valid_date(
            item.get("signal_date"), "signal_date", symbol
        )
        item["entry_price"] = _positive_price(
            item.get("entry_price"), "entry_price", symbol
        )
        item["peak_pct"] = _number(item.get("peak_pct"), 0.0)
        item["trough_pct"] = _number(item.get("trough_pct"), 0.0)

        if "checkpoint_reported" not in item:
            item["checkpoint_reported"] = False
            migrations += 1
        else:
            item["checkpoint_reported"] = bool(item["checkpoint_reported"])

        migrated_tracking.append(item)

    migrated_closed: list[dict[str, Any]] = []
    for index, original in enumerate(closed):
        if not isinstance(original, dict):
            raise ValueError(f"closed[{index}] must be a JSON object")

        item = dict(original)
        symbol = item.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"closed[{index}] has no valid symbol")

        item["signal_date"] = _valid_date(
            item.get("signal_date"), "signal_date", symbol
        )
        item["close_date"] = _valid_date(
            item.get("close_date"), "close_date", symbol
        )
        item["entry_price"] = _positive_price(
            item.get("entry_price"), "entry_price", symbol
        )
        item["close_price"] = _positive_price(
            item.get("close_price"), "close_price", symbol
        )

        if "final_pct" not in item:
            # Some older tracker builds named this field return_pct.
            if "return_pct" in item:
                item["final_pct"] = _number(item["return_pct"])
                migrations += 1
            else:
                raise ValueError(f"{symbol}: closed record missing final_pct")
        else:
            item["final_pct"] = _number(item["final_pct"])

        item["peak_pct"] = _number(item.get("peak_pct"), 0.0)
        item["trough_pct"] = _number(item.get("trough_pct"), 0.0)
        item["checkpoint_reported"] = bool(
            item.get("checkpoint_reported", True)
        )
        migrated_closed.append(item)

    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        migrations += 1

    state["schema_version"] = STATE_SCHEMA_VERSION
    state["tracking"] = migrated_tracking
    state["closed"] = migrated_closed
    return state, migrations


def _load() -> tuple[dict[str, Any], int]:
    path = _state_path()
    if not path.exists():
        return _empty_state(), 0

    try:
        with path.open(encoding="utf-8-sig") as handle:
            raw = json.load(handle)
        return _migrate_state(raw)
    except Exception as exc:
        # Do not silently replace unreadable history with an empty file.
        raise RuntimeError(
            f"tracker state unreadable; original file preserved: {exc}"
        ) from exc


def _save(state: dict[str, Any], now: datetime | None = None) -> None:
    """Prune old closed records and replace the state file atomically."""
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    current = now or datetime.now(IST)
    cutoff = (current - timedelta(days=HISTORY_KEEP_DAYS)).strftime("%Y-%m-%d")
    state["closed"] = [
        item
        for item in state["closed"]
        if item.get("close_date", "") >= cutoff
    ]
    state["schema_version"] = STATE_SCHEMA_VERSION

    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(
        f"[tracker] saved: {len(state['tracking'])} tracking, "
        f"{len(state['closed'])} closed"
    )


def _age(date_str: str, today: datetime) -> int:
    signal_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    return (today.date() - signal_date).days


# ============================================================
# CORE
# ============================================================

def _record(
    state: dict[str, Any], alerts: dict[str, dict[str, Any]], today: datetime
) -> list[dict[str, Any]]:
    """Add eligible new signals to tracking."""
    today_str = today.strftime("%Y-%m-%d")
    added: list[dict[str, Any]] = []
    skipped_types: dict[str, int] = {}

    active_symbols = {item["symbol"] for item in state["tracking"]}

    for symbol, info in alerts.items():
        signal = info.get("signal", "")
        price = info.get("price")

        try:
            price_value = float(price)
        except (TypeError, ValueError):
            continue
        if price_value <= 0:
            continue

        if signal not in TRACK_SIGNALS:
            skipped_types[signal] = skipped_types.get(signal, 0) + 1
            continue

        if symbol in active_symbols:
            continue

        recent = [
            item
            for item in state["closed"]
            if item["symbol"] == symbol
            and _age(item.get("close_date", item["signal_date"]), today)
            < COOLDOWN_DAYS
        ]
        if recent:
            continue

        entry = {
            "symbol": symbol,
            "signal": signal,
            "signal_date": today_str,
            "entry_price": round(price_value, 2),
            "peak_pct": 0.0,
            "trough_pct": 0.0,
            "checkpoint_reported": False,
        }
        state["tracking"].append(entry)
        active_symbols.add(symbol)
        added.append(entry)

    for signal, count in skipped_types.items():
        print(
            f"[tracker] NOT TRACKED: {signal!r} fired for {count} stock(s) "
            "— not in TRACK_SIGNALS"
        )

    return added


def _check(
    state: dict[str, Any], prices: dict[str, float], today: datetime
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Update active journeys and return checkpoint, close and drop events."""
    checkpoint_events: list[dict[str, Any]] = []
    closed_events: list[dict[str, Any]] = []
    dropped: list[str] = []
    still_tracking: list[dict[str, Any]] = []

    for item in state["tracking"]:
        symbol = item["symbol"]
        now_price = prices.get(symbol)
        age = _age(item["signal_date"], today)

        if now_price is None or now_price <= 0:
            if age > TRACK_DAYS + STALE_GRACE_DAYS:
                dropped.append(symbol)
                print(
                    f"[tracker] DROPPED {symbol} — no price for {age} days "
                    "(removed from watchlist?)"
                )
            else:
                still_tracking.append(item)
            continue

        pct = (float(now_price) - item["entry_price"]) / item["entry_price"] * 100
        item["peak_pct"] = max(item.get("peak_pct", 0.0), round(pct, 2))
        item["trough_pct"] = min(item.get("trough_pct", 0.0), round(pct, 2))

        # Use a range rather than age == 15. A day-15 weekend/holiday is then
        # reported once on the next successful weekday run.
        if (
            CHECKPOINT_DAY <= age < TRACK_DAYS
            and not item.get("checkpoint_reported", False)
        ):
            item["checkpoint_reported"] = True
            checkpoint_events.append(
                {
                    **item,
                    "current_price": round(float(now_price), 2),
                    "return_pct": round(pct, 2),
                    "days": age,
                }
            )

        if age >= TRACK_DAYS:
            record = dict(item)
            record.update(
                {
                    "close_date": today.strftime("%Y-%m-%d"),
                    "close_price": round(float(now_price), 2),
                    "final_pct": round(pct, 2),
                    "days_held": age,
                }
            )
            state["closed"].append(record)
            closed_events.append(record)
        else:
            still_tracking.append(item)

    state["tracking"] = still_tracking
    return checkpoint_events, closed_events, dropped


def _record_stats(state: dict[str, Any]) -> dict[str, Any] | None:
    closed = state["closed"]
    if not closed:
        return None

    finals = [item["final_pct"] for item in closed]
    peaks = [item.get("peak_pct", 0.0) for item in closed]
    wins = [value for value in finals if value > 0]
    troughs = [item.get("trough_pct", 0.0) for item in closed]

    return {
        "n": len(closed),
        "win_rate": len(wins) / len(closed) * 100,
        "avg_final": sum(finals) / len(finals),
        "avg_peak": sum(peaks) / len(peaks),
        "best_final": max(finals),
        "worst_final": min(finals),
        "best_peak": max(peaks),
        "worst_trough": min(troughs),
        "since": min(item["signal_date"] for item in closed),
    }


# ============================================================
# TELEGRAM TEXT
# ============================================================

def _fmt_date(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").strftime("%d %b")


def _build_text(
    checkpoints: list[dict[str, Any]],
    closed: list[dict[str, Any]],
    stats: dict[str, Any] | None,
) -> str:
    lines: list[str] = []

    if checkpoints:
        lines += ["", "⏱ <b>15-DAY CHECKPOINT</b>", "━━━━━━━━━━━━━━"]
        for item in sorted(
            checkpoints, key=lambda value: value["return_pct"], reverse=True
        ):
            name = html.escape(item["symbol"].replace(".NS", ""))
            signal = html.escape(item["signal"])
            mark = "📈" if item["return_pct"] > 0 else "📉"
            sign = "+" if item["return_pct"] > 0 else ""
            timing = (
                f" · reported day {item['days']}"
                if item["days"] > CHECKPOINT_DAY
                else ""
            )
            lines.append(
                f"<b>{name}</b> ({signal})  {mark}{timing}\n"
                f"Signalled {_fmt_date(item['signal_date'])} @ "
                f"₹{item['entry_price']:,.0f}\n"
                f"Now ₹{item['current_price']:,.0f}  "
                f"{sign}{item['return_pct']:.1f}%\n"
                f"Peak so far {item['peak_pct']:+.1f}% · "
                f"Low {item['trough_pct']:.1f}%"
            )
            lines.append("")

    if closed:
        lines += ["", "📆 <b>30-DAY FINAL</b>", "━━━━━━━━━━━━━━"]
        for item in sorted(
            closed, key=lambda value: value["final_pct"], reverse=True
        ):
            name = html.escape(item["symbol"].replace(".NS", ""))
            signal = html.escape(item["signal"])
            mark = "✅" if item["final_pct"] > 0 else "❌"
            sign = "+" if item["final_pct"] > 0 else ""

            opportunity = ""
            if item.get("peak_pct", 0) > item["final_pct"] + 0.5:
                left = item["peak_pct"] - item["final_pct"]
                opportunity = (
                    f" (peaked {item['peak_pct']:+.1f}%, left {left:.1f}%)"
                )
            if item.get("trough_pct", 0) < item["final_pct"] - 0.5:
                opportunity += (
                    f" (recovered from {item['trough_pct']:.1f}%)"
                )

            lines.append(
                f"<b>{name}</b> ({signal})\n"
                f"{_fmt_date(item['signal_date'])} @ "
                f"₹{item['entry_price']:,.0f} → ₹{item['close_price']:,.0f}\n"
                f"{sign}{item['final_pct']:.1f}% {mark}{opportunity}"
            )
            lines.append("")

    if stats:
        since = datetime.strptime(stats["since"], "%Y-%m-%d").strftime(
            "%d %b %Y"
        )
        lines += [
            "",
            "📋 <b>TRACK RECORD (30-day cycles)</b>",
            f"<i>Since {since}</i>",
            "━━━━━━━━━━━━━━",
            f"{stats['n']} signals closed · {stats['win_rate']:.0f}% positive",
            f"Avg final {stats['avg_final']:+.1f}% · "
            f"Avg peak {stats['avg_peak']:+.1f}%",
            f"Best final {stats['best_final']:+.1f}% · "
            f"Best peak {stats['best_peak']:+.1f}%",
            f"Worst final {stats['worst_final']:.1f}% · "
            f"Worst correction {stats['worst_trough']:.1f}%",
            "",
        ]

    if lines:
        lines.append(
            "<i>Journey tracking is a record of past signals, not a "
            "recommendation or a claim about future performance.</i>"
        )

    return "\n".join(lines)


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def track_and_report(
    alerts: dict[str, dict[str, Any]],
    prices: dict[str, float],
    now: datetime | None = None,
) -> str:
    """Update tracking state and return Telegram-ready HTML."""
    today = now or datetime.now(IST)
    state, migrations = _load()

    checkpoints, closed, dropped = _check(state, prices, today)
    new = _record(state, alerts, today)
    stats = _record_stats(state)

    text = _build_text(checkpoints, closed, stats)
    _save(state, now=today)

    print(
        f"[tracker] migrated={migrations} new={len(new)} "
        f"checkpoint={len(checkpoints)} closed={len(closed)} "
        f"dropped={len(dropped)}"
    )
    return text
