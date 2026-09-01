"""
=======================================================
  VISH_SCAN — SCANNER v7.3.1

  Reliability hotfix:
    - Legacy tracker failures can no longer block scanner alerts.
    - Telegram delivery is retried and failures fail the workflow visibly.
    - Telegram text is chunked safely, including journey-tracker text.
    - Duplicate watchlist symbols are removed before download.
    - Two identical full scans remain at 7:30 AM and 4:00 PM IST.

  Trading classification logic remains the same as v7.3.
=======================================================
"""

from __future__ import annotations

import html
import logging
import os
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf


try:
    from signal_tracker import track_and_report as _track_and_report

    TRACKER_IMPORT_ERROR = None
except Exception as exc:  # tracker must never prevent the main scan from loading
    _track_and_report = None
    TRACKER_IMPORT_ERROR = exc


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("CHAT_ID", "").strip()

IST = timezone(timedelta(hours=5, minutes=30))
BENCHMARK = "^NSEI"

# 97 entries were supplied in v7.3, but TIINDIA.NS appeared twice.
# This list therefore contains the same 96 unique stocks.
WATCHLIST = [
    "HBLENGINE.NS",
    "PARAS.NS",
    "ZENTEC.NS",
    "DATAPATTNS.NS",
    "MAZDOCK.NS",
    "HAL.NS",
    "BEL.NS",
    "ASTRAMICRO.NS",
    "BHARATFORG.NS",
    "MTARTECH.NS",
    "IDEAFORGE.NS",
    "M&M.NS",
    "ASHOKLEY.NS",
    "TVSMOTOR.NS",
    "BANCOINDIA.NS",
    "PRECWIRE.NS",
    "MOTHERSON.NS",
    "ENDURANCE.NS",
    "TIINDIA.NS",
    "BALUFORGE.NS",
    "SMLMAH.NS",
    "HAVELLS.NS",
    "POLYCAB.NS",
    "KEI.NS",
    "SCHNEIDER.NS",
    "CGPOWER.NS",
    "TRANSRAILL.NS",
    "TARIL.NS",
    "VOLTAMP.NS",
    "GVT&D.NS",
    "TRITURBINE.NS",
    "TDPOWERSYS.NS",
    "IONEXCHANG.NS",
    "TITAGARH.NS",
    "KEC.NS",
    "LT.NS",
    "ADANIGREEN.NS",
    "ADANIPOWER.NS",
    "JSWENERGY.NS",
    "INOXWIND.NS",
    "WAAREERTL.NS",
    "KPIL.NS",
    "JWL.NS",
    "INFY.NS",
    "WIPRO.NS",
    "DIXON.NS",
    "REDINGTON.NS",
    "KAYNES.NS",
    "NETWEB.NS",
    "BBOX.NS",
    "BHARTIARTL.NS",
    "INDUSTOWER.NS",
    "MOTILALOFS.NS",
    "ANGELONE.NS",
    "BAJFINANCE.NS",
    "AXISBANK.NS",
    "BSE.NS",
    "NSDL.NS",
    "CDSL.NS",
    "KFINTECH.NS",
    "ADANIPORTS.NS",
    "ANANTRAJ.NS",
    "CHALET.NS",
    "RATEGAIN.NS",
    "METROBRAND.NS",
    "BLS.NS",
    "BOROLTD.NS",
    "TIPSMUSIC.NS",
    "GOKULAGRO.NS",
    "VBL.NS",
    "LTFOODS.NS",
    "RADICO.NS",
    "GODFRYPHLP.NS",
    "ABDL.NS",
    "LLOYDSME.NS",
    "COALINDIA.NS",
    "RELIANCE.NS",
    "NATCOPHARM.NS",
    "RUBICON.NS",
    "YATHARTH.NS",
    "KIMS.NS",
    "NH.NS",
    "RAINBOW.NS",
    "KAJARIACER.NS",
    "AEGISLOG.NS",
    "PRICOLLTD.NS",
    "CCL.NS",
    "EMMVEE.NS",
    "IDFCFIRSTB.NS",
    "CPPLUS.NS",
    "ENRIN.NS",
    "APARINDS.NS",
    "MANKIND.NS",
    "HDFCBANK.NS",
    "ETERNAL.NS",
    "SWIGGY.NS",
]

# Gap stage bands
STAGE1_MIN, STAGE1_MAX = -20.0, -12.0
STAGE2_MIN, STAGE2_MAX = -12.0, -8.0
STAGE3_MIN, STAGE3_MAX = -8.0, -4.0
STAGE4_MIN, STAGE4_MAX = -4.0, -2.0

# Thresholds
GOLDEN_CROSS_LOOKBACK = 5
SLOPE_LOOKBACK = 3
MIN_SLOPE_MOVE = 0.15

RSI_OVERBOUGHT = 70
RSI_EXIT = 75
RSI_OVERSOLD = 30

VOLUME_CONFIRM = 1.5
VOL_SMOOTH_MIN_RATIO = 1.2
VOL_SMOOTH_MIN_DAYS = 2

BREAKOUT_3M_DAYS = 63
BREAKOUT_6M_DAYS = 126
SUPPORT_ZONE = 0.02

COIL_PROXIMITY = 0.05
COIL_RSI_MIN = 55
COIL_RSI_MAX = 68
COIL_MIN_HITS = 3

TELEGRAM_MAX_UNITS = 4096
TELEGRAM_CHUNK_UNITS = 3500
TELEGRAM_ATTEMPTS = 3


# ============================================================
# Telegram
# ============================================================

def _validate_telegram_config() -> None:
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("BOT_TOKEN")
    if not TELEGRAM_CHAT_ID:
        missing.append("CHAT_ID")
    if missing:
        raise RuntimeError(
            "Missing required GitHub secret(s): " + ", ".join(missing)
        )


def _telegram_units(value: str) -> int:
    """Telegram counts astral emoji as two UTF-16 code units."""
    return len(value.encode("utf-16-le")) // 2


def _split_long_line(line: str, max_units: int) -> list[str]:
    """Split an unexpectedly long plain line without exceeding max_units."""
    pieces = []
    current = []
    units = 0
    for char in line:
        char_units = _telegram_units(char)
        if current and units + char_units > max_units:
            pieces.append("".join(current))
            current = []
            units = 0
        current.append(char)
        units += char_units
    if current or not pieces:
        pieces.append("".join(current))
    return pieces


def build_telegram_chunks(items, max_units=TELEGRAM_CHUNK_UNITS):
    """Build chunks safely even when an item itself contains many newlines."""
    if not 1 <= max_units < TELEGRAM_MAX_UNITS:
        raise ValueError("max_units must be between 1 and 4095")

    lines = []
    for item in items:
        text = str(item)
        split = text.splitlines()
        lines.extend(split if split else [""])

    chunks = []
    current_lines = []
    current_units = 0

    for source_line in lines:
        line_parts = (
            _split_long_line(source_line, max_units)
            if _telegram_units(source_line) > max_units
            else [source_line]
        )
        for line in line_parts:
            addition = _telegram_units(line) + (1 if current_lines else 0)
            if current_lines and current_units + addition > max_units:
                chunks.append("\n".join(current_lines))
                current_lines = []
                current_units = 0

            current_lines.append(line)
            current_units += _telegram_units(line) + (1 if len(current_lines) > 1 else 0)

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def send_telegram(message):
    _validate_telegram_config()
    if not message:
        return
    if _telegram_units(message) > TELEGRAM_MAX_UNITS:
        raise ValueError("Telegram message exceeds the 4096-unit limit")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    last_error = None

    for attempt in range(1, TELEGRAM_ATTEMPTS + 1):
        try:
            response = requests.post(url, json=payload, timeout=20)
            response.raise_for_status()
            result = response.json()
            if not result.get("ok", False):
                raise RuntimeError("Telegram API returned ok=false")
            log.info("  Telegram: sent")
            return
        except Exception as exc:
            last_error = exc
            log.warning(
                "  Telegram attempt %d/%d failed: %s",
                attempt,
                TELEGRAM_ATTEMPTS,
                exc,
            )
            if attempt < TELEGRAM_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(
        f"Telegram delivery failed after {TELEGRAM_ATTEMPTS} attempts"
    ) from last_error


def send_telegram_chunked(items, max_units=TELEGRAM_CHUNK_UNITS):
    chunks = build_telegram_chunks(items, max_units=max_units)
    for index, chunk in enumerate(chunks, start=1):
        log.info("  Telegram chunk %d/%d", index, len(chunks))
        send_telegram(chunk)


def safe_tracking_report(alerts, prices):
    """Return tracker text without ever blocking the main scanner report."""
    if _track_and_report is None:
        log.error("Journey tracker import failed: %s", TRACKER_IMPORT_ERROR)
        return (
            "\n⚠️ <b>Journey tracker unavailable</b>\n"
            "Main scan completed. Check the GitHub Actions log."
        )

    try:
        return _track_and_report(alerts, prices)
    except Exception:
        log.exception("Journey tracker failed; main scanner will still be sent")
        return (
            "\n⚠️ <b>Journey tracker unavailable</b>\n"
            "Main scan completed. Tracking state was left unchanged; "
            "check the GitHub Actions log."
        )


# ============================================================
# Indicators
# ============================================================

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def rsi_simple(rsi):
    if rsi < 30:
        return f"{rsi:.0f} — Oversold"
    if rsi < 45:
        return f"{rsi:.0f} — Recovering"
    if rsi < 55:
        return f"{rsi:.0f} — Neutral"
    if rsi < 70:
        return f"{rsi:.0f} — Heated"
    return f"{rsi:.0f} — Overbought"


def vol_simple(vol_ratio):
    if vol_ratio >= 1.5:
        return f"High ({vol_ratio:.1f}x)"
    if vol_ratio >= 0.8:
        return f"OK ({vol_ratio:.1f}x)"
    return f"Low ({vol_ratio:.1f}x)"


def detect_trendline(prices):
    x = np.arange(len(prices))
    y = np.array(prices)
    slope, intercept = np.polyfit(x, y, 1)
    current = slope * (len(prices) - 1) + intercept
    return slope, current


def swing_lows(low_series, order=3):
    """Return local minima only."""
    values = low_series.values
    points = []
    for index in range(order, len(values) - order):
        window = values[index - order : index + order + 1]
        if values[index] == window.min():
            points.append((index, values[index]))
    return points


# ============================================================
# Batched fetch
# ============================================================

def _ticker_frame(raw, symbol):
    """Handle either ticker-first or price-first yfinance MultiIndex output."""
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw

    level_zero = raw.columns.get_level_values(0)
    if symbol in level_zero:
        return raw[symbol]

    level_one = raw.columns.get_level_values(1)
    if symbol in level_one:
        return raw.xs(symbol, axis=1, level=1)

    raise KeyError(symbol)


def fetch_all_data(symbols):
    unique_symbols = list(dict.fromkeys(symbols))
    tickers = unique_symbols + [BENCHMARK]
    log.info("Batch downloading %d tickers...", len(tickers))

    raw = None
    last_error = None
    for attempt in range(1, 4):
        try:
            raw = yf.download(
                tickers,
                period="1y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            if raw is None or raw.empty:
                raise RuntimeError("Yahoo Finance returned an empty dataframe")
            break
        except Exception as exc:
            last_error = exc
            log.warning("Yahoo download attempt %d/3 failed: %s", attempt, exc)
            if attempt < 3:
                time.sleep(3 * attempt)

    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance batch download failed") from last_error

    try:
        bench_close = _ticker_frame(raw, BENCHMARK)["Close"].dropna()
        if len(bench_close) < 21:
            raise ValueError("fewer than 21 benchmark rows")
        nifty_return_20d = float(
            bench_close.iloc[-1] / bench_close.iloc[-21] - 1
        ) * 100
    except Exception as exc:
        log.warning("Benchmark fetch failed, defaulting RS to neutral: %s", exc)
        nifty_return_20d = 0.0

    results = {}
    for symbol in unique_symbols:
        try:
            frame = _ticker_frame(raw, symbol).dropna(how="all")
            if frame.empty or len(frame) < 200:
                log.warning("  %s: insufficient data (%d rows)", symbol, len(frame))
                continue
            data = process_symbol(symbol, frame, nifty_return_20d)
            if data:
                results[symbol] = data
        except Exception as exc:
            log.error("  %s: error — %s", symbol, exc)
    return results


def process_symbol(symbol, df, nifty_return_20d):
    close = df["Close"].dropna()
    high = df["High"].dropna()
    low = df["Low"].dropna()
    volume = df["Volume"].dropna()

    if len(close) < 200:
        return None

    ema50_series = close.ewm(span=50, adjust=False).mean()
    ema200_series = close.ewm(span=200, adjust=False).mean()
    gap_series = ((ema50_series - ema200_series) / ema200_series * 100).round(2)

    gap_today = float(gap_series.iloc[-1])
    gap_prev = float(gap_series.iloc[-2])
    gap_3ago = float(gap_series.iloc[-1 - SLOPE_LOOKBACK])
    gap_closing = (gap_today - gap_3ago) > MIN_SLOPE_MOVE

    lookback_window = gap_series.iloc[-1 - GOLDEN_CROSS_LOOKBACK : -1]
    golden_cross = bool((lookback_window < 0).all() and gap_today >= 0)

    rsi_series = compute_rsi(close)
    rsi_today = round(float(rsi_series.iloc[-1]), 1)

    vol_avg20_series = volume.rolling(20).mean()
    vol_ratio_series = (volume / vol_avg20_series).round(2)
    vol_ratio_today = (
        float(vol_ratio_series.iloc[-1])
        if not np.isnan(vol_ratio_series.iloc[-1])
        else 0
    )

    last3_vol = vol_ratio_series.iloc[-3:].fillna(0)
    smoothed_vol_ok = (
        int((last3_vol >= VOL_SMOOTH_MIN_RATIO).sum()) >= VOL_SMOOTH_MIN_DAYS
    )

    price_today = float(close.iloc[-1])
    price_prev = float(close.iloc[-2])

    high_3m = float(high.iloc[-BREAKOUT_3M_DAYS:-1].max())
    high_6m = float(high.iloc[-BREAKOUT_6M_DAYS:-1].max())

    breakout_3m = price_today > high_3m and price_prev <= high_3m
    breakout_6m = price_today > high_6m and price_prev <= high_6m

    vol_confirm = (vol_ratio_today >= VOLUME_CONFIRM) or smoothed_vol_ok
    breakout_3m_confirmed = breakout_3m and vol_confirm
    breakout_6m_confirmed = breakout_6m and vol_confirm

    recent_low_series = low.iloc[-BREAKOUT_3M_DAYS:]
    points = swing_lows(recent_low_series, order=3)
    if len(points) >= 2:
        xs = np.array([point[0] for point in points])
        ys = np.array([point[1] for point in points])
        slope, intercept = np.polyfit(xs, ys, 1)
        trendline_support = slope * (len(recent_low_series) - 1) + intercept
    else:
        slope, trendline_support = detect_trendline(recent_low_series.values)

    near_support = (
        slope > 0
        and gap_today < 0
        and abs(price_today - trendline_support) / trendline_support <= SUPPORT_ZONE
    )

    recent_highs = high.iloc[-BREAKOUT_3M_DAYS:].values
    _, trendline_resistance = detect_trendline(recent_highs)
    trendline_breakout = (
        price_today > trendline_resistance
        and price_prev <= trendline_resistance
        and vol_confirm
    )

    near_high = (
        (high_3m - price_today) / high_3m <= COIL_PROXIMITY
        and price_today <= high_3m
    )
    rsi_building = COIL_RSI_MIN <= rsi_today <= COIL_RSI_MAX

    returns = close.pct_change()
    vol10 = returns.iloc[-10:].std()
    vol20 = returns.iloc[-20:].std()
    volatility_squeeze = (
        bool(vol10 < vol20 * 0.85) if vol20 and not np.isnan(vol20) else False
    )

    vol_recent3 = float(volume.iloc[-3:].mean())
    vol_prior10 = float(volume.iloc[-13:-3].mean())
    vol_trend_up = vol_recent3 > vol_prior10

    coil_hits = sum([near_high, rsi_building, volatility_squeeze, vol_trend_up])
    coiling = coil_hits >= COIL_MIN_HITS and not (
        breakout_3m_confirmed or breakout_6m_confirmed
    )

    try:
        stock_return_20d = float(close.iloc[-1] / close.iloc[-21] - 1) * 100
    except Exception:
        stock_return_20d = 0.0
    rs_positive = stock_return_20d > nifty_return_20d

    return {
        "symbol": symbol,
        "price": round(price_today, 2),
        "gap_today": gap_today,
        "gap_prev": gap_prev,
        "gap_closing": gap_closing,
        "golden_cross": golden_cross,
        "rsi": rsi_today,
        "vol_ratio": vol_ratio_today,
        "breakout_3m_confirmed": breakout_3m_confirmed,
        "breakout_6m_confirmed": breakout_6m_confirmed,
        "near_support": near_support,
        "trendline_breakout": trendline_breakout,
        "coiling": coiling,
        "rs_positive": rs_positive,
        "stock_return_20d": round(stock_return_20d, 1),
        "nifty_return_20d": round(nifty_return_20d, 1),
    }


# ============================================================
# Classification — unchanged from v7.3
# ============================================================

def classify(data):
    ticker = html.escape(data["symbol"].replace(".NS", ""))
    gap = data["gap_today"]
    price = data["price"]
    rsi = data["rsi"]
    vol_ratio = data["vol_ratio"]

    rsi_label = rsi_simple(rsi)
    vol_label = vol_simple(vol_ratio)
    rs_tag = (
        "✅ RS > Nifty"
        if data["rs_positive"]
        else "⚠️ RS below Nifty (beta-driven)"
    )
    price_str = f"₹{price:,.0f}"

    def stock_line(emoji, action_text, extra_note=""):
        note = f"\n📝 {extra_note}" if extra_note else ""
        return (
            f"\n<b>{ticker}</b>\n"
            f"💵 CMP      : {price_str}\n"
            f"📐 EMA Gap  : {gap:.1f}%\n"
            f"📊 RSI      : {rsi_label}\n"
            f"📦 Volume   : {vol_label}\n"
            f"📈 RS(20d)  : {rs_tag}\n"
            f"👉 {emoji} {action_text}{note}"
        )

    if data["golden_cross"]:
        note = (
            "RSI ≥ 75 — tighten stop / consider trimming"
            if rsi >= RSI_EXIT
            else ""
        )
        return "exit", stock_line(
            "🌟",
            "Golden Cross confirmed. Stay invested. Trail stop below 50 EMA.",
            note,
        )

    if gap >= 0:
        return None, ""

    alerts_fired = []
    if data["gap_closing"]:
        if STAGE4_MIN <= gap <= STAGE4_MAX:
            alerts_fired.append("wait")
        elif STAGE3_MIN <= gap <= STAGE3_MAX:
            alerts_fired.append("aggr")
        elif STAGE2_MIN <= gap <= STAGE2_MAX:
            alerts_fired.append("accum")
        elif STAGE1_MIN <= gap <= STAGE1_MAX:
            alerts_fired.append("watch")

    if data["breakout_6m_confirmed"] or data["breakout_3m_confirmed"]:
        alerts_fired.append("breakout")
    if data["trendline_breakout"]:
        alerts_fired.append("trendline_break")
    elif data["near_support"]:
        alerts_fired.append("support")
    if data["coiling"]:
        alerts_fired.append("coiling")

    if not alerts_fired:
        return None, ""

    has_breakout = any(
        item in alerts_fired for item in ["breakout", "trendline_break"]
    )
    has_support = "support" in alerts_fired

    rsi_note = ""
    if rsi > RSI_OVERBOUGHT:
        rsi_note = (
            "RSI overbought — size in gradually, don't chase the full "
            "position at once."
        )
    elif rsi < RSI_OVERSOLD:
        rsi_note = "RSI below 30 — verify no breakdown before entry."

    if has_breakout:
        return "buy_now", stock_line(
            "🔥", "Buy today. Breakout confirmed with volume.", rsi_note
        )

    support_note = " Also sitting near a support zone." if has_support else ""

    if "aggr" in alerts_fired:
        return "aggr", stock_line(
            "📈",
            f"Add aggressively. Cross expected in 2-4 weeks.{support_note}",
            rsi_note,
        )

    if "accum" in alerts_fired:
        return "accum", stock_line(
            "🟡",
            f"Start building. Buy 30-40% of planned amount.{support_note}",
            rsi_note,
        )

    if has_support:
        return "support", stock_line(
            "🛡", "Stock at support. Small entry with tight stop loss.", rsi_note
        )

    if "coiling" in alerts_fired:
        return "coiling", stock_line(
            "🌀",
            "UNCONFIRMED pre-breakout setup — squeeze + volume building near "
            "highs. Not yet backtest-validated, watch only.",
        )

    if "wait" in alerts_fired:
        return "wait", stock_line(
            "⏳", "Hold if invested. No fresh buying at this stage."
        )

    if "watch" in alerts_fired:
        return "watch", stock_line(
            "👁", "Too early to buy. Research fundamentals now."
        )

    return None, ""


# ============================================================
# Runner
# ============================================================

def run_scanner():
    log.info("=" * 55)
    log.info("  VISH_SCAN v7.3.1 — STARTING")
    log.info("=" * 55)

    buckets = {
        "exit": [],
        "buy_now": [],
        "aggr": [],
        "accum": [],
        "support": [],
        "coiling": [],
        "wait": [],
        "watch": [],
    }

    bucket_to_signal = {
        "buy_now": "Buy Now",
        "aggr": "Aggressive Accumulation",
        "accum": "Accumulate",
    }

    all_data = fetch_all_data(WATCHLIST)
    scanned = len(all_data)
    skipped = len(WATCHLIST) - scanned

    alerts = {}
    prices = {}

    for symbol, data in all_data.items():
        prices[symbol] = data["price"]
        bucket, line = classify(data)
        if bucket and bucket in buckets:
            buckets[bucket].append(line)
            log.info(
                "  %s → %s  gap=%s%%", symbol, bucket, data["gap_today"]
            )
            if bucket in bucket_to_signal:
                alerts[symbol] = {
                    "signal": bucket_to_signal[bucket],
                    "price": data["price"],
                }
        else:
            log.info("  %s → no alert  gap=%s%%", symbol, data["gap_today"])

    tracking_text = safe_tracking_report(alerts, prices)

    total = sum(len(values) for values in buckets.values())
    now_str = datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")

    if total == 0:
        body_lines = [
            f"✅ {scanned} stocks scanned ({skipped} skipped)",
            "😴 No actionable signals today.",
            "💡 Patience is the edge.",
        ]
        if tracking_text:
            body_lines.extend(tracking_text.splitlines())

        send_telegram_chunked(
            [
                "📋 <b>VISH_SCAN</b>",
                now_str,
                "━━━━━━━━━━━━━━━━━━━━━━",
                *body_lines,
            ]
        )
        return

    body_lines = []
    sections = [
        ("exit", "🌟 GOLDEN CROSS — Stay Invested & Trail Stop"),
        ("buy_now", "🔥 BUY NOW"),
        ("aggr", "📈 AGGRESSIVE ACCUMULATION"),
        ("accum", "🟡 ACCUMULATE"),
        ("support", "🛡 SUPPORT ZONE"),
        ("wait", "⏳ WAIT — Hold Only"),
        ("watch", "👁 WATCHLIST — Too Early"),
        (
            "coiling",
            "🌀 COILING — UNCONFIRMED, backtest still shows ~coin-flip odds",
        ),
    ]

    for key, heading in sections:
        if buckets[key]:
            body_lines.append("\n━━━━━━━━━━━━━━━━━━━━━━")
            body_lines.append(f"<b>{heading}</b>")
            body_lines.extend(buckets[key])

    body_lines.append("\n━━━━━━━━━━━━━━━━━━━━━━")
    body_lines.append("⚠️ Not SEBI advice. Personal use only.")

    if tracking_text:
        # Split the tracker report into actual lines. In v7.3 it was appended as
        # one oversized item, which could still breach Telegram's message limit.
        body_lines.extend(tracking_text.splitlines())

    header = [
        "📋 <b>VISH_SCAN</b>",
        now_str,
        f"{scanned} scanned ({skipped} skipped) · {total} alerts",
    ]

    send_telegram_chunked(header + body_lines)
    log.info("DONE — %d alerts sent.", total)
    log.info("=" * 55)


def _send_fatal_notice():
    """Best-effort notice for failures after this module has loaded."""
    try:
        timestamp = datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")
        send_telegram(
            "🚨 <b>VISH_SCAN FAILED</b>\n"
            f"{timestamp}\n"
            "No scanner report was completed. Check the GitHub Actions log."
        )
    except Exception:
        log.exception("Could not send fatal Telegram notice")


if __name__ == "__main__":
    try:
        run_scanner()
    except Exception:
        log.exception("VISH_SCAN failed")
        _send_fatal_notice()
        raise
