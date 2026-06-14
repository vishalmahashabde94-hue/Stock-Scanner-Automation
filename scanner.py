"""
=======================================================
  VISH_SCAN — SCANNER v5 FINAL
  Clean optimised Telegram output
  Fixed action text per stage
  EMA + RSI + Volume + Breakout + Trendline
=======================================================
"""

import os
import json
import time
import logging
import requests
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["CHAT_ID"]

STATE_FILE = "scanner_state.json"
IST = timezone(timedelta(hours=5, minutes=30))

WATCHLIST = [
    # Defence & Aerospace
    "HBLENGINE.NS",
    "PARASDYNE.NS",
    "ZENTEC.NS",
    "DATAPATTNS.NS",
    "MAZDOCK.NS",
    "HAL.NS",
    "BEL.NS",
    "ASTRAMICRO.NS",
    "BHARATFORG.NS",
    "MTARTECH.NS",
    "IDEAFORGE.NS",
    # Auto & Ancillary
    "M&M.NS",
    "ASHOKLEY.NS",
    "TVSMOTOR.NS",
    "BANCOINDIA.NS",
    "PRECWIRE.NS",
    "MOTHERSON.NS",
    "ENDURANCE.NS",
    "TIINDIA.NS",
    "MOTHERSONSUM.NS",
    # Electricals & Power
    "HAVELLS.NS",
    "POLYCAB.NS",
    "KEIIND.NS",
    "SCHNEIDER.NS",
    "CGPOWER.NS",
    "TRANSRAILL.NS",
    "TRIL.NS",
    "VOLTAMP.NS",
    "GVTD.NS",
    # Engineering & Industrial
    "TRITURBINE.NS",
    "TDPOWERSYS.NS",
    "IONEXCHANG.NS",
    "TITAGARH.NS",
    "KEC.NS",
    "LT.NS",
    # Renewable Energy
    "ADANIGREEN.NS",
    "ADANIPOWER.NS",
    "JSWENERGY.NS",
    "INOXWIND.NS",
    "WAAREERTL.NS",
    # Infrastructure & EPC
    "KPIL.NS",
    "JWL.NS",
    # Technology & Electronics
    "INFY.NS",
    "WIPRO.NS",
    "DIXON.NS",
    "REDINGTON.NS",
    "KAYNES.NS",
    "NETWEB.NS",
    "BBOX.NS",
    # Telecom
    "BHARTIARTL.NS",
    # Financial Services
    "MOTILALOFS.NS",
    "ANGELONE.NS",
    "BAJFINANCE.NS",
    "AXISBANK.NS",
    "BSE.NS",
    "NSDL.NS",
    # Adani Group
    "ADANIPORTS.NS",
    # Real Estate
    "ANANTRAJ.NS",
    # Food & Beverages
    "GOKULAGRO.NS",
    "VBL.NS",
    "LTFOODS.NS",
    "RADICO.NS",
    # Metals & Mining
    "LLOYDMETAL.NS",
    "COALINDIA.NS",
    "RELIANCE.NS",
    # Pharma
    "NATCOPHARM.NS",
    # Healthcare
    "YATHARTH.NS",
    "KIIMS.NS",
    "KIMS.NS",
    "NH.NS",
    # Ceramics
    "KAJARIACER.NS",
    # Logistics
    "AEGISLOG.NS",
    # Others
    "PRICOLLTD.NS",
    "CCL.NS",
]

STAGE1_MIN = -20.0
STAGE1_MAX = -12.0
STAGE2_MIN = -12.0
STAGE2_MAX =  -8.0
STAGE3_MIN =  -8.0
STAGE3_MAX =  -4.0
STAGE4_MIN =  -4.0
STAGE4_MAX =  -2.0

BREAKOUT_3M_DAYS = 63
BREAKOUT_6M_DAYS = 126
VOLUME_CONFIRM   = 1.5
SUPPORT_ZONE     = 0.02


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        log.info("  Telegram: sent")
    except Exception as e:
        log.error(f"  Telegram: FAILED — {e}")


def compute_rsi(close, period=14):
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def rsi_simple(rsi):
    """Simple 1-2 word RSI label."""
    if rsi < 30:
        return f"{rsi:.0f} — Oversold"
    elif rsi < 45:
        return f"{rsi:.0f} — Recovering"
    elif rsi < 55:
        return f"{rsi:.0f} — Neutral"
    elif rsi < 70:
        return f"{rsi:.0f} — Heated"
    else:
        return f"{rsi:.0f} — Overbought"


def vol_simple(vol_ratio):
    """Simple volume label."""
    if vol_ratio >= 1.5:
        return f"High ({vol_ratio:.1f}x)"
    elif vol_ratio >= 0.8:
        return f"OK ({vol_ratio:.1f}x)"
    else:
        return f"Low ({vol_ratio:.1f}x)"


def detect_trendline(prices):
    x = np.arange(len(prices))
    y = np.array(prices)
    slope, intercept = np.polyfit(x, y, 1)
    current = slope * (len(prices) - 1) + intercept
    return slope, current


def fetch_stock_data(symbol):
    try:
        df = yf.download(
            symbol,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if df.empty or len(df) < 200:
            log.warning(f"  {symbol}: insufficient data ({len(df)} rows)")
            return None

        close  = df["Close"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        ema50_today  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
        ema200_today = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
        ema50_prev   = float(close.ewm(span=50,  adjust=False).mean().iloc[-2])
        ema200_prev  = float(close.ewm(span=200, adjust=False).mean().iloc[-2])

        if ema200_today == 0 or ema200_prev == 0:
            return None

        gap_today = round((ema50_today - ema200_today) / ema200_today * 100, 2)
        gap_prev  = round((ema50_prev  - ema200_prev)  / ema200_prev  * 100, 2)

        rsi_series = compute_rsi(close)
        rsi_today  = round(float(rsi_series.iloc[-1]), 1)
        rsi_3ago   = round(float(rsi_series.iloc[-3]), 1)
        rsi_rising = rsi_today > rsi_3ago

        vol_today  = float(volume.iloc[-1])
        vol_avg20  = float(volume.rolling(20).mean().iloc[-1])
        vol_ratio  = round(vol_today / vol_avg20, 2) if vol_avg20 > 0 else 0

        lows = []
        for i in range(3):
            start = -(i + 1) * 5
            end   = -i * 5 if i > 0 else None
            chunk = close.iloc[start:end] if end else close.iloc[start:]
            lows.append(float(chunk.min()))
        higher_lows = lows[0] > lows[1] > lows[2]

        price_today = float(close.iloc[-1])
        price_prev  = float(close.iloc[-2])

        high_3m = float(high.iloc[-BREAKOUT_3M_DAYS:-1].max())
        high_6m = float(high.iloc[-BREAKOUT_6M_DAYS:-1].max())

        breakout_3m = price_today > high_3m and price_prev <= high_3m
        breakout_6m = price_today > high_6m and price_prev <= high_6m

        breakout_3m_confirmed = breakout_3m and vol_ratio >= VOLUME_CONFIRM
        breakout_6m_confirmed = breakout_6m and vol_ratio >= VOLUME_CONFIRM

        recent_lows = low.iloc[-BREAKOUT_3M_DAYS:].values
        slope, trendline_support = detect_trendline(recent_lows)

        near_support = (
            slope > 0 and
            gap_today < 0 and
            abs(price_today - trendline_support) / trendline_support <= SUPPORT_ZONE
        )

        recent_highs = high.iloc[-BREAKOUT_3M_DAYS:].values
        _, trendline_resistance = detect_trendline(recent_highs)
        trendline_breakout = (
            price_today > trendline_resistance and
            price_prev <= trendline_resistance and
            vol_ratio >= VOLUME_CONFIRM
        )

        return {
            "symbol":                symbol,
            "price":                 round(price_today, 2),
            "ema50":                 round(ema50_today, 2),
            "ema200":                round(ema200_today, 2),
            "gap_today":             gap_today,
            "gap_prev":              gap_prev,
            "rsi":                   rsi_today,
            "rsi_rising":            rsi_rising,
            "vol_ratio":             vol_ratio,
            "higher_lows":           higher_lows,
            "high_3m":               round(high_3m, 2),
            "high_6m":               round(high_6m, 2),
            "breakout_3m_confirmed": breakout_3m_confirmed,
            "breakout_6m_confirmed": breakout_6m_confirmed,
            "near_support":          near_support,
            "trendline_support":     round(trendline_support, 2),
            "trendline_breakout":    trendline_breakout,
        }

    except Exception as e:
        log.error(f"  {symbol}: error — {e}")
        return None


def classify(data):
    """
    Classify stock and return (bucket, formatted_line)
    Buckets: exit, buy_now, aggr, accum, accum_support,
             support, wait, watch
    """
    ticker  = data["symbol"].replace(".NS", "")
    gap     = data["gap_today"]
    gap_p   = data["gap_prev"]
    price   = data["price"]
    rsi     = data["rsi"]
    vol_r   = data["vol_ratio"]

    gap_closing = gap > gap_p

    rsi_lbl = rsi_simple(rsi)
    vol_lbl = vol_simple(vol_r)

    price_str = f"₹{price:,.0f}"

    def stock_line(action_emoji, action_text):
        return (
            f"\n<b>{ticker}</b>\n"
            f"💵 CMP      : {price_str}\n"
            f"📐 EMA Gap  : {gap:.1f}%\n"
            f"📊 RSI      : {rsi_lbl}\n"
            f"📦 Volume   : {vol_lbl}\n"
            f"👉 {action_emoji} {action_text}"
        )

    # ── Golden Cross ──────────────────────────────────────────────────────────
    if gap_p < 0 and gap >= 0:
        return "exit", stock_line(
            "💰", "Exit position. Golden Cross complete. Book profits."
        )

    # Skip stocks already above 200 EMA
    if gap >= 0:
        return None, ""

    alerts_fired = []

    if STAGE4_MIN <= gap <= STAGE4_MAX and gap_closing:
        alerts_fired.append("wait")
    elif STAGE3_MIN <= gap <= STAGE3_MAX and gap_closing:
        alerts_fired.append("aggr")
    elif STAGE2_MIN <= gap <= STAGE2_MAX and gap_closing:
        alerts_fired.append("accum")
    elif STAGE1_MIN <= gap <= STAGE1_MAX and gap_closing:
        alerts_fired.append("watch")

    if data["breakout_6m_confirmed"] or data["breakout_3m_confirmed"]:
        alerts_fired.append("breakout")

    if data["trendline_breakout"]:
        alerts_fired.append("trendline_break")
    elif data["near_support"]:
        alerts_fired.append("support")

    if not alerts_fired:
        return None, ""

    has_accum    = any(x in alerts_fired for x in ["aggr", "accum"])
    has_breakout = any(x in alerts_fired for x in ["breakout", "trendline_break"])
    has_support  = "support" in alerts_fired

    # Best combination first
    if has_breakout:
        return "buy_now", stock_line(
            "🔥", "Buy today. Breakout confirmed with volume."
        )

    if has_accum and has_support:
        return "accum_support", stock_line(
            "📈", "Low risk entry. Buy at current levels."
        )

    if "aggr" in alerts_fired:
        return "aggr", stock_line(
            "📈", "Add aggressively. Cross expected in 2-4 weeks."
        )

    if "accum" in alerts_fired:
        return "accum", stock_line(
            "📈", "Start building. Buy 30-40% of planned amount."
        )

    if has_support:
        return "support", stock_line(
            "🛡", "Stock at support. Small entry with tight stop loss."
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


def run_scanner():
    log.info("=" * 55)
    log.info("  VISH_SCAN v5 FINAL — STARTING")
    log.info("=" * 55)

    saved_state = load_state()
    new_state   = {}

    buckets = {
        "exit":         [],
        "buy_now":      [],
        "aggr":         [],
        "accum":        [],
        "accum_support":[],
        "support":      [],
        "wait":         [],
        "watch":        [],
    }

    scanned = 0
    skipped = 0

    for symbol in WATCHLIST:
        log.info(f"Scanning {symbol}...")
        data = fetch_stock_data(symbol)

        if data is None:
            skipped += 1
            time.sleep(1)
            continue

        scanned += 1
        new_state[symbol] = {"gap_pct": data["gap_today"]}
        bucket, line = classify(data)

        if bucket and bucket in buckets:
            buckets[bucket].append(line)
            log.info(f"  → {bucket}  gap={data['gap_today']}%")
        else:
            log.info(f"  → No alert  gap={data['gap_today']}%")

        time.sleep(0.8)

    save_state(new_state)

    total = sum(len(v) for v in buckets.values())
    now_ist = datetime.now(IST)
    hour    = now_ist.hour
    session = "🌅 Pre Market" if hour < 12 else "🌆 Post Market"
    now_str = now_ist.strftime("%d %b %Y, %I:%M %p IST")

    if total == 0:
        send_telegram(
            f"📋 <b>VISH_SCAN {session}</b>\n"
            f"{now_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ {scanned} stocks scanned\n"
            f"😴 No actionable signals today.\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 Patience is the edge."
        )
        return

    # Build single clean message
    msg = []
    msg.append(f"📋 <b>VISH_SCAN {session}</b>")
    msg.append(f"{now_str}")
    msg.append(f"{scanned} stocks scanned · {total} alerts")

    sections = [
        ("exit",          "💰 EXIT — Book Profits"),
        ("buy_now",       "🔥 BUY NOW"),
        ("aggr",          "📈 AGGRESSIVE ACCUMULATION"),
        ("accum_support", "📈 ACCUMULATE + SUPPORT"),
        ("accum",         "📈 ACCUMULATE"),
        ("support",       "🛡 SUPPORT ZONE"),
        ("wait",          "⏳ WAIT — Hold Only"),
        ("watch",         "👁 WATCHLIST — Too Early"),
    ]

    for key, heading in sections:
        if buckets[key]:
            msg.append(f"\n━━━━━━━━━━━━━━━━━━━━━━")
            msg.append(f"<b>{heading}</b>")
            for line in buckets[key]:
                msg.append(line)

    msg.append(f"\n━━━━━━━━━━━━━━━━━━━━━━")
    msg.append(f"⚠️ Not SEBI advice. Personal use only.")

    send_telegram("\n".join(msg))
    log.info(f"DONE — {total} alerts sent.")
    log.info("=" * 55)


if __name__ == "__main__":
    run_scanner()
