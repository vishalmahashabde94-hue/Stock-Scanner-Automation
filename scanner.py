"""
=======================================================
  VISH_SCAN — INDIAN STOCK SCANNER
  Final Version — EMA based, matches AngelOne/TradingView

  STRATEGY:
  Track 50 EMA closing towards 200 EMA from below.
  Buy quietly before the crowd discovers the stock.

  STAGES (50 EMA below 200 EMA, gap closing):
  Stage 1: 12-20% away → WATCHLIST
  Stage 2: 8-12% away  → ACCUMULATE
  Stage 3: 4-8% away   → AGGRESSIVE ACCUMULATION
  Stage 4: 2-4% away   → WAIT, do not buy
  Stage 5: Cross done  → GOLDEN CROSS, book profits

  SUPPORTING INDICATORS:
  RSI     — confirms oversold recovery
  Volume  — confirms smart money entering
  Momentum— confirms higher lows forming
=======================================================
"""

import os
import json
import time
import logging
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["CHAT_ID"]

STATE_FILE = "scanner_state.json"

WATCHLIST = [
    "HBLENGINE.NS",
    "PARASDYNE.NS",
    "ZENTEC.NS",
    "DATAPATTNS.NS",
    "MAZDOCK.NS",
    "M&M.NS",
    "ASHOKLEY.NS",
    "TVSMOTOR.NS",
    "BANCOINDIA.NS",
    "PRECWIRE.NS",
    "BHARTIARTL.NS",
    "SCHNEIDER.NS",
    "MOTILALOFS.NS",
    "ANGELONE.NS",
    "BAJFINANCE.NS",
    "AXISBANK.NS",
    "BSE.NS",
    "NSDL.NS",
    "ADANIGREEN.NS",
    "ADANIPOWER.NS",
    "ADANIPORTS.NS",
    "GOKULAGRO.NS",
    "VBL.NS",
    "LTFOODS.NS",
    "DIXON.NS",
    "LLOYDMETAL.NS",
    "NATCOPHARM.NS",
    "YATHARTH.NS",
    "KIIMS.NS",
    "NH.NS",
    "HAVELLS.NS",
]

# ── Stage thresholds ──────────────────────────────────────────────────────────
STAGE1_MIN = -20.0
STAGE1_MAX = -12.0
STAGE2_MIN = -12.0
STAGE2_MAX =  -8.0
STAGE3_MIN =  -8.0
STAGE3_MAX =  -4.0
STAGE4_MIN =  -4.0
STAGE4_MAX =  -2.0


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
        "chat_id":                  TELEGRAM_CHAT_ID,
        "text":                     message,
        "parse_mode":               "HTML",
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

def rsi_label(rsi):
    if rsi < 30:
        return f"🟢 {rsi:.1f} — Oversold (best zone)"
    elif rsi < 45:
        return f"🟢 {rsi:.1f} — Recovery zone (good)"
    elif rsi < 55:
        return f"🟡 {rsi:.1f} — Neutral"
    elif rsi < 70:
        return f"🟠 {rsi:.1f} — Heating up (caution)"
    else:
        return f"🔴 {rsi:.1f} — Overbought (avoid)"

def volume_label(vol_ratio):
    if vol_ratio >= 2.0:
        return f"🟢 {vol_ratio:.1f}x avg — Strong smart money"
    elif vol_ratio >= 1.3:
        return f"🟢 {vol_ratio:.1f}x avg — Volume building"
    elif vol_ratio >= 0.8:
        return f"🟡 {vol_ratio:.1f}x avg — Average volume"
    else:
        return f"🔴 {vol_ratio:.1f}x avg — Low volume (weak)"

def momentum_label(higher_lows, rsi_rising):
    if higher_lows and rsi_rising:
        return "🟢 Strong — Higher lows + RSI rising"
    elif higher_lows:
        return "🟡 Moderate — Higher lows forming"
    elif rsi_rising:
        return "🟡 Moderate — RSI recovering"
    else:
        return "🔴 Weak — No recovery pattern yet"

def fetch_stock_data(symbol):
    try:
        df = yf.download(
            symbol,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

        if df.empty or len(df) < 200:
            log.warning(f"  {symbol}: insufficient data ({len(df)} rows)")
            return None

        close  = df["Adj Close"].squeeze()
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

        vol_today      = float(volume.iloc[-1])
        vol_avg20      = float(volume.rolling(20).mean().iloc[-1])
        vol_ratio      = round(vol_today / vol_avg20, 2) if vol_avg20 > 0 else 0
        vol_avg_recent = float(volume.iloc[-10:].mean())
        vol_avg_older  = float(volume.iloc[-20:-10].mean())
        vol_increasing = vol_avg_recent > vol_avg_older

        lows = []
        for i in range(3):
            start = -(i + 1) * 5
            end   = -i * 5 if i > 0 else None
            chunk = close.iloc[start:end] if end else close.iloc[start:]
            lows.append(float(chunk.min()))
        higher_lows = lows[0] > lows[1] > lows[2]

        return {
            "symbol":         symbol,
            "price":          round(float(close.iloc[-1]), 2),
            "ema50":          round(ema50_today, 2),
            "ema200":         round(ema200_today, 2),
            "gap_today":      gap_today,
            "gap_prev":       gap_prev,
            "rsi":            rsi_today,
            "rsi_rising":     rsi_rising,
            "vol_ratio":      vol_ratio,
            "vol_increasing": vol_increasing,
            "higher_lows":    higher_lows,
        }

    except Exception as e:
        log.error(f"  {symbol}: error — {e}")
        return None


def classify_and_build_message(data):
    ticker  = data["symbol"].replace(".NS", "")
    gap     = data["gap_today"]
    gap_p   = data["gap_prev"]
    price   = data["price"]
    e50     = data["ema50"]
    e200    = data["ema200"]
    rsi     = data["rsi"]
    vol_r   = data["vol_ratio"]

    gap_closing = gap > gap_p

    rsi_line = rsi_label(rsi)
    vol_line = volume_label(vol_r)
    mom_line = momentum_label(data["higher_lows"], data["rsi_rising"])

    indicators = (
        f"📈 RSI      : {rsi_line}\n"
        f"📦 Volume   : {vol_line}\n"
        f"⚡ Momentum : {mom_line}"
    )

    # Golden Cross
    if gap_p < 0 and gap >= 0:
        msg = (
            f"🌟 <b>GOLDEN CROSS — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 Price   : ₹{price:,.2f}\n"
            f"📊 50 EMA  : ₹{e50:,.2f}\n"
            f"📉 200 EMA : ₹{e200:,.2f}\n"
            f"📐 Gap     : {gap:+.2f}% (was {gap_p:+.2f}%)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{indicators}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏁 <b>50 EMA just crossed ABOVE 200 EMA</b>\n"
            f"💰 If you accumulated earlier → BOOK PROFITS\n"
            f"🚫 Do NOT buy fresh at this stage\n"
            f"📌 Stock likely to run up — exit in parts"
        )
        return "golden_cross", msg

    # Already crossed — skip
    if gap >= 0:
        return None, ""

    # Stage 4 — Wait Zone
    if STAGE4_MIN <= gap <= STAGE4_MAX and gap_closing:
        msg = (
            f"⏳ <b>WAIT ZONE — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 Price   : ₹{price:,.2f}\n"
            f"📊 50 EMA  : ₹{e50:,.2f}\n"
            f"📉 200 EMA : ₹{e200:,.2f}\n"
            f"📐 Gap     : {gap:.2f}%  ↗ Almost closed\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{indicators}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ <b>Only 2-4% from Golden Cross</b>\n"
            f"🚫 Do NOT buy fresh — risk/reward unfavourable\n"
            f"✅ If already holding → stay invested\n"
            f"👁 Golden Cross expected very soon"
        )
        return "stage4", msg

    # Stage 3 — Aggressive Accumulation
    if STAGE3_MIN <= gap <= STAGE3_MAX and gap_closing:
        msg = (
            f"🔥 <b>AGGRESSIVE ACCUMULATION — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 Price   : ₹{price:,.2f}\n"
            f"📊 50 EMA  : ₹{e50:,.2f}\n"
            f"📉 200 EMA : ₹{e200:,.2f}\n"
            f"📐 Gap     : {gap:.2f}%  ↗ Closing fast\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{indicators}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💎 <b>4-8% from Golden Cross</b>\n"
            f"✅ Strong buy zone — add aggressively\n"
            f"📌 Golden Cross likely in 2-4 weeks\n"
            f"💡 Risk is low, reward is high at this stage"
        )
        return "stage3", msg

    # Stage 2 — Accumulate
    if STAGE2_MIN <= gap <= STAGE2_MAX and gap_closing:
        msg = (
            f"🟡 <b>ACCUMULATE — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 Price   : ₹{price:,.2f}\n"
            f"📊 50 EMA  : ₹{e50:,.2f}\n"
            f"📉 200 EMA : ₹{e200:,.2f}\n"
            f"📐 Gap     : {gap:.2f}%  ↗ Closing\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{indicators}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 <b>8-12% from Golden Cross</b>\n"
            f"✅ Start building position in parts\n"
            f"📌 Buy 30-40% of planned amount now\n"
            f"💡 Average down if price dips further"
        )
        return "stage2", msg

    # Stage 1 — Watchlist
    if STAGE1_MIN <= gap <= STAGE1_MAX and gap_closing:
        msg = (
            f"👁 <b>WATCHLIST — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 Price   : ₹{price:,.2f}\n"
            f"📊 50 EMA  : ₹{e50:,.2f}\n"
            f"📉 200 EMA : ₹{e200:,.2f}\n"
            f"📐 Gap     : {gap:.2f}%  ↗ Starting to close\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{indicators}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📋 <b>12-20% from Golden Cross</b>\n"
            f"⏳ Too early to buy — keep on radar\n"
            f"📌 Alert when gap closes to 8-12%\n"
            f"💡 Research the stock fundamentals now"
        )
        return "stage1", msg

    return None, ""


def run_scanner():
    log.info("=" * 55)
    log.info("  VISH_SCAN — STARTING")
    log.info("=" * 55)

    saved_state = load_state()
    new_state   = {}

    alerts = {
        "golden_cross": [],
        "stage4":       [],
        "stage3":       [],
        "stage2":       [],
        "stage1":       [],
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

        stage_key, message = classify_and_build_message(data)

        if stage_key:
            alerts[stage_key].append(message)
            log.info(f"  ALERT → {stage_key}  gap={data['gap_today']}%")
        else:
            log.info(f"  No alert  gap={data['gap_today']}%")

        time.sleep(0.8)

    save_state(new_state)

    total_alerts = sum(len(v) for v in alerts.values())
    now = datetime.now().strftime("%d %b %Y, %I:%M %p")

    if total_alerts == 0:
        send_telegram(
            f"📋 <b>VISH_SCAN Report — {now}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ Scanned : {scanned} stocks\n"
            f"😴 No stocks in any alert zone today.\n"
            f"🔁 Next scan in ~2 days.\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 Patience is the edge. Wait for the right setup."
        )
        return

    send_telegram(
        f"📋 <b>VISH_SCAN Report — {now}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📦 Scanned             : {scanned} stocks\n"
        f"🚨 Total Alerts        : {total_alerts}\n\n"
        f"🌟 Golden Cross        : {len(alerts['golden_cross'])}  → Book profits\n"
        f"⏳ Wait Zone  (2-4%)   : {len(alerts['stage4'])}  → Hold only\n"
        f"🔥 Aggr. Accum (4-8%)  : {len(alerts['stage3'])}  → Strong buy\n"
        f"🟡 Accumulate (8-12%)  : {len(alerts['stage2'])}  → Start buying\n"
        f"👁  Watchlist (12-20%)  : {len(alerts['stage1'])}  → Monitor only\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"↓ Individual stock alerts below ↓"
    )
    time.sleep(1)

    for stage in ["golden_cross", "stage3", "stage2", "stage4", "stage1"]:
        for msg in alerts[stage]:
            send_telegram(msg)
            time.sleep(0.8)

    log.info(f"DONE — {total_alerts} alerts sent to Telegram.")
    log.info("=" * 55)


if __name__ == "__main__":
    run_scanner()