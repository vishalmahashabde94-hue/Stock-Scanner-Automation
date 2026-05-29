"""
=======================================================
  VISH_SCAN — COMBINED SCANNER v4
  EMA Gap + Resistance Breakout + RSI + Volume
  61 stocks — Final Version
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
    # ── Defence & Aerospace ───────────────────────────────
    "HBLENGINE.NS",      # HBL Engineering
    "PARASDYNE.NS",      # Paras Defence
    "ZENTEC.NS",         # Zen Technologies
    "DATAPATTNS.NS",     # Data Patterns
    "MAZDOCK.NS",        # Mazagon Dock
    "HAL.NS",            # Hindustan Aeronautics
    "BEL.NS",            # Bharat Electronics
    "ASTRAMICRO.NS",     # Astra Microwave

    # ── Auto & Auto Ancillary ─────────────────────────────
    "M&M.NS",            # Mahindra & Mahindra
    "ASHOKLEY.NS",       # Ashok Leyland
    "TVSMOTOR.NS",       # TVS Motor
    "BANCOINDIA.NS",     # Banco Products
    "PRECWIRE.NS",       # Precision Wires
    "MOTHERSON.NS",      # Samvardhana Motherson
    "ENDURANCE.NS",      # Endurance Technologies
    "TIINDIA.NS",        # Tube Investments of India
    "BHARATFORG.NS",     # Bharat Forge

    # ── Electricals & Power ───────────────────────────────
    "HAVELLS.NS",        # Havells India
    "POLYCAB.NS",        # Polycab India
    "KEIIND.NS",         # KEI Industries
    "SCHNEIDER.NS",      # Schneider Electric
    "CGPOWER.NS",        # CG Power
    "TRANSRAILL.NS",     # Transrail Lighting
    "TRIL.NS",           # Transformer & Rectifier

    # ── Engineering & Industrial ──────────────────────────
    "TRITURBINE.NS",     # Triveni Turbine
    "TDPOWERSYS.NS",     # TD Power Systems
    "IONEXCHANG.NS",     # Ion Exchange India
    "TITAGARH.NS",       # Titagarh Rail Systems

    # ── Infrastructure & EPC ─────────────────────────────
    "KPIL.NS",           # Kalpataru Projects
    "JWL.NS",            # Jupiter Wagons

    # ── Technology ───────────────────────────────────────
    "INFY.NS",           # Infosys
    "WIPRO.NS",          # Wipro
    "DIXON.NS",          # Dixon Technologies
    "REDINGTON.NS",      # Redington India

    # ── Telecom ──────────────────────────────────────────
    "BHARTIARTL.NS",     # Bharti Airtel

    # ── Financial Services ────────────────────────────────
    "MOTILALOFS.NS",     # Motilal Oswal
    "ANGELONE.NS",       # Angel One
    "BAJFINANCE.NS",     # Bajaj Finance
    "AXISBANK.NS",       # Axis Bank
    "BSE.NS",            # BSE Ltd
    "NSDL.NS",           # NSDL

    # ── Adani Group ───────────────────────────────────────
    "ADANIGREEN.NS",     # Adani Green
    "ADANIPOWER.NS",     # Adani Power
    "ADANIPORTS.NS",     # Adani Ports

    # ── Food & Beverages ──────────────────────────────────
    "GOKULAGRO.NS",      # Gokul Agro
    "VBL.NS",            # Varun Beverages
    "LTFOODS.NS",        # LT Foods
    "RADICO.NS",         # Radico Khaitan

    # ── Metals & Mining ───────────────────────────────────
    "LLOYDMETAL.NS",     # Lloyd Metals
    "COALINDIA.NS",      # Coal India
    "RELIANCE.NS",       # Reliance Industries

    # ── Pharma ───────────────────────────────────────────
    "NATCOPHARM.NS",     # Natco Pharma

    # ── Healthcare ───────────────────────────────────────
    "YATHARTH.NS",       # Yatharth Hospital
    "KIIMS.NS",          # KIIMS
    "KIMS.NS",           # Krishna Institute
    "NH.NS",             # Narayana Hrudayalaya

    # ── Ceramics & Building Materials ────────────────────
    "KAJARIACER.NS",     # Kajaria Ceramics

    # ── Logistics ────────────────────────────────────────
    "AEGISLOG.NS",       # Aegis Logistics

    # ── Others ───────────────────────────────────────────
    "PRICOLLTD.NS",      # Pricol
    "CCL.NS",            # CCL Products
    "IDEAFORGE.NS",      # Ideaforge
    "MOTHERSONSUM.NS",   # Motherson Sumi
]

# EMA Stage thresholds
STAGE1_MIN = -20.0
STAGE1_MAX = -12.0
STAGE2_MIN = -12.0
STAGE2_MAX =  -8.0
STAGE3_MIN =  -8.0
STAGE3_MAX =  -4.0
STAGE4_MIN =  -4.0
STAGE4_MAX =  -2.0

# Breakout thresholds
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


def rsi_label(rsi):
    if rsi < 30:
        return f"🟢 {rsi:.1f} — Oversold (best zone)"
    elif rsi < 45:
        return f"🟢 {rsi:.1f} — Recovery zone (good)"
    elif rsi < 55:
        return f"🟡 {rsi:.1f} — Neutral"
    elif rsi < 70:
        return f"🟠 {rsi:.1f} — Heating up"
    else:
        return f"🔴 {rsi:.1f} — Overbought (avoid)"


def volume_label(vol_ratio):
    if vol_ratio >= 2.0:
        return f"🟢 {vol_ratio:.1f}x — Strong smart money"
    elif vol_ratio >= 1.5:
        return f"🟢 {vol_ratio:.1f}x — Volume building"
    elif vol_ratio >= 0.8:
        return f"🟡 {vol_ratio:.1f}x — Average volume"
    else:
        return f"🔴 {vol_ratio:.1f}x — Low volume (weak)"


def momentum_label(higher_lows, rsi_rising):
    if higher_lows and rsi_rising:
        return "🟢 Strong — Higher lows + RSI rising"
    elif higher_lows:
        return "🟡 Moderate — Higher lows forming"
    elif rsi_rising:
        return "🟡 Moderate — RSI recovering"
    else:
        return "🔴 Weak — No recovery pattern yet"


def detect_trendline(prices):
    x = np.arange(len(prices))
    y = np.array(prices)
    slope, intercept = np.polyfit(x, y, 1)
    current_trendline = slope * (len(prices) - 1) + intercept
    return slope, current_trendline


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
            "vol_increasing":        vol_increasing,
            "higher_lows":           higher_lows,
            "high_3m":               round(high_3m, 2),
            "high_6m":               round(high_6m, 2),
            "breakout_3m":           breakout_3m,
            "breakout_6m":           breakout_6m,
            "breakout_3m_confirmed": breakout_3m_confirmed,
            "breakout_6m_confirmed": breakout_6m_confirmed,
            "near_support":          near_support,
            "trendline_support":     round(trendline_support, 2),
            "trendline_breakout":    trendline_breakout,
        }

    except Exception as e:
        log.error(f"  {symbol}: error — {e}")
        return None


def classify_and_build_message(data):
    ticker = data["symbol"].replace(".NS", "")
    gap    = data["gap_today"]
    gap_p  = data["gap_prev"]
    price  = data["price"]
    e50    = data["ema50"]
    e200   = data["ema200"]
    rsi    = data["rsi"]
    vol_r  = data["vol_ratio"]

    gap_closing = gap > gap_p

    rsi_line = rsi_label(rsi)
    vol_line = volume_label(vol_r)
    mom_line = momentum_label(data["higher_lows"], data["rsi_rising"])

    indicators = (
        f"📈 RSI      : {rsi_line}\n"
        f"📦 Volume   : {vol_line}\n"
        f"⚡ Momentum : {mom_line}"
    )

    alerts_fired = []

    if gap_p < 0 and gap >= 0:
        msg = (
            f"🌟 <b>GOLDEN CROSS — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 Price   : ₹{price:,.2f}\n"
            f"📊 50 EMA  : ₹{e50:,.2f}\n"
            f"📉 200 EMA : ₹{e200:,.2f}\n"
            f"📐 Gap     : {gap:+.2f}%\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{indicators}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏁 <b>50 EMA crossed ABOVE 200 EMA</b>\n"
            f"💰 Book profits if already holding\n"
            f"🚫 Do NOT buy fresh at this stage"
        )
        return "golden_cross", msg

    if gap >= 0:
        pass
    else:
        if STAGE4_MIN <= gap <= STAGE4_MAX and gap_closing:
            alerts_fired.append("ema_stage4")
        elif STAGE3_MIN <= gap <= STAGE3_MAX and gap_closing:
            alerts_fired.append("ema_stage3")
        elif STAGE2_MIN <= gap <= STAGE2_MAX and gap_closing:
            alerts_fired.append("ema_stage2")
        elif STAGE1_MIN <= gap <= STAGE1_MAX and gap_closing:
            alerts_fired.append("ema_stage1")

    if data["breakout_6m_confirmed"]:
        alerts_fired.append("breakout_6m")
    elif data["breakout_3m_confirmed"]:
        alerts_fired.append("breakout_3m")
    elif data["breakout_6m"]:
        alerts_fired.append("breakout_6m_weak")
    elif data["breakout_3m"]:
        alerts_fired.append("breakout_3m_weak")

    if data["trendline_breakout"]:
        alerts_fired.append("trendline_breakout")
    elif data["near_support"]:
        alerts_fired.append("trendline_support")

    if not alerts_fired:
        return None, ""

    has_ema_accumulate = any(x in alerts_fired for x in
                             ["ema_stage2", "ema_stage3"])
    has_confirmed_breakout = any(x in alerts_fired for x in
                                 ["breakout_6m", "breakout_3m",
                                  "trendline_breakout"])
    has_support = "trendline_support" in alerts_fired

    signal_lines = []

    if "ema_stage4" in alerts_fired:
        signal_lines.append("⏳ EMA: Wait Zone (2-4% from cross)")
    elif "ema_stage3" in alerts_fired:
        signal_lines.append("🔥 EMA: Aggressive Accum (4-8% from cross)")
    elif "ema_stage2" in alerts_fired:
        signal_lines.append("🟡 EMA: Accumulate (8-12% from cross)")
    elif "ema_stage1" in alerts_fired:
        signal_lines.append("👁 EMA: Watchlist (12-20% from cross)")

    if "breakout_6m" in alerts_fired:
        signal_lines.append(
            f"🚀 6M Breakout: ₹{data['high_6m']:,.2f} broken ✅ Volume confirmed")
    elif "breakout_3m" in alerts_fired:
        signal_lines.append(
            f"📈 3M Breakout: ₹{data['high_3m']:,.2f} broken ✅ Volume confirmed")
    elif "breakout_6m_weak" in alerts_fired:
        signal_lines.append(
            f"⚠️ 6M Breakout: ₹{data['high_6m']:,.2f} broken ⚠️ Low volume")
    elif "breakout_3m_weak" in alerts_fired:
        signal_lines.append(
            f"⚠️ 3M Breakout: ₹{data['high_3m']:,.2f} broken ⚠️ Low volume")

    if "trendline_breakout" in alerts_fired:
        signal_lines.append("🚀 Trendline: Resistance broken ✅ Volume confirmed")
    elif "trendline_support" in alerts_fired:
        signal_lines.append(
            f"🛡 Trendline: Support at ₹{data['trendline_support']:,.2f}")

    signal_text = "\n".join(signal_lines)

    if has_ema_accumulate and has_confirmed_breakout:
        verdict = "🔥🔥 <b>HIGHEST CONVICTION BUY</b>"
        advice  = "EMA recovery + Breakout + Volume. Rare setup. Act now."
        key     = "highest"
    elif has_confirmed_breakout and has_support:
        verdict = "🚀 <b>STRONG BREAKOUT + SUPPORT</b>"
        advice  = "Breakout with trendline support. Strong risk/reward."
        key     = "strong_breakout"
    elif has_confirmed_breakout:
        verdict = "📈 <b>CONFIRMED BREAKOUT</b>"
        advice  = "Volume confirmed breakout. Good momentum entry."
        key     = "breakout"
    elif has_ema_accumulate and has_support:
        verdict = "💎 <b>ACCUMULATE + SUPPORT</b>"
        advice  = "EMA recovery zone + trendline support. Low risk entry."
        key     = "accumulate_support"
    elif "ema_stage3" in alerts_fired:
        verdict = "🔥 <b>AGGRESSIVE ACCUMULATION</b>"
        advice  = "4-8% from Golden Cross. Strong buy zone."
        key     = "stage3"
    elif "ema_stage2" in alerts_fired:
        verdict = "🟡 <b>ACCUMULATE</b>"
        advice  = "8-12% from Golden Cross. Start building position."
        key     = "stage2"
    elif "trendline_breakout" in alerts_fired:
        verdict = "🚀 <b>TRENDLINE BREAKOUT</b>"
        advice  = "Resistance trendline broken with volume. Good entry."
        key     = "trendline_breakout"
    elif "trendline_support" in alerts_fired:
        verdict = "🛡 <b>TRENDLINE SUPPORT</b>"
        advice  = "Stock at support. Good risk/reward for entry."
        key     = "support"
    elif "ema_stage4" in alerts_fired:
        verdict = "⏳ <b>WAIT ZONE</b>"
        advice  = "2-4% from Golden Cross. Hold if invested. No fresh buy."
        key     = "stage4"
    else:
        verdict = "👁 <b>WATCHLIST</b>"
        advice  = "Early signals. Keep on radar."
        key     = "stage1"

    msg = (
        f"{verdict} — <b>{ticker}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 Price   : ₹{price:,.2f}\n"
        f"📊 50 EMA  : ₹{e50:,.2f}\n"
        f"📉 200 EMA : ₹{e200:,.2f}\n"
        f"📐 EMA Gap : {gap:.2f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Signals Detected:</b>\n"
        f"{signal_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{indicators}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 {advice}"
    )
    return key, msg


def run_scanner():
    log.info("=" * 55)
    log.info("  VISH_SCAN COMBINED — STARTING")
    log.info("=" * 55)

    saved_state = load_state()
    new_state   = {}

    alerts = {
        "golden_cross":       [],
        "highest":            [],
        "strong_breakout":    [],
        "breakout":           [],
        "accumulate_support": [],
        "stage3":             [],
        "stage2":             [],
        "trendline_breakout": [],
        "support":            [],
        "stage4":             [],
        "stage1":             [],
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
        key, message = classify_and_build_message(data)

        if key:
            if key in alerts:
                alerts[key].append(message)
            log.info(f"  ALERT → {key}  gap={data['gap_today']}%")
        else:
            log.info(f"  No alert  gap={data['gap_today']}%")

        time.sleep(0.8)

    save_state(new_state)

    total_alerts = sum(len(v) for v in alerts.values())
    now = datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")

    if total_alerts == 0:
        send_telegram(
            f"📋 <b>VISH_SCAN Report — {now}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ Scanned : {scanned} stocks\n"
            f"😴 No signals today.\n"
            f"🔁 Next scan tomorrow.\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 Patience is the edge."
        )
        return
    
       # Build one single combined message
    lines = []
    lines.append(f"📋 <b>VISH_SCAN — {now}</b>")
    lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"📦 Scanned : {scanned} stocks")
    lines.append(f"🚨 Alerts  : {total_alerts}")
    lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")

    priority = [
        "golden_cross",
        "highest",
        "strong_breakout",
        "breakout",
        "accumulate_support",
        "stage3",
        "trendline_breakout",
        "stage2",
        "support",
        "stage4",
        "stage1",
    ]

    labels = {
        "golden_cross":       "🌟 GOLDEN CROSS",
        "highest":            "🔥🔥 HIGHEST CONVICTION",
        "strong_breakout":    "🚀 STRONG BREAKOUT",
        "breakout":           "📈 BREAKOUT",
        "accumulate_support": "💎 ACCUM + SUPPORT",
        "stage3":             "🔥 AGGR. ACCUMULATION",
        "trendline_breakout": "🚀 TRENDLINE BREAKOUT",
        "stage2":             "🟡 ACCUMULATE",
        "support":            "🛡 SUPPORT",
        "stage4":             "⏳ WAIT ZONE",
        "stage1":             "👁 WATCHLIST",
    }

    for key in priority:
        for msg in alerts[key]:
            lines_msg = msg.split("\n")
            ticker_line = lines_msg[0]
            price_line  = [l for l in lines_msg if "Price" in l]
            gap_line    = [l for l in lines_msg if "Gap" in l]
            rsi_line    = [l for l in lines_msg if "RSI" in l]
            vol_line    = [l for l in lines_msg if "Volume" in l]

            ticker = ticker_line.split("—")[-1].strip().replace("</b>","").replace("<b>","")
            price  = price_line[0].strip() if price_line else ""
            gap    = gap_line[0].strip()   if gap_line   else ""
            rsi    = rsi_line[0].strip()   if rsi_line   else ""
            vol    = vol_line[0].strip()   if vol_line   else ""

            lines.append(f"\n{labels.get(key, key)} — <b>{ticker}</b>")
            lines.append(price)
            lines.append(gap)
            lines.append(rsi)
            lines.append(vol)

    lines.append(f"\n━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"⚠️ Not SEBI advice. Personal use only.")

    send_telegram("\n".join(lines))
    log.info(f"DONE — {total_alerts} alerts sent.")
    log.info("=" * 55)


if __name__ == "__main__":
    run_scanner() 
