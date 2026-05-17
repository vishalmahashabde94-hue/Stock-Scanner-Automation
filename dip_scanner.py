"""
=======================================================
  VISH_SCAN — WEEKLY DIP SCANNER
  Runs every Friday at 3:30 PM IST
  Scans Monday to Friday price movement
  Alerts on 3%, 5%, 7.5%+ weekly dips
=======================================================
"""

import os
import time
import logging
import requests
import yfinance as yf
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["CHAT_ID"]

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

MILD_DIP     = 3.0
MODERATE_DIP = 5.0
STRONG_DIP   = 7.5


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
        return f"🟢 {vol_ratio:.1f}x — Strong accumulation"
    elif vol_ratio >= 1.3:
        return f"🟢 {vol_ratio:.1f}x — Volume building"
    elif vol_ratio >= 0.8:
        return f"🟡 {vol_ratio:.1f}x — Average volume"
    else:
        return f"🔴 {vol_ratio:.1f}x — Low volume (weak)"

def fetch_dip_data(symbol):
    try:
        df = yf.download(
            symbol,
            period="14d",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

        if df.empty or len(df) < 4:
            log.warning(f"  {symbol}: not enough data")
            return None

        close  = df["Adj Close"].squeeze()
        volume = df["Volume"].squeeze()

        week_data    = close.iloc[-5:] if len(close) >= 5 else close
        monday_price = float(week_data.iloc[0])
        friday_price = float(week_data.iloc[-1])

        if monday_price == 0:
            return None

        dip_pct = round((monday_price - friday_price) / monday_price * 100, 2)

        df_rsi    = yf.download(
            symbol,
            period="1mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        close_rsi  = df_rsi["Adj Close"].squeeze()
        rsi_series = compute_rsi(close_rsi)
        rsi_today  = round(float(rsi_series.iloc[-1]), 1)

        vol_today = float(volume.iloc[-1])
        vol_avg   = float(volume.mean())
        vol_ratio = round(vol_today / vol_avg, 2) if vol_avg > 0 else 0

        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        days_available = list(week_data.values)
        breakdown = ""
        for i, price in enumerate(days_available):
            if i < len(day_labels):
                change = round((price - monday_price) / monday_price * 100, 2)
                arrow  = "↓" if change < 0 else "↑"
                breakdown += f"  {day_labels[i]}: ₹{price:,.2f} ({arrow}{abs(change):.1f}%)\n"

        return {
            "symbol":       symbol,
            "monday_price": round(monday_price, 2),
            "friday_price": round(friday_price, 2),
            "dip_pct":      dip_pct,
            "rsi":          rsi_today,
            "vol_ratio":    vol_ratio,
            "breakdown":    breakdown.strip(),
        }

    except Exception as e:
        log.error(f"  {symbol}: error — {e}")
        return None

def classify_dip(data):
    ticker    = data["symbol"].replace(".NS", "")
    dip       = data["dip_pct"]
    mon       = data["monday_price"]
    fri       = data["friday_price"]
    rsi       = data["rsi"]
    vol_r     = data["vol_ratio"]
    breakdown = data["breakdown"]

    if dip < MILD_DIP:
        return None, ""

    if dip >= STRONG_DIP:
        level   = "strong"
        header  = f"🔴 <b>STRONG DIP — {ticker}</b>"
        advice  = "High priority entry opportunity. Check fundamentals before buying."
        urgency = f"Stock fell <b>{dip:.1f}%</b> this week — significant correction"
    elif dip >= MODERATE_DIP:
        level   = "moderate"
        header  = f"🟠 <b>MODERATE DIP — {ticker}</b>"
        advice  = "Good entry opportunity if EMA gap is also closing."
        urgency = f"Stock fell <b>{dip:.1f}%</b> this week — meaningful dip"
    else:
        level   = "mild"
        header  = f"🟡 <b>MILD DIP — {ticker}</b>"
        advice  = "Minor weakness. Watch for continuation or recovery."
        urgency = f"Stock fell <b>{dip:.1f}%</b> this week — minor pullback"

    msg = (
        f"{header}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 Monday  : ₹{mon:,.2f}\n"
        f"📅 Friday  : ₹{fri:,.2f}\n"
        f"📉 Weekly  : {urgency}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 This Week:\n{breakdown}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 RSI     : {rsi_label(rsi)}\n"
        f"📦 Volume  : {volume_label(vol_r)}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 {advice}"
    )
    return level, msg


def run_dip_scanner():
    log.info("=" * 55)
    log.info("  VISH_SCAN — WEEKLY DIP SCANNER STARTING")
    log.info("=" * 55)

    now = datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")

    dips = {
        "strong":   [],
        "moderate": [],
        "mild":     [],
    }

    scanned = 0
    skipped = 0

    for symbol in WATCHLIST:
        log.info(f"Scanning {symbol}...")
        data = fetch_dip_data(symbol)

        if data is None:
            skipped += 1
            time.sleep(1)
            continue

        scanned += 1
        level, message = classify_dip(data)

        if level:
            dips[level].append(message)
            log.info(f"  DIP → {level}  {data['dip_pct']}%")
        else:
            log.info(f"  No dip  {data['dip_pct']}%")

        time.sleep(0.8)

    total_dips = sum(len(v) for v in dips.values())

    if total_dips == 0:
        send_telegram(
            f"📊 <b>Weekly Dip Report — {now}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ Scanned : {scanned} stocks\n"
            f"😴 No significant dips this week.\n"
            f"📈 Market held up well — good sign.\n"
            f"🔁 Next report next Friday."
        )
        return

    send_telegram(
        f"📊 <b>Weekly Dip Report — {now}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📦 Scanned           : {scanned} stocks\n"
        f"🚨 Dips Detected     : {total_dips}\n\n"
        f"🔴 Strong  (7.5%+)   : {len(dips['strong'])}\n"
        f"🟠 Moderate (5-7.5%) : {len(dips['moderate'])}\n"
        f"🟡 Mild    (3-5%)    : {len(dips['mild'])}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"↓ Individual dip alerts below ↓"
    )
    time.sleep(1)

    for level in ["strong", "moderate", "mild"]:
        for msg in dips[level]:
            send_telegram(msg)
            time.sleep(0.8)

    log.info(f"DONE — {total_dips} dip alerts sent.")
    log.info("=" * 55)


if __name__ == "__main__":
    run_dip_scanner()
