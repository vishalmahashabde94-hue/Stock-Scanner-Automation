"""
=======================================================
  INDIAN STOCK DMA GAP SCANNER
  Built for: Vishal's Swing Trade Autopilot System
  Runs: Every 2 days via GitHub Actions
  Alerts: Telegram
=======================================================
"""

import os
import json
import time
import logging
import requests
import yfinance as yf
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

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

STAGE1_MIN = 10.0
STAGE2_MIN =  6.0
STAGE2_MAX =  7.0
STAGE3_MIN =  2.0
STAGE3_MAX =  3.0


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

def fetch_dma_data(symbol):
    try:
        df = yf.download(symbol, period="1y", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty or len(df) < 200:
            log.warning(f"  {symbol}: not enough data ({len(df)} rows)")
            return None

        close = df["Close"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()

        dma50_today  = float(close.rolling(50).mean().iloc[-1])
        dma200_today = float(close.rolling(200).mean().iloc[-1])
        dma50_prev   = float(close.rolling(50).mean().iloc[-2])
        dma200_prev  = float(close.rolling(200).mean().iloc[-2])

        if dma200_today == 0 or dma200_prev == 0:
            return None

        gap_today = round((dma50_today - dma200_today) / dma200_today * 100, 2)
        gap_prev  = round((dma50_prev  - dma200_prev)  / dma200_prev  * 100, 2)

        return {
            "symbol":    symbol,
            "price":     round(float(close.iloc[-1]), 2),
            "dma50":     round(dma50_today, 2),
            "dma200":    round(dma200_today, 2),
            "gap_today": gap_today,
            "gap_prev":  gap_prev,
        }
    except Exception as e:
        log.error(f"  {symbol}: error — {e}")
        return None

def classify_stock(data):
    ticker = data["symbol"].replace(".NS", "")
    gap    = data["gap_today"]
    gap_p  = data["gap_prev"]
    price  = data["price"]
    d50    = data["dma50"]
    d200   = data["dma200"]

    if gap_p < 0 and gap >= 0:
        return "golden_cross", (
            f"🌟 <b>GOLDEN CROSS — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Price   : ₹{price:,.2f}\n"
            f"📊 50 DMA  : ₹{d50:,.2f}\n"
            f"📉 200 DMA : ₹{d200:,.2f}\n"
            f"📐 Gap     : {gap:+.2f}%  (was {gap_p:+.2f}%)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ 50DMA just crossed ABOVE 200DMA\n"
            f"🔔 Strongest signal. Check volume before buying."
        )

    if STAGE3_MIN <= gap <= STAGE3_MAX:
        direction = "↘ Closing" if gap < gap_p else "→ Stable"
        return "stage3", (
            f"🟢 <b>BUY ZONE — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Price   : ₹{price:,.2f}\n"
            f"📊 50 DMA  : ₹{d50:,.2f}\n"
            f"📉 200 DMA : ₹{d200:,.2f}\n"
            f"📐 Gap     : +{gap:.2f}%  {direction}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ Gap in the 2-3% sweet spot\n"
            f"🔔 Golden Cross imminent. High-probability entry."
        )

    if STAGE2_MIN <= gap <= STAGE2_MAX:
        return "stage2", (
            f"🟡 <b>PREPARE — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Price   : ₹{price:,.2f}\n"
            f"📊 50 DMA  : ₹{d50:,.2f}\n"
            f"📉 200 DMA : ₹{d200:,.2f}\n"
            f"📐 Gap     : +{gap:.2f}%\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚡ Gap narrowing. Stay alert.\n"
            f"🔔 Next alert when gap reaches 2-3%."
        )

    if gap > STAGE1_MIN:
        return "stage1", (
            f"👁 <b>WATCH — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Price   : ₹{price:,.2f}\n"
            f"📊 50 DMA  : ₹{d50:,.2f}\n"
            f"📉 200 DMA : ₹{d200:,.2f}\n"
            f"📐 Gap     : +{gap:.2f}%\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 Large gap. On watchlist.\n"
            f"🔔 Alert when gap closes to 6-7%."
        )

    return None, ""


def run_scanner():
    log.info("=" * 50)
    log.info("  DMA SCANNER STARTING")
    log.info("=" * 50)

    saved_state = load_state()
    new_state   = {}
    alerts = {"golden_cross": [], "stage3": [], "stage2": [], "stage1": []}
    scanned = 0
    skipped = 0

    for symbol in WATCHLIST:
        log.info(f"Scanning {symbol}...")
        data = fetch_dma_data(symbol)

        if data is None:
            skipped += 1
            time.sleep(1)
            continue

        scanned += 1
        new_state[symbol] = {"gap_pct": data["gap_today"]}
        stage_key, message = classify_stock(data)

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
            f"📋 <b>DMA Scanner — {now}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ Scanned : {scanned} stocks\n"
            f"😴 No stocks in any alert zone today.\n"
            f"🔁 Next scan in ~2 days."
        )
        return

    send_telegram(
        f"📋 <b>DMA Scanner Report — {now}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📦 Scanned : {scanned} stocks\n"
        f"🚨 Alerts  : {total_alerts} triggered\n\n"
        f"🌟 Golden Cross : {len(alerts['golden_cross'])}\n"
        f"🟢 Buy Zone     : {len(alerts['stage3'])}\n"
        f"🟡 Prepare      : {len(alerts['stage2'])}\n"
        f"👁  Watch        : {len(alerts['stage1'])}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"↓ Individual alerts below ↓"
    )
    time.sleep(1)

    for stage in ["golden_cross", "stage3", "stage2", "stage1"]:
        for msg in alerts[stage]:
            send_telegram(msg)
            time.sleep(0.8)

    log.info(f"DONE — {total_alerts} alerts sent.")


if __name__ == "__main__":
    run_scanner()