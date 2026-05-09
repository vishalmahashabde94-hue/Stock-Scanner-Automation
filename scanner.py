"""
=======================================================
  INDIAN STOCK DMA GAP SCANNER — CORRECTED LOGIC
  Strategy: Buy BEFORE golden cross, sell after
  
  LOGIC:
  50DMA is BELOW 200DMA = negative gap (stock recovering)
  We track as gap CLOSES from -10% towards 0%
  
  WATCH     : gap worse than -10% (too early)
  PREPARE   : gap at -6% to -7%  (get ready)
  BUY ZONE  : gap at -2% to -3%  (buy here)
  GOLDEN X  : 50DMA crosses above 200DMA (book profits)
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

# Gap is NEGATIVE because 50DMA is below 200DMA
# Example: gap of -10 means 50DMA is 10% below 200DMA
WATCH_THRESHOLD   = -10.0   # gap worse than -10% → WATCH
PREPARE_MIN       =  -7.0   # gap between -7% and -6% → PREPARE
PREPARE_MAX       =  -6.0
BUY_MIN           =  -3.0   # gap between -3% and -2% → BUY ZONE
BUY_MAX           =  -2.0


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

        # Signed gap — negative means 50DMA is below 200DMA
        # This is what we WANT to track — stocks recovering upward
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
    gap    = data["gap_today"]   # negative = 50DMA below 200DMA
    gap_p  = data["gap_prev"]
    price  = data["price"]
    d50    = data["dma50"]
    d200   = data["dma200"]

    # Is the gap actually closing? (gap moving towards 0)
    # e.g. gap was -8% yesterday, now -6% = closing = good
    closing = gap > gap_p  # gap increasing towards 0 = recovering

    # ── GOLDEN CROSS ──────────────────────────────────────────────────────
    # Yesterday 50DMA was below 200DMA (negative gap)
    # Today 50DMA crossed above 200DMA (gap becomes positive)
    # = TIME TO BOOK PROFITS
    if gap_p < 0 and gap >= 0:
        return "golden_cross", (
            f"🌟 <b>GOLDEN CROSS — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Price   : ₹{price:,.2f}\n"
            f"📊 50 DMA  : ₹{d50:,.2f}\n"
            f"📉 200 DMA : ₹{d200:,.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏁 50DMA just crossed ABOVE 200DMA\n"
            f"💰 If you bought earlier → consider BOOKING PROFITS\n"
            f"🚫 Do NOT buy fresh at this stage"
        )

    # Only track stocks where 50DMA is BELOW 200DMA (negative gap)
    # AND the gap is closing (recovering)
    if gap >= 0:
        # 50DMA already above 200DMA — not our target stock right now
        return None, ""

    if not closing:
        # Gap is still widening — stock still falling, ignore
        return None, ""

    # ── BUY ZONE: gap between -3% and -2% ────────────────────────────────
    if BUY_MIN <= gap <= BUY_MAX:
        return "stage3", (
            f"🟢 <b>BUY ZONE — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Price   : ₹{price:,.2f}\n"
            f"📊 50 DMA  : ₹{d50:,.2f}\n"
            f"📉 200 DMA : ₹{d200:,.2f}\n"
            f"📐 Gap     : {gap:.2f}%  ↗ Closing fast\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ 50DMA only {abs(gap):.1f}% below 200DMA\n"
            f"🔔 Golden Cross expected soon — HIGH PRIORITY BUY"
        )

    # ── PREPARE: gap between -7% and -6% ─────────────────────────────────
    if PREPARE_MIN <= gap <= PREPARE_MAX:
        return "stage2", (
            f"🟡 <b>PREPARE — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Price   : ₹{price:,.2f}\n"
            f"📊 50 DMA  : ₹{d50:,.2f}\n"
            f"📉 200 DMA : ₹{d200:,.2f}\n"
            f"📐 Gap     : {gap:.2f}%  ↗ Closing\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚡ Gap narrowing steadily\n"
            f"🔔 Watch closely — BUY ZONE alert coming soon"
        )

    # ── WATCH: gap worse than -10%, but closing ───────────────────────────
    if gap <= WATCH_THRESHOLD:
        return "stage1", (
            f"👁 <b>WATCH — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Price   : ₹{price:,.2f}\n"
            f"📊 50 DMA  : ₹{d50:,.2f}\n"
            f"📉 200 DMA : ₹{d200:,.2f}\n"
            f"📐 Gap     : {gap:.2f}%  ↗ Starting to close\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 50DMA is recovering but gap still large\n"
            f"🔔 Too early to buy — keep watching"
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
        f"🌟 Golden Cross  : {len(alerts['golden_cross'])} → Book profits\n"
        f"🟢 Buy Zone      : {len(alerts['stage3'])} → Act now\n"
        f"🟡 Prepare       : {len(alerts['stage2'])} → Get ready\n"
        f"👁  Watch         : {len(alerts['stage1'])} → Too early\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"↓ Individual alerts below ↓"
    )
    time.sleep(1)

    for stage in ["golden_cross", "stage3", "stage2", "stage1"]:
        for msg in alerts[stage]:
            send_telegram(msg)
            time.sleep(0.8)

    log.info(f"DONE — {total_alerts} alerts sent.")
    log.info("=" * 50)


if __name__ == "__main__":
    run_scanner()