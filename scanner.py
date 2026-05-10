"""
=======================================================
  INDIAN STOCK DMA GAP SCANNER v2.0 — REVISED LOGIC
  
  FIXES:
  - Uses SMA (not EMA) to match AngelOne/Screener/TradingView
  
  NEW INDICATORS:
  - RSI(14)          : oversold bounce detection
  - Volume Surge     : vs 20-day avg volume
  - Momentum Score   : price vs 20 days ago
  - Composite Score  : ranks buy quality 0-100

  STRATEGY: Buy BEFORE golden cross, sell after
  
  DMA GAP STAGES (50SMA vs 200SMA):
  WATCH     : gap < -10%          (too early)
  PREPARE   : gap -7% to -6%     (get ready)
  BUY ZONE  : gap -3% to -2%     (buy here)
  GOLDEN X  : 50SMA crosses above 200SMA (book profits)

  COMPOSITE BUY SCORE (0-100):
  - DMA gap stage        : 40 pts max
  - RSI in sweet spot    : 25 pts max
  - Volume surge         : 20 pts max
  - Momentum positive    : 15 pts max
=======================================================
"""

import os
import json
import time
import logging
import requests
import numpy as np
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

# ── THRESHOLDS ─────────────────────────────────────────────────────────────
WATCH_THRESHOLD   = -10.0
PREPARE_MIN       =  -7.0
PREPARE_MAX       =  -6.0
BUY_MIN           =  -3.0
BUY_MAX           =  -2.0

RSI_OVERSOLD      =  40.0   # RSI below this = good setup (recovering from oversold)
RSI_OVERBOUGHT    =  70.0   # RSI above this = avoid fresh entry
VOLUME_SURGE_X    =   1.5   # volume > 1.5x 20-day avg = surge
MOMENTUM_DAYS     =  20     # price change over 20 days for momentum


# ── STATE ──────────────────────────────────────────────────────────────────
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── TELEGRAM ───────────────────────────────────────────────────────────────
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


# ── INDICATORS ─────────────────────────────────────────────────────────────
def compute_rsi(close_series, period=14):
    """Wilder's RSI — standard implementation matching TradingView"""
    delta = close_series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    # First average (simple)
    avg_gain = gain.iloc[:period+1].mean()
    avg_loss = loss.iloc[:period+1].mean()

    gains = [avg_gain]
    losses = [avg_loss]

    for i in range(period+1, len(close_series)):
        gains.append((gains[-1] * (period - 1) + gain.iloc[i]) / period)
        losses.append((losses[-1] * (period - 1) + loss.iloc[i]) / period)

    if losses[-1] == 0:
        return 100.0
    rs  = gains[-1] / losses[-1]
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)


def compute_volume_ratio(volume_series):
    """Today's volume vs 20-day average"""
    if len(volume_series) < 21:
        return None
    avg_vol   = float(volume_series.iloc[-21:-1].mean())
    today_vol = float(volume_series.iloc[-1])
    if avg_vol == 0:
        return None
    return round(today_vol / avg_vol, 2)


def compute_momentum(close_series, days=20):
    """% price change over last N days"""
    if len(close_series) < days + 1:
        return None
    price_now  = float(close_series.iloc[-1])
    price_then = float(close_series.iloc[-(days+1)])
    if price_then == 0:
        return None
    return round((price_now - price_then) / price_then * 100, 2)


def compute_score(gap, rsi, vol_ratio, momentum):
    """
    Composite buy score 0-100.
    Higher = better setup.
    """
    score = 0

    # ── DMA Gap stage (max 40 pts) ──────────────────────
    if gap is not None:
        if BUY_MIN <= gap <= BUY_MAX:
            score += 40       # best stage
        elif PREPARE_MIN <= gap <= PREPARE_MAX:
            score += 25
        elif WATCH_THRESHOLD <= gap < PREPARE_MIN:
            score += 10

    # ── RSI sweet spot (max 25 pts) ─────────────────────
    # Best: RSI 30-50 (recovering from oversold)
    # Good: RSI 50-60 (momentum building)
    # Avoid: RSI > 70 (overbought)
    if rsi is not None:
        if 30 <= rsi <= 50:
            score += 25
        elif 50 < rsi <= 60:
            score += 15
        elif rsi < 30:
            score += 10       # extremely oversold — may be falling knife
        elif 60 < rsi <= 70:
            score += 5
        # RSI > 70: 0 pts

    # ── Volume surge (max 20 pts) ───────────────────────
    if vol_ratio is not None:
        if vol_ratio >= 2.0:
            score += 20
        elif vol_ratio >= VOLUME_SURGE_X:
            score += 12
        elif vol_ratio >= 1.2:
            score += 5

    # ── Momentum (max 15 pts) ───────────────────────────
    if momentum is not None:
        if 0 < momentum <= 5:
            score += 15       # gentle positive momentum — ideal
        elif 5 < momentum <= 15:
            score += 10
        elif momentum > 15:
            score += 5        # may have run too fast
        elif -5 <= momentum < 0:
            score += 3        # slight pullback — ok
        # momentum < -5: 0 pts

    return score


def score_label(score):
    if score >= 80:
        return "🔥 EXCEPTIONAL"
    elif score >= 65:
        return "⭐ STRONG"
    elif score >= 50:
        return "✅ GOOD"
    elif score >= 35:
        return "🟡 MODERATE"
    else:
        return "👁 WEAK"


# ── DATA FETCH ─────────────────────────────────────────────────────────────
def fetch_data(symbol):
    try:
        df = yf.download(symbol, period="1y", interval="1d",
                         auto_adjust=True, progress=False)

        if df.empty or len(df) < 210:
            log.warning(f"  {symbol}: not enough data ({len(df)} rows)")
            return None

        close  = df["Close"]
        volume = df["Volume"]

        # Squeeze multi-index if needed
        if hasattr(close, "squeeze"):
            close  = close.squeeze()
        if hasattr(volume, "squeeze"):
            volume = volume.squeeze()

        # ── SMA (matches AngelOne / Screener / TradingView default) ──────
        sma50_series  = close.rolling(window=50,  min_periods=50).mean()
        sma200_series = close.rolling(window=200, min_periods=200).mean()

        sma50_today  = float(sma50_series.iloc[-1])
        sma200_today = float(sma200_series.iloc[-1])
        sma50_prev   = float(sma50_series.iloc[-2])
        sma200_prev  = float(sma200_series.iloc[-2])

        if np.isnan(sma50_today) or np.isnan(sma200_today):
            log.warning(f"  {symbol}: SMA has NaN — skipping")
            return None

        if sma200_today == 0 or sma200_prev == 0:
            return None

        gap_today = round((sma50_today - sma200_today) / sma200_today * 100, 2)
        gap_prev  = round((sma50_prev  - sma200_prev)  / sma200_prev  * 100, 2)

        # ── Additional indicators ─────────────────────────────────────────
        rsi        = compute_rsi(close)
        vol_ratio  = compute_volume_ratio(volume)
        momentum   = compute_momentum(close, days=MOMENTUM_DAYS)

        return {
            "symbol":    symbol,
            "ticker":    symbol.replace(".NS", ""),
            "price":     round(float(close.iloc[-1]), 2),
            "sma50":     round(sma50_today, 2),
            "sma200":    round(sma200_today, 2),
            "gap_today": gap_today,
            "gap_prev":  gap_prev,
            "rsi":       rsi,
            "vol_ratio": vol_ratio,
            "momentum":  momentum,
        }

    except Exception as e:
        log.error(f"  {symbol}: error — {e}")
        return None


# ── CLASSIFY ───────────────────────────────────────────────────────────────
def classify_stock(data):
    ticker    = data["ticker"]
    gap       = data["gap_today"]
    gap_p     = data["gap_prev"]
    price     = data["price"]
    d50       = data["sma50"]
    d200      = data["sma200"]
    rsi       = data["rsi"]
    vol_ratio = data["vol_ratio"]
    momentum  = data["momentum"]

    closing = gap > gap_p   # gap moving towards 0 = recovering

    # ── RSI / Volume / Momentum display helpers ───────────────────────────
    rsi_str    = f"{rsi:.1f}" if rsi is not None else "N/A"
    vol_str    = f"{vol_ratio:.1f}x" if vol_ratio is not None else "N/A"
    mom_str    = (f"+{momentum:.1f}%" if momentum and momentum > 0
                  else f"{momentum:.1f}%" if momentum is not None else "N/A")

    rsi_flag   = ("🔥" if rsi and rsi < 30 else
                  "✅" if rsi and rsi <= 50 else
                  "🟡" if rsi and rsi <= 60 else
                  "⚠️" if rsi and rsi <= 70 else "🔴")
    vol_flag   = "🔊" if vol_ratio and vol_ratio >= VOLUME_SURGE_X else "🔈"
    mom_flag   = "📈" if momentum and momentum > 0 else "📉"

    score = compute_score(gap, rsi, vol_ratio, momentum)
    label = score_label(score)

    def indicator_block():
        return (
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Indicators\n"
            f"{rsi_flag} RSI(14)   : {rsi_str}\n"
            f"{vol_flag} Volume    : {vol_str} vs avg\n"
            f"{mom_flag} Momentum  : {mom_str} (20d)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 Score : {score}/100  {label}"
        )

    # ── GOLDEN CROSS ──────────────────────────────────────────────────────
    if gap_p < 0 and gap >= 0:
        return "golden_cross", (
            f"🌟 <b>GOLDEN CROSS — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Price   : ₹{price:,.2f}\n"
            f"📊 50 SMA  : ₹{d50:,.2f}\n"
            f"📉 200 SMA : ₹{d200:,.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏁 50SMA just crossed ABOVE 200SMA\n"
            f"{indicator_block()}\n"
            f"💰 If you bought earlier → consider BOOKING PROFITS\n"
            f"🚫 Do NOT buy fresh at this stage"
        ), score

    # Only track stocks where 50SMA is below 200SMA AND gap is closing
    if gap >= 0:
        return None, "", 0

    if not closing:
        return None, "", 0

    # ── BUY ZONE ──────────────────────────────────────────────────────────
    if BUY_MIN <= gap <= BUY_MAX:
        return "stage3", (
            f"🟢 <b>BUY ZONE — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Price   : ₹{price:,.2f}\n"
            f"📊 50 SMA  : ₹{d50:,.2f}\n"
            f"📉 200 SMA : ₹{d200:,.2f}\n"
            f"📐 Gap     : {gap:.2f}%  ↗ Closing fast\n"
            f"{indicator_block()}\n"
            f"✅ 50SMA only {abs(gap):.1f}% below 200SMA\n"
            f"🔔 Golden Cross expected soon — HIGH PRIORITY BUY"
        ), score

    # ── PREPARE ───────────────────────────────────────────────────────────
    if PREPARE_MIN <= gap <= PREPARE_MAX:
        return "stage2", (
            f"🟡 <b>PREPARE — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Price   : ₹{price:,.2f}\n"
            f"📊 50 SMA  : ₹{d50:,.2f}\n"
            f"📉 200 SMA : ₹{d200:,.2f}\n"
            f"📐 Gap     : {gap:.2f}%  ↗ Closing\n"
            f"{indicator_block()}\n"
            f"⚡ Gap narrowing steadily\n"
            f"🔔 Watch closely — BUY ZONE alert coming soon"
        ), score

    # ── WATCH ─────────────────────────────────────────────────────────────
    if gap <= WATCH_THRESHOLD:
        return "stage1", (
            f"👁 <b>WATCH — {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Price   : ₹{price:,.2f}\n"
            f"📊 50 SMA  : ₹{d50:,.2f}\n"
            f"📉 200 SMA : ₹{d200:,.2f}\n"
            f"📐 Gap     : {gap:.2f}%  ↗ Starting to close\n"
            f"{indicator_block()}\n"
            f"📌 50SMA is recovering but gap still large\n"
            f"🔔 Too early to buy — keep watching"
        ), score

    return None, "", 0


# ── MAIN RUNNER ────────────────────────────────────────────────────────────
def run_scanner():
    log.info("=" * 55)
    log.info("  DMA SCANNER v2.0 STARTING  (SMA + RSI + Vol + Mom)")
    log.info("=" * 55)

    saved_state = load_state()
    new_state   = {}
    # Store (message, score) tuples per stage
    alerts = {"golden_cross": [], "stage3": [], "stage2": [], "stage1": []}
    scanned = 0
    skipped = 0

    for symbol in WATCHLIST:
        log.info(f"Scanning {symbol}...")
        data = fetch_data(symbol)

        if data is None:
            skipped += 1
            time.sleep(1)
            continue

        scanned += 1
        new_state[symbol] = {
            "gap_pct":  data["gap_today"],
            "rsi":      data["rsi"],
            "momentum": data["momentum"],
        }

        stage_key, message, score = classify_stock(data)

        if stage_key:
            alerts[stage_key].append((score, message))
            log.info(f"  ALERT → {stage_key}  gap={data['gap_today']}%  "
                     f"RSI={data['rsi']}  vol={data['vol_ratio']}x  "
                     f"score={score}/100")
        else:
            log.info(f"  No alert  gap={data['gap_today']}%")

        time.sleep(0.8)

    save_state(new_state)

    # Sort each stage by score descending (best opportunities first)
    for stage in alerts:
        alerts[stage].sort(key=lambda x: x[0], reverse=True)

    total_alerts = sum(len(v) for v in alerts.values())
    now = datetime.now().strftime("%d %b %Y, %I:%M %p")

    if total_alerts == 0:
        send_telegram(
            f"📋 <b>DMA Scanner v2.0 — {now}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ Scanned : {scanned} stocks\n"
            f"😴 No stocks in any alert zone today.\n"
            f"🔁 Next scan in ~2 days."
        )
        return

    send_telegram(
        f"📋 <b>DMA Scanner v2.0 — {now}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📦 Scanned : {scanned} stocks\n"
        f"🚨 Alerts  : {total_alerts} triggered\n\n"
        f"🌟 Golden Cross : {len(alerts['golden_cross'])} → Book profits\n"
        f"🟢 Buy Zone     : {len(alerts['stage3'])} → Act now\n"
        f"🟡 Prepare      : {len(alerts['stage2'])} → Get ready\n"
        f"👁  Watch        : {len(alerts['stage1'])} → Too early\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Scores = DMA(40) + RSI(25) + Vol(20) + Mom(15)\n"
        f"↓ Best opportunities first ↓"
    )
    time.sleep(1)

    for stage in ["golden_cross", "stage3", "stage2", "stage1"]:
        for score, msg in alerts[stage]:
            send_telegram(msg)
            time.sleep(0.8)

    log.info(f"DONE — {total_alerts} alerts sent.")
    log.info("=" * 55)


if __name__ == "__main__":
    run_scanner()
