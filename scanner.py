"""
=======================================================
  VISH_SCAN — SCANNER v7
  Backtested against 3 years of history (see
  backtest_vish_scan.py). Changes from v6 based on
  actual results, not just theory:

    1. RSI overbought no longer downgrades signals to
       Watch — backtest showed "watch_overbought" stocks
       actually outperformed (62.4% win rate, +8.36% avg
       at 40d). Now shown as a caution NOTE instead,
       keeping the original bucket and action.
    2. "Accumulate + Support" combo bucket REMOVED —
       backtest showed it losing money (33% win rate,
       -2.49% avg return at 40d, worst drawdown of any
       bucket). Falls back to the plain Aggressive/
       Accumulate signal instead.
    3. Coiling moved to a clearly-labeled "unconfirmed"
       section — backtest showed ~51-53% win rate across
       all horizons, essentially noise. Kept for further
       tuning, not for trusting yet.
    4. Golden Cross and Buy Now kept as top priority —
       these were the two strongest validated signals
       (63.5% and 57.9%+ win rates with the best
       risk/reward of any bucket).
    5. Fixed watchlist tickers that failed in backtest
       (wrong/outdated symbols, duplicates).

  Carried over from v6:
    - Golden Cross true 5-day lookback, no whipsaw
    - gap_closing uses 3-day slope, not 1-day noise
    - Relative strength vs Nifty tagged on every signal
    - Volume confirmation smoothed over last 3 days
    - Batched yfinance download

  v7.1 — added signal_tracker.py integration:
    - Tracks entry signals (Golden Cross, Buy Now,
      Aggressive Accumulation, Accumulate, Support Zone)
      and fires milestone alerts on Telegram.
    - Runs every scan, even on quiet days with zero new
      signals, because it checks stocks signalled weeks
      ago, not just today's.
=======================================================
"""

import os
import json
import logging
import html
import requests
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone, timedelta

from signal_tracker import track_and_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["CHAT_ID"]

IST = timezone(timedelta(hours=5, minutes=30))
BENCHMARK = "^NSEI"

WATCHLIST = [
    "HBLENGINE.NS",
    "PARAS.NS",
    "ZENTEC.NS", "DATAPATTNS.NS", "MAZDOCK.NS",
    "HAL.NS", "BEL.NS", "ASTRAMICRO.NS", "BHARATFORG.NS", "MTARTECH.NS", "IDEAFORGE.NS",
    "M&M.NS", "ASHOKLEY.NS", "TVSMOTOR.NS", "BANCOINDIA.NS", "PRECWIRE.NS",
    "MOTHERSON.NS", "ENDURANCE.NS", "TIINDIA.NS",
    # MOTHERSONSUM.NS removed — duplicate, old symbol for MOTHERSON.NS (renamed June 2022)
    "BALUFORGE.NS", "SMLMAH.NS",
    "HAVELLS.NS", "POLYCAB.NS", "KEI.NS", "SCHNEIDER.NS", "CGPOWER.NS",   # KEIIND -> KEI (correct NSE symbol)
    "TRANSRAILL.NS", "TARIL.NS", "VOLTAMP.NS", "GVT&D.NS",   # TRIL -> TARIL (Trans & Rectifiers)
    "TRITURBINE.NS", "TDPOWERSYS.NS", "IONEXCHANG.NS", "TITAGARH.NS", "KEC.NS", "LT.NS",
    "ADANIGREEN.NS", "ADANIPOWER.NS", "JSWENERGY.NS", "INOXWIND.NS", "WAAREERTL.NS",
    "KPIL.NS", "JWL.NS",
    "INFY.NS", "WIPRO.NS", "DIXON.NS", "REDINGTON.NS", "KAYNES.NS", "NETWEB.NS", "BBOX.NS",
    "BHARTIARTL.NS", "INDUSTOWER.NS",
    "MOTILALOFS.NS", "ANGELONE.NS", "BAJFINANCE.NS", "AXISBANK.NS", "BSE.NS",
    "NSDL.NS", "CDSL.NS", "KFINTECH.NS",
    "ADANIPORTS.NS",
    "ANANTRAJ.NS",
    "CHALET.NS", "RATEGAIN.NS",
    "METROBRAND.NS", "BLS.NS", "BOROLTD.NS",
    "TIPSMUSIC.NS",
    "GOKULAGRO.NS", "VBL.NS", "LTFOODS.NS", "RADICO.NS", "GODFRYPHLP.NS", "ABDL.NS",
    "LLOYDSME.NS", "COALINDIA.NS", "RELIANCE.NS",
    "NATCOPHARM.NS", "RUBICON.NS",
    "YATHARTH.NS", "KIMS.NS", "NH.NS", "RAINBOW.NS",   # KIIMS.NS removed — duplicate/typo of KIMS.NS
    "KAJARIACER.NS",
    "AEGISLOG.NS",
    "PRICOLLTD.NS", "CCL.NS",
]

# ── Gap stage bands (unchanged) ─────────────────────────────────────────────
STAGE1_MIN, STAGE1_MAX = -20.0, -12.0   # Watchlist
STAGE2_MIN, STAGE2_MAX = -12.0,  -8.0   # Accumulate
STAGE3_MIN, STAGE3_MAX =  -8.0,  -4.0   # Aggressive Accumulation
STAGE4_MIN, STAGE4_MAX =  -4.0,  -2.0   # Wait Zone

# ── New thresholds ───────────────────────────────────────────────────────
GOLDEN_CROSS_LOOKBACK = 5      # trading days
SLOPE_LOOKBACK         = 3      # days used for gap_closing confirmation
MIN_SLOPE_MOVE         = 0.15   # % — ignore sub-noise gap movement

RSI_OVERBOUGHT = 70
RSI_EXIT       = 75
RSI_OVERSOLD   = 30

VOLUME_CONFIRM       = 1.5      # single-day spike threshold (unchanged)
VOL_SMOOTH_MIN_RATIO = 1.2      # secondary smoothed threshold
VOL_SMOOTH_MIN_DAYS  = 2        # need 2 of last 3 days above smoothed ratio

BREAKOUT_3M_DAYS = 63
BREAKOUT_6M_DAYS = 126
SUPPORT_ZONE     = 0.02

COIL_PROXIMITY  = 0.05   # within 5% of 3M high, not yet broken
COIL_RSI_MIN    = 55
COIL_RSI_MAX    = 68
COIL_MIN_HITS   = 3       # need 3 of 4 coiling conditions


# ─────────────────────────────────────────────────────────────────────────
# Telegram
# ─────────────────────────────────────────────────────────────────────────
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
        # Log Telegram's actual error body (e.g. which char/offset broke HTML
        # parsing) so future failures are diagnosable straight from the log.
        body = ""
        try:
            body = r.text
        except Exception:
            pass
        log.error(f"  Telegram: FAILED — {e} | response: {body}")


def send_telegram_chunked(lines, max_chars=3800):
    """Join lines into chunks that stay under Telegram's 4096-char limit."""
    chunk = []
    length = 0
    for line in lines:
        line_len = len(line) + 1
        if length + line_len > max_chars and chunk:
            send_telegram("\n".join(chunk))
            chunk, length = [], 0
        chunk.append(line)
        length += line_len
    if chunk:
        send_telegram("\n".join(chunk))


# ─────────────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────────────
def compute_rsi(close, period=14):
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def rsi_simple(rsi):
    if rsi < 30:   return f"{rsi:.0f} — Oversold"
    if rsi < 45:   return f"{rsi:.0f} — Recovering"
    if rsi < 55:   return f"{rsi:.0f} — Neutral"
    if rsi < 70:   return f"{rsi:.0f} — Heated"
    return f"{rsi:.0f} — Overbought"


def vol_simple(vol_ratio):
    if vol_ratio >= 1.5: return f"High ({vol_ratio:.1f}x)"
    if vol_ratio >= 0.8: return f"OK ({vol_ratio:.1f}x)"
    return f"Low ({vol_ratio:.1f}x)"


def detect_trendline(prices):
    x = np.arange(len(prices))
    y = np.array(prices)
    slope, intercept = np.polyfit(x, y, 1)
    current = slope * (len(prices) - 1) + intercept
    return slope, current


def swing_lows(low_series, order=3):
    """Local minima only — a low that's lower than `order` candles on each side."""
    vals = low_series.values
    pts = []
    for i in range(order, len(vals) - order):
        window = vals[i - order:i + order + 1]
        if vals[i] == window.min():
            pts.append((i, vals[i]))
    return pts


# ─────────────────────────────────────────────────────────────────────────
# Batched fetch — one API call for the whole watchlist + benchmark
# ─────────────────────────────────────────────────────────────────────────
def fetch_all_data(symbols):
    tickers = symbols + [BENCHMARK]
    log.info(f"Batch downloading {len(tickers)} tickers...")
    raw = yf.download(
        tickers,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # Benchmark 20-day return
    try:
        bench_close = raw[BENCHMARK]["Close"].dropna()
        nifty_return_20d = float(bench_close.iloc[-1] / bench_close.iloc[-21] - 1) * 100
    except Exception as e:
        log.warning(f"Benchmark fetch failed, defaulting RS to neutral: {e}")
        nifty_return_20d = 0.0

    results = {}
    for symbol in symbols:
        try:
            df = raw[symbol].dropna(how="all")
            if df.empty or len(df) < 200:
                log.warning(f"  {symbol}: insufficient data ({len(df)} rows)")
                continue
            data = process_symbol(symbol, df, nifty_return_20d)
            if data:
                results[symbol] = data
        except Exception as e:
            log.error(f"  {symbol}: error — {e}")
    return results


def process_symbol(symbol, df, nifty_return_20d):
    close  = df["Close"].dropna()
    high   = df["High"].dropna()
    low    = df["Low"].dropna()
    volume = df["Volume"].dropna()

    if len(close) < 200:
        return None

    # Full EMA + gap series (no state file dependency)
    ema50_series  = close.ewm(span=50,  adjust=False).mean()
    ema200_series = close.ewm(span=200, adjust=False).mean()
    gap_series = ((ema50_series - ema200_series) / ema200_series * 100).round(2)

    gap_today = float(gap_series.iloc[-1])
    gap_prev  = float(gap_series.iloc[-2])
    gap_3ago  = float(gap_series.iloc[-1 - SLOPE_LOOKBACK])

    # 3-day slope confirmation instead of 1-day noise
    gap_closing = (gap_today - gap_3ago) > MIN_SLOPE_MOVE

    # True 5-day Golden Cross: negative for all 5 prior days, positive today, no whipsaw
    lookback_window = gap_series.iloc[-1 - GOLDEN_CROSS_LOOKBACK:-1]
    golden_cross = bool((lookback_window < 0).all() and gap_today >= 0)

    rsi_series = compute_rsi(close)
    rsi_today  = round(float(rsi_series.iloc[-1]), 1)

    vol_avg20_series = volume.rolling(20).mean()
    vol_ratio_series = (volume / vol_avg20_series).round(2)
    vol_ratio_today   = float(vol_ratio_series.iloc[-1]) if not np.isnan(vol_ratio_series.iloc[-1]) else 0

    # Smoothed volume confirmation: 2 of last 3 days above 1.2x
    last3_vol = vol_ratio_series.iloc[-3:].fillna(0)
    smoothed_vol_ok = int((last3_vol >= VOL_SMOOTH_MIN_RATIO).sum()) >= VOL_SMOOTH_MIN_DAYS

    price_today = float(close.iloc[-1])
    price_prev  = float(close.iloc[-2])

    high_3m = float(high.iloc[-BREAKOUT_3M_DAYS:-1].max())
    high_6m = float(high.iloc[-BREAKOUT_6M_DAYS:-1].max())

    breakout_3m = price_today > high_3m and price_prev <= high_3m
    breakout_6m = price_today > high_6m and price_prev <= high_6m

    vol_confirm = (vol_ratio_today >= VOLUME_CONFIRM) or smoothed_vol_ok
    breakout_3m_confirmed = breakout_3m and vol_confirm
    breakout_6m_confirmed = breakout_6m and vol_confirm

    # Support: fit trendline through actual swing lows only
    recent_low_series = low.iloc[-BREAKOUT_3M_DAYS:]
    pts = swing_lows(recent_low_series, order=3)
    if len(pts) >= 2:
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        slope, intercept = np.polyfit(xs, ys, 1)
        trendline_support = slope * (len(recent_low_series) - 1) + intercept
    else:
        slope, trendline_support = detect_trendline(recent_low_series.values)

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
        vol_confirm
    )

    # ── Pre-breakout "Coiling" signal ──────────────────────────────────
    near_high = (high_3m - price_today) / high_3m <= COIL_PROXIMITY and price_today <= high_3m
    rsi_building = COIL_RSI_MIN <= rsi_today <= COIL_RSI_MAX

    returns = close.pct_change()
    vol10 = returns.iloc[-10:].std()
    vol20 = returns.iloc[-20:].std()
    volatility_squeeze = bool(vol10 < vol20 * 0.85) if vol20 and not np.isnan(vol20) else False

    vol_recent3  = float(volume.iloc[-3:].mean())
    vol_prior10  = float(volume.iloc[-13:-3].mean())
    vol_trend_up = vol_recent3 > vol_prior10

    coil_hits = sum([near_high, rsi_building, volatility_squeeze, vol_trend_up])
    coiling = coil_hits >= COIL_MIN_HITS and not (breakout_3m_confirmed or breakout_6m_confirmed)

    # Relative strength vs Nifty
    try:
        stock_return_20d = float(close.iloc[-1] / close.iloc[-21] - 1) * 100
    except Exception:
        stock_return_20d = 0.0
    rs_positive = stock_return_20d > nifty_return_20d

    return {
        "symbol": symbol, "price": round(price_today, 2),
        "gap_today": gap_today, "gap_prev": gap_prev, "gap_closing": gap_closing,
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


# ─────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────
def classify(data):
    ticker = html.escape(data["symbol"].replace(".NS", ""))
    gap    = data["gap_today"]
    price  = data["price"]
    rsi    = data["rsi"]
    vol_r  = data["vol_ratio"]

    rsi_lbl = rsi_simple(rsi)
    vol_lbl = vol_simple(vol_r)
    rs_tag  = "✅ RS > Nifty" if data["rs_positive"] else "⚠️ RS below Nifty (beta-driven)"
    price_str = f"₹{price:,.0f}"

    def stock_line(emoji, action_text, extra_note=""):
        note = f"\n📝 {extra_note}" if extra_note else ""
        return (
            f"\n<b>{ticker}</b>\n"
            f"💵 CMP      : {price_str}\n"
            f"📐 EMA Gap  : {gap:.1f}%\n"
            f"📊 RSI      : {rsi_lbl}\n"
            f"📦 Volume   : {vol_lbl}\n"
            f"📈 RS(20d)  : {rs_tag}\n"
            f"👉 {emoji} {action_text}{note}"
        )

    # ── Golden Cross ────────────────────────────────────────────────────
    if data["golden_cross"]:
        note = "RSI ≥ 75 — tighten stop / consider trimming" if rsi >= RSI_EXIT else ""
        return "exit", stock_line(
            "🌟", "Golden Cross confirmed. Stay invested. Trail stop below 50 EMA.", note
        )

    if gap >= 0:
        return None, ""   # already above 200 EMA, no cross this run

    alerts_fired = []
    if data["gap_closing"]:
        if STAGE4_MIN <= gap <= STAGE4_MAX: alerts_fired.append("wait")
        elif STAGE3_MIN <= gap <= STAGE3_MAX: alerts_fired.append("aggr")
        elif STAGE2_MIN <= gap <= STAGE2_MAX: alerts_fired.append("accum")
        elif STAGE1_MIN <= gap <= STAGE1_MAX: alerts_fired.append("watch")

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

    has_breakout = any(x in alerts_fired for x in ["breakout", "trendline_break"])
    has_support  = "support" in alerts_fired

    # RSI notes only — backtest showed overbought signals still outperformed
    # (62.4% win rate, +8.36% avg at 40d), so we no longer downgrade the bucket.
    # We just flag it so you know to size in carefully rather than chase.
    overbought = rsi > RSI_OVERBOUGHT
    rsi_note = ""
    if overbought:
        rsi_note = "RSI overbought — size in gradually, don't chase the full position at once."
    elif rsi < RSI_OVERSOLD:
        rsi_note = "RSI below 30 — verify no breakdown before entry."

    if has_breakout:
        return "buy_now", stock_line("🔥", "Buy today. Breakout confirmed with volume.", rsi_note)

    # NOTE: the old "Accumulate + Support" combo bucket is removed — backtest
    # showed it losing money (33% win rate, -2.49% avg return at 40d, worst
    # drawdown of any bucket). A stock in this state now falls through to its
    # plain Aggressive/Accumulate signal, with support only mentioned as a note.
    support_note = " Also sitting near a support zone." if has_support else ""

    if "aggr" in alerts_fired:
        return "aggr", stock_line("📈", f"Add aggressively. Cross expected in 2-4 weeks.{support_note}", rsi_note)

    if "accum" in alerts_fired:
        return "accum", stock_line("🟡", f"Start building. Buy 30-40% of planned amount.{support_note}", rsi_note)

    if has_support:
        return "support", stock_line("🛡", "Stock at support. Small entry with tight stop loss.", rsi_note)

    if "coiling" in alerts_fired:
        # Backtest showed ~51-53% win rate at every horizon — essentially a
        # coin flip so far. Kept visible but clearly marked unconfirmed until
        # the thresholds are retuned and re-validated.
        return "coiling", stock_line(
            "🌀", "UNCONFIRMED pre-breakout setup — squeeze + volume building near highs. Not yet backtest-validated, watch only."
        )

    if "wait" in alerts_fired:
        return "wait", stock_line("⏳", "Hold if invested. No fresh buying at this stage.")

    if "watch" in alerts_fired:
        return "watch", stock_line("👁", "Too early to buy. Research fundamentals now.")

    return None, ""


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────
def run_scanner():
    log.info("=" * 55)
    log.info("  VISH_SCAN v7 — STARTING")
    log.info("=" * 55)

    buckets = {
        "exit": [], "buy_now": [], "aggr": [], "accum": [],
        "support": [], "coiling": [], "wait": [], "watch": [],
    }

    # Maps your v7 internal bucket keys to the signal names
    # signal_tracker.py's TRACK_SIGNALS expects.
    BUCKET_TO_SIGNAL = {
        "exit":    "Golden Cross",
        "buy_now": "Buy Now",
        "aggr":    "Aggressive Accumulation",
        "accum":   "Accumulate",
        "support": "Support Zone",
    }

    all_data = fetch_all_data(WATCHLIST)
    scanned = len(all_data)
    skipped = len(WATCHLIST) - scanned

    alerts = {}   # only stocks that fired a tracked signal this run
    prices = {}   # every scanned stock, tracker needs this for milestone checks

    for symbol, data in all_data.items():
        prices[symbol] = data["price"]
        bucket, line = classify(data)
        if bucket and bucket in buckets:
            buckets[bucket].append(line)
            log.info(f"  {symbol} → {bucket}  gap={data['gap_today']}%")
            if bucket in BUCKET_TO_SIGNAL:
                alerts[symbol] = {"signal": BUCKET_TO_SIGNAL[bucket], "price": data["price"]}
        else:
            log.info(f"  {symbol} → no alert  gap={data['gap_today']}%")

    # Runs every time — even on quiet days — because it's checking
    # milestones on signals from weeks ago, not just today's new ones.
    tracking_text = track_and_report(alerts, prices)

    total = sum(len(v) for v in buckets.values())
    now_ist = datetime.now(IST)
    session = "🌅 Pre Market" if now_ist.hour < 12 else "🌆 Post Market"
    now_str = now_ist.strftime("%d %b %Y, %I:%M %p IST")

    if total == 0:
        base_msg = (
            f"📋 <b>VISH_SCAN {session}</b>\n{now_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ {scanned} stocks scanned ({skipped} skipped)\n"
            f"😴 No actionable signals today.\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 Patience is the edge."
        )
        send_telegram(base_msg + tracking_text)
        return

    msg = [
        f"📋 <b>VISH_SCAN {session}</b>",
        f"{now_str}",
        f"{scanned} scanned ({skipped} skipped) · {total} alerts",
    ]

    sections = [
        ("exit",     "🌟 GOLDEN CROSS — Stay Invested & Trail Stop"),
        ("buy_now",  "🔥 BUY NOW"),
        ("aggr",     "📈 AGGRESSIVE ACCUMULATION"),
        ("accum",    "🟡 ACCUMULATE"),
        ("support",  "🛡 SUPPORT ZONE"),
        ("wait",     "⏳ WAIT — Hold Only"),
        ("watch",    "👁 WATCHLIST — Too Early"),
        ("coiling",  "🌀 COILING — UNCONFIRMED, backtest still shows ~coin-flip odds"),
    ]

    for key, heading in sections:
        if buckets[key]:
            msg.append(f"\n━━━━━━━━━━━━━━━━━━━━━━")
            msg.append(f"<b>{heading}</b>")
            msg.extend(buckets[key])

    msg.append(f"\n━━━━━━━━━━━━━━━━━━━━━━")
    msg.append(f"⚠️ Not SEBI advice. Personal use only.")

    if tracking_text:
        msg.append(tracking_text)

    send_telegram_chunked(msg)
    log.info(f"DONE — {total} alerts sent.")
    log.info("=" * 55)


if __name__ == "__main__":
    run_scanner()
