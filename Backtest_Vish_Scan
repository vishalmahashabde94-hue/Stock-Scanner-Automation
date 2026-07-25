"""
=======================================================
  VISH_SCAN — BACKTEST MODULE
  Validates v6 signal logic against historical data.

  For every day in the last N years, re-runs the same
  classification logic used live (gap stages, Golden
  Cross, breakout+volume, coiling, support) and then
  measures forward returns at 5/10/20/40 trading days
  to answer: "does this signal actually work?"

  Run standalone — no Telegram credentials required.
  Output: console summary + CSV of every signal fired
  with its forward return, for your own deeper analysis.
=======================================================
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── Same watchlist as the live scanner ──────────────────────────────────
WATCHLIST = [
    "HBLENGINE.NS", "PARASDYNE.NS", "ZENTEC.NS", "DATAPATTNS.NS", "MAZDOCK.NS",
    "HAL.NS", "BEL.NS", "ASTRAMICRO.NS", "BHARATFORG.NS", "MTARTECH.NS", "IDEAFORGE.NS",
    "M&M.NS", "ASHOKLEY.NS", "TVSMOTOR.NS", "BANCOINDIA.NS", "PRECWIRE.NS",
    "MOTHERSON.NS", "ENDURANCE.NS", "TIINDIA.NS", "MOTHERSONSUM.NS",
    "BALUFORGE.NS", "SMLISUZU.NS",
    "HAVELLS.NS", "POLYCAB.NS", "KEIIND.NS", "SCHNEIDER.NS", "CGPOWER.NS",
    "TRANSRAILL.NS", "TRIL.NS", "VOLTAMP.NS", "GVT&D.NS",
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
    "LLOYDMETAL.NS", "COALINDIA.NS", "RELIANCE.NS",
    "NATCOPHARM.NS", "RUBICON.NS",
    "YATHARTH.NS", "KIIMS.NS", "KIMS.NS", "NH.NS", "RAINBOW.NS",
    "KAJARIACER.NS",
    "AEGISLOG.NS",
    "PRICOLLTD.NS", "CCL.NS",
]

BACKTEST_PERIOD = "3y"
HORIZONS = [5, 10, 20, 40]   # trading days forward

# ── Same thresholds as vish_scan_v6.py ──────────────────────────────────
STAGE1_MIN, STAGE1_MAX = -20.0, -12.0
STAGE2_MIN, STAGE2_MAX = -12.0,  -8.0
STAGE3_MIN, STAGE3_MAX =  -8.0,  -4.0
STAGE4_MIN, STAGE4_MAX =  -4.0,  -2.0

GOLDEN_CROSS_LOOKBACK = 5
SLOPE_LOOKBACK        = 3
MIN_SLOPE_MOVE        = 0.15

RSI_OVERBOUGHT = 70
RSI_OVERSOLD   = 30

VOLUME_CONFIRM       = 1.5
VOL_SMOOTH_MIN_RATIO = 1.2
VOL_SMOOTH_MIN_DAYS  = 2

BREAKOUT_3M_DAYS = 63
BREAKOUT_6M_DAYS = 126
SUPPORT_ZONE     = 0.02

COIL_PROXIMITY = 0.05
COIL_RSI_MIN   = 55
COIL_RSI_MAX   = 68
COIL_MIN_HITS  = 3

# NOTE on fidelity to the live scanner:
# Support detection here uses a rolling-low proxy (20-day rolling min,
# rising) instead of the live scanner's swing-low linear-regression
# trendline. A per-day regression fit for every stock/day in a 3-year
# backtest is expensive and the rolling-min proxy captures the same
# "near a rising floor" idea closely enough for validation purposes.
# Everything else (gap stages, Golden Cross, breakout+volume, coiling,
# RSI gate) matches the live logic exactly.


def compute_rsi(close, period=14):
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def fetch_all(symbols):
    log.info(f"Downloading {len(symbols)} tickers, period={BACKTEST_PERIOD}...")
    raw = yf.download(
        symbols, period=BACKTEST_PERIOD, interval="1d",
        auto_adjust=True, progress=False, group_by="ticker", threads=True,
    )
    return raw


def build_signal_frame(symbol, df):
    """Vectorized recreation of the live classify() logic across full history."""
    df = df.dropna(how="all")
    if len(df) < 210:
        return None

    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    gap = ((ema50 - ema200) / ema200 * 100)

    gap_3ago = gap.shift(SLOPE_LOOKBACK)
    gap_closing = (gap - gap_3ago) > MIN_SLOPE_MOVE

    prior_window_neg = gap.shift(1).rolling(GOLDEN_CROSS_LOOKBACK).apply(
        lambda x: np.all(x < 0), raw=True
    ).astype(bool)
    golden_cross = prior_window_neg & (gap >= 0)

    rsi = compute_rsi(close)

    vol_avg20 = volume.rolling(20).mean()
    vol_ratio = volume / vol_avg20
    last3_hits = vol_ratio.rolling(3).apply(
        lambda x: (x >= VOL_SMOOTH_MIN_RATIO).sum(), raw=True
    )
    smoothed_vol_ok = last3_hits >= VOL_SMOOTH_MIN_DAYS
    vol_confirm = (vol_ratio >= VOLUME_CONFIRM) | smoothed_vol_ok

    high_3m_prior = high.shift(1).rolling(BREAKOUT_3M_DAYS).max()
    high_6m_prior = high.shift(1).rolling(BREAKOUT_6M_DAYS).max()
    breakout_3m = (close > high_3m_prior) & (close.shift(1) <= high_3m_prior)
    breakout_6m = (close > high_6m_prior) & (close.shift(1) <= high_6m_prior)
    breakout_confirmed = (breakout_3m | breakout_6m) & vol_confirm

    # Support proxy — rolling 20-day low, rising, price within 2% of it
    roll_low20 = low.rolling(20).min()
    roll_low20_rising = roll_low20 > roll_low20.shift(5)
    near_support = (
        roll_low20_rising & (gap < 0) &
        ((close - roll_low20).abs() / roll_low20 <= SUPPORT_ZONE)
    )

    # Coiling / pre-breakout
    near_high = ((high_3m_prior - close) / high_3m_prior <= COIL_PROXIMITY) & (close <= high_3m_prior)
    rsi_building = rsi.between(COIL_RSI_MIN, COIL_RSI_MAX)
    ret = close.pct_change()
    vol10 = ret.rolling(10).std()
    vol20 = ret.rolling(20).std()
    squeeze = vol10 < (vol20 * 0.85)
    vol_recent3 = volume.rolling(3).mean()
    vol_prior10 = volume.shift(3).rolling(10).mean()
    vol_trend_up = vol_recent3 > vol_prior10
    coil_hits = near_high.astype(int) + rsi_building.astype(int) + squeeze.astype(int) + vol_trend_up.astype(int)
    coiling = (coil_hits >= COIL_MIN_HITS) & ~breakout_confirmed

    out = pd.DataFrame({
        "close": close, "gap": gap, "gap_closing": gap_closing,
        "golden_cross": golden_cross, "rsi": rsi, "vol_ratio": vol_ratio,
        "breakout_confirmed": breakout_confirmed, "near_support": near_support,
        "coiling": coiling,
    })
    return out


def assign_bucket(row):
    """Mirrors classify() priority order in vish_scan_v6.py."""
    gap = row["gap"]

    if row["golden_cross"]:
        return "exit_golden_cross"
    if gap >= 0:
        return None

    overbought = row["rsi"] > RSI_OVERBOUGHT

    stage = None
    if row["gap_closing"]:
        if STAGE4_MIN <= gap <= STAGE4_MAX: stage = "wait"
        elif STAGE3_MIN <= gap <= STAGE3_MAX: stage = "aggr"
        elif STAGE2_MIN <= gap <= STAGE2_MAX: stage = "accum"
        elif STAGE1_MIN <= gap <= STAGE1_MAX: stage = "watch"

    if row["breakout_confirmed"]:
        return "watch_overbought" if overbought else "buy_now"

    if stage in ("aggr", "accum") and row["near_support"]:
        return "watch_overbought" if overbought else "accum_support"

    if stage == "aggr":
        return "watch_overbought" if overbought else "aggr"
    if stage == "accum":
        return "watch_overbought" if overbought else "accum"

    if row["near_support"]:
        return "support"
    if row["coiling"]:
        return "coiling"
    if stage == "wait":
        return "wait"
    if stage == "watch":
        return "watch"

    return None


def extract_events(sig_df, symbol):
    """Only count the FIRST day a bucket fires (avoid counting a 20-day
    persistent Accumulate stage as 20 separate 'signals')."""
    sig_df = sig_df.copy()
    sig_df["bucket"] = sig_df.apply(assign_bucket, axis=1)
    sig_df["is_new"] = (sig_df["bucket"] != sig_df["bucket"].shift(1)) & sig_df["bucket"].notna()

    events = []
    idx = sig_df.index
    close = sig_df["close"]
    for i, (ts, is_new) in enumerate(zip(idx, sig_df["is_new"])):
        if not is_new:
            continue
        bucket = sig_df["bucket"].iloc[i]
        entry_price = close.iloc[i]
        row = {"symbol": symbol, "date": ts, "bucket": bucket, "entry_price": entry_price}
        for h in HORIZONS:
            if i + h < len(close):
                fwd = close.iloc[i + h]
                row[f"ret_{h}d"] = round((fwd / entry_price - 1) * 100, 2)
                window = close.iloc[i:i + h + 1]
                row[f"mdd_{h}d"] = round((window.min() / entry_price - 1) * 100, 2)
            else:
                row[f"ret_{h}d"] = np.nan
                row[f"mdd_{h}d"] = np.nan
        events.append(row)
    return events


def summarize(events_df):
    print("\n" + "=" * 78)
    print("  BACKTEST SUMMARY — signal accuracy by bucket")
    print("=" * 78)

    for bucket, group in events_df.groupby("bucket"):
        print(f"\n▶ {bucket}   (n={len(group)} signals)")
        header = f"{'Horizon':>8} | {'Win Rate':>9} | {'Avg Ret':>8} | {'Median':>8} | {'Avg MaxDD':>10}"
        print(header)
        print("-" * len(header))
        for h in HORIZONS:
            col = f"ret_{h}d"
            mdd_col = f"mdd_{h}d"
            valid = group[col].dropna()
            if valid.empty:
                continue
            win_rate = (valid > 0).mean() * 100
            avg_ret = valid.mean()
            median_ret = valid.median()
            avg_mdd = group[mdd_col].dropna().mean()
            print(f"{h:>6}d  | {win_rate:>8.1f}% | {avg_ret:>7.2f}% | {median_ret:>7.2f}% | {avg_mdd:>9.2f}%")

    print("\n" + "=" * 78)
    print("  Read this as: for each bucket, does the stock actually move up")
    print("  in the following N days, or is the signal noise?")
    print("  Avg MaxDD shows the typical drawdown risk you'd have sat through.")
    print("=" * 78)


def run_backtest():
    raw = fetch_all(WATCHLIST)
    all_events = []

    for symbol in WATCHLIST:
        try:
            df = raw[symbol] if len(WATCHLIST) > 1 else raw
            sig_df = build_signal_frame(symbol, df)
            if sig_df is None:
                log.warning(f"  {symbol}: insufficient data, skipped")
                continue
            events = extract_events(sig_df, symbol)
            all_events.extend(events)
            log.info(f"  {symbol}: {len(events)} signal events")
        except Exception as e:
            log.error(f"  {symbol}: error — {e}")

    if not all_events:
        log.warning("No events generated — check data availability.")
        return

    events_df = pd.DataFrame(all_events)
    events_df.to_csv("backtest_signals.csv", index=False)
    log.info(f"Saved {len(events_df)} raw signal events to backtest_signals.csv")

    summarize(events_df)


if __name__ == "__main__":
    run_backtest()
