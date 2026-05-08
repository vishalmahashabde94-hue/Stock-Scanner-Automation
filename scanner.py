
import os

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
import os
import yfinance as yf
import pandas as pd
import requests
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": text}
    requests.get(url, params=params)

stocks = [
    "HBLENGINE.NS",
    "M&M.NS",
    "BHARTIARTL.NS",
    "BANCOINDIA.NS",
    "PARAS.NS",
    "ZENTEC.NS",
    "GOKULAGRO.NS",
    "MOTILALOFS.NS",
    "ANGELONE.NS",
    "PRECWIRE.NS",
    "SCHNEIDER.NS",
    "ASHOKLEY.NS",
    "TVSMOTOR.NS",
    "BAJFINANCE.NS",
    "ADANIGREEN.NS",
    "ADANIPOWER.NS",
    "ADANIPORTS.NS",
    "DATAPATTNS.NS",
    "LLOYDSME.NS",
    "VBL.NS",
    "DIXON.NS",
    "BSE.NS",
    "NATCOPHARM.NS",
    "AXISBANK.NS"
]

for stock_name in stocks:
    print(f"\nChecking: {stock_name}")

    try:
        data = yf.download(stock_name, period="1y", progress=False, auto_adjust=False)

        if data.empty or len(data) < 210:
            print("Not enough data")
            continue

        close = data["Close"]
        volume = data["Volume"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]

        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        avg_volume_20 = volume.rolling(20).mean()

        latest_price = float(close.iloc[-1])
        latest_50 = float(ma50.iloc[-1])
        latest_200 = float(ma200.iloc[-1])

        prev_50 = float(ma50.iloc[-2])
        prev_200 = float(ma200.iloc[-2])

        old_50 = float(ma50.iloc[-5])
        old_200 = float(ma200.iloc[-5])

        latest_volume = float(volume.iloc[-1])
        avg_vol = float(avg_volume_20.iloc[-1])

        gap_today = ((latest_200 - latest_50) / latest_200) * 100
        gap_5_days_ago = ((old_200 - old_50) / old_200) * 100

        price_distance_from_50 = ((latest_price - latest_50) / latest_50) * 100
        price_distance_from_200 = ((latest_price - latest_200) / latest_200) * 100

        volume_ratio = latest_volume / avg_vol if avg_vol > 0 else 0

        gap_shrinking = gap_today < gap_5_days_ago
        ma50_rising = latest_50 > old_50
        price_near_zone = abs(price_distance_from_50) <= 10 and abs(price_distance_from_200) <= 10

        fresh_crossover = latest_50 >= latest_200 and prev_50 < prev_200

        stage = None

        if latest_50 < latest_200 and 5 < gap_today <= 12 and gap_shrinking and ma50_rising:
            stage = "🟡 STAGE 1: WATCHLIST ZONE"

        elif latest_50 < latest_200 and 1 < gap_today <= 5 and gap_shrinking and ma50_rising and price_near_zone:
            stage = "🟠 STAGE 2: STRONG CONVERGENCE ZONE"

        elif ((0 <= gap_today <= 1) or fresh_crossover) and ma50_rising and price_near_zone:
            stage = "🔴 STAGE 3: ACTION ZONE"

        if stage:
            msg = (
                f"{stage}\n\n"
                f"Stock: {stock_name}\n"
                f"Price: {latest_price:.2f}\n"
                f"50 DMA: {latest_50:.2f}\n"
                f"200 DMA: {latest_200:.2f}\n\n"
                f"DMA Gap Today: {gap_today:.2f}%\n"
                f"DMA Gap 5 Days Ago: {gap_5_days_ago:.2f}%\n"
                f"Price vs 50 DMA: {price_distance_from_50:.2f}%\n"
                f"Price vs 200 DMA: {price_distance_from_200:.2f}%\n"
                f"Volume vs 20D Avg: {volume_ratio:.2f}x\n\n"
                f"Signal Meaning: 50 DMA and 200 DMA are moving closer."
            )

            print(msg)
            send_message(msg)

        else:
            print(
                f"No setup | "
                f"Gap Today: {gap_today:.2f}% | "
                f"Gap 5D Ago: {gap_5_days_ago:.2f}% | "
                f"Volume: {volume_ratio:.2f}x"
            )

    except Exception as e:
        print(f"{stock_name}: Error {e}")
