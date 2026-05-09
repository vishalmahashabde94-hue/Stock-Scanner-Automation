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
    try:
        data = yf.download(stock_name, period="1y", progress=False, auto_adjust=False)

        if data.empty or len(data) < 210:
            print(f"{stock_name}: Not enough data")
            continue

        close = data["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        latest_price = float(close.iloc[-1])
        latest_50 = float(ma50.iloc[-1])
        latest_200 = float(ma200.iloc[-1])

        prev_50 = float(ma50.iloc[-2])
        prev_200 = float(ma200.iloc[-2])

        gap_percent = abs((latest_200 - latest_50) / latest_200) * 100

        stage = None

        if latest_50 >= latest_200 and prev_50 < prev_200:
            stage = "🚀 GOLDEN CROSSOVER HAPPENED"

        elif latest_50 > latest_200:
            stage = "✅ ALREADY ABOVE 200 DMA"

        elif 8 <= gap_percent <= 12:
            stage = "🟡 STAGE 1: 10% GAP WATCHLIST ZONE"

        elif 5 <= gap_percent < 8:
            stage = "🟠 STAGE 2: 6–7% CONVERGENCE ZONE"

        elif 0 < gap_percent < 5:
            stage = "🔴 STAGE 3: 2–3% ACTION ZONE"

        if stage:
            msg = (
                f"{stage}\n\n"
                f"Stock: {stock_name}\n"
                f"Price: {latest_price:.2f}\n"
                f"50 DMA: {latest_50:.2f}\n"
                f"200 DMA: {latest_200:.2f}\n"
                f"DMA Gap: {gap_percent:.2f}%\n\n"
                f"Meaning: Tracking 50DMA vs 200DMA movement."
            )

            print(msg)
            send_message(msg)

        else:
            print(
                f"{stock_name}: No setup | "
                f"50DMA: {latest_50:.2f} | "
                f"200DMA: {latest_200:.2f} | "
                f"Gap: {gap_percent:.2f}%"
            )

    except Exception as e:
        print(f"{stock_name}: Error {e}")