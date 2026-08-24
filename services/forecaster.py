import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Modeli ve Scaler'ları yüklüyoruz
model = load_model("models/gru_model.keras")
scaler_x = joblib.load("models/gru_scaler_x.pkl")
scaler_y = joblib.load("models/gru_scaler_y.pkl")


def compute_RSI(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))


def compute_MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def add_features(df):
    df = df.sort_values("Date").copy()
    cols = ["Open", "High", "Low", "Close", "Volume", "Adjusted Close"]
    df[cols] = df[cols].ffill()

    # Trend Göstergeleri
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_60"] = df["Close"].rolling(60).mean()
    df["MA_ratio"] = (df["MA_20"] / df["MA_60"]) - 1

    # Momentum Göstergeleri
    df["Daily_Return"] = df["Close"].pct_change()
    df["RSI_14"] = compute_RSI(df["Close"], 14)
    macd, macd_signal, macd_hist = compute_MACD(df["Close"])
    df["MACD"] = macd
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist

    # Volatilite ve Hacim Göstergeleri
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std()
    df["Volume_MA_20"] = df["Volume"].rolling(20).mean()
    df["Volume_Change"] = df["Volume"].pct_change()

    return df


def get_stock_data(symbol: str):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1y").reset_index()

    if data.empty or len(data) < 60:
        raise ValueError(f"{symbol} için yeterli borsa verisi çekilemedi.")

    if "Adj Close" in data.columns:
        data = data.rename(columns={"Adj Close": "Adjusted Close"})
    elif "Adjusted Close" not in data.columns:
        data["Adjusted Close"] = data["Close"]

    last_real_close = float(data["Close"].iloc[-1])

    # Göstergeleri ekle ve temizle
    df = add_features(data)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if len(df) < 60:
        raise ValueError(f"{symbol} için yeterli veri kalmadı.")

    # Log dönüşümleri
    price_cols = ["Open", "High", "Low", "Close", "Adjusted Close", "MA_20", "MA_60"]
    df[price_cols] = np.log(df[price_cols])
    df["Volume"] = np.log1p(df["Volume"])
    df["Volume_MA_20"] = np.log1p(df["Volume_MA_20"])

    target_col = "Adjusted Close"
    x_cols = [
        "Close", "Open", "High", "Low", "Volume",
        "MA_20", "MA_60", "MA_ratio", "Daily_Return",
        "Volatility_20", "Volume_MA_20", "Volume_Change",
        "RSI_14", "MACD", "MACD_signal", "MACD_hist"
    ]

    # Son 60 günü alıyoruz
    last_60_df = df.tail(60)
    target_matrix = last_60_df[[target_col]].values  # (60, 1)
    features_matrix = last_60_df[x_cols].values       # (60, 16)

    return target_matrix, features_matrix, last_real_close


def predict_stock_price(symbol: str, days: int = 1):
    # 1. Ham verileri al
    target_matrix, features_matrix, last_real_close = get_stock_data(symbol)

    # 2. İlgili scaler'lar ile ayrı ayrı ölçekle
    scaled_target = scaler_y.transform(target_matrix)       # (60, 1)
    scaled_features = scaler_x.transform(features_matrix)   # (60, 16)

    # 3. İkisini yan yana birleştirerek 17 sütun elde et (60, 17)
    scaled_17 = np.hstack([scaled_target, scaled_features])

    # 4. Modeli çalıştır: (1, 60, 17)
    input_tensor = scaled_17.reshape(1, 60, 17)
    pred_scaled = model.predict(input_tensor, verbose=0)

    # 5. scaler_y ile ters dönüşüm yap ve log'dan çıkar (np.exp)
    pred_log = scaler_y.inverse_transform(pred_scaled)
    real_price = float(np.exp(pred_log)[0, 0])
    predicted_price = round(real_price, 2)

    trend = "Yükseliş" if predicted_price > last_real_close else "Düşüş"

    return {
        "symbol": symbol.upper(),
        "forecast_days": 1,
        "predictions": [predicted_price],
        "trend": trend,
        "last_close_price": round(last_real_close, 2)
    }


if __name__ == "__main__":
    sonuc = predict_stock_price("AAPL", days=1)
    print("\n--- Model Tahmin Çıktısı ---")
    print(sonuc)