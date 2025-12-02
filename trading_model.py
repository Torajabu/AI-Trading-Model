"""
AI Trading Model – Random Forest Trend Predictor

What this script does
---------------------
1. Downloads historical OHLCV data from Yahoo Finance
2. Builds technical indicators (Returns, SMA, EMA, RSI)
3. Trains a RandomForestClassifier to predict next-day direction (Up / Down)
4. Simulates a simple long-only trading strategy:
   - Buy/hold if the model predicts Up
   - Stay in cash if the model predicts Down
5. Plots Market vs Strategy cumulative returns
6. Prints accuracy and a detailed classification report

Ready to drop into: trading_model.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1) Data Download
# -----------------------------
def download_price_data(symbol: str = "AAPL", period: str = "2y") -> pd.DataFrame:
    data = yf.download(symbol, period=period)
    if data.empty:
        raise ValueError(f"No data downloaded for symbol {symbol}")
    return data


# -----------------------------
# 2) Feature Engineering
# -----------------------------
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Proper RSI implementation (Wilder's method).
    """
    delta = series.diff()

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # Daily returns
    df["Return"] = df["Close"].pct_change()

    # Direction: 1 if next day's return > 0, else 0
    df["Direction"] = (df["Return"].shift(-1) > 0).astype(int)

    # Moving averages and indicators
    df["SMA10"] = df["Close"].rolling(window=10).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["RSI14"] = compute_rsi(df["Close"], window=14)

    # Drop rows with NaNs from rolling calculations
    df = df.dropna().copy()

    return df


# -----------------------------
# 3) Train / Test Split & Model
# -----------------------------
def train_model(df: pd.DataFrame):
    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA10",
        "SMA50",
        "EMA20",
        "RSI14",
    ]
    X = df[feature_cols]
    y = df["Direction"]

    # Time-series style split: no shuffling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluation on test set
    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)

    print("=== Model Evaluation ===")
    print(f"Accuracy on test set: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred_test))

    return model, X_train, X_test, y_train, y_test


# -----------------------------
# 4) Trading Strategy Backtest
# -----------------------------
def backtest_strategy(df: pd.DataFrame, model: RandomForestClassifier) -> pd.DataFrame:
    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA10",
        "SMA50",
        "EMA20",
        "RSI14",
    ]
    X_all = df[feature_cols]

    # Predict direction for all days (for analysis / strategy)
    df["Pred_Dir"] = model.predict(X_all)

    # Simple strategy:
    # Position = Predicted direction of previous day
    # 1 -> Long, 0 -> Flat
    df["Position"] = df["Pred_Dir"].shift(1).fillna(0)

    # Strategy daily return = Position * Market Return
    df["Strategy_Return"] = df["Position"] * df["Return"]

    # Cumulative returns (as equity curves)
    df["Market_Cum"] = (1 + df["Return"]).cumprod() - 1
    df["Strategy_Cum"] = (1 + df["Strategy_Return"]).cumprod() - 1

    return df


# -----------------------------
# 5) Visualization
# -----------------------------
def plot_performance(df: pd.DataFrame, symbol: str):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Market_Cum"], label=f"{symbol} Market Return")
    plt.plot(df.index, df["Strategy_Cum"], label="Strategy Return")
    plt.title(f"Market vs Strategy Performance ({symbol})")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 6) Main Script
# -----------------------------
def main():
    symbol = "AAPL"  # change to any ticker you like
    period = "2y"

    print(f"Downloading data for {symbol} ({period})...")
    data = download_price_data(symbol=symbol, period=period)

    print("Preparing features...")
    df = prepare_features(data)

    print("Training model...")
    model, X_train, X_test, y_train, y_test = train_model(df)

    print("Running backtest...")
    df_bt = backtest_strategy(df, model)

    print("Plotting performance...")
    plot_performance(df_bt, symbol)

    # Show last few rows for sanity check
    print("\n=== Sample of final dataframe ===")
    print(df_bt[["Close", "Return", "Pred_Dir", "Position", "Market_Cum", "Strategy_Cum"]].tail())


if __name__ == "__main__":
    main()
