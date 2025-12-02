# Dataset Information

This project automatically downloads financial market data using
**Yahoo Finance (yfinance)** — no external files or accounts are required.

## 📥 Data Source  
- Library: `yfinance`
- Ticker Symbol: `AAPL` (modifiable)
- Period: `2 years`

## 📊 Columns Included  
The dataset includes the following fields:

| Column  | Description |
|---------|-------------|
| Open    | Opening price of the day |
| High    | Highest price of the day |
| Low     | Lowest price of the day |
| Close   | Final price of the day |
| Volume  | Number of shares traded |
| Return  | Percentage daily return |
| Direction | Next-day direction label (1 = Up, 0 = Down) |
| SMA10   | 10-day Simple Moving Average |
| SMA50   | 50-day Simple Moving Average |
| EMA20   | 20-day Exponential Moving Average |
| RSI14   | Relative Strength Index (14-day) |

## 🧮 How the Dataset is Generated

The script performs:

1. **Download OHLCV data**  
2. **Create technical indicators**  
3. **Generate the supervised learning label**  
4. **Clean NaN rows**  
5. **Send final DataFrame for modeling**

## 🔒 No Manual Download Required  
Running `trading_model.py` automatically produces the dataset during execution.

