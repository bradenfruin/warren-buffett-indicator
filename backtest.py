import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------
# 1) Load Fear & Greed Index (CSV)
# ------------------------------


FGI_CSV_URL = "https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/main/fear-greed-2011-2023.csv"

SELL_MULTIPLIER = 0.5




def fetch_fgi_csv(start_date="2011-01-01"):
    df = pd.read_csv(FGI_CSV_URL)

    # --- detect date column ---
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"Couldn't find a date column. Columns: {list(df.columns)}")

    # --- detect FGI/value column ---
    value_col = None
    # prefer columns that clearly mean fear+greed
    for c in df.columns:
        cl = c.lower()
        if ("fear" in cl and "greed" in cl) or cl in ("fgi", "fear_greed", "fear&greed", "value", "index"):
            if c != date_col:
                value_col = c
                break
    # fallback: first numeric column that isn't the date col
    if value_col is None:
        numeric_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            value_col = numeric_cols[0]

    if value_col is None:
        raise ValueError(f"Couldn't find an FGI/value column. Columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df[date_col])
    df["fgi"] = pd.to_numeric(df[value_col], errors="coerce")

    df = df[["date", "fgi"]].dropna()
    df = df[df["date"] >= pd.to_datetime(start_date)]
    df = df.sort_values("date").reset_index(drop=True)

    # sanity check: show range so you KNOW you extended past 2020
    print(f"FGI date range: {df['date'].min().date()} → {df['date'].max().date()}")

    return df

# ------------------------------
# 2) Fetch SPY data (robust)
# ------------------------------

def fetch_spy_history(start_date="2011-01-01"):
    df = yf.download(
        "SPY",
        start=start_date,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError("No SPY data returned from Yahoo Finance.")

    # FIX: yfinance can return MultiIndex columns in some environments
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten: keep the first level names (Open/High/Low/Close/Volume)
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # After flattening, we should have columns like: Date, Open, High, Low, Close, Volume
    if "Close" not in df.columns:
        raise ValueError(f"Expected 'Close' column but got columns: {list(df.columns)}")

    df.rename(columns={"Date": "date", "Close": "spy_close"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])

    return df[["date", "spy_close"]].dropna()

# ------------------------------
# 3) Cubic Fear & Greed Rule
# ------------------------------

def y_raw_from_fgi(x):
    """
    y = -((x - 60)^3) / 2160

    +y => buy % of cash
    -y => sell % of invested value
    """
    x = max(0, min(100, float(x)))
    return -((x - 60) ** 3) / 2160

# ------------------------------
# 4) Backtest Strategy
# ------------------------------

def backtest_fgi_cubic(start_cash=100_000, start_date="2011-01-01"):

    fgi = fetch_fgi_csv(start_date)
    spy = fetch_spy_history(start_date)

    df = pd.merge(fgi, spy, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    cash = float(start_cash)
    shares = 0.0
    records = []

    for _, row in df.iterrows():
        price = float(row["spy_close"])
        fgi_val = float(row["fgi"])
        y = y_raw_from_fgi(fgi_val)

        invested_value = shares * price

        # BUY: y% of current cash
        if y > 0:
            pct = min(100.0, y)
            dollars = cash * (pct / 100.0)
            shares += dollars / price
            cash -= dollars
            action = "BUY"

        # SELL: |y|% of current invested value
        elif y < 0:
            pct = min(100.0, SELL_MULTIPLIER * abs(y))   # <-- sell half as aggressively
            dollars = invested_value * (pct / 100.0)
            sell_shares = min(shares, dollars / price)
            shares -= sell_shares
            cash += sell_shares * price
            action = "SELL"


        else:
            action = "HOLD"

        equity = cash + shares * price

        records.append({
            "date": row["date"],
            "FGI": fgi_val,
            "y_raw_%": y,
            "action": action,
            "cash_$": cash,
            "invested_$": shares * price,
            "equity_$": equity
        })

    return pd.DataFrame(records)

# ------------------------------
# 5) Buy & Hold SPY Benchmark
# ------------------------------

def buy_and_hold_spy(spy_df, start_cash=100_000):
    first_price = float(spy_df["spy_close"].iloc[0])
    shares = start_cash / first_price
    bh = spy_df.copy()
    bh["bh_equity_$"] = shares * bh["spy_close"]
    return bh[["date", "bh_equity_$"]]

# ------------------------------
# 6) Performance Metrics
# ------------------------------

def total_return(series):
    return series.iloc[-1] / series.iloc[0] - 1

def max_drawdown(series):
    peak = series.cummax()
    return (series / peak - 1).min()

def cagr(series, dates):
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1

def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd

def running_max_drawdown(dd: pd.Series) -> pd.Series:
    # Most negative drawdown seen so far (non-increasing)
    return dd.cummin()


# ------------------------------
# 7) Run Everything
# ------------------------------

if __name__ == "__main__":

    START_CASH = 100_000
    START_DATE = "2011-01-01"

    results = backtest_fgi_cubic(START_CASH, START_DATE)

    spy = fetch_spy_history(START_DATE)
    bh = buy_and_hold_spy(spy, START_CASH)

    comparison = results.merge(bh, on="date", how="inner")

    strat = comparison["equity_$"]
    bh_eq = comparison["bh_equity_$"]
    dates = comparison["date"]
    final_row = comparison.iloc[-1]
    
    #================================
    # DD stuff
    #===============================
    
    # Compute drawdowns for both curves
    comparison["dd_fgi"] = drawdown_series(comparison["equity_$"])
    comparison["dd_bh"]  = drawdown_series(comparison["bh_equity_$"])

    # Compute running max drawdown (optional but matches "max DD vs time")
    comparison["maxdd_fgi"] = running_max_drawdown(comparison["dd_fgi"])
    comparison["maxdd_bh"]  = running_max_drawdown(comparison["dd_bh"])

    print("\n=== FINAL BACKTEST RESULTS ===\n")

    print("FGI Cubic Strategy:")
    print(f"  Final Cash:   ${final_row['cash_$']:,.2f}")
    print(f"  Final Equity: ${final_row['equity_$']:,.2f}")
    
    print("\nBuy & Hold SPY:")
    print(f"  Final Equity: ${final_row['bh_equity_$']:,.2f}")

    print(f"FGI Strategy Max Drawdown: {max_drawdown(strat):.2%}")
    print(f"Buy & Hold Max Drawdown:   {max_drawdown(bh_eq):.2%}\n")

    print(f"FGI Strategy CAGR: {cagr(strat, dates):.2%}")
    print(f"Buy & Hold CAGR:   {cagr(bh_eq, dates):.2%}\n")
    #==========================================
    # Graph Stuff
    #==========================================
    os.makedirs("report", exist_ok=True)



    # -------------------------
    # Plot 1: Drawdown (underwater) over time
    # -------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(comparison["date"], comparison["dd_fgi"] * 100, label="FGI Strategy DD")
    plt.plot(comparison["date"], comparison["dd_bh"] * 100, label="Buy & Hold DD")
    plt.title("Drawdown Over Time (Underwater Plot)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("report/drawdown.png", dpi=200, bbox_inches="tight")
    plt.close()


    # -------------------------
    # Plot 2: Running Max Drawdown to date (max DD vs time)
    # -------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(comparison["date"], comparison["maxdd_fgi"] * 100, label="FGI Strategy Max DD to Date")
    plt.plot(comparison["date"], comparison["maxdd_bh"] * 100, label="Buy & Hold Max DD to Date")
    plt.title("Running Maximum Drawdown (Worst So Far)")
    plt.xlabel("Date")
    plt.ylabel("Max Drawdown to Date (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("report/drawdown.png", dpi=200, bbox_inches="tight")
    plt.close()

    
    comparison["date"] = pd.to_datetime(comparison["date"])

    plt.figure(figsize=(10, 6))
    plt.plot(comparison["date"], comparison["equity_$"], label="FGI Cubic Strategy")
    plt.plot(comparison["date"], comparison["bh_equity_$"], label="Buy & Hold SPY")

    plt.title("Equity Curve: FGI Strategy vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("report/drawdown.png", dpi=200, bbox_inches="tight")
    plt.close()



