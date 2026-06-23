"""
Stock Multi-Factor Strategy Engine
Combines momentum, mean-reversion, trend, and volatility factors.
Walk-forward validated. Daily bars via yfinance.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# ── Factors ──────────────────────────────────────────────────────────
def momentum_factor(close: pd.Series, period: int = 20) -> pd.Series:
    """Rate of change. Positive = uptrend."""
    return close.pct_change(period)

def rsi_factor(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI normalized to [-1, 1]. Low RSI = oversold (buy signal)."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return -(rsi - 50) / 50  # normalized: -1 (overbought) to +1 (oversold)

def trend_factor(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """MA crossover. Bullish when fast > slow."""
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    return (ma_fast - ma_slow) / close * 100

def volatility_factor(close: pd.Series, period: int = 20) -> pd.Series:
    """High vol = fear = potential reversal (contrarian)."""
    ret = close.pct_change()
    vol = ret.rolling(period).std()
    vol_rank = vol.rolling(252).rank(pct=True)
    return vol_rank - 0.5  # center around 0

def volume_factor(volume: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Volume spike with price confirmation."""
    vol_ma = volume.rolling(period).mean()
    vol_ratio = volume / vol_ma
    price_dir = np.sign(close.pct_change())
    return vol_ratio * price_dir


# ── Signal Combiner ──────────────────────────────────────────────────
def compute_composite_signal(df: pd.DataFrame, weights: dict = None) -> pd.Series:
    """Compute weighted composite signal from multiple factors."""
    if weights is None:
        weights = {
            "momentum": 0.30,
            "rsi": 0.25,
            "trend": 0.25,
            "volatility": 0.10,
            "volume": 0.10,
        }

    close = df["close"]
    volume = df.get("volume", pd.Series(1, index=close.index))

    factors = {}
    factors["momentum"] = momentum_factor(close, 20).fillna(0)
    factors["rsi"] = rsi_factor(close, 14).fillna(0)
    factors["trend"] = trend_factor(close, 20, 50).fillna(0)
    factors["volatility"] = volatility_factor(close, 20).fillna(0)
    factors["volume"] = volume_factor(volume, close, 20).fillna(0)

    # Normalize each factor to [-1, 1]
    for name in factors:
        f = factors[name]
        f_max = f.abs().quantile(0.99)
        if f_max > 0:
            factors[name] = (f / f_max).clip(-1, 1)

    # Weighted sum
    signal = pd.Series(0.0, index=close.index)
    for name, w in weights.items():
        signal += w * factors[name]

    return signal


# ── Backtest ─────────────────────────────────────────────────────────
def run_stock_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    threshold: float = 0.3,
    stop_loss_pct: float = 0.03,
    take_profit_pct: float = 0.06,
    initial_capital: float = 10000,
    start_date: str = None,
    end_date: str = None,
) -> dict:
    """Simple daily bar backtest with SL/TP."""
    close = df["close"]
    
    if start_date:
        mask = (df.index >= start_date)
        if end_date:
            mask = mask & (df.index <= end_date)
    elif end_date:
        mask = df.index <= end_date
    else:
        mask = pd.Series(True, index=df.index)
    
    signal = signal[mask]
    close = close[mask]

    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity = [{"date": str(close.index[0]), "equity": capital}]

    for i in range(1, len(close) - 1):
        current_price = close.iloc[i]
        current_signal = signal.iloc[i]
        date = str(close.index[i])

        if position == 0:
            # Entry
            if current_signal > threshold:
                position = 1
                entry_price = current_price
            elif current_signal < -threshold:
                position = -1
                entry_price = current_price
        else:
            # Check exit conditions
            pnl_pct = (current_price / entry_price - 1) * position
            
            # Stop loss
            if pnl_pct < -stop_loss_pct:
                trades.append({
                    "entry_date": str(close.index[i-1]),
                    "exit_date": date,
                    "side": "LONG" if position > 0 else "SHORT",
                    "entry": round(entry_price, 2),
                    "exit": round(current_price, 2),
                    "pnl_pct": round(pnl_pct * 100, 2),
                    "reason": "stop_loss",
                })
                capital *= (1 + pnl_pct)
                position = 0
            # Take profit
            elif pnl_pct > take_profit_pct:
                trades.append({
                    "entry_date": str(close.index[i-1]),
                    "exit_date": date,
                    "side": "LONG" if position > 0 else "SHORT",
                    "entry": round(entry_price, 2),
                    "exit": round(current_price, 2),
                    "pnl_pct": round(pnl_pct * 100, 2),
                    "reason": "take_profit",
                })
                capital *= (1 + pnl_pct)
                position = 0
            # Signal reversal
            elif (position > 0 and current_signal < -threshold) or \
                 (position < 0 and current_signal > threshold):
                trades.append({
                    "entry_date": str(close.index[i-1]),
                    "exit_date": date,
                    "side": "LONG" if position > 0 else "SHORT",
                    "entry": round(entry_price, 2),
                    "exit": round(current_price, 2),
                    "pnl_pct": round(pnl_pct * 100, 2),
                    "reason": "signal_flip",
                })
                capital *= (1 + pnl_pct)
                position = 0

        equity.append({"date": date, "equity": round(capital, 2)})

    # Close any open position
    if position != 0:
        pnl_pct = (close.iloc[-1] / entry_price - 1) * position
        capital *= (1 + pnl_pct)

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    total_return = (capital / initial_capital - 1) * 100
    win_rate = (tdf["pnl_pct"] > 0).mean() * 100 if not tdf.empty else 0
    avg_win = tdf[tdf["pnl_pct"] > 0]["pnl_pct"].mean() if not tdf.empty else 0
    avg_loss = tdf[tdf["pnl_pct"] < 0]["pnl_pct"].mean() if not tdf.empty else 0
    
    eq_df = pd.DataFrame(equity)
    peak = eq_df["equity"].cummax()
    dd = (eq_df["equity"] - peak) / peak * 100
    max_dd = dd.min()

    return {
        "total_return_pct": round(total_return, 2),
        "trades": len(tdf),
        "win_rate_pct": round(win_rate, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "max_dd_pct": round(max_dd, 2),
        "final_capital": round(capital, 2),
        "trades_df": tdf,
        "equity_df": eq_df,
    }
