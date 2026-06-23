"""
Market regime classifier for BTC futures.

8 regimes, priority-matching (first match wins):
  trend_up       — ADX > 25, EMA20 > EMA50 > EMA200
  trend_down     — ADX > 25, EMA20 < EMA50 < EMA200
  squeeze        — BB width < 10th percentile
  post_breakout  — within 12 bars after Donchian breakout
  range          — ADX < 20, BB width < 30th percentile
  high_vol       — ATR > 70th percentile
  low_vol        — ATR < 30th percentile
  weekend        — Saturday/Sunday (UTC)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ALL_REGIMES = [
    "trend_up", "trend_down", "squeeze", "post_breakout",
    "range", "high_vol", "low_vol", "weekend",
]


def classify_regime(df: pd.DataFrame, adx_threshold: float = 25.0) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    adx = df["adx14"].values
    ema20 = df["ema20"].values
    ema50 = df["ema50"].values
    ema200 = df["ema200"].values
    atr = df["atr14"].values
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    strong_adx = adx > adx_threshold
    weak_adx = adx < 20.0
    bullish = (ema20 > ema50) & (ema50 > ema200)
    bearish = (ema20 < ema50) & (ema50 < ema200)

    atr_rank = pd.Series(atr).rolling(500, min_periods=100).rank(pct=True).values
    high_vol = atr_rank > 0.7
    low_vol = atr_rank < 0.3

    bb_width = df.get("bb_width", pd.Series(np.zeros(n)))
    bb_rank = bb_width.rolling(500, min_periods=100).rank(pct=True).values
    squeeze_mask = bb_rank < 0.10
    range_mask = weak_adx & (bb_rank < 0.30) & ~squeeze_mask

    # Post-breakout: bars after a Donchian breakout
    dh20 = df.get("donchian_high_20", pd.Series(np.full(n, np.nan))).values
    dl20 = df.get("donchian_low_20", pd.Series(np.full(n, np.nan))).values
    breakout = (close > dh20) | (close < dl20)
    post_breakout = np.zeros(n, dtype=bool)
    for i in range(n):
        if breakout[i]:
            post_breakout[i+1:min(n, i+13)] = True
    post_breakout = post_breakout & (atr_rank > 0.5)

    # Weekend
    dow = pd.to_datetime(df["open_time"]).dt.dayofweek.values
    weekend = dow >= 5

    # Assign — priority order
    regime = np.full(n, None, dtype=object)
    regime[strong_adx & bullish] = "trend_up"
    regime[strong_adx & bearish] = "trend_down"
    regime[squeeze_mask] = "squeeze"
    regime[post_breakout] = "post_breakout"
    regime[range_mask] = "range"
    regime[high_vol] = "high_vol"
    regime[low_vol] = "low_vol"
    regime[weekend] = "weekend"

    df["regime"] = regime
    return df


def regime_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    counts = df["regime"].value_counts()
    result = {"total_bars": int(total)}
    for r in ALL_REGIMES:
        result[f"{r}_pct"] = round(float(counts.get(r, 0)) / total * 100, 1)
    unclassified = total - sum(counts.get(r, 0) for r in ALL_REGIMES)
    result["unclassified_pct"] = round(float(unclassified) / total * 100, 1)
    return result


def regime_returns_by_fold(df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """Per-regime trade performance summary."""
    if trades_df.empty:
        return pd.DataFrame()

    trades_df = trades_df.copy()
    trades_df["entry_ts"] = pd.to_datetime(trades_df["entry_time"])
    df = df.copy()
    df["ot"] = pd.to_datetime(df["open_time"])

    results = []
    for _, trade in trades_df.iterrows():
        et = trade["entry_ts"]
        matching = df[df["ot"] <= et].tail(1)
        if matching.empty:
            continue
        results.append({
            "regime": matching.iloc[0]["regime"],
            "net_pnl": trade.get("net_pnl", 0),
            "r_multiple": trade.get("r_multiple", 0),
            "side": trade.get("side", ""),
        })

    if not results:
        return pd.DataFrame()

    rd = pd.DataFrame(results)
    summary = rd.groupby("regime").agg(
        trades=("net_pnl", "count"),
        total_pnl=("net_pnl", "sum"),
        mean_r=("r_multiple", "mean"),
        win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
    ).round(3)
    summary["pct_of_trades"] = (summary["trades"] / summary["trades"].sum() * 100).round(1)
    return summary.sort_values("total_pnl", ascending=False)
