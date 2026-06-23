#!/usr/bin/env python3
"""
Stock Multi-Factor Pipeline
3 stocks: QLD, SOXL, TQQQ
Walk-forward: Train on 2023-2024, Validate on 2025, Test on 2026
"""
import sys
sys.path.insert(0, ".")
import yfinance as yf
import pandas as pd
import numpy as np
from src.stock_factors import compute_composite_signal, run_stock_backtest

STOCKS = ["QLD", "SOXL", "TQQQ"]

# Walk-forward splits
SPLITS = {
    "train": ("2023-01-01", "2025-01-01"),
    "validate": ("2025-01-01", "2026-01-01"),
    "test": ("2026-01-01", "2026-06-21"),
}

# Factor weight candidates to test
WEIGHT_COMBOS = [
    {"momentum": 0.35, "rsi": 0.25, "trend": 0.25, "volatility": 0.10, "volume": 0.05},
    {"momentum": 0.30, "rsi": 0.30, "trend": 0.20, "volatility": 0.10, "volume": 0.10},
    {"momentum": 0.25, "rsi": 0.25, "trend": 0.30, "volatility": 0.10, "volume": 0.10},
    {"momentum": 0.40, "rsi": 0.20, "trend": 0.20, "volatility": 0.10, "volume": 0.10},
    {"momentum": 0.20, "rsi": 0.20, "trend": 0.20, "volatility": 0.20, "volume": 0.20},
]

def run_stock_pipeline(symbol: str):
    print(f"\n{'='*60}")
    print(f"STOCK: {symbol}")
    print(f"{'='*60}")
    
    # Download data
    df = yf.download(symbol, start="2023-01-01", end="2026-06-21", progress=False)
    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    
    # Compute daily vol for context
    ret = df["close"].pct_change().dropna()
    daily_vol = ret.std() * 100
    print(f"  Daily vol: {daily_vol:.2f}% | Total return: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:+.1f}%")

    # Find best factor weights using train period
    best_score = -999
    best_weights = None
    
    for weights in WEIGHT_COMBOS:
        signal = compute_composite_signal(df, weights)
        result = run_stock_backtest(
            df, signal, 
            start_date=SPLITS["train"][0], end_date=SPLITS["train"][1]
        )
        # Score: return - penalty for drawdown
        score = result["total_return_pct"] + result["max_dd_pct"] * 0.5
        if score > best_score:
            best_score = score
            best_weights = weights.copy()
    
    print(f"  Best weights (train): {best_weights}")
    
    # Full walk-forward evaluation with best weights
    signal = compute_composite_signal(df, best_weights)
    
    results = {}
    for split_name, (start, end) in SPLITS.items():
        result = run_stock_backtest(df, signal, start_date=start, end_date=end)
        results[split_name] = result
    
    # Print summary
    print(f"\n  {'Split':<12} {'Return':>10} {'Trades':>8} {'WR':>8} {'MDD':>8} {'AvgW':>8} {'AvgL':>8}")
    print(f"  {'-'*60}")
    all_positive = True
    for split_name in ["train", "validate", "test"]:
        r = results[split_name]
        flag = "✅" if r["total_return_pct"] > 0 else "❌"
        if r["total_return_pct"] <= 0:
            all_positive = False
        print(f"  {flag} {split_name:<9} {r['total_return_pct']:>+8.2f}% {r['trades']:>6}  "
              f"{r['win_rate_pct']:>5.0f}% {r['max_dd_pct']:>6.2f}% "
              f"{r['avg_win_pct']:>+6.2f}% {r['avg_loss_pct']:>+6.2f}%")
    
    test_r = results["test"]
    print(f"\n  Gap (train-test): {results['train']['total_return_pct'] - test_r['total_return_pct']:+.2f}%")
    print(f"  Overfitting: {'YES ⚠️' if results['train']['total_return_pct'] - test_r['total_return_pct'] > 10 else 'NO ✅'}")
    print(f"  All splits positive: {all_positive}")
    
    return {
        "symbol": symbol,
        "daily_vol": daily_vol,
        "best_weights": best_weights,
        "results": results,
        "all_positive": all_positive,
    }

if __name__ == "__main__":
    print("STOCK MULTI-FACTOR PIPELINE")
    print(f"Stocks: {STOCKS}")
    print(f"Walk-forward: Train={SPLITS['train']}, Val={SPLITS['validate']}, Test={SPLITS['test']}")
    
    all_results = {}
    for sym in STOCKS:
        all_results[sym] = run_stock_pipeline(sym)
    
    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Stock':<8} {'Vol':>8} {'Train':>10} {'Val':>10} {'Test':>10} {'All+':>6} {'Best Factor'}")
    print(f"  {'-'*70}")
    for sym, res in all_results.items():
        r = res["results"]
        all_pos = "✅" if res["all_positive"] else "❌"
        top_factor = max(res["best_weights"], key=res["best_weights"].get)
        print(f"  {sym:<8} {res['daily_vol']:>6.2f}% {r['train']['total_return_pct']:>+8.2f}% "
              f"{r['validate']['total_return_pct']:>+8.2f}% {r['test']['total_return_pct']:>+8.2f}% "
              f"{all_pos:<6} {top_factor}")
