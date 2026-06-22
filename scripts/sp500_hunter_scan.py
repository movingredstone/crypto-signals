#!/usr/bin/env python3
"""S&P500-beater hunter scan: broad fold eval + stress for additional liquid small-cap futures.

Purpose: find strategies that can plausibly beat a ~10% annual S&P500 benchmark on small capital,
without dropping to fee-killed 15m timeframes.
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

# Liquid-ish, sub-$100-ish majors not already in the live 3-coin portfolio.
# 4h+8h only: enough signals without 15m fee/slippage trap.
COINS = [
    ("ADAUSDT",  {"atr_stop_mults": [1.5, 2.0, 2.5, 3.0, 4.0], "take_profit_rs": [2.0, 3.0, 4.0, 5.0]}),
    ("LINKUSDT", {"atr_stop_mults": [1.2, 1.5, 2.0, 2.5, 3.0], "take_profit_rs": [1.8, 2.0, 2.5, 3.0, 4.0]}),
    ("NEARUSDT", {"atr_stop_mults": [1.5, 2.0, 2.5, 3.0, 4.0], "take_profit_rs": [2.0, 3.0, 4.0, 5.0]}),
    ("DOTUSDT",  {"atr_stop_mults": [1.5, 2.0, 2.5, 3.0, 4.0], "take_profit_rs": [2.0, 3.0, 4.0, 5.0]}),
    ("HBARUSDT", {"atr_stop_mults": [1.5, 2.0, 2.5, 3.0, 4.0], "take_profit_rs": [2.0, 3.0, 4.0, 5.0]}),
]

if __name__ == "__main__":
    import pandas as pd
    from src.fold_evaluator import evaluate_folds
    from src.optimizer import _run_stress as run_stress

    summary = []
    for i, (symbol, focus) in enumerate(COINS, 1):
        print(f"\n{'='*70}\n[{i}/{len(COINS)}] {symbol}\n{'='*70}", flush=True)
        result = evaluate_folds(
            symbol=symbol,
            intervals=["4h", "8h"],
            experiments=1500,
            workers=9,
            top_n=40,
            seed=20260622,
            config_path="config.yaml",
            output_dir="results/optimization",
            mode="baseline",
            focus_override=focus,
        )
        df = result.get("df", pd.DataFrame())
        csv_path = result.get("csv_path", "")
        if df.empty or not csv_path or not Path(csv_path).exists():
            print(f"{symbol}: NO BASELINE RESULTS", flush=True)
            continue
        print(f"{symbol} baseline rows={len(df)} 5/7+={(df['pos_folds']>=5).sum()} 7/7+={(df['pos_folds']>=7).sum()}", flush=True)
        stress = run_stress(
            symbol=symbol,
            intervals=["4h", "8h"],
            experiments=40,
            workers=9,
            output_dir=Path("results/optimization"),
            top_n=40,
            seed=20260622,
            config_path="config.yaml",
            source_top_path=csv_path,
        )
        stress_path = stress.get("top_candidates_path", "")
        if not stress_path or not Path(stress_path).exists():
            print(f"{symbol}: NO STRESS RESULTS", flush=True)
            continue
        s = pd.read_csv(stress_path)
        top = s.sort_values(["pos_folds", "mean_return", "mean_pf"], ascending=[False, False, False]).head(5)
        for _, r in top.iterrows():
            print(f"  {r['family']}/{r['interval']}: folds={int(r['pos_folds'])}/7 ret={r['mean_return']:+.2f}% PF={r['mean_pf']:.2f} trades={int(r['total_trades'])}", flush=True)
        best = top.iloc[0]
        summary.append({
            "symbol": symbol,
            "stress_path": stress_path,
            "family": best["family"],
            "interval": best["interval"],
            "folds": int(best["pos_folds"]),
            "mean_return": float(best["mean_return"]),
            "mean_pf": float(best["mean_pf"]),
            "trades": int(best["total_trades"]),
        })

    out = Path("results/optimization/sp500_hunter_scan_summary.csv")
    pd.DataFrame(summary).to_csv(out, index=False)
    print(f"\nSAVED {out}", flush=True)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
