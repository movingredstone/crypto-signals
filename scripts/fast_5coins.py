#!/usr/bin/env python3
"""Fast 5-coin pipeline: broad fold eval (2K exp, 4h+8h) + stress for each."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, ".")

COINS = [
    ("ETHUSDT",  None),           # BTC-like, default ranges
    ("BNBUSDT",  None),           # similar vol to BTC
    ("SOLUSDT",  {"atr_stop_mults": [1.5,2.0,2.5,3.0,4.0], "take_profit_rs": [2.0,3.0,4.0,5.0]}),
    ("XRPUSDT",  {"atr_stop_mults": [1.5,2.0,2.5,3.0,4.0], "take_profit_rs": [2.0,3.0,4.0,5.0]}),
    ("AVAXUSDT", {"atr_stop_mults": [1.5,2.0,2.5,3.0,4.0], "take_profit_rs": [2.0,3.0,4.0,5.0]}),
]

if __name__ == "__main__":
    from src.fold_evaluator import evaluate_folds
    from src.optimizer import _run_stress as run_stress
    import pandas as pd

    for symbol, focus in COINS:
        print(f"\n{'='*60}")
        print(f"COIN: {symbol}")
        print(f"{'='*60}")

        # 1. Broad fold eval
        print(f"\n[1/2] Broad fold eval (2K exp, 4h+8h)...")
        result = evaluate_folds(
            symbol=symbol, intervals=["4h", "8h"],
            experiments=2000, workers=9, top_n=50, seed=42,
            config_path="config.yaml", output_dir="results/optimization",
            mode="baseline", focus_override=focus,
        )
        df = result.get("df", pd.DataFrame())
        if df.empty:
            print(f"  {symbol}: NO RESULTS")
            continue
        surv = int((df["pos_folds"] >= 5).sum())
        surv7 = int((df["pos_folds"] >= 7).sum())
        print(f"  {symbol}: {len(df)} results, {surv} 5/7+, {surv7} 7/7")

        # 2. Stress on top 50
        csv_path = result.get("csv_path", "")
        if not csv_path or not Path(csv_path).exists():
            print(f"  {symbol}: SKIP stress (no CSV)")
            continue

        print(f"\n[2/2] Stress test (top 50)...")
        stress = run_stress(
            symbol=symbol, intervals=["4h", "8h"],
            experiments=50, workers=9, output_dir=Path("results/optimization"),
            top_n=50, seed=42, config_path="config.yaml",
            source_top_path=csv_path,
        )

        stress_df = pd.read_csv(stress.get("top_candidates_path", ""))
        stress_surv6 = int((stress_df["pos_folds"] >= 6).sum())
        stress_surv5 = int((stress_df["pos_folds"] >= 5).sum())

        print(f"  {symbol} stress: {len(stress_df)} candidates, "
              f"{stress_surv6} 6/7+, {stress_surv5} 5/7+")

        # Print top 3 stress survivors
        top3 = stress_df.sort_values("mean_return", ascending=False).head(3)
        for _, row in top3.iterrows():
            print(f"    {row['family']}/{row['interval']}: "
                  f"{int(row['pos_folds'])}/7, +{row['mean_return']:.2f}%, "
                  f"PF={row['mean_pf']:.2f}, {int(row['total_trades'])}t")

    print(f"\n{'='*60}")
    print("ALL 5 COINS COMPLETE")
    print(f"{'='*60}")
