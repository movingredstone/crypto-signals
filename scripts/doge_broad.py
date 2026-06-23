#!/usr/bin/env python3
"""DOGE high-vol broad search with expanded parameter ranges."""
import sys
sys.path.insert(0, ".")

# Expanded axes for high-volatility coins (wider stops, bigger targets)
HIGH_VOL_FOCUS = {
    "atr_stop_mults": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    "take_profit_rs": [2.0, 2.5, 3.0, 4.0, 5.0],
    "lookbacks": [48, 96, 144, 192],
    "max_holding_bars": [12, 24, 48, 72],
    "volume_mins": [1.2, 1.5, 2.0],
}

if __name__ == "__main__":
    from src.fold_evaluator import evaluate_folds, format_fold_report

    result = evaluate_folds(
        symbol="DOGEUSDT",
        intervals=["1h", "4h", "8h", "1d"],
        experiments=3000,
        workers=9,
        top_n=75,
        seed=42,
        config_path="config.yaml",
        output_dir="results/optimization",
        mode="baseline",
        focus_override=HIGH_VOL_FOCUS,
    )

    df = result["df"]
    if not df.empty:
        print(format_fold_report(df, "DOGEUSDT", result["mode"], ["1h","4h","8h","1d"], 3000))
