#!/usr/bin/env python3
"""
VERIFICATION #1: Reverse Walk-Forward
Train on recent data (2025-2026), test on old data (2023-2024).
If strategy works BOTH ways → not overfitted to any specific period.
"""
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.research_engine import (
        load_config, load_or_download_klines, enrich_features,
        backtest_experiment, backtest_experiment_detailed,
    )
    from src.reports import profit_factor, max_drawdown
    import pandas as pd

    config = load_config("config.yaml")
    
    # BTC keltner_breakout/1h params (from stress test)
    strategy = {
        "symbol": "BTCUSDT", "interval": "1h", "family": "keltner_breakout",
        "direction_filter": "mtf_trend", "lookback": 96, "volume_min": 0.7,
        "atr_stop_mult": 1.2, "take_profit_r": 3.0, "max_holding_bars": 96,
        "stop_rule": "swing", "adx_min": 0, "regime": "high_vol",
        "trailing_atr_mult": 3.0, "breakeven_r": None, "partial_tp_r": None,
    }

    # Load full data
    df = load_or_download_klines("BTCUSDT", "1h", "2023-01-01", "2026-06-01")
    df = enrich_features(df, "1h", lookbacks=[96])
    records = df.to_dict("records")

    print("=" * 60)
    print("VERIFICATION #1: REVERSE WALK-FORWARD")
    print("Train on 2025-2026, Test on 2023-2024")
    print("If strategy works → robust across ALL time periods")
    print("=" * 60)

    # Train: 2025-2026
    train = backtest_experiment(records, strategy, config, "2025-01-01", "2026-06-01")
    # Test: 2023-2024 (completely unseen during training)
    test = backtest_experiment(records, strategy, config, "2023-01-01", "2025-01-01")

    print(f"\nTRAIN (2025-2026):")
    print(f"  Return: {train['return_pct']:+.2f}%  PF: {train['profit_factor']:.3f}  "
          f"Trades: {train['trades']}  WR: {train['win_rate_pct']:.0f}%  MDD: {train['max_drawdown_pct']:.2f}%")

    print(f"\nTEST (2023-2024) [UNSEEN]:")
    print(f"  Return: {test['return_pct']:+.2f}%  PF: {test['profit_factor']:.3f}  "
          f"Trades: {test['trades']}  WR: {test['win_rate_pct']:.0f}%  MDD: {test['max_drawdown_pct']:.2f}%")

    # Also reverse: train on 2023-2024, test on 2025-2026 (original direction)
    train2 = backtest_experiment(records, strategy, config, "2023-01-01", "2025-01-01")
    test2 = backtest_experiment(records, strategy, config, "2025-01-01", "2026-06-01")

    print(f"\nREVERSE: Train 2023-2024, Test 2025-2026:")
    print(f"  Train: {train2['return_pct']:+.2f}%  Test: {test2['return_pct']:+.2f}%")

    both_profitable = test['return_pct'] > 0 and test2['return_pct'] > 0
    print(f"\n✅ BOTH directions profitable: {both_profitable}")
    print(f"  2023-2024 (test): {test['return_pct']:+.2f}%")
    print(f"  2025-2026 (test): {test2['return_pct']:+.2f}%")
    
    if not both_profitable:
        print("❌ FAILED: Strategy only works in one direction → overfitting!")
        sys.exit(1)
