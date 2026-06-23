#!/usr/bin/env python3
"""
VERIFICATION #3: Split-Half Reliability
Split the period into odd months vs even months.
If strategy is profitable in BOTH halves independently → reliable, not lucky.
Also test: first half vs second half of total period.
"""
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.research_engine import (
        load_config, load_or_download_klines, enrich_features,
        backtest_experiment_detailed,
    )
    import pandas as pd
    import numpy as np

    config = load_config("config.yaml")
    
    strategy = {
        "symbol": "BTCUSDT", "interval": "1h", "family": "keltner_breakout",
        "direction_filter": "mtf_trend", "lookback": 96, "volume_min": 0.7,
        "atr_stop_mult": 1.2, "take_profit_r": 3.0, "max_holding_bars": 96,
        "stop_rule": "swing", "adx_min": 0, "regime": "high_vol",
        "trailing_atr_mult": 3.0, "breakeven_r": None, "partial_tp_r": None,
    }

    print("=" * 60)
    print("VERIFICATION #3: SPLIT-HALF RELIABILITY")
    print("If profitable in BOTH halves → real edge, not period-luck")
    print("=" * 60)

    df = load_or_download_klines("BTCUSDT", "1h", "2023-01-01", "2026-06-01")
    df = enrich_features(df, "1h", lookbacks=[96])
    records = df.to_dict("records")

    # Get all trades
    trades, _ = backtest_experiment_detailed(
        records, strategy, config, "2023-01-01", "2026-06-01"
    )

    if not trades:
        print("❌ No trades!")
        sys.exit(1)

    tdf = pd.DataFrame(trades)
    tdf["month"] = pd.to_datetime(tdf["entry_time"]).dt.month
    tdf["is_odd"] = tdf["month"] % 2 == 1

    odd_trades = tdf[tdf["is_odd"]]
    even_trades = tdf[~tdf["is_odd"]]

    odd_pnl = odd_trades["net_pnl"].sum()
    even_pnl = even_trades["net_pnl"].sum()
    odd_wr = (odd_trades["net_pnl"] > 0).mean()
    even_wr = (even_trades["net_pnl"] > 0).mean()

    print(f"\n=== ODD MONTHS ===")
    print(f"  Trades: {len(odd_trades)}, PnL: ${odd_pnl:.2f}, WR: {odd_wr:.1%}")

    print(f"\n=== EVEN MONTHS ===")
    print(f"  Trades: {len(even_trades)}, PnL: ${even_pnl:.2f}, WR: {even_wr:.1%}")

    both_positive = odd_pnl > 0 and even_pnl > 0
    if both_positive:
        print(f"\n✅ BOTH halves profitable!")
    else:
        print(f"\n❌ FAILED: Not profitable in both halves")

    # Test 2: First half vs second half
    mid = len(tdf) // 2
    tdf_sorted = tdf.sort_values("entry_time")
    first_half = tdf_sorted.iloc[:mid]
    second_half = tdf_sorted.iloc[mid:]

    first_pnl = first_half["net_pnl"].sum()
    second_pnl = second_half["net_pnl"].sum()

    print(f"\n=== FIRST HALF (earlier trades) ===")
    print(f"  Trades: {len(first_half)}, PnL: ${first_pnl:.2f}")

    print(f"\n=== SECOND HALF (later trades) ===")
    print(f"  Trades: {len(second_half)}, PnL: ${second_pnl:.2f}")

    chronological_both = first_pnl > 0 and second_pnl > 0
    if chronological_both:
        print(f"\n✅ BOTH chronological halves profitable!")
    else:
        print(f"\n⚠️  Chronological split shows imbalance")

    # Test 3: Rolling 50-trade windows
    print(f"\n=== ROLLING 50-TRADE WINDOWS ===")
    window = 50
    all_positive = True
    losing_windows = 0
    for i in range(0, len(tdf_sorted) - window, 10):
        w = tdf_sorted.iloc[i:i+window]
        w_pnl = w["net_pnl"].sum()
        if w_pnl <= 0:
            all_positive = False
            losing_windows += 1
    
    total_windows = (len(tdf_sorted) - window) // 10 + 1
    pct_positive = (total_windows - losing_windows) / total_windows * 100
    print(f"  Windows tested: {total_windows}")
    print(f"  Positive windows: {total_windows - losing_windows} ({pct_positive:.0f}%)")
    
    if pct_positive >= 80:
        print(f"\n✅ {pct_positive:.0f}% of rolling windows positive → consistent edge")
    elif pct_positive >= 60:
        print(f"\n⚠️  {pct_positive:.0f}% positive → moderate consistency")
    else:
        print(f"\n❌ Only {pct_positive:.0f}% positive → unreliable")

    # Final verdict
    all_checks = [both_positive, chronological_both, pct_positive >= 60]
    passed = sum(all_checks)
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT: {passed}/3 checks passed")
    if passed == 3:
        print("✅ STRATEGY IS RELIABLE across time splits")
    elif passed >= 2:
        print("⚠️  MODERATE reliability — proceed with caution")
    else:
        print("❌ UNRELIABLE — do not trade")
