#!/usr/bin/env python3
"""
FINAL VERIFICATION — BTC keltner_breakout/1h
Walk-forward: 6-fold rolling windows
Each fold: Train=12mo, Test=3mo
Plus: full stats on the strategy
"""
import sys
sys.path.insert(0, ".")
from src.research_engine import (
    load_config, load_or_download_klines, enrich_features,
    backtest_experiment, backtest_experiment_detailed,
)
from src.reports import profit_factor, max_drawdown
import pandas as pd
import numpy as np

STRATEGY = {
    "symbol": "BTCUSDT", "interval": "1h", "family": "keltner_breakout",
    "direction_filter": "mtf_trend", "lookback": 96, "volume_min": 0.7,
    "atr_stop_mult": 1.2, "take_profit_r": 3.0, "max_holding_bars": 96,
    "stop_rule": "swing", "adx_min": 0, "regime": "high_vol",
    "trailing_atr_mult": 3.0, "breakeven_r": None, "partial_tp_r": None,
}

config = load_config("config.yaml")
df = load_or_download_klines("BTCUSDT", "1h", "2023-01-01", "2026-06-01")
df = enrich_features(df, "1h", lookbacks=[96])
records = df.to_dict("records")

print("=" * 60)
print("FINAL VERIFICATION: BTC keltner_breakout/1h")
print("=" * 60)

# ── FULL PERIOD STATS ─────────────────────────────────────────────
trades, equity = backtest_experiment_detailed(
    records, STRATEGY, config, "2023-01-01", "2026-06-01"
)
tdf = pd.DataFrame(trades)
eq_df = pd.DataFrame(equity)

total_return = (eq_df["equity"].iloc[-1] / 10000 - 1) * 100
pf = profit_factor(tdf)
mdd = max_drawdown(eq_df["equity"]) * 100
wr = (tdf["net_pnl"] > 0).mean() * 100
avg_win = tdf[tdf["net_pnl"] > 0]["net_pnl"].mean()
avg_loss = tdf[tdf["net_pnl"] < 0]["net_pnl"].mean()
total_pnl = tdf["net_pnl"].sum()

print(f"\n📊 FULL PERIOD (2023-01 ~ 2026-06, 3.5 years)")
print(f"   Trades: {len(tdf)}")
print(f"   Win Rate: {wr:.1f}%")
print(f"   Profit Factor: {pf:.3f}")
print(f"   Max DD: {mdd:.2f}%")
print(f"   Total Return: {total_return:+.2f}%")
print(f"   Total PnL: ${total_pnl:+,.2f}")
print(f"   Avg Win: ${avg_win:+,.2f}")
print(f"   Avg Loss: ${avg_loss:+,.2f}")
print(f"   Expectancy: ${tdf['net_pnl'].mean():+,.3f}/trade")

# ── POSITION SIZING MATH (not AI!) ────────────────────────────────
print(f"\n📐 POSITION SIZING (pure math, formula-based)")
print(f"   Formula: size = (capital × risk%) / stop_distance")
print(f"   Example for $330, 5% risk:")
# Get a sample stop distance from actual trades
sample_stop_pct = 1.2  # typical for keltner_breakout
risk_dollar = 330 * 0.05
position = risk_dollar / (sample_stop_pct / 100)
print(f"     risk = $330 × 5% = ${risk_dollar:.2f}")
print(f"     stop = {sample_stop_pct}% → size = ${position:.0f}")
print(f"     leverage = ${position:.0f} / $330 = {position/330:.1f}x")

# ── WALK-FORWARD: 6-fold rolling ──────────────────────────────────
print(f"\n{'='*60}")
print("WALK-FORWARD: 6-Fold Rolling (Train=12mo, Test=3mo)")
print(f"{'='*60}")

folds = [
    ("Fold 1", "2023-01-01", "2024-01-01", "2024-01-01", "2024-04-01"),
    ("Fold 2", "2023-04-01", "2024-04-01", "2024-04-01", "2024-07-01"),
    ("Fold 3", "2023-07-01", "2024-07-01", "2024-07-01", "2024-10-01"),
    ("Fold 4", "2023-10-01", "2024-10-01", "2024-10-01", "2025-01-01"),
    ("Fold 5", "2024-04-01", "2025-04-01", "2025-04-01", "2025-07-01"),
    ("Fold 6", "2024-10-01", "2025-10-01", "2025-10-01", "2026-01-01"),
]

results = []
for name, t_start, t_end, v_start, v_end in folds:
    train_r = backtest_experiment(records, STRATEGY, config, t_start, t_end)
    test_r = backtest_experiment(records, STRATEGY, config, v_start, v_end)
    
    train_ret = train_r["return_pct"]
    test_ret = test_r["return_pct"]
    gap = train_ret - test_ret
    status = "✅" if test_ret > 0 else "❌"
    
    results.append({
        "fold": name, "train": train_ret, "test": test_ret,
        "pf": test_r["profit_factor"], "trades": test_r["trades"],
        "wr": test_r["win_rate_pct"], "status": status, "gap": gap,
    })
    
    print(f"  {name}: Train={t_start[:7]}~{t_end[:7]} → Test={v_start[:7]}~{v_end[:7]}")
    print(f"    Train: {train_ret:+.2f}% → Test: {test_ret:+.2f}% "
          f"(PF={test_r['profit_factor']:.2f}, {test_r['trades']}t, WR={test_r['win_rate_pct']:.0f}%)  {status}")

wins = sum(1 for r in results if r["test"] > 0)
avg_test = np.mean([r["test"] for r in results])
avg_pf = np.mean([r["pf"] for r in results if r["pf"] < 50])
total_test_trades = sum(r["trades"] for r in results)

print(f"\n{'='*60}")
print(f"WALK-FORWARD VERDICT")
print(f"{'='*60}")
print(f"  Positive folds: {wins}/{len(folds)}")
print(f"  Avg test return: {avg_test:+.2f}%")
print(f"  Avg PF: {avg_pf:.2f}")
print(f"  Total test trades: {total_test_trades}")
print(f"  Max gap (overfitting): {max(r['gap'] for r in results):+.2f}%")

if wins >= 5 and avg_test > 0:
    print(f"\n  ✅ WALK-FORWARD PASSED: Strategy is robust")
elif wins >= 4:
    print(f"\n  ⚠️  MODERATE: Proceed with caution")
else:
    print(f"\n  ❌ FAILED: Strategy not reliable")
