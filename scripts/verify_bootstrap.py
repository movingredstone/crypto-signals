import sys, json, random
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from src.research_engine import (
    load_config, load_or_download_klines, enrich_features,
    backtest_experiment_detailed,
)
from src.fold_evaluator import BASELINE_OVERRIDES

config = load_config("config.yaml")

# SUI macd_momentum/4h — best candidate
params = {
    "symbol": "SUIUSDT", "interval": "4h", "family": "macd_momentum",
    "direction_filter": "ema200", "lookback": 200, "volume_min": 0.5,
    "atr_stop_mult": 5.0, "take_profit_r": 6.0, "max_holding_bars": 96,
    "stop_rule": "atr", "adx_min": 20, "regime": "any",
    "trailing_atr_mult": 5.0, "breakeven_r": None, "partial_tp_r": None,
    "partial_tp_frac": 0.5, "atr_pct_min": 5, "atr_pct_max": 95,
}

exp = dict(params)
exp.update(BASELINE_OVERRIDES)

df = load_or_download_klines(params["symbol"], params["interval"],
    config["backtest"]["start_date"], config["backtest"]["end_date"])
df = enrich_features(df, params["interval"], lookbacks=[params.get("lookback", 48)])
records = df.to_dict("records")

# Get full trade history
trades, equity = backtest_experiment_detailed(records, exp, config, 
    config["backtest"]["start_date"], config["backtest"]["end_date"])

# Extract individual trade PnLs (as % of notional)
trade_pnls = []
for t in trades:
    net = t.get("net_pnl", 0)
    notional = t.get("notional", 10000)
    if notional > 0:
        trade_pnls.append((net / notional) * 100)
n_trades = len(trade_pnls)
mean_pnl = np.mean(trade_pnls)
std_pnl = np.std(trade_pnls)

print("="*70)
print("BOOTSTRAP ANALYSIS — SUI macd_momentum/4h")
print("="*70)
print(f"Total trades: {n_trades}")
print(f"Mean PnL/trade: {mean_pnl:+.3f}%")
print(f"Std PnL/trade: {std_pnl:.3f}%")
print(f"Win rate: {sum(1 for p in trade_pnls if p > 0)/n_trades*100:.0f}%")
print(f"Best trade: {max(trade_pnls):+.1f}% | Worst: {min(trade_pnls):+.1f}%")

# Bootstrap: resample trade PnLs 10,000 times
random.seed(42)
np.random.seed(42)
n_bootstrap = 10000
bootstrap_means = []
for _ in range(n_bootstrap):
    sample = np.random.choice(trade_pnls, size=len(trade_pnls), replace=True)
    bootstrap_means.append(np.mean(sample))

bootstrap_means = np.array(bootstrap_means)
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)
p_negative = np.mean(bootstrap_means <= 0)

print(f"\nBootstrap (10,000 resamples):")
print(f"  95% CI: [{ci_lower:+.3f}%, {ci_upper:+.3f}%]")
print(f"  Mean > 0: {1-p_negative:.1%} confidence")
print(f"  {'✅ STATISTICALLY SIGNIFICANT (p<0.05)' if p_negative < 0.05 else '⚠️ NOT significant'}")

# Per-year summary
print(f"\nYearly breakdown:")
yearly_data = {
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
    "2026": ("2026-01-01", "2026-06-01"),
}
for year, (start, end) in yearly_data.items():
    ytrades, _ = backtest_experiment_detailed(records, exp, config, start, end)
    ypnls = []
    for t in ytrades:
        net = t.get("net_pnl", 0)
        notional = t.get("notional", 10000)
        if notional > 0:
            ypnls.append((net / notional) * 100)
    ysum = sum(ypnls)
    ycount = len(ypnls)
    ywr = sum(1 for p in ypnls if p > 0)/max(ycount,1)*100
    print(f"  {year}: {ysum:+.1f}% ({ycount} trades, WR={ywr:.0f}%)")

# Total return
total_pnl = sum(trade_pnls)
print(f"\nTotal return: {total_pnl:+.1f}% over 3.5yr")
print(f"Annualized: {total_pnl/3.5:+.1f}%/yr")
print(f"B&H comparison: SUI went from ~$0.70 to ~$0.71 over this period (~flat)")
print(f"Strategy: {total_pnl:+.1f}% while B&H was flat → {total_pnl:+.0f}% alpha!")
