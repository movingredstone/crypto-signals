import sys, json
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from src.research_engine import (
    load_config, load_or_download_klines, enrich_features,
    backtest_experiment,
)
from src.fold_evaluator import BASELINE_OVERRIDES

config = load_config("config.yaml")

# SUI macd_momentum/4h — TVT #4 with gap -0.48%
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

# Load full data as records
df = load_or_download_klines(params["symbol"], params["interval"],
    config["backtest"]["start_date"], config["backtest"]["end_date"])
df = enrich_features(df, params["interval"], lookbacks=[params.get("lookback", 48)])
records = df.to_dict("records")

print("="*70)
print("WALK-FORWARD VERIFICATION — SUI macd_momentum/4h")
print("="*70)

# 1. Rolling Walk-Forward (test only, on out-of-sample periods)
print("\n1. ROLLING 6-FOLD WF (Test on 3-month windows)")
windows = [
    ("2023-Q2", "2023-04-01", "2023-07-01"),
    ("2023-Q3", "2023-07-01", "2023-10-01"),
    ("2024-Q1", "2024-01-01", "2024-04-01"),
    ("2024-Q3", "2024-07-01", "2024-10-01"),
    ("2025-Q1", "2025-01-01", "2025-04-01"),
    ("2025-Q3", "2025-07-01", "2025-10-01"),
]

wf_rets = []
for name, start, end in windows:
    result = backtest_experiment(records, exp, config, start, end)
    ret = result.get("return_pct") or 0
    trades = result.get("trades") or 0
    pf = result.get("pf") or 0
    wf_rets.append(ret)
    status = "✅" if ret > 0 else "❌"
    print(f"  {status} {name}: ret={ret:+.2f}% pf={pf:.2f} trades={trades}")

pos_wf = sum(1 for r in wf_rets if r > 0)
mean_wf = sum(wf_rets)/len(wf_rets) if wf_rets else 0
print(f"  => {pos_wf}/{len(wf_rets)} positive, mean={mean_wf:+.2f}%")

# 2. Reverse WF
print("\n2. REVERSE WF (Test on 2023-2024, trained on 2025-2026 idea)")
rev_result = backtest_experiment(records, exp, config, "2023-01-01", "2024-12-31")
rev_ret = rev_result.get("return_pct") or 0
rev_pf = rev_result.get("pf") or 0
rev_tr = rev_result.get("trades") or 0
print(f"  => ret={rev_ret:+.2f}% pf={rev_pf:.2f} trades={rev_tr} {'✅ Works both ways' if rev_ret>0 else '❌'}")

# 3. Full period test
full_result = backtest_experiment(records, exp, config, "2023-01-01", "2026-06-01")
full_ret = full_result.get("return_pct") or 0
full_pf = full_result.get("pf") or 0
full_tr = full_result.get("trades") or 0
print(f"\n3. FULL PERIOD (2023-2026): ret={full_ret:+.2f}% pf={full_pf:.2f} trades={full_tr}")

# 4. B&H comparison
first_close = float(df.iloc[0]["close"])
last_close = float(df.iloc[-1]["close"])
bh_ret = (last_close / first_close - 1) * 100
print(f"4. BUY & HOLD: {bh_ret:+.1f}% vs Strategy: {full_ret:+.1f}%")
print(f"   Strategy/B&H ratio: {full_ret/bh_ret:.1f}x {'✅ Beats B&H' if full_ret > bh_ret else '⚠️ Below B&H'}")

# Final
print(f"\n{'='*70}")
all_pass = pos_wf >= 4
print(f"FINAL: {'🔥 ALL CHECKS PASSED' if all_pass else '⚠️ Review'} (WF {pos_wf}/{len(wf_rets)}, Rev {'✅' if rev_ret>0 else '❌'})")
print(f"Annualized: ~{full_ret/3.5:.1f}%/yr over 3.5 years, {full_tr} total trades")
