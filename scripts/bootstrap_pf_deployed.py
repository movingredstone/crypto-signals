"""
Trade-level bootstrap of PROFIT FACTOR for the 3 DEPLOYED strategies.

Resolves audit OPEN #2. Unlike the old scripts/verify_bootstrap.py this:
  - uses the EXACT deployed params (SUI#1 / XRP / DOGE, post-audit portfolio),
  - bootstraps Profit Factor (PF), not just mean PnL,
  - reports a 95% CI for PF, P(PF>1), and P(PF >= the claimed forward PF),
  - runs all three strategies,
  - is honest: PF is scale-invariant, so it is identical at 0.5% or 5% risk.

Data: spot OHLCV from data-api.binance.vision (fapi is geo-blocked in CI/local).
Spot vs perp 4h candles differ marginally; this is a robustness check on PF,
not a P&L reproduction. Caveat noted in output.
"""
import sys, json, time, urllib.request
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from src.research_engine import load_config, enrich_features, backtest_experiment_detailed
from src.fold_evaluator import BASELINE_OVERRIDES

VISION = "https://data-api.binance.vision/api/v3/klines"
START, END = "2023-01-01", "2026-06-01"
KCOLS = ["open_time","open","high","low","close","volume","close_time","quote_volume",
         "number_of_trades","taker_buy_base_volume","taker_buy_quote_volume","ignore"]

# Deployed params (must match STRATEGIES in paper_trader_github.py)
DEPLOYED = {
    "SUI #1": dict(symbol="SUIUSDT", interval="4h", family="macd_momentum",
        direction_filter="ema200", lookback=200, volume_min=0.3, atr_stop_mult=2.5,
        take_profit_r=10.0, max_holding_bars=48, stop_rule="atr", adx_min=15,
        regime="any", trailing_atr_mult=3.0, breakeven_r=None, partial_tp_r=None,
        partial_tp_frac=0.5),
    "XRP": dict(symbol="XRPUSDT", interval="4h", family="macd_momentum",
        direction_filter="price_ema100", lookback=96, volume_min=0.0, atr_stop_mult=2.5,
        take_profit_r=10.0, max_holding_bars=144, stop_rule="atr", adx_min=15,
        regime="any", trailing_atr_mult=3.0, breakeven_r=None, partial_tp_r=None,
        partial_tp_frac=0.5),
    "DOGE": dict(symbol="DOGEUSDT", interval="4h", family="macd_momentum",
        direction_filter="price_ema100", lookback=96, volume_min=0.5, atr_stop_mult=5.0,
        take_profit_r=4.0, max_holding_bars=48, stop_rule="swing", adx_min=15,
        regime="any", trailing_atr_mult=3.0, breakeven_r=None, partial_tp_r=None,
        partial_tp_frac=0.5),
}
CLAIMED_PF = {"SUI #1": 1.919, "XRP": 1.332, "DOGE": 1.660}  # 2025-07..2026-06 forward


def fetch_klines(symbol, interval, start, end):
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    rows, cur = [], start_ms
    while cur < end_ms:
        url = f"{VISION}?symbol={symbol}&interval={interval}&startTime={cur}&endTime={end_ms}&limit=1000"
        data = json.load(urllib.request.urlopen(url, timeout=30))
        if not data:
            break
        rows.extend(data)
        cur = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=KCOLS)
    for c in ["open","high","low","close","volume","quote_volume",
              "taker_buy_base_volume","taker_buy_quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.drop_duplicates("open_time").reset_index(drop=True)


def profit_factor(pnls):
    g = sum(p for p in pnls if p > 0)
    l = -sum(p for p in pnls if p < 0)
    if l <= 0:
        return float("inf")
    return g / l


def bootstrap_pf(pnls, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.array(pnls)
    out = []
    for _ in range(n):
        s = rng.choice(arr, size=len(arr), replace=True)
        out.append(profit_factor(s))
    return np.array(out)


def main():
    config = load_config("config.yaml")
    print("=" * 72)
    print("BOOTSTRAP PROFIT FACTOR — deployed portfolio (audit OPEN #2)")
    print(f"Data: spot {START}..{END} (data-api.binance.vision). PF is scale-invariant.")
    print("=" * 72)
    summary = {}
    for name, params in DEPLOYED.items():
        df = fetch_klines(params["symbol"], params["interval"], START, END)
        df = enrich_features(df, params["interval"], lookbacks=[params["lookback"]])
        records = df.to_dict("records")
        exp = dict(params); exp.update(BASELINE_OVERRIDES)
        trades, _ = backtest_experiment_detailed(records, exp, config, START, END)
        pnls = [t["net_pnl"] for t in trades if t.get("notional", 0) > 0]
        n = len(pnls)
        if n < 10:
            print(f"\n{name}: only {n} trades — too few to bootstrap."); continue
        pf = profit_factor(pnls)
        wr = sum(1 for p in pnls if p > 0) / n * 100
        boot = bootstrap_pf(pnls)
        finite = boot[np.isfinite(boot)]
        lo, med, hi = np.percentile(finite, [2.5, 50, 97.5])
        p_gt1 = float(np.mean(boot > 1.0))
        claim = CLAIMED_PF.get(name)
        p_ge_claim = float(np.mean(boot >= claim)) if claim else None
        print(f"\n── {name} ({params['symbol']}) ──")
        print(f"  trades={n}  win_rate={wr:.0f}%  full-sample PF={pf:.3f}")
        print(f"  bootstrap PF: median={med:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")
        print(f"  P(PF>1)={p_gt1:.1%}", end="")
        if claim:
            print(f"   P(PF>=claimed {claim})={p_ge_claim:.1%}")
        else:
            print()
        verdict = "ROBUST" if lo > 1.0 else ("FRAGILE (CI includes <=1)" if med > 1.0 else "FAIL")
        print(f"  -> {verdict}")
        summary[name] = dict(n=n, wr=round(wr,1), pf=round(pf,3),
            ci=[round(lo,3), round(hi,3)], median=round(med,3),
            p_gt1=round(p_gt1,4), p_ge_claim=p_ge_claim, verdict=verdict)
    print("\n" + "=" * 72)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
