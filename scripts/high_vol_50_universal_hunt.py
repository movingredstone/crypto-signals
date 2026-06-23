#!/usr/bin/env python3
"""High-volatility 50-coin universal strategy hunt.

Goal: find a broadly applicable method for volatile Binance USDT perpetuals.

This is a screening pipeline, not live-trading permission:
1) Use 50 liquid/high-vol symbols with enough history for meaningful folds.
2) Search 4h/8h only (skip 15m fee trap; 1h can be stage-2 if needed).
3) Wider stop/target axes for high-vol coins.
4) Baseline fold evaluation -> stress retest -> R1/R2/R4/R5 honesty gates.
5) Save universal family/timeframe ranking across all survivors.

Run from repo root:
  PYTHONUNBUFFERED=1 python scripts/high_vol_50_universal_hunt.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, ".")

import pandas as pd

from src.fold_evaluator import evaluate_folds
from src.optimizer import _run_stress as run_stress

# Current high-vol/liquid Binance USDT perpetuals with history starting no later than 2024-06.
# Excludes too-new symbols whose first candles are 2025/2026 because they cannot pass honest WF/folds.
SYMBOLS = [
    "BICOUSDT", "IDUSDT", "BELUSDT", "TNSRUSDT", "WUSDT", "SAGAUSDT", "JTOUSDT", "ALICEUSDT", "ENAUSDT", "DYDXUSDT",
    "RIFUSDT", "DYMUSDT", "WLDUSDT", "TIAUSDT", "AXSUSDT", "SUIUSDT", "JUPUSDT", "STGUSDT", "APTUSDT", "ETHUSDT",
    "ZECUSDT", "WIFUSDT", "FETUSDT", "ARBUSDT", "ETHFIUSDT", "XMRUSDT", "AAVEUSDT", "NEARUSDT", "FILUSDT", "BTCUSDT",
    "PENDLEUSDT", "SOLUSDT", "TAOUSDT", "UNIUSDT", "AVAXUSDT", "LINKUSDT", "DASHUSDT", "BCHUSDT", "CHZUSDT", "INJUSDT",
    "1000PEPEUSDT", "ATOMUSDT", "TONUSDT", "XLMUSDT", "SANDUSDT", "ICPUSDT", "ADAUSDT", "ORDIUSDT", "LDOUSDT", "BNBUSDT",
]

INTERVALS = ["4h", "8h"]
EXPERIMENTS = int(os.environ.get("HV50_EXPERIMENTS", "450"))
TOP_N = int(os.environ.get("HV50_TOP_N", "30"))
WORKERS = int(os.environ.get("HV50_WORKERS", "9"))
SEED = int(os.environ.get("HV50_SEED", "20260622"))
OUTPUT_DIR = Path("results/optimization")
SUPPORTED = {"macd_momentum", "trend_pullback"}

# Wider, high-vol friendly exits; the random optimizer still samples all available families.
FOCUS = {
    "atr_stop_mults": [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "take_profit_rs": [1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0],
    "max_holding_bars": [12, 24, 36, 48, 72],
    "adx_mins": [0, 10, 15, 20, 25, 30],
}


def latest_stress_path(symbol: str) -> str | None:
    paths = sorted(glob.glob(str(OUTPUT_DIR / f"{symbol}_stress_*_fold_flat.csv")))
    return paths[-1] if paths else None


def honest_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    pf_cols = [c for c in df.columns if c.endswith("_pf") and c[:4].isdigit()]
    # R1/R2: >=5/7 positive folds, R4: enough trades, R5: no absurd PF artifacts.
    x = df[(df["pos_folds"] >= 5) & (df["total_trades"] >= 40)].copy()
    if x.empty:
        return x

    def honest(row) -> bool:
        vals = []
        for c in pf_cols:
            try:
                vals.append(float(row[c]))
            except Exception:
                pass
        try:
            mean_pf = float(row["mean_pf"])
        except Exception:
            return False
        return bool(vals) and all(math.isfinite(v) and v <= 20 for v in vals) and math.isfinite(mean_pf) and mean_pf < 10

    return x[x.apply(honest, axis=1)].copy()


def summarize_symbol(symbol: str, stress_path: str, rows: List[Dict]) -> None:
    df = pd.read_csv(stress_path)
    r1 = int((df["pos_folds"] >= 5).sum()) if "pos_folds" in df else 0
    r4 = int(((df["pos_folds"] >= 5) & (df["total_trades"] >= 40)).sum()) if "total_trades" in df else 0
    survivors = honest_filter(df)
    deployable_count = int(survivors["family"].isin(SUPPORTED).sum()) if not survivors.empty else 0
    print(f"{symbol}: stress_rows={len(df)} R1/R2={r1} R4={r4} R5={len(survivors)} deployable={deployable_count}", flush=True)
    if survivors.empty:
        return
    # Keep top 10 per symbol for global family/method analysis.
    survivors = survivors.sort_values(["mean_return", "pos_folds", "mean_pf", "total_trades"], ascending=[False, False, False, False]).head(10)
    for _, r in survivors.iterrows():
        try:
            params = json.loads(r.get("params_json", "{}"))
        except Exception:
            params = {}
        rows.append({
            "symbol": symbol,
            "deployable": bool(r["family"] in SUPPORTED),
            "family": r["family"],
            "interval": r["interval"],
            "pos_folds": int(r["pos_folds"]),
            "mean_return": float(r["mean_return"]),
            "mean_pf": float(r["mean_pf"]),
            "total_trades": int(r["total_trades"]),
            "lookback": params.get("lookback"),
            "direction_filter": params.get("direction_filter"),
            "stop_rule": params.get("stop_rule"),
            "atr_stop_mult": params.get("atr_stop_mult"),
            "take_profit_r": params.get("take_profit_r"),
            "max_holding_bars": params.get("max_holding_bars"),
            "params_json": r.get("params_json", "{}"),
            "stress_path": stress_path,
        })


def family_universal_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby(["family", "interval", "deployable"], as_index=False).agg(
        symbols=("symbol", "nunique"),
        candidates=("symbol", "count"),
        avg_return=("mean_return", "mean"),
        median_return=("mean_return", "median"),
        avg_pf=("mean_pf", "mean"),
        median_pf=("mean_pf", "median"),
        avg_folds=("pos_folds", "mean"),
        total_trades=("total_trades", "sum"),
    )
    # Universal score rewards breadth first, then robustness/return/PF/trades.
    g["universal_score"] = (
        g["symbols"] * 3.0
        + g["avg_folds"] * 1.2
        + g["median_return"] * 1.5
        + g["median_pf"] * 1.0
        + (g["total_trades"].clip(upper=1000) / 1000.0) * 2.0
        + g["deployable"].astype(int) * 2.0
    )
    return g.sort_values(["universal_score", "symbols", "median_return"], ascending=[False, False, False])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict] = []
    errors = []
    print(f"High-vol 50 universal hunt: symbols={len(SYMBOLS)} intervals={INTERVALS} experiments={EXPERIMENTS} workers={WORKERS}", flush=True)
    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"\n{'='*90}\n[{i}/{len(SYMBOLS)}] {symbol}\n{'='*90}", flush=True)
        try:
            # Always run a fresh compact baseline for this 50-coin pass. It is intentionally screen-level.
            result = evaluate_folds(
                symbol=symbol,
                intervals=INTERVALS,
                experiments=EXPERIMENTS,
                workers=WORKERS,
                top_n=TOP_N,
                seed=SEED,
                config_path="config.yaml",
                output_dir=str(OUTPUT_DIR),
                mode="baseline",
                focus_override=FOCUS,
            )
            baseline_path = result.get("csv_path", "")
            if not baseline_path or not Path(baseline_path).exists():
                raise RuntimeError("baseline CSV missing")
            bdf = result.get("df", pd.DataFrame())
            print(f"baseline_rows={len(bdf)} pos>=5={int((bdf['pos_folds']>=5).sum()) if not bdf.empty else 0}", flush=True)
            stress = run_stress(
                symbol=symbol,
                intervals=INTERVALS,
                experiments=TOP_N,
                workers=WORKERS,
                output_dir=OUTPUT_DIR,
                top_n=TOP_N,
                seed=SEED,
                config_path="config.yaml",
                source_top_path=baseline_path,
            )
            stress_path = stress.get("top_candidates_path", "")
            if not stress_path or not Path(stress_path).exists():
                raise RuntimeError("stress CSV missing")
            summarize_symbol(symbol, stress_path, all_rows)
        except Exception as e:
            print(f"ERROR {symbol}: {e!r}", flush=True)
            errors.append({"symbol": symbol, "error": repr(e)})

        partial = pd.DataFrame(all_rows)
        partial_path = OUTPUT_DIR / "high_vol_50_universal_survivors_partial.csv"
        partial.to_csv(partial_path, index=False)
        if not partial.empty:
            family_universal_summary(partial).to_csv(OUTPUT_DIR / "high_vol_50_universal_family_partial.csv", index=False)
            print("Current universal top families:", flush=True)
            print(family_universal_summary(partial).head(8).to_string(index=False), flush=True)

    survivors = pd.DataFrame(all_rows)
    survivors_path = OUTPUT_DIR / "high_vol_50_universal_survivors.csv"
    family_path = OUTPUT_DIR / "high_vol_50_universal_family.csv"
    survivors.to_csv(survivors_path, index=False)
    fam = family_universal_summary(survivors)
    fam.to_csv(family_path, index=False)
    print(f"\nSAVED {survivors_path}")
    print(f"SAVED {family_path}")
    if not survivors.empty:
        print("\nTOP 30 CANDIDATES")
        print(survivors.sort_values(["mean_return", "pos_folds", "mean_pf"], ascending=[False, False, False]).head(30)[[
            "symbol", "family", "interval", "deployable", "pos_folds", "mean_return", "mean_pf", "total_trades", "direction_filter", "stop_rule", "atr_stop_mult", "take_profit_r"
        ]].to_string(index=False))
        print("\nUNIVERSAL FAMILY RANKING")
        print(fam.head(20).to_string(index=False))
    if errors:
        err_path = OUTPUT_DIR / "high_vol_50_universal_errors.csv"
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"\nERRORS saved {err_path}")
        print(pd.DataFrame(errors).to_string(index=False))


if __name__ == "__main__":
    main()
