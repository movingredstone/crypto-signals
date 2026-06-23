#!/usr/bin/env python3
"""Stage-2 large search for high-vol universal crypto strategies.

Reads the 50-coin stage-1 survivor CSV, selects the strongest symbols, then reruns a
larger random search per selected symbol. This is the expensive pass intended to move
from "screening" to "serious candidates".

Run from repo root after scripts/high_vol_50_universal_hunt.py completes:
  PYTHONUNBUFFERED=1 HV_STAGE2_EXPERIMENTS=3000 HV_STAGE2_TOP_N=75 python scripts/high_vol_stage2_large_hunt.py
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

OUTPUT_DIR = Path("results/optimization")
STAGE1_PATHS = [
    OUTPUT_DIR / "high_vol_50_universal_survivors.csv",
    OUTPUT_DIR / "high_vol_50_universal_survivors_partial.csv",
]
INTERVALS = ["4h", "8h"]
EXPERIMENTS = int(os.environ.get("HV_STAGE2_EXPERIMENTS", "3000"))
TOP_N = int(os.environ.get("HV_STAGE2_TOP_N", "75"))
WORKERS = int(os.environ.get("HV_STAGE2_WORKERS", "9"))
MAX_SYMBOLS = int(os.environ.get("HV_STAGE2_SYMBOLS", "18"))
SEED = int(os.environ.get("HV_STAGE2_SEED", "20260623"))
SUPPORTED = {"macd_momentum", "trend_pullback"}

FOCUS = {
    "atr_stop_mults": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "take_profit_rs": [1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0],
    "max_holding_bars": [12, 18, 24, 36, 48, 72, 96],
    "adx_mins": [0, 10, 15, 20, 25, 30],
}

# If stage-1 yields too few deployable candidates, seed the pass with known high-vol names.
FALLBACK_SYMBOLS = [
    "ARBUSDT", "SUIUSDT", "AVAXUSDT", "DOGEUSDT", "SOLUSDT", "BICOUSDT", "SAGAUSDT", "ENAUSDT", "WLDUSDT", "JTOUSDT",
    "APTUSDT", "FETUSDT", "NEARUSDT", "FILUSDT", "INJUSDT", "LINKUSDT", "DOTUSDT", "OPUSDT",
]


def load_stage1() -> pd.DataFrame:
    for p in STAGE1_PATHS:
        if p.exists():
            df = pd.read_csv(p)
            if not df.empty:
                return df
    return pd.DataFrame()


def select_symbols(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return FALLBACK_SYMBOLS[:MAX_SYMBOLS]
    x = df.copy()
    x["deployable_bonus"] = x["deployable"].astype(bool).astype(int) * 2.0
    x["score"] = (
        x["deployable_bonus"]
        + x["pos_folds"].astype(float) * 1.2
        + x["mean_return"].astype(float) * 1.5
        + x["mean_pf"].astype(float) * 0.8
        + x["total_trades"].astype(float).clip(upper=150) / 150.0
    )
    best = x.sort_values(["score", "deployable", "mean_return", "mean_pf"], ascending=[False, False, False, False])
    symbols = []
    for s in best["symbol"].tolist() + FALLBACK_SYMBOLS:
        if s not in symbols:
            symbols.append(s)
        if len(symbols) >= MAX_SYMBOLS:
            break
    return symbols


def honest_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    pf_cols = [c for c in df.columns if c.endswith("_pf") and c[:4].isdigit()]
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


def collect_rows(symbol: str, stress_path: str, rows: List[Dict]) -> None:
    df = pd.read_csv(stress_path)
    survivors = honest_filter(df)
    r1 = int((df["pos_folds"] >= 5).sum()) if not df.empty else 0
    r4 = int(((df["pos_folds"] >= 5) & (df["total_trades"] >= 40)).sum()) if not df.empty else 0
    deployable = int(survivors["family"].isin(list(SUPPORTED)).sum()) if not survivors.empty else 0
    print(f"{symbol}: stress_rows={len(df)} R1/R2={r1} R4={r4} R5={len(survivors)} deployable={deployable}", flush=True)
    if survivors.empty:
        return
    survivors = survivors.sort_values(["mean_return", "pos_folds", "mean_pf", "total_trades"], ascending=[False, False, False, False]).head(15)
    for _, r in survivors.iterrows():
        try:
            params = json.loads(str(r.get("params_json", "{}")))
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


def universal_summary(df: pd.DataFrame) -> pd.DataFrame:
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
    g["universal_score"] = (
        g["symbols"] * 3.0
        + g["avg_folds"] * 1.2
        + g["median_return"] * 1.5
        + g["median_pf"]
        + (g["total_trades"].clip(upper=1200) / 1200.0) * 2.0
        + g["deployable"].astype(int) * 2.0
    )
    return g.sort_values(["universal_score", "symbols", "median_return"], ascending=[False, False, False])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stage1 = load_stage1()
    symbols = select_symbols(stage1)
    print(f"Stage-2 large hunt: symbols={len(symbols)} experiments={EXPERIMENTS}/interval top_n={TOP_N} workers={WORKERS}", flush=True)
    print("Symbols: " + ", ".join(symbols), flush=True)
    rows: List[Dict] = []
    errors = []
    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'='*90}\nSTAGE2 [{i}/{len(symbols)}] {symbol}\n{'='*90}", flush=True)
        try:
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
            collect_rows(symbol, stress_path, rows)
        except Exception as e:
            print(f"ERROR {symbol}: {e!r}", flush=True)
            errors.append({"symbol": symbol, "error": repr(e)})
        partial = pd.DataFrame(rows)
        partial.to_csv(OUTPUT_DIR / "high_vol_stage2_large_survivors_partial.csv", index=False)
        if not partial.empty:
            summary = universal_summary(partial)
            summary.to_csv(OUTPUT_DIR / "high_vol_stage2_large_family_partial.csv", index=False)
            print("Current stage-2 universal top:", flush=True)
            print(summary.head(10).to_string(index=False), flush=True)

    out = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "high_vol_stage2_large_survivors.csv"
    fam_path = OUTPUT_DIR / "high_vol_stage2_large_family.csv"
    out.to_csv(out_path, index=False)
    fam = universal_summary(out)
    fam.to_csv(fam_path, index=False)
    print(f"\nSAVED {out_path}")
    print(f"SAVED {fam_path}")
    if not out.empty:
        print("\nTOP STAGE-2 CANDIDATES")
        print(out.sort_values(["mean_return", "pos_folds", "mean_pf"], ascending=[False, False, False]).head(40)[[
            "symbol", "family", "interval", "deployable", "pos_folds", "mean_return", "mean_pf", "total_trades", "direction_filter", "stop_rule", "atr_stop_mult", "take_profit_r"
        ]].to_string(index=False))
        print("\nSTAGE-2 UNIVERSAL FAMILY RANKING")
        print(fam.head(20).to_string(index=False))
    if errors:
        err_path = OUTPUT_DIR / "high_vol_stage2_large_errors.csv"
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"\nERRORS saved {err_path}")
        print(pd.DataFrame(errors).to_string(index=False))


if __name__ == "__main__":
    main()
