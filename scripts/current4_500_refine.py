#!/usr/bin/env python3
"""Extra 500-experiment refinement for the current 4 deployed paper symbols.

Runs a fresh random-search pass with a new seed on UNI/NEAR/SOL/ADA, stress-tests the
best candidates, applies R1/R2/R4/R5 honesty filters, and writes a concise summary.
"""
from __future__ import annotations

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
SYMBOLS = os.environ.get("REFINE_SYMBOLS", "UNIUSDT,NEARUSDT,SOLUSDT,ADAUSDT").split(",")
INTERVALS = ["4h", "8h"]
EXPERIMENTS = int(os.environ.get("REFINE_EXPERIMENTS", "500"))
TOP_N = int(os.environ.get("REFINE_TOP_N", "75"))
WORKERS = int(os.environ.get("REFINE_WORKERS", "9"))
SEED = int(os.environ.get("REFINE_SEED", "20260624"))
SUPPORTED = {"macd_momentum", "trend_pullback"}

FOCUS = {
    "atr_stop_mults": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "take_profit_rs": [1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0],
    "max_holding_bars": [12, 18, 24, 36, 48, 72, 96],
    "adx_mins": [0, 10, 15, 20, 25, 30],
}


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


def collect(symbol: str, stress_path: str, rows: List[Dict]) -> None:
    df = pd.read_csv(stress_path)
    survivors = honest_filter(df)
    r1 = int((df["pos_folds"] >= 5).sum()) if not df.empty else 0
    r4 = int(((df["pos_folds"] >= 5) & (df["total_trades"] >= 40)).sum()) if not df.empty else 0
    deployable = int(survivors["family"].isin(list(SUPPORTED)).sum()) if not survivors.empty else 0
    print(f"{symbol}: stress_rows={len(df)} R1/R2={r1} R4={r4} R5={len(survivors)} deployable={deployable}", flush=True)
    if survivors.empty:
        return
    survivors = survivors.sort_values(["mean_return", "pos_folds", "mean_pf", "total_trades"], ascending=[False, False, False, False]).head(20)
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
            "regime": params.get("regime"),
            "stop_rule": params.get("stop_rule"),
            "atr_stop_mult": params.get("atr_stop_mult"),
            "take_profit_r": params.get("take_profit_r"),
            "max_holding_bars": params.get("max_holding_bars"),
            "params_json": r.get("params_json", "{}"),
            "stress_path": stress_path,
        })


def summary(df: pd.DataFrame) -> pd.DataFrame:
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
    return g.sort_values(["symbols", "median_return", "median_pf"], ascending=[False, False, False])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    print(f"Current-4 500-refine: symbols={SYMBOLS} experiments={EXPERIMENTS}/interval workers={WORKERS} seed={SEED}", flush=True)
    for i, symbol in enumerate(SYMBOLS, 1):
        symbol = symbol.strip()
        if not symbol:
            continue
        print(f"\n{'='*90}\nREFINE [{i}/{len(SYMBOLS)}] {symbol}\n{'='*90}", flush=True)
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
        collect(symbol, stress.get("top_candidates_path", ""), rows)
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "current4_500_refine_survivors_partial.csv", index=False)

    out = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "current4_500_refine_survivors.csv"
    fam_path = OUTPUT_DIR / "current4_500_refine_family.csv"
    out.to_csv(out_path, index=False)
    fam = summary(out)
    fam.to_csv(fam_path, index=False)
    print(f"\nSAVED {out_path}")
    print(f"SAVED {fam_path}")
    if not out.empty:
        print("\nTOP DEPLOYABLE SURVIVORS")
        print(out[out.deployable].sort_values(["mean_return", "pos_folds", "mean_pf"], ascending=[False, False, False]).head(30)[[
            "symbol", "family", "interval", "pos_folds", "mean_return", "mean_pf", "total_trades", "direction_filter", "regime", "stop_rule", "atr_stop_mult", "take_profit_r"
        ]].to_string(index=False))
        print("\nFAMILY SUMMARY")
        print(fam.to_string(index=False))


if __name__ == "__main__":
    main()
