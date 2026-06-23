#!/usr/bin/env python3
"""Expanded S&P500-hurdle crypto scan.

Searches additional liquid small-unit Binance USDT futures for candidates that can
clear a harder hurdle than the current portfolio. This is still paper/research only.

Design:
- Avoid 15m fee trap.
- Include 1h/4h/8h for more opportunities, but final deployment must still pass WF.
- Stress-test only top baseline candidates.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd
from src.fold_evaluator import evaluate_folds
from src.optimizer import _run_stress as run_stress

# Mostly sub-$50, liquid/high-beta futures. Skip already-scanned 13 names.
COINS = [
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
    "FILUSDT",
    "INJUSDT",
    "ATOMUSDT",
    "TRXUSDT",
    "SEIUSDT",
    "FETUSDT",
    "WIFUSDT",
    "TONUSDT",
    "AAVEUSDT",  # higher unit price, but liquid; can be filtered later for small-cap practicality
]

# Wider stops/targets for high-vol alts. Keep search compact enough to iterate.
FOCUS = {
    "atr_stop_mults": [1.2, 1.5, 2.0, 2.5, 3.0, 4.0],
    "take_profit_rs": [1.8, 2.0, 2.5, 3.0, 4.0, 5.0],
}

DEPLOYABLE = {"macd_momentum", "trend_pullback"}
HURDLE_Q = 2.41  # S&P500 ~10% annual converted to 3-month return

def honest_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    pf_cols = [c for c in df.columns if c.endswith("_pf") and c[:4].isdigit()]
    x = df[(df["pos_folds"] >= 5) & (df["total_trades"] >= 40)].copy()
    if x.empty:
        return x
    def honest(row):
        vals = []
        for c in pf_cols:
            try:
                v = float(row[c])
                vals.append(v)
            except Exception:
                pass
        try:
            mean_pf = float(row["mean_pf"])
        except Exception:
            return False
        return bool(vals) and all(math.isfinite(v) and v <= 20 for v in vals) and math.isfinite(mean_pf) and mean_pf < 10
    return x[x.apply(honest, axis=1)].copy()


def main():
    out_dir = Path("results/optimization")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    errors = []
    for i, symbol in enumerate(COINS, 1):
        print(f"\n{'='*80}\n[{i}/{len(COINS)}] {symbol} expanded scan: 1h/4h/8h, 800 exp each\n{'='*80}", flush=True)
        try:
            result = evaluate_folds(
                symbol=symbol,
                intervals=["1h", "4h", "8h"],
                experiments=800,
                workers=9,
                top_n=35,
                seed=20260622,
                config_path="config.yaml",
                output_dir=str(out_dir),
                mode="baseline",
                focus_override=FOCUS,
            )
            csv_path = result.get("csv_path", "")
            df = result.get("df", pd.DataFrame())
            if df.empty or not csv_path or not Path(csv_path).exists():
                print(f"{symbol}: no baseline rows", flush=True)
                errors.append((symbol, "no baseline rows"))
                continue
            print(
                f"{symbol} baseline rows={len(df)} pos>=5={(df['pos_folds']>=5).sum()} pos=7={(df['pos_folds']>=7).sum()}",
                flush=True,
            )
            stress = run_stress(
                symbol=symbol,
                intervals=["1h", "4h", "8h"],
                experiments=35,
                workers=9,
                output_dir=out_dir,
                top_n=35,
                seed=20260622,
                config_path="config.yaml",
                source_top_path=csv_path,
            )
            stress_path = stress.get("top_candidates_path", "")
            if not stress_path or not Path(stress_path).exists():
                print(f"{symbol}: no stress path", flush=True)
                errors.append((symbol, "no stress path"))
                continue
            s = pd.read_csv(stress_path)
            survivors = honest_filter(s)
            top = s.sort_values(["pos_folds", "mean_return", "mean_pf", "total_trades"], ascending=[False, False, False, False]).head(6)
            print("Top raw stress:", flush=True)
            for _, r in top.iterrows():
                mark = "DEPLOY" if r["family"] in DEPLOYABLE else "ENGINE"
                print(
                    f"  {mark:6} {r['family']}/{r['interval']} folds={int(r['pos_folds'])}/7 "
                    f"ret={float(r['mean_return']):+.2f}% PF={float(r['mean_pf']):.2f} trades={int(r['total_trades'])}",
                    flush=True,
                )
            print(f"Survivors R1/R2/R4/R5={len(survivors)}; deployable={survivors['family'].isin(DEPLOYABLE).sum() if not survivors.empty else 0}", flush=True)
            for _, r in survivors.sort_values(["mean_return", "pos_folds", "mean_pf"], ascending=[False, False, False]).head(8).iterrows():
                summary.append({
                    "symbol": symbol,
                    "stress_path": stress_path,
                    "deployable": r["family"] in DEPLOYABLE,
                    "family": r["family"],
                    "interval": r["interval"],
                    "pos_folds": int(r["pos_folds"]),
                    "mean_return": float(r["mean_return"]),
                    "mean_pf": float(r["mean_pf"]),
                    "total_trades": int(r["total_trades"]),
                    "beats_sp500_fold_mean": float(r["mean_return"]) >= HURDLE_Q,
                })
        except Exception as e:
            print(f"ERROR {symbol}: {e}", flush=True)
            errors.append((symbol, repr(e)))
    summary_df = pd.DataFrame(summary)
    summary_path = out_dir / "sp500_expand_scan_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSAVED {summary_path}", flush=True)
    if not summary_df.empty:
        print("\nTOP SURVIVORS BY MEAN_RETURN", flush=True)
        print(summary_df.sort_values(["mean_return", "pos_folds", "mean_pf"], ascending=[False, False, False]).head(30).to_string(index=False), flush=True)
        print("\nDEPLOYABLE ONLY", flush=True)
        d = summary_df[summary_df.deployable]
        print(d.sort_values(["mean_return", "pos_folds", "mean_pf"], ascending=[False, False, False]).head(20).to_string(index=False), flush=True)
    if errors:
        err_path = out_dir / "sp500_expand_scan_errors.csv"
        pd.DataFrame(errors, columns=["symbol", "error"]).to_csv(err_path, index=False)
        print(f"\nERRORS saved {err_path}", flush=True)
        print(pd.DataFrame(errors, columns=["symbol", "error"]).to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
