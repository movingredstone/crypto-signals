#!/usr/bin/env python3
"""Rolling walk-forward for high-vol stage-2 survivors.

After high_vol_stage2_large_hunt.py finishes, this script tests whether the selected
candidate pool can generalize forward. It selects using TRAIN ONLY, then tests the next
3 months. It reports both:
- all families (including breakout families that need paper-trader engine work)
- deployable-only families (macd_momentum, trend_pullback)

Run:
  PYTHONUNBUFFERED=1 python scripts/high_vol_stage2_walk_forward.py
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, ".")

import pandas as pd
from src.research_engine import load_config, load_or_download_klines, enrich_features, backtest_experiment

OUTPUT_DIR = Path("results/optimization")
STAGE2_SURVIVORS = OUTPUT_DIR / "high_vol_stage2_large_survivors.csv"
STAGE2_PARTIAL = OUTPUT_DIR / "high_vol_stage2_large_survivors_partial.csv"
SUPPORTED = {"macd_momentum", "trend_pullback"}
SP500_Q = 2.41
WINDOWS = [
    ("WF1", "2023-01-01", "2024-01-01", "2024-01-01", "2024-04-01"),
    ("WF2", "2023-04-01", "2024-04-01", "2024-04-01", "2024-07-01"),
    ("WF3", "2023-07-01", "2024-07-01", "2024-07-01", "2024-10-01"),
    ("WF4", "2023-10-01", "2024-10-01", "2024-10-01", "2025-01-01"),
    ("WF5", "2024-01-01", "2025-01-01", "2025-01-01", "2025-04-01"),
    ("WF6", "2024-04-01", "2025-04-01", "2025-04-01", "2025-07-01"),
    ("WF7", "2024-07-01", "2025-07-01", "2025-07-01", "2025-10-01"),
    ("WF8", "2024-10-01", "2025-10-01", "2025-10-01", "2026-01-01"),
    ("WF9", "2025-01-01", "2026-01-01", "2026-01-01", "2026-04-01"),
]


def honest_df(df: pd.DataFrame) -> pd.DataFrame:
    pf_cols = [c for c in df.columns if c.endswith("_pf") and c[:4].isdigit()]
    r = df[(df.pos_folds >= 5) & (df.total_trades >= 40)].copy()
    if r.empty:
        return r

    def honest(row) -> bool:
        vals = []
        for c in pf_cols:
            try:
                vals.append(float(row[c]))
            except Exception:
                pass
        try:
            mpf = float(row.mean_pf)
        except Exception:
            return False
        return bool(vals) and all(math.isfinite(v) and v <= 20 for v in vals) and math.isfinite(mpf) and mpf < 10

    return r[r.apply(honest, axis=1)].copy()


def symbols_from_stage2() -> List[str]:
    p = STAGE2_SURVIVORS if STAGE2_SURVIVORS.exists() else STAGE2_PARTIAL
    if p.exists():
        df = pd.read_csv(p)
        if not df.empty:
            # Rank symbols by deployable candidates first, then return/PF/folds.
            df["score"] = (
                df["deployable"].astype(bool).astype(int) * 2
                + df["mean_return"].astype(float) * 1.5
                + df["mean_pf"].astype(float) * 0.7
                + df["pos_folds"].astype(float)
            )
            g = df.groupby("symbol").agg(score=("score", "max")).sort_values("score", ascending=False)
            return g.index.tolist()
    # Fallback: latest stage-2 stress symbols by filenames.
    syms = []
    for path in sorted(glob.glob(str(OUTPUT_DIR / "*_stress_*_fold_flat.csv"))):
        sym = Path(path).name.split("_stress_")[0]
        if sym not in syms:
            syms.append(sym)
    return syms


def load_candidates(symbol: str, deployable_only: bool) -> List[Dict]:
    paths = sorted(glob.glob(str(OUTPUT_DIR / f"{symbol}_stress_*_fold_flat.csv")))
    if not paths:
        return []
    path = paths[-1]
    df = honest_df(pd.read_csv(path))
    if deployable_only:
        df = df[df.family.isin(list(SUPPORTED))].copy()
    out = []
    for _, r in df.iterrows():
        try:
            exp = json.loads(str(r.params_json))
        except Exception:
            continue
        exp["symbol"] = symbol
        exp["interval"] = str(r.interval)
        exp["family"] = str(r.family)
        exp["_stress_path"] = path
        exp["_stress_mean_return"] = float(r.mean_return)
        exp["_stress_pos_folds"] = int(r.pos_folds)
        exp["_stress_total_trades"] = int(r.total_trades)
        exp["_stress_mean_pf"] = float(r.mean_pf)
        out.append(exp)
    return out


def score_train(m: Dict) -> float:
    trades = int(m.get("trades", 0))
    ret = float(m.get("return_pct", 0.0))
    pf = float(m.get("profit_factor", m.get("pf", 0.0)) or 0.0)
    if trades < 12 or ret <= 0 or not math.isfinite(pf) or pf < 1.15:
        return -999.0
    # Prefer robust train return, honest PF, enough trades; avoid pure sparse lottery picks.
    return ret * 0.45 + min(pf, 5) * 0.35 + min(trades, 80) / 80 * 0.6


def run_mode(mode: str, deployable_only: bool) -> pd.DataFrame:
    cfg = load_config("config.yaml")
    symbols = symbols_from_stage2()
    print(f"\nMODE={mode} deployable_only={deployable_only} symbols={len(symbols)}", flush=True)
    rows = []
    for si, symbol in enumerate(symbols, 1):
        cands = load_candidates(symbol, deployable_only)
        if not cands:
            print(f"[{si}/{len(symbols)}] {symbol}: no candidates", flush=True)
            continue
        by_interval: Dict[str, List[Dict]] = {}
        for exp in cands:
            by_interval.setdefault(exp["interval"], []).append(exp)
        records_by_interval = {}
        for interval, exps in by_interval.items():
            lookbacks = sorted({int(e.get("lookback", 20)) for e in exps})
            df = load_or_download_klines(symbol, interval, "2023-01-01", "2026-06-01")
            df = enrich_features(df, interval, lookbacks=lookbacks)
            records_by_interval[interval] = df.to_dict("records")
        print(f"[{si}/{len(symbols)}] {symbol}: candidates={len(cands)}", flush=True)
        total = 0.0
        pos = 0
        n = 0
        trades = 0
        for w, tr_s, tr_e, te_s, te_e in WINDOWS:
            best = None
            for exp in cands:
                rec = records_by_interval[exp["interval"]]
                train = backtest_experiment(rec, exp, cfg, tr_s, tr_e)
                sc = score_train(train)
                if best is None or sc > best[0]:
                    best = (sc, exp, train)
            if best is None or best[0] <= -999:
                print(f"  {w}: no train-qualified candidate", flush=True)
                continue
            sc, exp, train = best
            test = backtest_experiment(records_by_interval[exp["interval"]], exp, cfg, te_s, te_e)
            pf = test.get("profit_factor", test.get("pf", 0))
            total += float(test["return_pct"])
            pos += float(test["return_pct"]) > 0
            n += 1
            trades += int(test["trades"])
            rows.append({
                "mode": mode,
                "symbol": symbol,
                "window": w,
                "family": exp["family"],
                "interval": exp["interval"],
                "supported": exp["family"] in SUPPORTED,
                "train_ret": train["return_pct"],
                "train_trades": train["trades"],
                "test_ret": test["return_pct"],
                "test_pf": pf,
                "test_trades": test["trades"],
                "stress_mean_return": exp["_stress_mean_return"],
                "stress_pos_folds": exp["_stress_pos_folds"],
                "stress_total_trades": exp["_stress_total_trades"],
                "stress_mean_pf": exp["_stress_mean_pf"],
                "stress_path": exp["_stress_path"],
                "params": json.dumps(exp),
            })
            print(f"  {w}: {exp['family']}/{exp['interval']} train={train['return_pct']:+.2f}%/{train['trades']}t -> TEST={test['return_pct']:+.2f}% PF={pf:.2f} {test['trades']}t", flush=True)
        if n:
            print(f"  summary {symbol}: pos={pos}/{n} avg_test={total/n:+.2f}% summed={total:+.2f}% trades={trades} beats_q={total/n>=SP500_Q}", flush=True)
    return pd.DataFrame(rows)


def print_summary(out: pd.DataFrame) -> None:
    if out.empty:
        print("No WF rows")
        return
    for mode, sub in out.groupby("mode"):
        print(f"\nOVERALL BY SYMBOL — {mode}")
        g = sub.groupby("symbol").agg(
            avg_test=("test_ret", "mean"),
            sum_test=("test_ret", "sum"),
            pos=("test_ret", lambda s: int((s > 0).sum())),
            n=("test_ret", "count"),
            trades=("test_trades", "sum"),
            supported_rate=("supported", "mean"),
        ).sort_values("avg_test", ascending=False)
        g["beats_sp500_q"] = g["avg_test"] >= SP500_Q
        print(g.head(30).to_string())
        print(f"\nBEATS S&P500 Q — {mode}")
        print(g[g.beats_sp500_q].to_string())
        print(f"\nFAMILY PICKS — {mode}")
        fg = sub.groupby(["family", "interval"]).agg(picks=("symbol", "count"), avg_test=("test_ret", "mean"), pos=("test_ret", lambda s: int((s > 0).sum()))).sort_values(["picks", "avg_test"], ascending=[False, False])
        print(fg.head(20).to_string())


def main() -> None:
    all_mode = run_mode("all_families", deployable_only=False)
    dep_mode = run_mode("deployable_only", deployable_only=True)
    out = pd.concat([all_mode, dep_mode], ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "high_vol_stage2_walk_forward_results.csv"
    out.to_csv(out_path, index=False)
    print_summary(out)
    print(f"SAVED {out_path}")


if __name__ == "__main__":
    main()
