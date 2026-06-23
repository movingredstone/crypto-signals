"""
Fold-based strategy evaluation with stress testing.

Evaluates trading strategies across 7 half-year folds (2023H1~2026H1)
with per-fold metrics: return, MDD, PF, trades, win rate, fee/slippage.
Stress mode applies 8 harsher conditions on survivors.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.research_engine import (
    ALL_FAMILIES,
    DEFAULT_AXES,
    backtest_experiment,
    backtest_experiment_detailed,
    build_experiments,
    enrich_features,
    load_config,
    load_or_download_klines,
    format_progress_message,
)

# ── 7 Half-year folds ──────────────────────────────────────────────
FOLDS = [
    ("2023 H1", "2023-01-01", "2023-07-01"),
    ("2023 H2", "2023-07-01", "2024-01-01"),
    ("2024 H1", "2024-01-01", "2024-07-01"),
    ("2024 H2", "2024-07-01", "2025-01-01"),
    ("2025 H1", "2025-01-01", "2025-07-01"),
    ("2025 H2", "2025-07-01", "2026-01-01"),
    ("2026 H1", "2026-01-01", "2026-06-01"),
]

# ── Stress overrides (Phase 6) ─────────────────────────────────────
STRESS_OVERRIDES = {
    "slippage_mult": 2.0,       # 2x slippage
    "fee_mult": 2.5,            # taker fee (0.0005 / 0.0002 = 2.5x maker)
    "entry_delay": 1,           # 1 bar delay
    "conservative_stop": True,  # extra slippage on stops
    "sl_priority": True,        # SL wins over TP when both triggered
    "funding_include": True,    # deduct funding costs
    "max_leverage": 1.0,        # 1x spot-level leverage
    "max_trades_per_day": 1,    # max 1 trade/day
    "consecutive_loss_limit": 3,  # stop after 3 consecutive losses
}

# ── Default (baseline) overrides ───────────────────────────────────
BASELINE_OVERRIDES = {
    "slippage_mult": 1.0,
    "fee_mult": 1.0,
    "entry_delay": 0,
    "conservative_stop": False,
    "sl_priority": True,        # already default in simulate_exit
    "funding_include": False,
    "max_leverage": 2.0,
    "max_trades_per_day": 3,
    "consecutive_loss_limit": 999,  # effectively disabled
}

# ── Worker globals ─────────────────────────────────────────────────
_WORKER_RECORDS = None
_WORKER_CONFIG = None
_WORKER_FOLDS = None
_WORKER_OVERRIDES = None


def _init_worker(records, config, folds, overrides):
    global _WORKER_RECORDS, _WORKER_CONFIG, _WORKER_FOLDS, _WORKER_OVERRIDES
    _WORKER_RECORDS = records
    _WORKER_CONFIG = config
    _WORKER_FOLDS = folds
    _WORKER_OVERRIDES = overrides


def _run_experiment_folds(item):
    """Run one experiment across all 7 folds. Returns dict with per-fold metrics."""
    idx, exp = item
    results = {
        "experiment_id": idx,
        "symbol": exp["symbol"],
        "interval": exp["interval"],
        "family": exp["family"],
        "params_json": json.dumps(exp, ensure_ascii=False),
    }

    # Apply overrides to exp
    exp_mod = dict(exp)
    exp_mod["slippage_mult"] = _WORKER_OVERRIDES["slippage_mult"]
    exp_mod["fee_mult"] = _WORKER_OVERRIDES["fee_mult"]
    exp_mod["entry_delay"] = _WORKER_OVERRIDES.get("entry_delay", 0)
    exp_mod["conservative_stop"] = _WORKER_OVERRIDES.get("conservative_stop", False)
    exp_mod["sl_priority"] = _WORKER_OVERRIDES.get("sl_priority", True)
    exp_mod["funding_include"] = _WORKER_OVERRIDES.get("funding_include", False)
    exp_mod["max_leverage"] = _WORKER_OVERRIDES.get("max_leverage", 2.0)
    exp_mod["max_trades_per_day"] = _WORKER_OVERRIDES.get("max_trades_per_day", 3)
    exp_mod["consecutive_loss_limit"] = _WORKER_OVERRIDES.get("consecutive_loss_limit", 999)

    fold_returns = []
    fold_mdds = []
    fold_pfs = []
    fold_trades = []
    fold_wins = []
    fold_fees = []
    fold_slippage = []

    for fold_name, fold_start, fold_end in _WORKER_FOLDS:
        bt = backtest_experiment(
            _WORKER_RECORDS, exp_mod, _WORKER_CONFIG, fold_start, fold_end
        )
        fold_returns.append(bt["return_pct"])
        fold_mdds.append(-abs(bt["max_drawdown_pct"]))
        fold_pfs.append(bt["profit_factor"])
        fold_trades.append(bt["trades"])
        fold_wins.append(bt["win_rate_pct"])
        fold_fees.append(bt["fees"])
        fold_slippage.append(bt["slippage_cost"])

    # Survival: folds with positive return
    pos_folds = sum(1 for r in fold_returns if r > 0)
    survival_pct = round(pos_folds / len(fold_returns) * 100, 1)

    results.update({
        "fold_returns": json.dumps(fold_returns),
        "fold_mdds": json.dumps(fold_mdds),
        "fold_pfs": json.dumps(fold_pfs),
        "fold_trades": json.dumps(fold_trades),
        "fold_wins": json.dumps(fold_wins),
        "fold_fees": json.dumps(fold_fees),
        "fold_slippage": json.dumps(fold_slippage),
        "pos_folds": pos_folds,
        "total_folds": len(fold_returns),
        "survival_pct": survival_pct,
        "mean_return": round(float(np.mean(fold_returns)), 2),
        "median_return": round(float(np.median(fold_returns)), 2),
        "min_return": round(float(np.min(fold_returns)), 2),
        "max_return": round(float(np.max(fold_returns)), 2),
        "std_return": round(float(np.std(fold_returns)), 2),
        "mean_pf": round(float(np.mean([p for p in fold_pfs if p > 0])), 3) if any(p > 0 for p in fold_pfs) else 0.0,
        "mean_mdd": round(float(np.mean(fold_mdds)), 2),
        "max_mdd": round(float(np.min(fold_mdds)), 2),
        "total_trades": int(sum(fold_trades)),
    })

    return results


def evaluate_folds(
    symbol: str = "BTCUSDT",
    intervals: list[str] | None = None,
    experiments: int = 5000,
    workers: int = 9,
    top_n: int = 50,
    seed: int = 42,
    config_path: str = "config.yaml",
    output_dir: str = "results/fold_eval",
    mode: str = "baseline",  # "baseline" or "stress"
    source_top_path: str | None = None,
    focus_override: dict | None = None,
):
    """Evaluate experiments across 7 half-year folds.

    Args:
        mode: "baseline" (normal conditions) or "stress" (Phase 6 harsh conditions)
        source_top_path: If provided, load experiments from this CSV instead of
            generating random ones.
        focus_override: If provided, merged into the focus dict for
            build_experiments(). Use for custom parameter ranges (e.g. high-vol coins).
    """
    if intervals is None:
        intervals = ["15m", "1h", "4h"]

    config = load_config(config_path)
    overrides = STRESS_OVERRIDES if mode == "stress" else BASELINE_OVERRIDES

    mode_label = "STRESS" if mode == "stress" else "BASELINE"
    print(f"\n{'='*60}")
    print(f"FOLD EVALUATION — {mode_label} MODE")
    print(f"Symbol: {symbol}  |  Intervals: {intervals}")
    print(f"Folds: {len(FOLDS)} half-years ({FOLDS[0][0]} → {FOLDS[-1][0]})")
    if source_top_path:
        print(f"Source: {source_top_path} (loading specific candidates)")
    else:
        print(f"Experiments: {experiments}/interval  |  Workers: {workers}")
    print(f"{'='*60}\n")

    # ── Load experiments from source or generate random ──────────────
    all_exps_by_interval: dict[str, list[dict]] = {}

    if source_top_path and Path(source_top_path).exists():
        # Load and deduplicate experiments from flat CSV
        source_df = pd.read_csv(source_top_path)
        # Sort by survival then mean return to pick the best
        if "survival_pct" in source_df.columns and "mean_return" in source_df.columns:
            source_df = source_df.sort_values(
                ["survival_pct", "mean_return"], ascending=[False, False]
            )
        source_df = source_df.head(top_n)

        for interval in intervals:
            int_df = source_df[source_df["interval"] == interval]
            seen_params = set()
            exps = []
            for _, row in int_df.iterrows():
                try:
                    params = json.loads(row["params_json"])
                except Exception:
                    params = {}
                # Strip stress overrides from params (they'll be applied at backtest time)
                for k in list(params.keys()):
                    if k in ("fee_mult", "slippage_mult", "entry_delay",
                             "conservative_stop", "max_leverage",
                             "max_trades_per_day", "consecutive_loss_limit",
                             "funding_include"):
                        del params[k]
                param_key = json.dumps(params, sort_keys=True)
                if param_key not in seen_params:
                    seen_params.add(param_key)
                    exps.append(params)
            all_exps_by_interval[interval] = exps
            print(f"  {interval}: {len(exps)} unique experiments loaded from source")
        experiments = max(len(v) for v in all_exps_by_interval.values()) if all_exps_by_interval else 0
    else:
        for interval in intervals:
            focus = {"families": list(ALL_FAMILIES)}
            # Merge focus_override if provided (for custom parameter ranges)
            if focus_override:
                for k, v in focus_override.items():
                    focus[k] = v
            exps = build_experiments(symbol, interval, max_experiments=experiments, focus=focus, seed=seed)
            all_exps_by_interval[interval] = exps

    print(f"\nWorkers: {workers}")

    all_results = []

    for interval in intervals:
        exps = all_exps_by_interval.get(interval, [])
        if not exps:
            print(f"\n── Interval: {interval} — no experiments, skipping ──")
            continue

        print(f"\n── Interval: {interval} ({len(exps)} experiments) ──")

        # Load data (full range for warmup + all folds)
        data_start = config["backtest"]["start_date"]
        data_end = config["backtest"]["end_date"]
        df = load_or_download_klines(symbol, interval, data_start, data_end)

        # Collect all lookbacks needed
        lookbacks_used = sorted({int(e.get("lookback", 20)) for e in exps})
        df = enrich_features(df, interval, lookbacks=lookbacks_used)

        records = df.to_dict("records")
        items = list(enumerate(exps, start=1))

        print(f"  Experiments: {len(items)}  |  Bars: {len(records)}")

        if workers <= 1:
            _init_worker(records, config, FOLDS, overrides)
            for item in items:
                all_results.append(_run_experiment_folds(item))
                if item[0] % 50 == 0 or item[0] == len(items):
                    print("  " + format_progress_message("evaluated", item[0], len(items), "experiments"))
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(records, config, FOLDS, overrides),
            ) as executor:
                futures = [executor.submit(_run_experiment_folds, item) for item in items]
                for done_count, future in enumerate(as_completed(futures), start=1):
                    all_results.append(future.result())
                    if done_count % 100 == 0 or done_count == len(items):
                        print("  " + format_progress_message("evaluated", done_count, len(items), "experiments"))

    # ── Build results DataFrame ────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        print("\nNo results generated.")
        return {"df": results_df, "path": "", "mode": mode}

    # Sort by survival rate, then mean return
    results_df = results_df.sort_values(
        ["survival_pct", "mean_return"], ascending=[False, False]
    ).reset_index(drop=True)

    # ── Save artifacts ─────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # ── Best candidate detailed trades ──────────────────────────────
    # Find the best candidate (highest survival, then mean return)
    if not results_df.empty:
        best = results_df.iloc[0]
        best_exp = json.loads(best["params_json"])
        best_interval = best["interval"]

        # Run detailed backtest for best candidate
        best_exp["fee_mult"] = overrides.get("fee_mult", 1.0)
        best_exp["slippage_mult"] = overrides.get("slippage_mult", 1.0)
        best_exp["entry_delay"] = overrides.get("entry_delay", 0)
        best_exp["conservative_stop"] = overrides.get("conservative_stop", False)
        best_exp["max_leverage"] = overrides.get("max_leverage", 2.0)
        best_exp["max_trades_per_day"] = overrides.get("max_trades_per_day", 3)
        best_exp["consecutive_loss_limit"] = overrides.get("consecutive_loss_limit", 999)
        best_exp["funding_include"] = overrides.get("funding_include", False)

        # Load data for best candidate's interval
        data_start = config["backtest"]["start_date"]
        data_end = config["backtest"]["end_date"]
        best_df = load_or_download_klines(symbol, best_interval, data_start, data_end)
        best_lookbacks = sorted({int(best_exp.get("lookback", 20))})
        best_df = enrich_features(best_df, best_interval, lookbacks=best_lookbacks)

        # Run backtest across all folds, collecting detailed trades + equity
        all_trades = []
        all_equity = []
        for fold_name, fold_start, fold_end in FOLDS:
            bt_trades, bt_equity = backtest_experiment_detailed(
                best_df.to_dict("records"),
                best_exp,
                config,
                fold_start,
                fold_end,
            )
            for t in bt_trades:
                t["fold"] = fold_name
            all_trades.extend(bt_trades)
            for e in bt_equity:
                e["fold"] = fold_name
            all_equity.extend(bt_equity)

        if all_trades:
            trades_df_out = pd.DataFrame(all_trades)
            trades_path = out_dir / f"{symbol}_{mode}_{ts}_best_candidate_trades.csv"
            trades_df_out.to_csv(trades_path, index=False)
            print(f"  Best trades:  {trades_path} ({len(all_trades)} trades)")

            equity_df_out = pd.DataFrame(all_equity)
            equity_path = out_dir / f"{symbol}_{mode}_{ts}_best_candidate_equity.csv"
            equity_df_out.to_csv(equity_path, index=False)
            print(f"  Best equity:  {equity_path} ({len(all_equity)} bars)")
        else:
            trades_path = None
            equity_path = None
            print(f"  Best trades:  no trades generated")
    else:
        trades_path = None

    # Build flat DataFrame with fold-level columns
    fold_names = [f[0] for f in FOLDS]
    flat_rows = []
    for _, row in results_df.iterrows():
        flat = {
            "experiment_id": row["experiment_id"],
            "symbol": row["symbol"],
            "interval": row["interval"],
            "family": row["family"],
            "params_json": row["params_json"],
        }
        # Fold-level metrics
        returns = json.loads(row["fold_returns"])
        mdds = json.loads(row["fold_mdds"])
        pfs = json.loads(row["fold_pfs"])
        trades = json.loads(row["fold_trades"])
        wins = json.loads(row["fold_wins"])
        fees = json.loads(row["fold_fees"])
        slippage = json.loads(row["fold_slippage"])
        for j, fn in enumerate(fold_names):
            key = fn.replace(" ", "_").lower()  # e.g. 2023_h1
            flat[f"{key}_return_pct"] = returns[j]
            flat[f"{key}_mdd_pct"] = mdds[j]
            flat[f"{key}_pf"] = pfs[j]
            flat[f"{key}_trades"] = trades[j]
            flat[f"{key}_win_rate_pct"] = wins[j]
            flat[f"{key}_fees"] = fees[j]
            flat[f"{key}_slippage_cost"] = slippage[j]
        # Summary
        flat["pos_folds"] = int(row["pos_folds"])
        flat["total_folds"] = int(row["total_folds"])
        flat["survival_pct"] = row["survival_pct"]
        flat["mean_return"] = row["mean_return"]
        flat["median_return"] = row["median_return"]
        flat["min_return"] = row["min_return"]
        flat["max_return"] = row["max_return"]
        flat["std_return"] = row["std_return"]
        flat["mean_pf"] = row["mean_pf"]
        flat["mean_mdd"] = row["mean_mdd"]
        flat["max_mdd"] = row["max_mdd"]
        flat["total_trades"] = int(row["total_trades"])
        flat_rows.append(flat)

    flat_df = pd.DataFrame(flat_rows)

    # Flattened CSV
    csv_flat_path = out_dir / f"{symbol}_{mode}_{ts}_fold_flat.csv"
    flat_df.to_csv(csv_flat_path, index=False)

    # Summary JSON (matching optimizer format)
    summary = {
        "symbol": symbol,
        "stage": mode,
        "intervals": intervals,
        "experiments": experiments,
        "workers": workers,
        "generated_at_utc": ts,
        "folds": [f[0] for f in FOLDS],
        "top_candidates_path": str(csv_flat_path),
        "summary_path": str(out_dir / f"{symbol}_{mode}_{ts}_summary.json"),
        "markdown_path": str(out_dir / f"{symbol}_{mode}_{ts}_fold_report.md"),
        "survival_counts": {
            "7_of_7": int((results_df["pos_folds"] >= 7).sum()),
            "6_of_7": int((results_df["pos_folds"] >= 6).sum()),
            "5_of_7": int((results_df["pos_folds"] >= 5).sum()),
            "4_of_7": int((results_df["pos_folds"] >= 4).sum()),
            "total": int(len(results_df)),
        },
        "top_candidates": flat_df.head(50).to_dict(orient="records"),
    }
    summary_path = out_dir / f"{symbol}_{mode}_{ts}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Full results with fold arrays as JSON (for programmatic use)
    json_full_path = out_dir / f"{symbol}_{mode}_{ts}_fold_full.json"
    results_df.to_json(json_full_path, orient="records", indent=2, force_ascii=False)

    # Markdown report
    report_path = out_dir / f"{symbol}_{mode}_{ts}_fold_report.md"
    report = format_fold_report(results_df, symbol, mode, intervals, experiments)
    report_path.write_text(report, encoding="utf-8")

    # Best candidate markdown
    best_md_path = out_dir / f"{symbol}_{mode}_{ts}_best_candidate.md"
    if not results_df.empty:
        best_md = format_best_candidate_md(
            results_df.iloc[0], symbol, mode, 
            trades_path=str(trades_path) if trades_path else None,
            equity_path=str(equity_path) if equity_path else None,
        )
        best_md_path.write_text(best_md, encoding="utf-8")
        print(f"  Best MD:      {best_md_path}")
    else:
        best_md_path = None

    print(f"\nResults saved:")
    print(f"  Flat CSV:     {csv_flat_path}")
    print(f"  Summary JSON: {summary_path}")
    print(f"  Full JSON:    {json_full_path}")
    print(f"  Report:       {report_path}")

    return {
        "df": results_df,
        "csv_path": str(csv_flat_path),
        "summary_path": str(summary_path),
        "json_path": str(json_full_path),
        "report_path": str(report_path),
        "mode": mode,
    }


def format_fold_report(df: pd.DataFrame, symbol: str, mode: str, intervals: list, experiments: int) -> str:
    """Generate Markdown report."""
    from textwrap import dedent

    mode_label = "STRESS" if mode == "stress" else "BASELINE"
    total = len(df)

    # Count survivors
    survivors_5of7 = int((df["pos_folds"] >= 5).sum())
    survivors_6of7 = int((df["pos_folds"] >= 6).sum())
    survivors_7of7 = int((df["pos_folds"] >= 7).sum())

    lines = [
        f"# Fold Evaluation Report — {symbol} ({mode_label})",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Intervals:** {', '.join(intervals)}",
        f"**Experiments per interval:** {experiments}",
        f"**Folds:** 2023H1, 2023H2, 2024H1, 2024H2, 2025H1, 2025H2, 2026H1",
        "",
        "## Survival Summary",
        "",
        f"| Threshold | Count | % |",
        f"|-----------|-------|---|",
        f"| All 7/7 positive | {survivors_7of7} | {survivors_7of7/total*100:.1f}% |",
        f"| ≥ 6/7 positive | {survivors_6of7} | {survivors_6of7/total*100:.1f}% |",
        f"| ≥ 5/7 positive | {survivors_5of7} | {survivors_5of7/total*100:.1f}% |",
        f"| Total experiments | {total} | 100% |",
        "",
        "## Top 20 by Survival Rate",
        "",
    ]

    top_n = min(20, total)
    top = df.head(top_n)

    for i, (_, row) in enumerate(top.iterrows(), 1):
        params = json.loads(row["params_json"])
        lines.append(f"### #{i} — {row['family']} / {row['interval']}")
        lines.append(f"**Survival:** {int(row['pos_folds'])}/7 ({row['survival_pct']:.0f}%)  |  "
                     f"Mean Return: {row['mean_return']:+.2f}%  |  "
                     f"Median: {row['median_return']:+.2f}%  |  "
                     f"Min/Max: {row['min_return']:+.2f}% / {row['max_return']:+.2f}%")
        lines.append(f"**Mean PF:** {row['mean_pf']:.3f}  |  "
                     f"Mean MDD: {row['mean_mdd']:.2f}%  |  "
                     f"Worst MDD: {row['max_mdd']:.2f}%  |  "
                     f"Total Trades: {int(row['total_trades'])}")
        lines.append("")
        lines.append("| Fold | Return | MDD | PF | Trades | Win% | Fees | Slippage |")
        lines.append("|------|--------|-----|----|--------|------|------|----------|")
        fold_names = [f[0] for f in FOLDS]
        returns = json.loads(row["fold_returns"])
        mdds = json.loads(row["fold_mdds"])
        pfs = json.loads(row["fold_pfs"])
        trades = json.loads(row["fold_trades"])
        wins = json.loads(row["fold_wins"])
        fees = json.loads(row["fold_fees"])
        slippage = json.loads(row["fold_slippage"])
        for j in range(len(fold_names)):
            sign = "✅" if returns[j] > 0 else "❌"
            lines.append(
                f"| {sign} {fold_names[j]} | {returns[j]:+.2f}% | {mdds[j]:.2f}% | "
                f"{pfs[j]:.3f} | {trades[j]} | {wins[j]:.0f}% | ${fees[j]:.0f} | ${slippage[j]:.0f} |"
            )
        lines.append("")
        # Key parameters
        key_params = {k: v for k, v in params.items()
                      if k not in ("symbol", "interval", "fee_mult", "slippage_mult",
                                   "entry_delay", "conservative_stop", "sl_priority",
                                   "funding_include", "max_leverage", "max_trades_per_day",
                                   "consecutive_loss_limit")}
        lines.append(f"**Params:** `{json.dumps(key_params, ensure_ascii=False)}`")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Stress Conditions" if mode == "stress" else "## Baseline Conditions",
        "",
    ])
    if mode == "stress":
        for k, v in STRESS_OVERRIDES.items():
            lines.append(f"- **{k}:** {v}")
    else:
        for k, v in BASELINE_OVERRIDES.items():
            lines.append(f"- **{k}:** {v}")

    lines.extend([
        "",
        "> ⚠️ This is research/backtest output, not investment advice.",
    ])

    return "\n".join(lines)


def format_best_candidate_md(row, symbol: str, mode: str, trades_path: str = None, equity_path: str = None) -> str:
    """Generate a focused Markdown report for the single best candidate."""
    params = json.loads(row["params_json"])
    fold_names = [f[0] for f in FOLDS]
    returns = json.loads(row["fold_returns"])
    mdds = json.loads(row["fold_mdds"])
    pfs = json.loads(row["fold_pfs"])
    trades = json.loads(row["fold_trades"])
    wins = json.loads(row["fold_wins"])
    fees = json.loads(row["fold_fees"])
    slippage = json.loads(row["fold_slippage"])

    mode_label = "STRESS" if mode == "stress" else "BASELINE"

    lines = [
        f"[investmentsystem Fold Evaluation — Best Candidate]",
        "",
        f"Symbol: {symbol}",
        f"Stage: {mode_label}",
        f"Interval: {row['interval']}",
        f"Family: {row['family']}",
        f"Survival: {int(row['pos_folds'])}/7 folds ({row['survival_pct']:.0f}%)",
        "",
        "## Parameters",
        "",
        "```json",
        json.dumps(params, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Fold-by-Fold Metrics",
        "",
        "| Fold | Return | MDD | PF | Trades | Win% | Fees | Slippage |",
        "|------|--------|-----|----|--------|------|------|----------|",
    ]

    for j, fn in enumerate(fold_names):
        sign = "✅" if returns[j] > 0 else "❌"
        lines.append(
            f"| {sign} {fn} | {returns[j]:+.2f}% | {mdds[j]:.2f}% | "
            f"{pfs[j]:.3f} | {trades[j]} | {wins[j]:.0f}% | ${fees[j]:.0f} | ${slippage[j]:.0f} |"
        )

    lines.extend([
        "",
        "## Aggregate",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean Return | {row['mean_return']:+.2f}% |",
        f"| Median Return | {row['median_return']:+.2f}% |",
        f"| Min Return | {row['min_return']:+.2f}% |",
        f"| Max Return | {row['max_return']:+.2f}% |",
        f"| Std Return | {row['std_return']:.2f}% |",
        f"| Mean PF | {row.get('mean_pf') or 0:.3f} |",
        f"| Mean MDD | {row.get('mean_mdd') or 0:.2f}% |",
        f"| Worst MDD | {row.get('max_mdd') or 0:.2f}% |",
        f"| Total Trades | {int(row['total_trades'])} |",
        "",
    ])

    if mode == "stress":
        lines.append("## Stress Conditions Applied")
        lines.append("")
        for k, v in STRESS_OVERRIDES.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    lines.extend([
        "## Artifact Files",
        "",
    ])
    if trades_path:
        lines.append(f"- Trades: `{trades_path}`")
    if equity_path:
        lines.append(f"- Equity: `{equity_path}`")
    lines.extend([
        "",
        "> ⚠️ This is research/backtest output, not investment advice.",
    ])

    return "\n".join(lines)
