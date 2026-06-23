"""
Regime-aware strategy evaluation.

Splits market data into regimes FIRST, then evaluates strategy performance
per regime. Enables regime-switching meta strategies.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

from src.regime_classifier import classify_regime
from src.research_engine import (
    backtest_experiment,
    enrich_features,
    load_config,
    load_or_download_klines,
)


def evaluate_per_regime(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    experiments: int = 500,
    workers: int = 4,
    config_path: str = "config.yaml",
    seed: int = 42,
) -> dict:
    """Run strategy evaluation split by market regime.

    Returns:
        {
            "regime_distribution": {...},
            "per_regime_top": {
                "strong_trend_up": [top candidates],
                "strong_trend_down": [...],
                ...
            }
        }
    """
    config = load_config(config_path)
    start_date = config["backtest"]["start_date"]
    end_date = config["backtest"]["end_date"]

    # Load data and classify regime
    df = load_or_download_klines(symbol, interval, start_date, end_date)
    df = enrich_features(df, interval, lookbacks=[20, 48, 96, 144])
    df = classify_regime(df)
    records = df.to_dict("records")

    # Split records by regime
    regimes = ["trend_up", "trend_down", "squeeze", "range", "high_vol", "low_vol"]
    regime_records = {}
    for r in regimes:
        regime_records[r] = [
            rec for rec in records if rec.get("regime") == r
        ]
    
    # Build experiments (sample from full search space)
    from src.research_engine import build_experiments
    exps = build_experiments(symbol, interval, max_experiments=experiments, seed=seed)

    # Evaluate each regime
    results = {}
    for regime_name, recs in regime_records.items():
        if len(recs) < 500:
            results[regime_name] = {"note": f"insufficient bars ({len(recs)}), skipped"}
            continue

        print(f"  {regime_name}: {len(recs)} bars, {len(exps)} experiments")

        rows = []
        for idx, exp in enumerate(exps, start=1):
            bt = backtest_experiment(recs, exp, config, start_date, end_date)
            rows.append({
                "experiment_id": idx,
                "family": exp["family"],
                "params_json": json.dumps(exp, ensure_ascii=False),
                "return_pct": bt["return_pct"],
                "mdd_pct": bt["max_drawdown_pct"],
                "pf": bt["profit_factor"],
                "trades": bt["trades"],
                "win_rate_pct": bt["win_rate_pct"],
                "avg_r": bt["avg_r"],
            })

        df_r = pd.DataFrame(rows)
        if not df_r.empty:
            df_r = df_r.sort_values("return_pct", ascending=False).reset_index(drop=True)

        results[regime_name] = {
            "bars": len(recs),
            "top_10": df_r.head(10).to_dict(orient="records") if not df_r.empty else [],
        }

    # Regime distribution
    dist = df["regime"].value_counts(normalize=True).to_dict()
    dist = {k: round(v * 100, 1) for k, v in dist.items()}

    return {
        "symbol": symbol,
        "interval": interval,
        "experiments": experiments,
        "regime_distribution": dist,
        "per_regime": results,
    }


def format_regime_report(report: dict) -> str:
    """Generate readable regime-aware strategy report."""
    lines = [
        f"[Regime-Aware Strategy Analysis]",
        f"Symbol: {report['symbol']} | Interval: {report['interval']}",
        f"Experiments: {report['experiments']}",
        "",
        "## Regime Distribution",
        "",
        "| Regime | % of Time |",
        "|--------|-----------|",
    ]
    for regime, pct in sorted(report["regime_distribution"].items(), key=lambda x: -x[1]):
        lines.append(f"| {regime} | {pct}% |")

    lines.extend(["", "## Best Strategy per Regime", ""])

    per_regime = report.get("per_regime", {})
    for regime, data in sorted(per_regime.items()):
        if isinstance(data, dict) and "top_10" in data:
            top = data["top_10"]
            if not top:
                continue
            best = top[0]
            lines.extend([
                f"### {regime} ({data['bars']} bars)",
                f"**Best:** {best['family']} — Return: {best['return_pct']:+.2f}% | "
                f"PF: {best['pf']:.3f} | MDD: {best['mdd_pct']:.2f}% | "
                f"Trades: {best['trades']} | Win: {best['win_rate_pct']:.0f}%",
                "",
            ])
            # Top 3 per regime
            for i, r in enumerate(top[:3], 1):
                params = json.loads(r["params_json"])
                key = f"{r['family']} ({params.get('direction_filter','?')}, lookback={params.get('lookback','?')})"
                lines.append(
                    f"  {i}. {key}: {r['return_pct']:+.2f}% PF={r['pf']:.3f} MDD={r['mdd_pct']:.2f}% T={r['trades']}"
                )
            lines.append("")

    # Across-regime consistency
    lines.extend([
        "## Cross-Regime Consistency",
        "",
        "Which families work across multiple regimes?",
        "",
    ])
    regime_families = defaultdict(set)
    for regime, data in per_regime.items():
        if isinstance(data, dict) and "top_10" in data:
            for r in data["top_10"][:5]:
                regime_families[regime].add(r["family"])

    family_scores = defaultdict(int)
    for regime, families in regime_families.items():
        for f in families:
            family_scores[f] += 1

    lines.append("| Family | Regimes Covered |")
    lines.append("|--------|-----------------|")
    for fam, count in sorted(family_scores.items(), key=lambda x: -x[1]):
        if count >= 2:
            lines.append(f"| {fam} | {count} |")

    return "\n".join(lines)
