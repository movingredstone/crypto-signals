"""
TVT (Train / Validate / Test) forward cross-validation.

Evaluates strategy candidates on a strict forward time split to detect
overfitting. The candidate parameters come from prior optimization;
TVT checks whether performance holds up on unseen future data.

Split: Train = 2023, Validate = 2024, Test = 2025-01 → data end.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.research_engine import (
    load_config,
    load_or_download_klines,
    enrich_features,
    backtest_experiment,
)

# ── TVT splits (forward time, no look-ahead) ────────────────────────
# These must be fully within config backtest date range.
TVT_SPLITS = {
    "train": ("2023-01-01", "2024-10-01"),      # 21 months
    "validate": ("2024-10-01", "2025-07-01"),    # 9 months
    "test": ("2025-07-01", "2026-06-01"),        # 11 months
}


def run_tvt(
    candidates: list[dict],
    symbol: str = "BTCUSDT",
    config_path: str = "config.yaml",
    output_dir: str = "results/tvt",
) -> dict:
    """Run TVT forward cross-validation on specific strategy candidates.

    Args:
        candidates: List of strategy experiment dicts.
        symbol: Trading pair.
        config_path: YAML config path.
        output_dir: Where to save CSV and report.

    Returns:
        Dict with 'df', 'csv_path', 'report_path'.
    """
    config = load_config(config_path)
    data_start = config["backtest"]["start_date"]
    data_end = config["backtest"]["end_date"]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    data_cache: dict[str, list[dict]] = {}

    for idx, exp in enumerate(candidates):
        interval = exp.get("interval", "1h")
        family = exp.get("family", "unknown")

        if interval not in data_cache:
            df = load_or_download_klines(symbol, interval, data_start, data_end)
            lookbacks = sorted({int(exp.get("lookback", 20))})
            df = enrich_features(df, interval, lookbacks=lookbacks)
            data_cache[interval] = df.to_dict("records")

        records = data_cache[interval]

        row = {
            "candidate_id": idx + 1,
            "symbol": symbol,
            "interval": interval,
            "family": family,
            "params_json": json.dumps(exp, ensure_ascii=False),
        }

        splits_ok = 0
        for split_name, (split_start, split_end) in TVT_SPLITS.items():
            bt = backtest_experiment(records, exp, config, split_start, split_end)
            ret = bt.get("return_pct", 0.0)
            pf = bt.get("profit_factor", 0.0)
            mdd = bt.get("max_drawdown_pct", 0.0)
            trades = bt.get("trades", 0)
            win_rate = bt.get("win_rate_pct", 0.0)
            fees = bt.get("fees", 0.0)
            slippage_cost = bt.get("slippage_cost", 0.0)

            row[f"{split_name}_return_pct"] = ret
            row[f"{split_name}_pf"] = pf
            row[f"{split_name}_mdd_pct"] = mdd
            row[f"{split_name}_trades"] = trades
            row[f"{split_name}_win_rate_pct"] = win_rate
            row[f"{split_name}_fees"] = fees
            row[f"{split_name}_slippage_cost"] = slippage_cost

            if ret > 0:
                splits_ok += 1

        train_ret = row["train_return_pct"]
        val_ret = row["validate_return_pct"]
        test_ret = row["test_return_pct"]

        real_avg = (val_ret + test_ret) / 2 if (val_ret + test_ret) != 0 else 0
        gap = train_ret - real_avg
        row["train_vs_real_gap"] = gap
        row["splits_positive"] = splits_ok
        row["total_trades"] = sum(
            row[f"{s}_trades"] for s in TVT_SPLITS
        )
        row["stability_std"] = pd.Series([train_ret, val_ret, test_ret]).std()

        results.append(row)

    df = pd.DataFrame(results)
    if df.empty:
        return {"df": df, "csv_path": "", "report_path": ""}

    df = df.sort_values(
        ["splits_positive", "train_vs_real_gap", "test_return_pct"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    csv_path = out_dir / f"{symbol}_TVT_{ts}.csv"
    df.to_csv(csv_path, index=False)

    report_path = out_dir / f"{symbol}_TVT_{ts}.md"
    report = format_tvt_report(df, symbol, len(candidates))
    report_path.write_text(report, encoding="utf-8")

    print(f"\nTVT Results saved:")
    print(f"  CSV:    {csv_path}")
    print(f"  Report: {report_path}")

    return {
        "df": df,
        "csv_path": str(csv_path),
        "report_path": str(report_path),
    }


def format_tvt_report(df: pd.DataFrame, symbol: str, total_candidates: int) -> str:
    """Generate Markdown TVT report."""
    lines = [
        f"# TVT Forward Cross-Validation — {symbol}",
        "",
        f"**Candidates tested:** {total_candidates}",
        f"**Split:** Train=2023 | Validate=2024 | Test=2025-2026",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## How to read",
        "",
        "- **train_vs_real_gap**: positive = train outperforms real (val+test) — overfitting",
        "- **splits_positive**: how many of the 3 splits are profitable",
        "- **stability_std**: lower = more consistent across time periods",
        "- **Ideal**: all 3 positive, low gap, low stability_std",
        "",
        "## Results",
        "",
        "| # | Strategy | Train | Val | Test | Gap | 3/3? | Std | Trades |",
        "|---|----------|-------|-----|------|-----|------|-----|--------|",
    ]

    for _, row in df.iterrows():
        name = f"{row['family']}/{row['interval']}"
        train_r = row["train_return_pct"]
        val_r = row["validate_return_pct"]
        test_r = row["test_return_pct"]
        gap = row["train_vs_real_gap"]
        all3 = "Y" if row["splits_positive"] >= 3 else str(row["splits_positive"])
        std = row["stability_std"]
        trades = int(row["total_trades"])

        lines.append(
            f"| {int(row['candidate_id'])} | {name} | {train_r:+.2f}% | {val_r:+.2f}% | "
            f"{test_r:+.2f}% | {gap:+.2f}% | {all3} | {std:.2f}% | {trades} |"
        )

    lines.extend([
        "",
        "## Detailed Per-Candidate",
        "",
    ])

    for _, row in df.iterrows():
        name = f"{row['family']} / {row['interval']}"
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Split | Return | PF | MDD | Trades | Win% | Fees | Slippage |")
        lines.append("|-------|--------|----|-----|--------|------|------|----------|")
        for split in ["train", "validate", "test"]:
            r = row[f"{split}_return_pct"]
            pf = row[f"{split}_pf"]
            mdd = row[f"{split}_mdd_pct"]
            t = int(row[f"{split}_trades"])
            wr = row[f"{split}_win_rate_pct"]
            fees = row[f"{split}_fees"]
            sl = row[f"{split}_slippage_cost"]
            lines.append(
                f"| {split.capitalize()} | {r:+.2f}% | {pf:.3f} | {mdd:.2f}% | "
                f"{t} | {wr:.0f}% | ${fees:.0f} | ${sl:.0f} |"
            )
        lines.append("")
        try:
            params = json.loads(row["params_json"])
            key_params = {k: v for k, v in params.items()
                          if k not in ("symbol", "interval", "family",
                                       "fee_mult", "slippage_mult", "entry_delay",
                                       "conservative_stop", "max_leverage",
                                       "max_trades_per_day", "consecutive_loss_limit",
                                       "funding_include")}
            lines.append(f"**Params:** `{json.dumps(key_params, ensure_ascii=False)}`")
        except Exception:
            pass

        gap = row["train_vs_real_gap"]
        std = row["stability_std"]
        all3 = "Yes" if row["splits_positive"] >= 3 else f"No ({int(row['splits_positive'])}/3)"
        lines.extend([
            "",
            f"- Train vs Real Gap: {gap:+.2f}%",
            f"- Stability Std: {std:.2f}%",
            f"- All 3 splits positive: {all3}",
            "",
            "---",
            "",
        ])

    lines.extend([
        "",
        "## Interpretation Guide",
        "",
        "| Signal | Meaning |",
        "|--------|---------|",
        "| All 3 splits positive | Robust strategy, works across regimes |",
        "| Gap < +0.5% | No significant overfitting |",
        "| Gap > +2% | Possible overfitting — great on train, weak on real data |",
        "| Std < 1% | Stable across time |",
        "| Std > 3% | Inconsistent — regime-dependent |",
        "",
        "> This is research/backtest output, not investment advice.",
    ])

    return "\n".join(lines)
