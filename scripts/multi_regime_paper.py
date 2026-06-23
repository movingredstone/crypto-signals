#!/usr/bin/env python3
"""
Multi-Regime Paper Trading

Runs each of the 3 top strategies in its 2 preferred regimes with split capital.
N=6 total runs, each gets $10,000/6 = $1,666.67 capital allocation.
Compares multi-regime results to baseline paper results.

Preferred regimes per strategy (from regime_mapping.md):
  vwap_pullback/4h  → high_vol + trend_up
  rsi_momentum/8h   → low_vol + trend_down
  keltner_breakout/1h → weekend + post_breakout

Output: results/paper/multi_regime/
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.binance_data import load_or_download_klines
from src.regime_classifier import classify_regime, ALL_REGIMES
from src.research_engine import enrich_features, backtest_experiment_detailed
from src.reports import profit_factor, max_drawdown

# ── Paper period ───────────────────────────────────────────────────────────
PAPER_START = "2026-01-01"
PAPER_END = "2026-06-01"
DATA_START = "2023-01-01"  # need lookback for indicator warmup

# ── Capital allocation ─────────────────────────────────────────────────────
TOTAL_CAPITAL = 10_000.0
NUM_RUNS = 6  # 3 strategies × 2 regimes each
CAPITAL_PER_RUN = round(TOTAL_CAPITAL / NUM_RUNS, 2)  # $1,666.67

# ── Base config (mirrors regime_analysis.py + paper_trader.py) ─────────────
BASE_CONFIG = {
    "fees": {"maker": 0.0002, "taker": 0.0005},
    "slippage": {"BTCUSDT": 0.0002, "default": 0.0004},
    "backtest": {"initial_capital": CAPITAL_PER_RUN},
    "risk": {
        "risk_per_trade": 0.005,
        "max_leverage": 2,
        "daily_loss_limit": 0.01,
        "weekly_loss_limit": 0.03,
        "max_trades_per_day": 3,
        "max_open_positions": 1,
        "min_stop_distance_pct": 0.001,
    },
}

# ── Strategy definitions (from paper_trader.py TOP3_STRATEGIES) ────────────

STRATEGIES = [
    {
        "name": "vwap_pullback/4h",
        "symbol": "BTCUSDT",
        "interval": "4h",
        "family": "vwap_pullback",
        "direction_filter": "price_ema100",
        "lookback": 144,
        "volume_min": 1.5,
        "atr_stop_mult": 1.2,
        "take_profit_r": 1.2,
        "max_holding_bars": 48,
        "stop_rule": "atr",
        "adx_min": 20,
        "regime": "any",  # overridden per run
        "trailing_atr_mult": None,
        "breakeven_r": 1.0,
        "partial_tp_r": 1.0,
        "partial_tp_frac": 0.5,
        "funding_max_z": None,
        "atr_pct_min": 5,
        "atr_pct_max": 95,
        "tolerance_pct": 0.006,
        "pullback_ref": "ema20",
        # Preferred regimes
        "preferred_regimes": ["high_vol", "trend_up"],
    },
    {
        "name": "rsi_momentum/8h",
        "symbol": "BTCUSDT",
        "interval": "8h",
        "family": "rsi_momentum",
        "direction_filter": "supertrend",
        "lookback": 48,
        "volume_min": 1.0,
        "atr_stop_mult": 2.0,
        "take_profit_r": 3.0,
        "max_holding_bars": 72,
        "stop_rule": "atr",
        "adx_min": 0,
        "regime": "any",
        "trailing_atr_mult": 3.0,
        "breakeven_r": None,
        "partial_tp_r": None,
        "partial_tp_frac": 0.5,
        "funding_max_z": None,
        "atr_pct_min": 5,
        "atr_pct_max": 95,
        "rsi_mid": 50,
        "preferred_regimes": ["low_vol", "trend_down"],
    },
    {
        "name": "keltner_breakout/1h",
        "symbol": "BTCUSDT",
        "interval": "1h",
        "family": "keltner_breakout",
        "direction_filter": "mtf_trend",
        "lookback": 96,
        "volume_min": 0.7,
        "atr_stop_mult": 1.2,
        "take_profit_r": 3.0,
        "max_holding_bars": 96,
        "stop_rule": "swing",
        "adx_min": 0,
        "regime": "any",
        "trailing_atr_mult": 3.0,
        "breakeven_r": None,
        "partial_tp_r": None,
        "partial_tp_frac": 0.5,
        "funding_max_z": None,
        "atr_pct_min": 5,
        "atr_pct_max": 95,
        "preferred_regimes": ["weekend", "post_breakout"],
    },
]

# ── Regimes that have native support in confirmation_ok ────────────────────
NATIVE_REGIMES = {"high_vol", "low_vol"}


def _build_exp(strategy: dict, regime: str) -> dict:
    """Build experiment dict from strategy + regime override."""
    exp = {k: v for k, v in strategy.items()
           if k not in ("name", "preferred_regimes")}
    exp["regime"] = regime
    return exp


def _post_filter_trades(
    trades: list[dict],
    records: list[dict],
    target_regime: str,
) -> list[dict]:
    """Filter trades to only those whose entry bar's classified regime matches target_regime."""
    if not trades:
        return []

    # Build lookup: open_time → regime
    time_to_regime = {}
    for rec in records:
        regime_val = rec.get("regime", "unknown")
        ot = rec.get("open_time")
        if ot is not None:
            time_to_regime[str(ot)] = regime_val

    filtered = []
    for t in trades:
        et = t.get("entry_time")
        # Try exact match, then fall back to nearest bar at or before entry
        entry_regime = time_to_regime.get(str(et), None)
        if entry_regime is None:
            # Find closest bar at or before entry_time
            entry_ts = pd.Timestamp(et)
            best = "unknown"
            for ot_str, reg in time_to_regime.items():
                ot_ts = pd.Timestamp(ot_str)
                if ot_ts <= entry_ts and (entry_ts - ot_ts) < pd.Timedelta(hours=24):
                    best = reg
            entry_regime = best

        if entry_regime == target_regime:
            filtered.append(t)

    return filtered


def _compute_summary(trades: list[dict], initial_capital: float) -> dict:
    """Compute summary stats from a list of trades."""
    if not trades:
        return {
            "return_pct": 0.0,
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_r": 0.0,
            "net_pnl": 0.0,
            "total_fees": 0.0,
            "final_equity": initial_capital,
        }

    df = pd.DataFrame(trades)

    # Build equity curve from trades
    df_sorted = df.sort_values("entry_time")
    equity = initial_capital
    points = []
    for _, row in df_sorted.iterrows():
        points.append({"time": str(row["entry_time"]), "equity": round(equity, 2)})
        equity += float(row["net_pnl"])
        points.append({"time": str(row["exit_time"]), "equity": round(equity, 2)})

    eq_df = pd.DataFrame(points)
    final_equity = equity

    return {
        "return_pct": round((final_equity / initial_capital - 1) * 100, 2),
        "trades": len(df),
        "win_rate_pct": round(float((df["net_pnl"] > 0).mean() * 100), 2),
        "profit_factor": round(profit_factor(df), 3),
        "max_drawdown_pct": round(
            max_drawdown(eq_df["equity"]) * 100, 2
        ) if not eq_df.empty and "equity" in eq_df.columns else 0.0,
        "avg_r": round(float(df["r_multiple"].mean()), 3) if "r_multiple" in df.columns else 0.0,
        "net_pnl": round(float(df["net_pnl"].sum()), 2),
        "total_fees": round(float(df["fee"].sum()), 2) if "fee" in df.columns else 0.0,
        "final_equity": round(final_equity, 2),
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = project_root / "results" / "paper" / "multi_regime"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  MULTI-REGIME PAPER TRADING")
    print("=" * 70)
    print(f"  Period: {PAPER_START} → {PAPER_END}")
    print(f"  Total capital: ${TOTAL_CAPITAL:,.2f}")
    print(f"  Capital per run: ${CAPITAL_PER_RUN:,.2f} (N={NUM_RUNS})")
    print()

    all_regime_trades = []  # all trades across all regime runs
    strategy_combined = {}   # strategy_name → combined trades & summary
    regime_run_details = []  # per-run details for reporting

    for strat in STRATEGIES:
        name = strat["name"]
        symbol = strat["symbol"]
        interval = strat["interval"]
        preferred_regimes = strat["preferred_regimes"]

        print(f"\n{'─' * 70}")
        print(f"  Strategy: {name}")
        print(f"  Regimes: {preferred_regimes}")
        print(f"{'─' * 70}")

        # Load & prepare data
        print(f"  Loading {symbol} {interval} data ({DATA_START} → {PAPER_END})...")
        df_raw = load_or_download_klines(symbol, interval, DATA_START, PAPER_END)
        print(f"  Loaded {len(df_raw)} bars")
        df = enrich_features(df_raw, interval)
        df = classify_regime(df)
        records = df.to_dict("records")

        strategy_trades = []

        for regime in preferred_regimes:
            run_label = f"{name} @ {regime}"
            print(f"  [{run_label}] Running backtest...", end=" ", flush=True)

            # Determine the regime param for the backtest engine
            if regime in NATIVE_REGIMES:
                bt_regime = regime  # native filtering via confirmation_ok
            else:
                bt_regime = "any"   # run unfiltered, post-filter afterward

            exp = _build_exp(strat, bt_regime)

            # Create config with the correct capital
            run_config = dict(BASE_CONFIG)
            run_config["backtest"] = dict(BASE_CONFIG["backtest"])
            run_config["backtest"]["initial_capital"] = CAPITAL_PER_RUN

            trades, equity_curve = backtest_experiment_detailed(
                records, exp, run_config, PAPER_START, PAPER_END
            )

            # Post-filter if regime not natively supported
            if regime not in NATIVE_REGIMES:
                trades = _post_filter_trades(trades, records, regime)

            # Tag trades
            for t in trades:
                t["strategy"] = name
                t["regime"] = regime

            summary = _compute_summary(trades, CAPITAL_PER_RUN)
            print(f"{len(trades)} trades, PnL=${summary['net_pnl']:+,.2f}, "
                  f"Return={summary['return_pct']:+.2f}%, PF={summary['profit_factor']}")

            regime_run_details.append({
                "strategy": name,
                "regime": regime,
                "run_label": run_label,
                "capital": CAPITAL_PER_RUN,
                "trades": len(trades),
                "summary": summary,
            })

            strategy_trades.extend(trades)
            all_regime_trades.extend(trades)

        # ── Combine the 2 regime runs for this strategy ──────────────────
        combined_summary = _compute_summary(strategy_trades, CAPITAL_PER_RUN * 2)
        strategy_combined[name] = {
            "trades": strategy_trades,
            "summary": combined_summary,
            "capital": CAPITAL_PER_RUN * 2,
        }

        print(f"  [{name}] COMBINED: {len(strategy_trades)} trades, "
              f"PnL=${combined_summary['net_pnl']:+,.2f}, "
              f"Return={combined_summary['return_pct']:+.2f}%, "
              f"PF={combined_summary['profit_factor']}")

    # ── Combine all 3 strategies ────────────────────────────────────────
    all_trades = []
    for name, data in strategy_combined.items():
        all_trades.extend(data["trades"])

    total_combined = _compute_summary(all_trades, TOTAL_CAPITAL)
    total_trades = len(all_trades)

    print(f"\n{'═' * 70}")
    print(f"  ALL STRATEGIES COMBINED: {total_trades} trades")
    print(f"  PnL: ${total_combined['net_pnl']:+,.2f}")
    print(f"  Return: {total_combined['return_pct']:+.2f}%")
    print(f"  Profit Factor: {total_combined['profit_factor']}")
    print(f"  Win Rate: {total_combined['win_rate_pct']:.1f}%")
    print(f"  Max Drawdown: {total_combined['max_drawdown_pct']:.2f}%")
    print(f"  Avg R: {total_combined['avg_r']}")
    print(f"  Fees: ${total_combined['total_fees']:,.2f}")
    print(f"  Final Equity: ${total_combined['final_equity']:,.2f}")
    print(f"{'═' * 70}")

    # ── Baseline comparison ─────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  COMPARISON TO BASELINE PAPER")
    print(f"{'═' * 70}")
    print(f"  {'Metric':<25} {'Baseline':>12} {'Multi-Regime':>12} {'Delta':>12}")
    print(f"  {'─' * 25} {'─' * 12} {'─' * 12} {'─' * 12}")

    baseline = {"trades": 47, "return_pct": 2.58, "pf": 1.40}
    comparisons = [
        ("Trades", baseline["trades"], total_trades, total_trades - baseline["trades"]),
        ("Return %", baseline["return_pct"], total_combined["return_pct"],
         round(total_combined["return_pct"] - baseline["return_pct"], 2)),
        ("Profit Factor", baseline["pf"], total_combined["profit_factor"],
         round(total_combined["profit_factor"] - baseline["pf"], 3)),
    ]

    for label, base, multi, delta in comparisons:
        sign = "+" if delta > 0 else ""
        print(f"  {label:<25} {base:>12} {multi:>12} {sign}{delta:>11}")

    print(f"{'═' * 70}")

    # ── Save trades CSV ─────────────────────────────────────────────────
    if all_regime_trades:
        trades_df = pd.DataFrame(all_regime_trades)
        desired_cols = [
            "strategy", "regime", "entry_time", "exit_time", "side",
            "entry_price", "exit_price", "stop_loss", "take_profit",
            "reason", "notional", "leverage",
            "gross_pnl", "fee", "slippage_cost", "funding_cost", "net_pnl",
            "equity_after", "r_multiple", "holding_bars",
        ]
        existing_cols = [c for c in desired_cols if c in trades_df.columns]
        remaining_cols = [c for c in trades_df.columns if c not in existing_cols]
        trades_df = trades_df[existing_cols + remaining_cols]
        trades_path = out_dir / f"BTCUSDT_multi_regime_{ts}_trades.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"\n  Trades saved: {trades_path}")

    # ── Save summary JSON ───────────────────────────────────────────────
    result = {
        "timestamp": ts,
        "period": f"{PAPER_START} ~ {PAPER_END}",
        "symbol": "BTCUSDT",
        "total_capital": TOTAL_CAPITAL,
        "capital_per_run": CAPITAL_PER_RUN,
        "num_runs": NUM_RUNS,
        "baseline": baseline,
        "multi_regime_combined": total_combined,
        "strategy_combined": {
            name: data["summary"] for name, data in strategy_combined.items()
        },
        "regime_run_details": regime_run_details,
    }

    summary_path = out_dir / f"BTCUSDT_multi_regime_{ts}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Summary saved: {summary_path}")

    # ── Save detailed report ────────────────────────────────────────────
    report_lines = [
        "=" * 70,
        "  MULTI-REGIME PAPER TRADING REPORT",
        "=" * 70,
        f"  Period: {PAPER_START} → {PAPER_END}",
        f"  Symbol: BTCUSDT",
        f"  Total Capital: ${TOTAL_CAPITAL:,.2f}",
        f"  Capital per Run: ${CAPITAL_PER_RUN:,.2f} (N={NUM_RUNS})",
        f"  Run: {ts}",
        "",
        "-" * 70,
        "  PER-REGIME RUN DETAILS",
        "-" * 70,
    ]

    for rd in regime_run_details:
        s = rd["summary"]
        report_lines.append(f"  {rd['run_label']}:")
        report_lines.append(f"    Trades: {s['trades']}, PnL: ${s['net_pnl']:+,.2f}, "
                            f"Return: {s['return_pct']:+.2f}%, PF: {s['profit_factor']}, "
                            f"WR: {s['win_rate_pct']:.1f}%")

    report_lines.extend([
        "",
        "-" * 70,
        "  STRATEGY COMBINED (2 regimes each)",
        "-" * 70,
    ])

    for name, data in strategy_combined.items():
        s = data["summary"]
        report_lines.append(f"  {name}:")
        report_lines.append(f"    Trades: {s['trades']}, PnL: ${s['net_pnl']:+,.2f}, "
                            f"Return: {s['return_pct']:+.2f}%, PF: {s['profit_factor']}, "
                            f"WR: {s['win_rate_pct']:.1f}%, MDD: {s['max_drawdown_pct']:.2f}%")

    report_lines.extend([
        "",
        "=" * 70,
        "  ALL COMBINED",
        "=" * 70,
        f"  Trades: {total_combined['trades']}",
        f"  PnL: ${total_combined['net_pnl']:+,.2f}",
        f"  Return: {total_combined['return_pct']:+.2f}%",
        f"  Profit Factor: {total_combined['profit_factor']}",
        f"  Win Rate: {total_combined['win_rate_pct']:.1f}%",
        f"  Max Drawdown: {total_combined['max_drawdown_pct']:.2f}%",
        f"  Avg R: {total_combined['avg_r']}",
        f"  Fees: ${total_combined['total_fees']:,.2f}",
        f"  Final Equity: ${total_combined['final_equity']:,.2f}",
        "",
        "=" * 70,
        "  BASELINE COMPARISON",
        "=" * 70,
        f"  Baseline: {baseline['trades']} trades, +{baseline['return_pct']}%, PF={baseline['pf']}",
        f"  Multi-Regime: {total_trades} trades, {total_combined['return_pct']:+.2f}%, PF={total_combined['profit_factor']}",
        f"  Delta: {total_trades - baseline['trades']:+d} trades, "
        f"{total_combined['return_pct'] - baseline['return_pct']:+.2f}%, "
        f"PF {total_combined['profit_factor'] - baseline['pf']:+.3f}",
        "=" * 70,
    ])

    report_path = out_dir / f"BTCUSDT_multi_regime_{ts}_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"  Report saved: {report_path}")

    # Print report to stdout
    print("\n")
    print("\n".join(report_lines))

    return result


if __name__ == "__main__":
    main()
