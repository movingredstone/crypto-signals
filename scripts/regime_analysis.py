#!/usr/bin/env python3
"""Analyze regime performance for top 3 strategies.

Runs detailed backtest for each strategy across full period (2023-01-01 to 2026-06-01),
classifies each trade's regime at entry using classify_regime, and computes per-regime
return/PF/win_rate/trade count. Saves results to results/tvt/regime_mapping.md.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.binance_data import load_or_download_klines
from src.regime_classifier import classify_regime, ALL_REGIMES
from src.research_engine import enrich_features, backtest_experiment_detailed

# ── Top 3 strategies ─────────────────────────────────────────────────────
STRATEGIES = [
    {
        "name": "vwap_pullback/4h",
        "params": {
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
            "regime": "any",
            "trailing_atr_mult": None,
            "breakeven_r": 1.0,
            "partial_tp_r": 1.0,
            "partial_tp_frac": 0.5,
            "funding_max_z": None,
            "atr_pct_min": 5,
            "atr_pct_max": 95,
            "tolerance_pct": 0.006,
            "pullback_ref": "ema20",
        },
    },
    {
        "name": "rsi_momentum/8h",
        "params": {
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
        },
    },
    {
        "name": "keltner_breakout/1h",
        "params": {
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
            "regime": "high_vol",
            "trailing_atr_mult": 3.0,
            "breakeven_r": None,
            "partial_tp_r": None,
            "partial_tp_frac": 0.5,
            "funding_max_z": None,
            "atr_pct_min": 5,
            "atr_pct_max": 95,
        },
    },
]

# ── Config ────────────────────────────────────────────────────────────────
CONFIG = {
    "fees": {"maker": 0.0002, "taker": 0.0005},
    "slippage": {"BTCUSDT": 0.0002, "default": 0.0004},
    "backtest": {"initial_capital": 10000},
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

FULL_START = "2023-01-01"
FULL_END = "2026-06-01"


def main():
    all_results = []

    for strat in STRATEGIES:
        name = strat["name"]
        exp = strat["params"]
        interval = exp["interval"]
        symbol = exp["symbol"]

        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"  Loading data for {symbol} {interval}...")

        # Load kline data
        df_raw = load_or_download_klines(symbol, interval, FULL_START, FULL_END)
        print(f"  Loaded {len(df_raw)} bars")

        # Enrich features (includes indicators + extra EMAs, BB, MACD, Donchian, etc.)
        print("  Enriching features...")
        df = enrich_features(df_raw, interval)
        print(f"  Columns after enrich: {len(df.columns)}")

        # Classify regime
        print("  Classifying regimes...")
        df = classify_regime(df)

        # Convert to list of dicts for backtest engine
        records = df.to_dict("records")

        # Run detailed backtest
        print("  Running backtest_experiment_detailed...")
        trades, equity_curve = backtest_experiment_detailed(
            records, exp, CONFIG, FULL_START, FULL_END
        )
        print(f"  Got {len(trades)} trades, {len(equity_curve)} equity points")

        if not trades:
            print(f"  WARNING: No trades for {name}")
            all_results.append({
                "strategy": name,
                "total_trades": 0,
                "overall_return_pct": 0.0,
                "overall_pf": 0.0,
                "overall_win_rate": 0.0,
                "per_regime": {},
            })
            continue

        trades_df = pd.DataFrame(trades)
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
        df["open_time"] = pd.to_datetime(df["open_time"])

        # Map each trade to its entry regime
        regime_labels = []
        for _, trade in trades_df.iterrows():
            et = trade["entry_time"]
            # Find the bar at or just before entry time
            matching = df[df["open_time"] <= et].tail(1)
            if matching.empty:
                regime_labels.append("unknown")
            else:
                regime_labels.append(matching.iloc[0].get("regime", "unknown"))
        trades_df["entry_regime"] = regime_labels

        # Overall stats
        initial_capital = CONFIG["backtest"]["initial_capital"]
        final_equity = float(trades_df["equity_after"].iloc[-1]) if not trades_df.empty else initial_capital
        overall_return = (final_equity / initial_capital - 1) * 100
        gross_profits = trades_df[trades_df["net_pnl"] > 0]["net_pnl"].sum()
        gross_losses = abs(trades_df[trades_df["net_pnl"] < 0]["net_pnl"].sum())
        overall_pf = gross_profits / gross_losses if gross_losses > 0 else float("inf")
        overall_win_rate = (trades_df["net_pnl"] > 0).mean() * 100 if not trades_df.empty else 0.0

        print(f"  Overall: Return={overall_return:.1f}%, PF={overall_pf:.2f}, WinRate={overall_win_rate:.1f}%, Trades={len(trades_df)}")

        # Per-regime stats
        per_regime = {}
        for regime in ALL_REGIMES:
            mask = trades_df["entry_regime"] == regime
            rt = trades_df[mask]
            if rt.empty:
                per_regime[regime] = {
                    "trades": 0,
                    "return_pct": 0.0,
                    "pf": 0.0,
                    "win_rate": 0.0,
                    "avg_r": 0.0,
                    "total_pnl": 0.0,
                }
                continue

            # Per-regime return: sum of net_pnl / initial_capital * 100
            total_pnl = float(rt["net_pnl"].sum())
            return_pct = (total_pnl / initial_capital) * 100

            g_prof = rt[rt["net_pnl"] > 0]["net_pnl"].sum()
            g_loss = abs(rt[rt["net_pnl"] < 0]["net_pnl"].sum())
            pf = g_prof / g_loss if g_loss > 0 else float("inf")
            wr = (rt["net_pnl"] > 0).mean() * 100
            avg_r = float(rt["r_multiple"].mean())

            per_regime[regime] = {
                "trades": len(rt),
                "return_pct": round(return_pct, 1),
                "pf": round(pf, 2),
                "win_rate": round(wr, 1),
                "avg_r": round(avg_r, 3),
                "total_pnl": round(total_pnl, 2),
            }
            print(f"    {regime:16s}: trades={len(rt):3d}  return={return_pct:+6.1f}%  PF={pf:.2f}  WR={wr:.1f}%")

        # Unknown regime
        mask = trades_df["entry_regime"] == "unknown"
        rt = trades_df[mask]
        if not rt.empty:
            per_regime["unknown"] = {
                "trades": len(rt),
                "return_pct": round(float(rt["net_pnl"].sum()) / initial_capital * 100, 1),
                "pf": round(
                    float(rt[rt["net_pnl"] > 0]["net_pnl"].sum()) / abs(float(rt[rt["net_pnl"] < 0]["net_pnl"].sum()))
                    if abs(float(rt[rt["net_pnl"] < 0]["net_pnl"].sum())) > 0 else 0.0,
                    2,
                ),
                "win_rate": round(float((rt["net_pnl"] > 0).mean() * 100), 1),
                "avg_r": round(float(rt["r_multiple"].mean()), 3),
                "total_pnl": round(float(rt["net_pnl"].sum()), 2),
            }

        all_results.append({
            "strategy": name,
            "total_trades": len(trades_df),
            "overall_return_pct": round(overall_return, 1),
            "overall_pf": round(overall_pf, 2) if overall_pf != float("inf") else "∞",
            "overall_win_rate": round(overall_win_rate, 1),
            "per_regime": per_regime,
        })

    # ── Build markdown output ─────────────────────────────────────────────
    lines = []
    lines.append("# Regime Performance Analysis — Top 3 Strategies")
    lines.append("")
    lines.append("**BTCUSDT | 2023-01-01 → 2026-06-01 | Baseline Config**")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("| Strategy | Total Trades | Return % | Profit Factor | Win Rate | Best Regime | Worst Regime |")
    lines.append("|----------|-------------:|---------:|--------------:|---------:|-------------|--------------|")
    for r in all_results:
        strat = r["strategy"]
        trades = r["total_trades"]
        ret = r["overall_return_pct"]
        pf = r["overall_pf"]
        wr = r["overall_win_rate"]
        pr = r["per_regime"]

        # Find best and worst regime by return_pct (only regimes with trades)
        best_regime = "N/A"
        worst_regime = "N/A"
        best_ret = float("-inf")
        worst_ret = float("inf")
        for reg, stats in pr.items():
            if stats["trades"] == 0:
                continue
            ret_val = stats["return_pct"]
            if ret_val > best_ret:
                best_ret = ret_val
                best_regime = reg
            if ret_val < worst_ret:
                worst_ret = ret_val
                worst_regime = reg

        pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) else str(pf)
        lines.append(f"| {strat} | {trades} | {ret:+.1f}% | {pf_str} | {wr:.1f}% | {best_regime} | {worst_regime} |")

    lines.append("")

    # Per-strategy detailed tables
    for r in all_results:
        strat = r["strategy"]
        pr = r["per_regime"]
        lines.append(f"## {strat}")
        lines.append("")
        lines.append(f"**Overall:** {r['total_trades']} trades | {r['overall_return_pct']:+.1f}% return | PF={r['overall_pf']} | Win Rate={r['overall_win_rate']:.1f}%")
        lines.append("")
        lines.append("| Regime | Trades | Return % | Profit Factor | Win Rate % | Avg R | Total PnL $ |")
        lines.append("|--------|-------:|---------:|--------------:|----------:|------:|------------:|")

        # Sort by return_pct descending
        sorted_regimes = sorted(pr.items(), key=lambda x: x[1]["return_pct"], reverse=True)
        for regime, stats in sorted_regimes:
            if stats["trades"] == 0:
                continue
            pf_str = f"{stats['pf']:.2f}" if stats['pf'] != float('inf') else "∞"
            lines.append(
                f"| {regime} | {stats['trades']} | {stats['return_pct']:+.1f}% | {pf_str} | {stats['win_rate']:.1f}% | {stats['avg_r']:.3f} | {stats['total_pnl']:+.2f} |"
            )
        lines.append("")

    # ── Cross-strategy regime comparison ──────────────────────────────────
    lines.append("## Cross-Strategy Regime Matrix")
    lines.append("")
    lines.append("| Regime | vwap_pullback/4h | rsi_momentum/8h | keltner_breakout/1h | Best Strategy |")
    lines.append("|--------|-----------------:|----------------:|--------------------:|---------------|")

    strat_names = [r["strategy"] for r in all_results]
    for regime in ALL_REGIMES:
        row = f"| {regime} "
        best_strat = None
        best_ret = float("-inf")
        for r in all_results:
            stats = r["per_regime"].get(regime, {})
            if stats.get("trades", 0) == 0:
                row += "| — "
            else:
                pf_str = f"{stats['pf']:.1f}" if stats['pf'] != float('inf') else "∞"
                row += f"| {stats['return_pct']:+.1f}% (PF={pf_str}, {stats['trades']}t) "
                if stats["return_pct"] > best_ret:
                    best_ret = stats["return_pct"]
                    best_strat = r["strategy"]
        row += f"| {best_strat or '—'} |"
        lines.append(row)

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**Legend:** PF = Profit Factor, t = trade count, — = no trades in this regime")
    lines.append("")
    lines.append("*Generated by scripts/regime_analysis.py*")

    output = "\n".join(lines)

    # Save
    out_dir = project_root / "results" / "tvt"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "regime_mapping.md"
    out_path.write_text(output)
    print(f"\n✅ Saved to {out_path}")

    # Also print to stdout
    print("\n" + output)


if __name__ == "__main__":
    main()
