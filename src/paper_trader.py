"""
Paper Trading Module
Runs the top 3 strategies on forward 2026 data (2026-01-01 to 2026-06-01).
Generates detailed trade logs and summary statistics.
Saves output to results/paper/.
"""
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
import yaml

from src.binance_data import load_or_download_klines
from src.research_engine import (
    enrich_features,
    backtest_experiment_detailed,
    load_config,
)
from src.reports import profit_factor, max_drawdown


# ── Top 3 Strategies (hardcoded, arrived via optimization pipeline) ──────────

TOP3_STRATEGIES = [
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
        "regime": "high_vol",
        "trailing_atr_mult": 3.0,
        "breakeven_r": None,
        "partial_tp_r": None,
        "partial_tp_frac": 0.5,
        "funding_max_z": None,
        "atr_pct_min": 5,
        "atr_pct_max": 95,
    },
]

# ── Optimized mode: regime filter + risk-parity ─────────────────────────────
# Each strategy only trades in its best regimes (from regime mapping analysis)

OPTIMIZED_REGIME_MAP = {
    "vwap_pullback/4h": "high_vol",     # PF=6.08, WR=87.5% in high_vol
    "rsi_momentum/8h": "low_vol",       # PF=3.50, +9.6% in low_vol
    "keltner_breakout/1h": "post_breakout",  # PF=1.39, +3.0% in post_breakout
}

# Risk-parity weights (from correlation analysis Sharpes)
# vwap=1.58, rsi=1.11, keltner=0.73
OPTIMIZED_RISK_WEIGHTS = {
    "vwap_pullback/4h": 0.46,       # 1.58 / 3.42
    "rsi_momentum/8h": 0.32,        # 1.11 / 3.42
    "keltner_breakout/1h": 0.22,    # 0.73 / 3.42
}

# ── Forward window (never seen during optimization) ─────────────────────────

PAPER_START = "2026-01-01"
PAPER_END = "2026-06-01"


def _build_exp(strategy: dict) -> dict:
    """Convert a strategy dict into an experiment dict for the backtest engine."""
    exp = {k: v for k, v in strategy.items() if k not in ("name",)}
    return exp


def _compute_strategy_summary(
    trades: list[dict],
    equity_curve: list[dict],
    initial_capital: float,
) -> dict:
    """Compute per-strategy summary stats from trade list and equity curve."""
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
    eq_df = pd.DataFrame(equity_curve)

    final_equity = float(df["equity_after"].iloc[-1]) if "equity_after" in df.columns else initial_capital

    return {
        "return_pct": round((final_equity / initial_capital - 1) * 100, 2),
        "trades": int(len(df)),
        "win_rate_pct": round(float((df["net_pnl"] > 0).mean() * 100), 2),
        "profit_factor": round(profit_factor(df), 3),
        "max_drawdown_pct": round(max_drawdown(eq_df["equity"]) * 100, 2) if not eq_df.empty and "equity" in eq_df.columns else 0.0,
        "avg_r": round(float(df["r_multiple"].mean()), 3) if "r_multiple" in df.columns else 0.0,
        "net_pnl": round(float(df["net_pnl"].sum()), 2),
        "total_fees": round(float(df["fee"].sum()), 2) if "fee" in df.columns else 0.0,
        "final_equity": round(final_equity, 2),
    }


def _compute_combined_summary(
    all_trades: pd.DataFrame,
    initial_capital: float,
) -> dict:
    """Compute combined summary across all strategies."""
    if all_trades.empty:
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

    # Simulate combined equity curve: start with initial_capital, add all PnLs in time order
    df = all_trades.sort_values("entry_time").reset_index(drop=True)
    equity = initial_capital
    equity_points = [{"time": str(df["entry_time"].iloc[0]), "equity": equity}]

    for _, row in df.iterrows():
        equity += float(row["net_pnl"])
        equity_points.append({"time": str(row["exit_time"]), "equity": equity})

    eq_df = pd.DataFrame(equity_points)

    final_equity = equity
    win_rate = float((df["net_pnl"] > 0).mean() * 100)
    pf = profit_factor(df)

    return {
        "return_pct": round((final_equity / initial_capital - 1) * 100, 2),
        "trades": int(len(df)),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 3),
        "max_drawdown_pct": round(max_drawdown(eq_df["equity"]) * 100, 2),
        "avg_r": round(float(df["r_multiple"].mean()), 3) if "r_multiple" in df.columns else 0.0,
        "net_pnl": round(float(df["net_pnl"].sum()), 2),
        "total_fees": round(float(df["fee"].sum()), 2) if "fee" in df.columns else 0.0,
        "final_equity": round(final_equity, 2),
    }


def run_paper_trading(
    config_path: str = "config.yaml",
    output_dir: str = "results/paper",
    mode: str = "baseline",
    end_date: str = None,
) -> dict:
    """Run paper trading on all 3 strategies and produce output files.

    Args:
        mode: "baseline" (trade all regimes, equal weight)
              "optimized" (regime filter + risk-parity sizing)
        end_date: End date for the paper trading window (YYYY-MM-DD).
                  Defaults to today's date if not provided.

    Returns a dict with paths and summary stats.
    """
    config = load_config(config_path)
    initial_capital = float(config["backtest"]["initial_capital"])

    # ── Resolve end date: use explicit parameter, or default to today ──────
    if end_date is None:
        paper_end = datetime.now().strftime("%Y-%m-%d")
    else:
        paper_end = end_date

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_trades = []
    strategy_summaries = {}

    # Group strategies by interval to avoid re-downloading data
    intervals_needed = {}
    for s in TOP3_STRATEGIES:
        intervals_needed.setdefault(s["interval"], []).append(s)

    print(f"[Paper Trading] Running {len(TOP3_STRATEGIES)} strategies on {PAPER_START} ~ {paper_end}")
    print(f"[Paper Trading] Mode: {mode.upper()}")
    if mode == "optimized":
        print(f"[Paper Trading] Regime filters: {OPTIMIZED_REGIME_MAP}")
        print(f"[Paper Trading] Risk weights: {OPTIMIZED_RISK_WEIGHTS}")
    print(f"[Paper Trading] Intervals: {sorted(intervals_needed.keys())}")
    print()

    for interval, strategies in intervals_needed.items():
        symbol = strategies[0]["symbol"]

        # Load data (need lookback from 2023 to have indicators populated)
        print(f"  Loading data: {symbol} {interval} ...")
        df = load_or_download_klines(symbol, interval, "2023-01-01", paper_end)
        df = enrich_features(df, interval)

        for strat in strategies:
            name = strat["name"]
            exp = _build_exp(strat)

            # ── Apply optimized mode overrides ───────────────────────
            if mode == "optimized":
                if name in OPTIMIZED_REGIME_MAP:
                    exp["regime"] = OPTIMIZED_REGIME_MAP[name]
                    print(f"  [optimized] {name} → regime={exp['regime']}")
            # ─────────────────────────────────────────────────────────

            print(f"  Running: {name} ... ", end="", flush=True)

            trades, equity_curve = backtest_experiment_detailed(
                df.to_dict("records"), exp, config, PAPER_START, paper_end
            )

            # Tag each trade with the strategy name
            for t in trades:
                t["strategy"] = name

            all_trades.extend(trades)

            summary = _compute_strategy_summary(trades, equity_curve, initial_capital)
            strategy_summaries[name] = summary

            print(f"{len(trades)} trades, {summary['return_pct']}% return, PF={summary['profit_factor']}")

    # ── Save combined trade log ──────────────────────────────────────────
    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        # Reorder columns for readability
        desired_cols = [
            "strategy", "entry_time", "exit_time", "side",
            "entry_price", "exit_price", "stop_loss", "take_profit",
            "reason", "notional", "leverage",
            "gross_pnl", "fee", "slippage_cost", "funding_cost", "net_pnl",
            "equity_after", "r_multiple", "holding_bars",
        ]
        existing_cols = [c for c in desired_cols if c in trades_df.columns]
        remaining_cols = [c for c in trades_df.columns if c not in existing_cols]
        trades_df = trades_df[existing_cols + remaining_cols]

    trades_path = output_path / f"BTCUSDT_paper_{ts}_trades.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"\n[Paper Trading] Trades saved: {trades_path} ({len(trades_df)} total)")

    # ── Compute combined summary ─────────────────────────────────────────
    if mode == "optimized":
        combined = _compute_risk_parity_combined(trades_df, initial_capital, strategy_summaries)
    else:
        combined = _compute_combined_summary(trades_df, initial_capital)

    # ── Build full results dict ──────────────────────────────────────────
    result = {
        "timestamp": ts,
        "period": f"{PAPER_START} ~ {paper_end}",
        "symbol": "BTCUSDT",
        "initial_capital": initial_capital,
        "strategies": strategy_summaries,
        "combined": combined,
        "paths": {
            "trades": str(trades_path),
        },
    }

    # ── Save summary JSON ───────────────────────────────────────────────
    summary_path = output_path / f"BTCUSDT_paper_{ts}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"[Paper Trading] Summary saved: {summary_path}")

    # ── Save equity curves per strategy ─────────────────────────────────
    # We reconstruct them from trade logs since equity_curve is already in time order
    equity_dir = output_path / "equity"
    equity_dir.mkdir(parents=True, exist_ok=True)

    for strategy_name in strategy_summaries:
        strat_trades = trades_df[trades_df["strategy"] == strategy_name].sort_values("entry_time")
        if strat_trades.empty:
            continue

        equity = initial_capital
        points = []
        for _, row in strat_trades.iterrows():
            points.append({"time": str(row["entry_time"]), "equity": round(equity, 2)})
            equity += float(row["net_pnl"])
            points.append({"time": str(row["exit_time"]), "equity": round(equity, 2)})

        eq_df = pd.DataFrame(points)
        eq_path = equity_dir / f"{strategy_name.replace('/', '_')}_{ts}_equity.csv"
        eq_df.to_csv(eq_path, index=False)

    # ── Combined equity ─────────────────────────────────────────────────
    combined_df = trades_df.sort_values("entry_time")
    equity = initial_capital
    points = []
    for _, row in combined_df.iterrows():
        points.append({"time": str(row["entry_time"]), "equity": round(equity, 2)})
        equity += float(row["net_pnl"])
        points.append({"time": str(row["exit_time"]), "equity": round(equity, 2)})

    if points:
        eq_combined_path = equity_dir / f"BTCUSDT_combined_{ts}_equity.csv"
        pd.DataFrame(points).to_csv(eq_combined_path, index=False)

    result["paths"]["equity_dir"] = str(equity_dir)

    return result


def _compute_risk_parity_combined(
    all_trades: pd.DataFrame,
    initial_capital: float,
    strategy_summaries: dict,
) -> dict:
    """Compute combined summary with risk-parity weighting per strategy."""
    if all_trades.empty:
        return _compute_combined_summary(all_trades, initial_capital)

    # Allocate capital per strategy based on risk-parity weights
    allocations = {}
    for name, weight in OPTIMIZED_RISK_WEIGHTS.items():
        allocations[name] = initial_capital * weight

    # Build weighted equity curve
    all_times = set()
    strat_equity = {}
    for name, alloc in allocations.items():
        strat_trades = all_trades[all_trades["strategy"] == name].sort_values("entry_time")
        equity = alloc
        points = {"times": [], "equities": []}
        for _, row in strat_trades.iterrows():
            points["times"].append(str(row["entry_time"]))
            points["equities"].append(equity)
            equity += float(row["net_pnl"])
            points["times"].append(str(row["exit_time"]))
            points["equities"].append(equity)
        strat_equity[name] = points
        all_times.update(points["times"])

    # Merge into combined equity at each unique time
    sorted_times = sorted(all_times)
    combined_equity = []
    last_vals = {name: alloc for name, alloc in allocations.items()}

    for t in sorted_times:
        total = 0.0
        for name in allocations:
            pts = strat_equity[name]
            # Find last known equity at or before this time
            val = last_vals[name]
            for i in range(len(pts["times"])):
                if pts["times"][i] > t:
                    break
                val = pts["equities"][i]
            last_vals[name] = val
            total += val
        combined_equity.append({"time": t, "equity": total})

    eq_df = pd.DataFrame(combined_equity)
    final_equity = eq_df["equity"].iloc[-1] if not eq_df.empty else initial_capital

    # Weighted PnL
    total_pnl = sum(
        float(all_trades[all_trades["strategy"] == name]["net_pnl"].sum())
        for name in allocations
    )
    total_trades = len(all_trades)
    win_rate = float((all_trades["net_pnl"] > 0).mean() * 100)
    pf = profit_factor(all_trades)

    return {
        "return_pct": round((final_equity / initial_capital - 1) * 100, 2),
        "trades": total_trades,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 3),
        "max_drawdown_pct": round(max_drawdown(eq_df["equity"]) * 100, 2),
        "avg_r": round(float(all_trades["r_multiple"].mean()), 3) if "r_multiple" in all_trades.columns else 0.0,
        "net_pnl": round(total_pnl, 2),
        "total_fees": round(float(all_trades["fee"].sum()), 2) if "fee" in all_trades.columns else 0.0,
        "final_equity": round(final_equity, 2),
        "allocations": {k: round(v, 2) for k, v in allocations.items()},
    }


def format_paper_report(result: dict) -> str:
    """Format a paper trading result into a readable text report."""
    lines = [
        "=" * 65,
        "  PAPER TRADING REPORT",
        "=" * 65,
        f"  Symbol:      {result['symbol']}",
        f"  Period:      {result['period']}",
        f"  Capital:     {result['initial_capital']:,.2f} USDT",
        f"  Run:         {result['timestamp']}",
        "",
        "-" * 65,
        "  PER-STRATEGY RESULTS",
        "-" * 65,
    ]

    for name, s in result["strategies"].items():
        lines.extend([
            f"  {name}:",
            f"    Trades:  {s['trades']}",
            f"    Return:  {s['return_pct']}%",
            f"    PF:      {s['profit_factor']}",
            f"    Win Rate:{s['win_rate_pct']}%",
            f"    MDD:     {s['max_drawdown_pct']}%",
            f"    Avg R:   {s['avg_r']}",
            f"    Net PnL: {s['net_pnl']:,.2f} USDT",
            f"    Fees:    {s['total_fees']:,.2f} USDT",
            f"    Final:   {s['final_equity']:,.2f} USDT",
            "",
        ])

    c = result["combined"]
    lines.extend([
        "-" * 65,
        "  COMBINED RESULTS",
        "-" * 65,
        f"  Trades:      {c['trades']}",
        f"  Return:      {c['return_pct']}%",
        f"  PF:          {c['profit_factor']}",
        f"  Win Rate:    {c['win_rate_pct']}%",
        f"  MDD:         {c['max_drawdown_pct']}%",
        f"  Avg R:       {c['avg_r']}",
        f"  Net PnL:     {c['net_pnl']:,.2f} USDT",
        f"  Fees:        {c['total_fees']:,.2f} USDT",
        f"  Final Equity:{c['final_equity']:,.2f} USDT",
        "",
        "-" * 65,
        "  FILES",
        "-" * 65,
        f"  Trades CSV:   {result['paths']['trades']}",
        f"  Equity Dir:   {result['paths'].get('equity_dir', 'N/A')}",
        "=" * 65,
    ])

    return "\n".join(lines)
