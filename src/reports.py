import json
from pathlib import Path
import pandas as pd


def max_drawdown(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0

    running_max = equity_series.cummax()
    dd = equity_series / running_max - 1
    return float(dd.min())


def profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0

    gross_profit = trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum()
    gross_loss = -trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum()

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return float(gross_profit / gross_loss)


def compute_summary(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    initial_capital: float,
    symbol: str,
    interval: str,
    period: str,
) -> dict:
    if trades.empty:
        return {
            "symbol": symbol,
            "interval": interval,
            "period": period,
            "initial_capital": initial_capital,
            "final_equity": initial_capital,
            "return_pct": 0.0,
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_r": 0.0,
            "total_fee": 0.0,
            "total_slippage_cost": 0.0,
            "net_pnl": 0.0,
            "score_buckets": {},
        }

    final_equity = float(trades["equity_after"].iloc[-1])

    score_buckets = {}
    for bucket, group in trades.groupby("score_bucket"):
        score_buckets[str(bucket)] = {
            "trades": int(len(group)),
            "win_rate_pct": round(float((group["net_pnl"] > 0).mean() * 100), 2),
            "profit_factor": round(profit_factor(group), 3),
            "avg_r": round(float(group["r_multiple"].mean()), 3),
            "net_pnl": round(float(group["net_pnl"].sum()), 2),
        }

    return {
        "symbol": symbol,
        "interval": interval,
        "period": period,
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "return_pct": round((final_equity / initial_capital - 1) * 100, 2),
        "trades": int(len(trades)),
        "win_rate_pct": round(float((trades["net_pnl"] > 0).mean() * 100), 2),
        "profit_factor": round(profit_factor(trades), 3),
        "max_drawdown_pct": round(max_drawdown(equity["equity"]) * 100, 2) if not equity.empty else 0.0,
        "avg_r": round(float(trades["r_multiple"].mean()), 3),
        "total_fee": round(float(trades["fee"].sum()), 2),
        "total_slippage_cost": round(float(trades["slippage_cost"].sum()), 2),
        "net_pnl": round(float(trades["net_pnl"].sum()), 2),
        "score_buckets": score_buckets,
    }


def save_summary(summary: dict, symbol: str, interval: str):
    Path("results/reports").mkdir(parents=True, exist_ok=True)

    path = Path("results/reports") / f"{symbol}_{interval}_summary.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return path


def format_summary(summary: dict) -> str:
    lines = [
        "[investmentsystem Backtest]",
        f"Symbol: {summary['symbol']}",
        f"Interval: {summary['interval']}",
        f"Period: {summary['period']}",
        "",
        f"Initial Capital: {summary['initial_capital']:,.2f} USDT",
        f"Final Equity: {summary['final_equity']:,.2f} USDT",
        f"Return: {summary['return_pct']}%",
        f"Net PnL after costs: {summary['net_pnl']:,.2f} USDT",
        "",
        f"Trades: {summary['trades']}",
        f"Win Rate: {summary['win_rate_pct']}%",
        f"Profit Factor: {summary['profit_factor']}",
        f"Max Drawdown: {summary['max_drawdown_pct']}%",
        f"Average R: {summary['avg_r']}",
        "",
        f"Fees: -{summary['total_fee']:,.2f} USDT",
        f"Slippage Cost: -{summary['total_slippage_cost']:,.2f} USDT",
        "",
        "[Score Buckets]",
    ]

    if not summary["score_buckets"]:
        lines.append("No trades.")
    else:
        for bucket, row in summary["score_buckets"].items():
            lines.append(
                f"{bucket}: trades={row['trades']}, "
                f"PF={row['profit_factor']}, "
                f"avgR={row['avg_r']}, "
                f"pnl={row['net_pnl']}"
            )

    return "\n".join(lines)
