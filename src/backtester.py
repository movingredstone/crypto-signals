from pathlib import Path
import json
import pandas as pd
import yaml

from src.binance_data import load_or_download_klines
from src.indicators import add_indicators
from src.strategies import generate_candidates
from src.filters import hard_filter
from src.scoring import score_candidate, score_bucket
from src.risk import calculate_position_notional
from src.reports import compute_summary, save_summary


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def simulate_exit(df, entry_i, direction, entry, stop, take, max_holding_bars=96):
    end_i = min(len(df) - 1, entry_i + max_holding_bars)

    for j in range(entry_i, end_i + 1):
        high = float(df.iloc[j]["high"])
        low = float(df.iloc[j]["low"])

        if direction == "LONG":
            stop_hit = low <= stop
            take_hit = high >= take

            # Conservative rule: stop wins if both touched
            if stop_hit:
                return j, stop, "stop_loss"
            if take_hit:
                return j, take, "take_profit"

        else:
            stop_hit = high >= stop
            take_hit = low <= take

            if stop_hit:
                return j, stop, "stop_loss"
            if take_hit:
                return j, take, "take_profit"

    return end_i, float(df.iloc[end_i]["close"]), "time_exit"


def calculate_gross_pnl(direction, entry, exit_price, notional):
    qty = notional / entry

    if direction == "LONG":
        return (exit_price - entry) * qty

    return (entry - exit_price) * qty


def run_backtest(symbol: str, interval: str, config_path: str = "config.yaml"):
    config = load_config(config_path)

    start_date = config["backtest"]["start_date"]
    end_date = config["backtest"]["end_date"]
    initial_capital = float(config["backtest"]["initial_capital"])
    score_threshold = int(config["backtest"]["score_threshold"])

    fee_rate = float(config["fees"]["taker"])
    slippage_rate = float(config["slippage"].get(symbol, config["slippage"]["default"]))

    df = load_or_download_klines(symbol, interval, start_date, end_date)
    df = add_indicators(df)

    equity = initial_capital
    trades = []
    equity_curve = []

    max_trades_per_day = int(config["risk"]["max_trades_per_day"])
    trades_by_day = {}
    in_position_until = -1

    for i in range(250, len(df) - 2):
        now = df.iloc[i]["open_time"]
        day_key = str(pd.Timestamp(now).date())

        equity_curve.append({
            "time": str(now),
            "equity": equity,
        })

        if i <= in_position_until:
            continue

        if trades_by_day.get(day_key, 0) >= max_trades_per_day:
            continue

        candidates = generate_candidates(df, i, symbol, interval)
        if not candidates:
            continue

        best_candidate = None
        best_scored = None

        for candidate in candidates:
            ok, reasons = hard_filter(candidate, df.iloc[i], config)
            if not ok:
                continue

            scored = score_candidate(candidate, df.iloc[i])

            if scored["score"] < score_threshold:
                continue

            if best_candidate is None or scored["score"] > best_scored["score"]:
                best_candidate = candidate
                best_scored = scored

        if best_candidate is None:
            continue

        entry_i = i + 1
        raw_entry = float(df.iloc[entry_i]["open"])

        # Unfavorable slippage
        if best_candidate.direction == "LONG":
            entry = raw_entry * (1 + slippage_rate)
            stop = float(best_candidate.stop_loss)

            if stop >= entry:
                continue

            risk = entry - stop
            take = entry + best_candidate.risk_reward_ratio * risk

        else:
            entry = raw_entry * (1 - slippage_rate)
            stop = float(best_candidate.stop_loss)

            if stop <= entry:
                continue

            risk = stop - entry
            take = entry - best_candidate.risk_reward_ratio * risk

            if take <= 0:
                continue

        plan = calculate_position_notional(
            equity=equity,
            entry=entry,
            stop_loss=stop,
            score=best_scored["score"],
            config=config,
        )

        notional = plan["position_notional"]

        if notional <= 0:
            continue

        exit_i, raw_exit, exit_reason = simulate_exit(
            df=df,
            entry_i=entry_i,
            direction=best_candidate.direction,
            entry=entry,
            stop=stop,
            take=take,
        )

        # Unfavorable exit slippage
        if best_candidate.direction == "LONG":
            exit_price = raw_exit * (1 - slippage_rate)
        else:
            exit_price = raw_exit * (1 + slippage_rate)

        gross_pnl = calculate_gross_pnl(
            direction=best_candidate.direction,
            entry=entry,
            exit_price=exit_price,
            notional=notional,
        )

        entry_fee = notional * fee_rate
        exit_fee = notional * fee_rate
        total_fee = entry_fee + exit_fee

        # Approximate explicit slippage cost
        slippage_cost = notional * slippage_rate * 2

        net_pnl = gross_pnl - total_fee
        equity += net_pnl

        initial_risk = notional * abs(entry - stop) / entry
        r_multiple = net_pnl / initial_risk if initial_risk > 0 else 0

        trade = {
            "symbol": symbol,
            "interval": interval,
            "strategy": best_candidate.strategy,
            "direction": best_candidate.direction,
            "signal_time": str(df.iloc[i]["open_time"]),
            "entry_time": str(df.iloc[entry_i]["open_time"]),
            "exit_time": str(df.iloc[exit_i]["open_time"]),
            "entry": entry,
            "stop_loss": stop,
            "take_profit": take,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "score": best_scored["score"],
            "score_bucket": score_bucket(best_scored["score"]),
            "score_breakdown": json.dumps(best_scored["breakdown"], ensure_ascii=False),
            "position_notional": notional,
            "leverage_used": plan["leverage_used"],
            "gross_pnl": gross_pnl,
            "fee": total_fee,
            "slippage_cost": slippage_cost,
            "net_pnl": net_pnl,
            "r_multiple": r_multiple,
            "equity_after": equity,
            "reason": best_candidate.reason,
        }

        trades.append(trade)
        trades_by_day[day_key] = trades_by_day.get(day_key, 0) + 1
        in_position_until = exit_i

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    Path("results/trades").mkdir(parents=True, exist_ok=True)
    Path("results/equity").mkdir(parents=True, exist_ok=True)

    trades_path = Path("results/trades") / f"{symbol}_{interval}_trades.csv"
    equity_path = Path("results/equity") / f"{symbol}_{interval}_equity.csv"

    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)

    summary = compute_summary(
        trades=trades_df,
        equity=equity_df,
        initial_capital=initial_capital,
        symbol=symbol,
        interval=interval,
        period=f"{start_date} ~ {end_date}",
    )

    summary_path = save_summary(summary, symbol, interval)

    summary["paths"] = {
        "trades": str(trades_path),
        "equity": str(equity_path),
        "summary": str(summary_path),
    }

    return summary
