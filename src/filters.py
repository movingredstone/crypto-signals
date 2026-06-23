import math
import pandas as pd


def hard_filter(candidate, row, config) -> tuple[bool, list[str]]:
    reasons = []

    if candidate is None:
        return False, ["no candidate"]

    prices = [
        candidate.entry_ref,
        candidate.stop_loss,
        candidate.take_profit_ref,
    ]

    if not all(math.isfinite(x) for x in prices):
        reasons.append("non-finite price")

    if candidate.stop_loss <= 0:
        reasons.append("invalid stop loss")

    if candidate.take_profit_ref <= 0:
        reasons.append("invalid take profit")

    if candidate.direction == "LONG" and candidate.stop_loss >= candidate.entry_ref:
        reasons.append("long stop is not below entry")

    if candidate.direction == "SHORT" and candidate.stop_loss <= candidate.entry_ref:
        reasons.append("short stop is not above entry")

    stop_distance = abs(candidate.entry_ref - candidate.stop_loss) / candidate.entry_ref
    min_stop = float(config["risk"]["min_stop_distance_pct"])

    if stop_distance < min_stop:
        reasons.append("stop distance too small")

    if candidate.risk_reward_ratio < 1.2:
        reasons.append("risk reward too low")

    required_indicators = [
        "ema20",
        "ema50",
        "ema200",
        "rsi14",
        "atr14",
        "vwap",
        "volume_ratio",
    ]

    for col in required_indicators:
        if col not in row or pd.isna(row[col]):
            reasons.append(f"missing indicator: {col}")

    return len(reasons) == 0, reasons
