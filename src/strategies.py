from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class TradeCandidate:
    signal_time: str
    symbol: str
    interval: str
    strategy: str
    direction: str
    entry_ref: float
    stop_loss: float
    take_profit_ref: float
    risk_reward_ratio: float
    reason: str

    def to_dict(self):
        return asdict(self)


def near(value: float, target: float, tolerance_pct: float) -> bool:
    if pd.isna(value) or pd.isna(target) or value == 0:
        return False

    return abs(value - target) / value <= tolerance_pct


def pullback_continuation(
    df: pd.DataFrame,
    i: int,
    symbol: str,
    interval: str,
    take_profit_r: float = 1.8,
):
    row = df.iloc[i]

    required = [
        "close",
        "ema20",
        "ema50",
        "ema200",
        "rsi14",
        "atr14",
        "vwap",
        "recent_swing_high",
        "recent_swing_low",
        "volume_ratio",
    ]

    if any(pd.isna(row[col]) for col in required):
        return None

    close = float(row["close"])
    atr = float(row["atr14"])

    # ATR 기준으로 EMA/VWAP 근처인지 판단
    tolerance = max(0.003, min(0.01, atr / close * 0.8))

    near_ema20 = near(close, float(row["ema20"]), tolerance)
    near_vwap = near(close, float(row["vwap"]), tolerance)

    is_near_pullback_zone = near_ema20 or near_vwap

    # LONG: 상승 추세에서 눌림목
    if (
        close > float(row["ema200"])
        and float(row["ema20"]) > float(row["ema50"])
        and is_near_pullback_zone
        and 45 <= float(row["rsi14"]) <= 65
        and float(row["volume_ratio"]) >= 0.45
    ):
        stop = min(float(row["recent_swing_low"]), close - 1.2 * atr)

        if stop <= 0 or stop >= close:
            return None

        risk = close - stop
        take = close + take_profit_r * risk

        return TradeCandidate(
            signal_time=str(row["open_time"]),
            symbol=symbol,
            interval=interval,
            strategy="pullback_continuation",
            direction="LONG",
            entry_ref=close,
            stop_loss=stop,
            take_profit_ref=take,
            risk_reward_ratio=take_profit_r,
            reason="Bullish trend + pullback near EMA20/VWAP + RSI continuation zone",
        )

    # SHORT: 하락 추세에서 되돌림
    if (
        close < float(row["ema200"])
        and float(row["ema20"]) < float(row["ema50"])
        and is_near_pullback_zone
        and 35 <= float(row["rsi14"]) <= 55
        and float(row["volume_ratio"]) >= 0.45
    ):
        stop = max(float(row["recent_swing_high"]), close + 1.2 * atr)

        if stop <= close:
            return None

        risk = stop - close
        take = close - take_profit_r * risk

        if take <= 0:
            return None

        return TradeCandidate(
            signal_time=str(row["open_time"]),
            symbol=symbol,
            interval=interval,
            strategy="pullback_continuation",
            direction="SHORT",
            entry_ref=close,
            stop_loss=stop,
            take_profit_ref=take,
            risk_reward_ratio=take_profit_r,
            reason="Bearish trend + pullback near EMA20/VWAP + RSI continuation zone",
        )

    return None


def generate_candidates(df: pd.DataFrame, i: int, symbol: str, interval: str):
    candidates = []

    candidate = pullback_continuation(
        df=df,
        i=i,
        symbol=symbol,
        interval=interval,
    )

    if candidate is not None:
        candidates.append(candidate)

    return candidates
