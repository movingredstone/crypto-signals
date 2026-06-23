import pandas as pd


def score_candidate(candidate, row: pd.Series) -> dict:
    breakdown = {}

    direction = candidate.direction
    close = float(row["close"])

    # 1. Trend fit: 20점
    if direction == "LONG":
        trend_fit = 20 if close > row["ema200"] and row["ema20"] > row["ema50"] else 5
        ema_alignment = 15 if row["ema20"] > row["ema50"] else 6
    else:
        trend_fit = 20 if close < row["ema200"] and row["ema20"] < row["ema50"] else 5
        ema_alignment = 15 if row["ema20"] < row["ema50"] else 6

    breakdown["trend_fit"] = trend_fit
    breakdown["ema_alignment"] = ema_alignment

    # 2. Pullback quality: 15점
    ema20_dist = abs(close - float(row["ema20"])) / close
    vwap_dist = abs(close - float(row["vwap"])) / close
    best_dist = min(ema20_dist, vwap_dist)

    if best_dist <= 0.003:
        pullback_quality = 15
    elif best_dist <= 0.006:
        pullback_quality = 12
    elif best_dist <= 0.01:
        pullback_quality = 8
    else:
        pullback_quality = 3

    breakdown["pullback_quality"] = pullback_quality

    # 3. Risk-reward: 20점
    rr = float(candidate.risk_reward_ratio)

    if rr >= 2.0:
        risk_reward = 20
    elif rr >= 1.7:
        risk_reward = 17
    elif rr >= 1.5:
        risk_reward = 13
    else:
        risk_reward = 5

    breakdown["risk_reward"] = risk_reward

    # 4. Volume condition: 10점
    volume_ratio = float(row["volume_ratio"])

    if 0.7 <= volume_ratio <= 2.5:
        volume_score = 10
    elif 0.45 <= volume_ratio < 0.7 or 2.5 < volume_ratio <= 4.0:
        volume_score = 6
    else:
        volume_score = 2

    breakdown["volume"] = volume_score

    # 5. Volatility condition: 10점
    atr_pct = float(row["atr_pct"]) if not pd.isna(row["atr_pct"]) else 50.0

    if 20 <= atr_pct <= 80:
        volatility = 10
    elif 10 <= atr_pct < 20 or 80 < atr_pct <= 90:
        volatility = 6
    else:
        volatility = 2

    breakdown["volatility"] = volatility

    # 6. Level quality: 10점
    level_quality = 10 if best_dist <= 0.006 else 5
    breakdown["level_quality"] = level_quality

    # 7. Execution quality: 5점
    breakdown["execution_quality"] = 5

    # 8. Funding/crowding proxy: 5점
    # MVP에서는 funding/OI 데이터를 아직 안 쓰므로 기본 5점 부여
    breakdown["funding_proxy"] = 5

    total_score = int(sum(breakdown.values()))

    return {
        "score": total_score,
        "breakdown": breakdown,
    }


def score_bucket(score: int) -> str:
    if score >= 80:
        return "80+"
    if score >= 70:
        return "70-79"
    if score >= 60:
        return "60-69"
    if score >= 50:
        return "50-59"
    return "<50"
