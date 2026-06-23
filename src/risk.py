def size_multiplier_from_score(score: int) -> float:
    if score >= 80:
        return 1.0
    if score >= 70:
        return 0.8
    if score >= 60:
        return 0.5
    return 0.0


def calculate_position_notional(
    equity: float,
    entry: float,
    stop_loss: float,
    score: int,
    config: dict,
) -> dict:
    risk_cfg = config["risk"]

    base_risk_pct = float(risk_cfg["risk_per_trade"])
    max_leverage = float(risk_cfg["max_leverage"])

    multiplier = size_multiplier_from_score(score)
    risk_pct = base_risk_pct * multiplier

    max_loss = equity * risk_pct
    stop_distance_pct = abs(entry - stop_loss) / entry

    if stop_distance_pct <= 0 or multiplier <= 0:
        position_notional = 0.0
    else:
        position_notional = max_loss / stop_distance_pct

    leverage_cap = equity * max_leverage
    position_notional = min(position_notional, leverage_cap)

    leverage_used = position_notional / equity if equity > 0 else 0.0

    return {
        "risk_pct": risk_pct,
        "max_loss": max_loss,
        "stop_distance_pct": stop_distance_pct,
        "position_notional": position_notional,
        "leverage_used": leverage_used,
        "size_multiplier": multiplier,
    }
