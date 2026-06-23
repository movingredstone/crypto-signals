from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import re
import json
import math
import random
import pandas as pd
import numpy as np
import yaml

from src.binance_data import load_or_download_klines
from src.indicators import add_indicators
from src.reports import profit_factor, max_drawdown


TRAIN_START = "2023-01-01"
TRAIN_END = "2025-01-01"

VAL_START = "2025-01-01"
VAL_END = "2026-01-01"

TEST_START = "2026-01-01"
TEST_END = "2026-06-01"

# Higher timeframe map for multi-timeframe (MTF) trend filter.
HTF_PANDAS = {
    "15m": "1h",
    "30m": "2h",
    "1h": "4h",
    "2h": "8h",
    "4h": "1D",
    "6h": "1D",
    "8h": "1D",
    "12h": "1D",
    "1d": "1W",
}

DEFAULT_DONCHIAN = [20, 48, 96, 144]


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_mtf_trend(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = HTF_PANDAS.get(interval)
    if rule is None:
        df["htf_trend"] = 0
        return df

    s = df.set_index("open_time")["close"]
    htf = s.resample(rule, label="right", closed="right").last().dropna()

    if len(htf) < 50:
        df["htf_trend"] = 0
        return df

    ema_f = htf.ewm(span=20, adjust=False, min_periods=20).mean()
    ema_s = htf.ewm(span=50, adjust=False, min_periods=50).mean()
    trend = np.sign(ema_f - ema_s)

    htf_df = pd.DataFrame({"htf_time": htf.index, "htf_trend": trend.values}).dropna()

    base = df[["open_time"]].copy().sort_values("open_time")
    merged = pd.merge_asof(
        base,
        htf_df.sort_values("htf_time"),
        left_on="open_time",
        right_on="htf_time",
        direction="backward",
    )
    df["htf_trend"] = merged["htf_trend"].fillna(0).values
    return df


def enrich_features(df: pd.DataFrame, interval: str, lookbacks=None) -> pd.DataFrame:
    df = add_indicators(df.copy())

    # Extra EMAs
    for span in [10, 30, 100]:
        df[f"ema{span}"] = df["close"].ewm(span=span, adjust=False, min_periods=span).mean()

    # Bollinger Bands
    ma20 = df["close"].rolling(20, min_periods=20).mean()
    sd20 = df["close"].rolling(20, min_periods=20).std()
    df["bb_mid"] = ma20
    df["bb_upper"] = ma20 + 2 * sd20
    df["bb_lower"] = ma20 - 2 * sd20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_width_pct"] = df["bb_width"].rolling(300, min_periods=100).rank(pct=True) * 100

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = df["close"].ewm(span=26, adjust=False, min_periods=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ROC
    for n in [6, 12, 24, 48]:
        df[f"roc{n}"] = df["close"].pct_change(n)

    # Donchian / range levels (dynamic: cover default + any requested lookback)
    lbset = set(DEFAULT_DONCHIAN)
    if lookbacks:
        lbset |= {int(x) for x in lookbacks}
    for lb in sorted(lbset):
        if lb < 2:
            continue
        df[f"donchian_high_{lb}"] = df["high"].rolling(lb, min_periods=lb).max().shift(1)
        df[f"donchian_low_{lb}"] = df["low"].rolling(lb, min_periods=lb).min().shift(1)

    # VWAP deviation in ATR units
    df["vwap_atr_dev"] = (df["close"] - df["vwap"]) / df["atr14"]

    # Session factors
    df["hour"] = df["open_time"].dt.hour
    df["dow"] = df["open_time"].dt.dayofweek

    # Multi-timeframe trend
    df = add_mtf_trend(df, interval)

    return df


def add_funding(df: pd.DataFrame, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Merge funding rate (8h) onto klines as funding_rate + funding_z. Network."""
    from src.binance_data import load_or_download_funding

    f = load_or_download_funding(symbol, start_date, end_date)
    if f.empty:
        df["funding_rate"] = 0.0
        df["funding_z"] = 0.0
        return df

    merged = pd.merge_asof(
        df.sort_values("open_time"),
        f.sort_values("funding_time"),
        left_on="open_time",
        right_on="funding_time",
        direction="backward",
    )
    df["funding_rate"] = merged["funding_rate"].fillna(0.0).values
    m = df["funding_rate"].rolling(90, min_periods=20).mean()
    s = df["funding_rate"].rolling(90, min_periods=20).std()
    df["funding_z"] = ((df["funding_rate"] - m) / s.replace(0, np.nan)).fillna(0.0)
    return df


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def near(value: float, target: float, tolerance_pct: float) -> bool:
    if not math.isfinite(value) or not math.isfinite(target) or value == 0:
        return False
    return abs(value - target) / value <= tolerance_pct


# --- Custom-strategy rule builder -------------------------------------------
# DeepSeek can invent NEW entry strategies (beyond the fixed families) by emitting
# predicate lists over these whitelisted indicator columns. Pure data, interpreted
# here — never executed as code.
WHITELIST_COLS = {
    "open", "high", "low", "close", "volume",
    "ema10", "ema20", "ema50", "ema100", "ema200",
    "rsi14", "atr14", "atr_pct",
    "vwap", "vwap_atr_dev", "volume_ratio",
    "bb_mid", "bb_upper", "bb_lower", "bb_width_pct",
    "macd", "macd_signal", "macd_hist",
    "roc6", "roc12", "roc24", "roc48",
    "adx14", "plus_di14", "minus_di14",
    "supertrend_dir", "kc_mid", "kc_upper", "kc_lower",
    "rv", "rv_pct", "htf_trend", "hour", "dow",
    "recent_swing_high", "recent_swing_low",
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "funding_rate", "funding_z",
}

_DONCHIAN_RE = re.compile(r"^donchian_(?:high|low)_\d+$")
_VALID_OPS = {">", "<", ">=", "<="}


def _is_allowed_col(name) -> bool:
    return isinstance(name, str) and (name in WHITELIST_COLS or bool(_DONCHIAN_RE.match(name)))


def _eval_pred(row, pred) -> bool | None:
    """Evaluate one predicate {left, op, right}. Returns bool, or None if invalid."""
    if not isinstance(pred, dict):
        return None
    left = pred.get("left")
    op = pred.get("op")
    right = pred.get("right")

    if not _is_allowed_col(left) or op not in _VALID_OPS:
        return None

    lv = safe_float(row.get(left))

    if isinstance(right, bool):
        return None
    if isinstance(right, (int, float)):
        rv = float(right)
    elif _is_allowed_col(right):
        rv = safe_float(row.get(right))
    else:
        return None

    if not math.isfinite(lv) or not math.isfinite(rv):
        return False

    if op == ">":
        return lv > rv
    if op == "<":
        return lv < rv
    if op == ">=":
        return lv >= rv
    return lv <= rv


def _eval_rule(row, preds) -> bool:
    """All predicates must hold. Any invalid predicate -> no trade (fail safe)."""
    if not preds:
        return False
    for p in preds:
        r = _eval_pred(row, p)
        if r is None or r is False:
            return False
    return True


def sanitize_custom_rules(rules) -> list[dict]:
    """Keep only structurally valid rules with at least one valid LONG predicate."""
    out = []
    if not isinstance(rules, list):
        return out
    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        longs = [p for p in (rule.get("long") or []) if _pred_shape_ok(p)]
        shorts = [p for p in (rule.get("short") or []) if _pred_shape_ok(p)]
        if not longs:
            continue
        out.append({
            "name": str(rule.get("name", f"custom_{idx}"))[:40],
            "long": longs,
            "short": shorts,
        })
    return out


def _pred_shape_ok(pred) -> bool:
    if not isinstance(pred, dict):
        return False
    if not _is_allowed_col(pred.get("left")) or pred.get("op") not in _VALID_OPS:
        return False
    right = pred.get("right")
    if isinstance(right, bool):
        return False
    return isinstance(right, (int, float)) or _is_allowed_col(right)


def custom_rule_donchian_lookbacks(rules) -> set:
    """Extract donchian_<high|low>_N column references so those columns get built."""
    needed = set()
    if not isinstance(rules, list):
        return needed
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        for key in ("long", "short"):
            for p in (rule.get(key) or []):
                if not isinstance(p, dict):
                    continue
                for side_val in (p.get("left"), p.get("right")):
                    if isinstance(side_val, str) and _DONCHIAN_RE.match(side_val):
                        try:
                            needed.add(int(side_val.rsplit("_", 1)[1]))
                        except Exception:
                            pass
    return needed


def direction_allowed(row, side: str, direction_filter: str) -> bool:
    close = safe_float(row["close"])

    if direction_filter == "none":
        return True

    if direction_filter == "ema200":
        if side == "LONG":
            return close > row["ema200"]
        return close < row["ema200"]

    if direction_filter == "ema_stack":
        if side == "LONG":
            return row["ema20"] > row["ema50"] > row["ema200"]
        return row["ema20"] < row["ema50"] < row["ema200"]

    if direction_filter == "ema_fast_stack":
        if side == "LONG":
            return row["ema10"] > row["ema20"] > row["ema50"]
        return row["ema10"] < row["ema20"] < row["ema50"]

    if direction_filter == "price_ema100":
        if side == "LONG":
            return close > row["ema100"]
        return close < row["ema100"]

    if direction_filter == "supertrend":
        d = safe_float(row.get("supertrend_dir"), 0)
        return d > 0 if side == "LONG" else d < 0

    if direction_filter == "mtf_trend":
        t = safe_float(row.get("htf_trend"), 0)
        return t > 0 if side == "LONG" else t < 0

    return False


def entry_trigger(records, i: int, side: str, exp: dict) -> bool:
    row = records[i]
    prev = records[i - 1]
    close = safe_float(row["close"])
    atr = safe_float(row["atr14"])

    if not math.isfinite(close) or not math.isfinite(atr) or atr <= 0:
        return False

    family = exp["family"]
    lookback = exp.get("lookback", 20)
    tol = exp.get("tolerance_pct", 0.006)

    if family == "trend_pullback":
        target = safe_float(row[exp.get("pullback_ref", "ema20")])
        return near(close, target, tol)

    if family == "vwap_pullback":
        return near(close, safe_float(row["vwap"]), tol)

    if family == "donchian_breakout":
        if side == "LONG":
            return close > safe_float(row[f"donchian_high_{lookback}"])
        return close < safe_float(row[f"donchian_low_{lookback}"])

    if family == "range_breakout":
        if side == "LONG":
            return row["close"] > row[f"donchian_high_{lookback}"] and row["close"] > row["open"]
        return row["close"] < row[f"donchian_low_{lookback}"] and row["close"] < row["open"]

    if family == "bollinger_reversion":
        rsi = safe_float(row["rsi14"])
        if side == "LONG":
            return close < safe_float(row["bb_lower"]) and rsi <= exp.get("rsi_low", 35)
        return close > safe_float(row["bb_upper"]) and rsi >= exp.get("rsi_high", 65)

    if family == "vwap_reversion":
        dev = safe_float(row["vwap_atr_dev"])
        threshold = exp.get("vwap_dev_atr", 1.5)
        if side == "LONG":
            return dev <= -threshold
        return dev >= threshold

    if family == "rsi_momentum":
        mid = exp.get("rsi_mid", 50)
        if side == "LONG":
            return safe_float(prev["rsi14"]) < mid <= safe_float(row["rsi14"])
        return safe_float(prev["rsi14"]) > mid >= safe_float(row["rsi14"])

    if family == "macd_momentum":
        if side == "LONG":
            return row["macd_hist"] > 0 and row["macd_hist"] > prev["macd_hist"]
        return row["macd_hist"] < 0 and row["macd_hist"] < prev["macd_hist"]

    if family == "volatility_breakout":
        atr_pct = safe_float(row["atr_pct"], 50)
        if atr_pct < exp.get("atr_breakout_min", 70):
            return False
        if side == "LONG":
            return close > safe_float(row[f"donchian_high_{lookback}"])
        return close < safe_float(row[f"donchian_low_{lookback}"])

    if family == "squeeze_breakout":
        prev_squeeze = safe_float(prev["bb_width_pct"], 50) <= exp.get("squeeze_pct", 25)
        if not prev_squeeze:
            return False
        if side == "LONG":
            return close > safe_float(row[f"donchian_high_{lookback}"])
        return close < safe_float(row[f"donchian_low_{lookback}"])

    if family == "supertrend_flip":
        prev_d = safe_float(prev.get("supertrend_dir"), 0)
        cur_d = safe_float(row.get("supertrend_dir"), 0)
        if side == "LONG":
            return prev_d < 0 and cur_d > 0
        return prev_d > 0 and cur_d < 0

    if family == "keltner_breakout":
        if side == "LONG":
            return close > safe_float(row.get("kc_upper"))
        return close < safe_float(row.get("kc_lower"))

    if family == "keltner_reversion":
        rsi = safe_float(row["rsi14"])
        if side == "LONG":
            return close < safe_float(row.get("kc_lower")) and rsi <= exp.get("rsi_low", 35)
        return close > safe_float(row.get("kc_upper")) and rsi >= exp.get("rsi_high", 65)

    if family == "custom":
        # DeepSeek-invented strategy: interpret whitelisted predicate lists.
        preds = exp.get("custom_long") if side == "LONG" else exp.get("custom_short")
        return _eval_rule(row, preds)

    return False


def funding_ok(row, side: str, exp: dict) -> bool:
    """Optional funding-rate gate (contrarian crowding filter).
    Only active when exp['funding_max_z'] is set; otherwise always passes,
    so runs without funding data are unaffected.
    """
    fmax = exp.get("funding_max_z")
    if fmax is None:
        return True
    fz = safe_float(row.get("funding_z"), 0.0)
    if not math.isfinite(fz):
        return True
    if side == "LONG":
        return fz <= fmax
    return fz >= -fmax


def confirmation_ok(row, exp: dict) -> bool:
    volume_ratio = safe_float(row["volume_ratio"], 0)
    atr_pct = safe_float(row["atr_pct"], 50)

    if volume_ratio < exp.get("volume_min", 0.0):
        return False
    if atr_pct < exp.get("atr_pct_min", 0):
        return False
    if atr_pct > exp.get("atr_pct_max", 100):
        return False

    # ADX trend-strength gate
    adx_min = exp.get("adx_min", 0)
    if adx_min and adx_min > 0:
        adx = safe_float(row.get("adx14"), 0)
        if not math.isfinite(adx) or adx < adx_min:
            return False

    # Volatility regime gate (realized-vol percentile)
    regime = exp.get("regime", "any")
    if regime and regime != "any":
        rv_pct = safe_float(row.get("rv_pct"), 50)
        if not math.isfinite(rv_pct):
            return False
        if regime == "low_vol" and rv_pct > 50:
            return False
        if regime == "high_vol" and rv_pct < 50:
            return False

    # Time-of-day gate
    hours = exp.get("hours_allowed")
    if hours:
        if int(safe_float(row.get("hour"), -1)) not in hours:
            return False

    # Weekday-only gate
    if exp.get("weekday_only"):
        if int(safe_float(row.get("dow"), 0)) >= 5:
            return False

    return True


def make_stop_take(row, side: str, entry: float, exp: dict):
    atr = safe_float(row["atr14"])
    if not math.isfinite(atr) or atr <= 0:
        return None, None

    atr_mult = exp.get("atr_stop_mult", 1.5)
    tp_r = exp.get("take_profit_r", 2.0)
    stop_rule = exp.get("stop_rule", "atr")

    if side == "LONG":
        atr_stop = entry - atr_mult * atr
        swing_stop = safe_float(row["recent_swing_low"])

        if stop_rule == "atr":
            stop = atr_stop
        elif stop_rule == "swing":
            stop = swing_stop
        else:
            stop = min(atr_stop, swing_stop)

        if not math.isfinite(stop) or stop <= 0 or stop >= entry:
            return None, None

        risk = entry - stop
        take = entry + tp_r * risk
        return stop, take

    atr_stop = entry + atr_mult * atr
    swing_stop = safe_float(row["recent_swing_high"])

    if stop_rule == "atr":
        stop = atr_stop
    elif stop_rule == "swing":
        stop = swing_stop
    else:
        stop = max(atr_stop, swing_stop)

    if not math.isfinite(stop) or stop <= entry:
        return None, None

    risk = stop - entry
    take = entry - tp_r * risk

    if take <= 0:
        return None, None

    return stop, take


def simulate_fills(records, entry_i, side, entry, stop, take, exp):
    """Bar-by-bar exit simulation supporting:
    - fixed stop / final take-profit / time exit
    - optional ATR trailing stop (trailing_atr_mult)
    - optional breakeven move (breakeven_r)
    - optional partial take-profit (partial_tp_r + partial_tp_frac)

    Returns list of fills: (fraction, raw_price, reason, bar_index).
    Fractions always sum to 1.0 so fees/slippage totals are unchanged.
    """
    max_holding = exp.get("max_holding_bars", 72)
    end_i = min(len(records) - 1, entry_i + max_holding)

    risk = abs(entry - stop)
    if risk <= 0:
        return [(1.0, entry, "invalid", entry_i)]

    trail_mult = exp.get("trailing_atr_mult")
    be_r = exp.get("breakeven_r")
    ptp_r = exp.get("partial_tp_r")
    ptp_frac = float(exp.get("partial_tp_frac", 0.5))

    long = side == "LONG"
    cur_stop = stop
    remaining = 1.0
    partial_done = False
    fills = []

    for j in range(entry_i, end_i + 1):
        rec = records[j]
        high = safe_float(rec["high"])
        low = safe_float(rec["low"])
        atr_j = safe_float(rec["atr14"])

        # 1) stop first (pessimistic intrabar). cur_stop is set from prior bars.
        if long and low <= cur_stop:
            fills.append((remaining, cur_stop, "stop_loss", j))
            remaining = 0.0
            break
        if (not long) and high >= cur_stop:
            fills.append((remaining, cur_stop, "stop_loss", j))
            remaining = 0.0
            break

        # 2) partial take-profit
        if (not partial_done) and ptp_r is not None:
            ptp_level = entry + ptp_r * risk if long else entry - ptp_r * risk
            hit = high >= ptp_level if long else low <= ptp_level
            if hit:
                frac = min(ptp_frac, remaining)
                fills.append((frac, ptp_level, "partial_tp", j))
                remaining -= frac
                partial_done = True
                cur_stop = max(cur_stop, entry) if long else min(cur_stop, entry)
                if remaining <= 1e-9:
                    break

        # 3) final take-profit
        tp_hit = high >= take if long else low <= take
        if tp_hit:
            fills.append((remaining, take, "take_profit", j))
            remaining = 0.0
            break

        # 4) breakeven move
        if be_r is not None:
            be_level = entry + be_r * risk if long else entry - be_r * risk
            reached = high >= be_level if long else low <= be_level
            if reached:
                cur_stop = max(cur_stop, entry) if long else min(cur_stop, entry)

        # 5) trailing stop update (end of bar -> affects next bar's stop check)
        if trail_mult is not None and math.isfinite(atr_j) and atr_j > 0:
            if long:
                cur_stop = max(cur_stop, high - trail_mult * atr_j)
            else:
                cur_stop = min(cur_stop, low + trail_mult * atr_j)

    if remaining > 1e-9:
        fills.append((remaining, safe_float(records[end_i]["close"]), "time_exit", end_i))

    return fills


def backtest_experiment(records, exp: dict, config: dict, start: str, end: str) -> dict:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    n = len(records)

    fee_rate = float(config["fees"]["taker"]) * float(exp.get("fee_mult", 1.0))
    symbol = exp["symbol"]
    base_slippage = float(config["slippage"].get(symbol, config["slippage"]["default"]))
    slippage_rate = base_slippage * float(exp.get("slippage_mult", 1.0))

    initial_capital = float(config["backtest"]["initial_capital"])
    equity = initial_capital

    risk_per_trade = float(config["risk"]["risk_per_trade"])
    # ── Stress: allow exp to override leverage / trades-per-day ──
    max_leverage = float(exp.get("max_leverage", config["risk"]["max_leverage"]))
    max_trades_per_day = int(exp.get("max_trades_per_day", config["risk"]["max_trades_per_day"]))
    consecutive_loss_limit = int(exp.get("consecutive_loss_limit", 999))
    entry_delay = int(exp.get("entry_delay", 0))
    conservative_stop = bool(exp.get("conservative_stop", False))

    trades = []
    equity_curve = []
    trades_by_day = {}
    in_position_until = -1
    consecutive_losses = 0

    min_i = 350

    for i in range(min_i, n - 2 - entry_delay):
        row = records[i]
        now = row["open_time"]

        if now < start_ts or now >= end_ts:
            continue

        day_key = str(pd.Timestamp(now).date())

        equity_curve.append({"time": str(now), "equity": equity})

        if i <= in_position_until:
            continue

        if consecutive_losses >= consecutive_loss_limit:
            continue

        if trades_by_day.get(day_key, 0) >= max_trades_per_day:
            continue

        if not confirmation_ok(row, exp):
            continue

        chosen_side = None

        for side in ["LONG", "SHORT"]:
            if not direction_allowed(row, side, exp.get("direction_filter", "none")):
                continue
            if not funding_ok(row, side, exp):
                continue
            if entry_trigger(records, i, side, exp):
                chosen_side = side
                break

        if chosen_side is None:
            continue

        entry_i = i + 1 + entry_delay
        if entry_i >= len(records) - 1:
            continue
        raw_entry = safe_float(records[entry_i]["open"])

        if chosen_side == "LONG":
            entry = raw_entry * (1 + slippage_rate)
        else:
            entry = raw_entry * (1 - slippage_rate)

        stop, take = make_stop_take(row, chosen_side, entry, exp)

        # ── Stress: conservative stop (extra slippage against position) ──
        if conservative_stop and stop is not None:
            if chosen_side == "LONG":
                stop = stop * (1 - slippage_rate)  # worse fill on stop
            else:
                stop = stop * (1 + slippage_rate)

        if stop is None or take is None:
            continue

        stop_distance_pct = abs(entry - stop) / entry
        if stop_distance_pct <= 0:
            continue

        max_loss = equity * risk_per_trade
        notional = max_loss / stop_distance_pct
        notional = min(notional, equity * max_leverage)
        if notional <= 0:
            continue

        fills = simulate_fills(records, entry_i, chosen_side, entry, stop, take, exp)
        exit_i = fills[-1][3]
        reason = fills[-1][2]

        # ── Stress: funding cost deduction ──
        funding_cost = 0.0
        if exp.get("funding_include"):
            for b in range(entry_i, exit_i + 1):
                if b < len(records):
                    fr = safe_float(records[b].get("funding_rate"), 0)
                    funding_cost += notional * abs(fr)

        qty = notional / entry
        gross_pnl = 0.0
        for frac, raw_price, _reason, _bar in fills:
            if chosen_side == "LONG":
                exit_price = raw_price * (1 - slippage_rate)
                gross_pnl += (exit_price - entry) * qty * frac
            else:
                exit_price = raw_price * (1 + slippage_rate)
                gross_pnl += (entry - exit_price) * qty * frac

        fee = notional * fee_rate * 2
        slippage_cost = notional * slippage_rate * 2
        net_pnl = gross_pnl - fee - funding_cost

        equity += net_pnl

        initial_risk = notional * stop_distance_pct
        r_multiple = net_pnl / initial_risk if initial_risk > 0 else 0

        # ── Stress: consecutive loss tracking ──
        if net_pnl < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0

        trades.append({
            "side": chosen_side,
            "entry_time": str(records[entry_i]["open_time"]),
            "exit_time": str(records[exit_i]["open_time"]),
            "reason": reason,
            "notional": notional,
            "gross_pnl": gross_pnl,
            "fee": fee,
            "slippage_cost": slippage_cost,
            "net_pnl": net_pnl,
            "r_multiple": r_multiple,
            "equity_after": equity,
        })

        trades_by_day[day_key] = trades_by_day.get(day_key, 0) + 1
        in_position_until = exit_i

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    if trades_df.empty:
        return {
            "return_pct": 0.0,
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_r": 0.0,
            "net_pnl": 0.0,
            "fees": 0.0,
            "slippage_cost": 0.0,
        }

    final_equity = float(trades_df["equity_after"].iloc[-1])

    return {
        "return_pct": round((final_equity / initial_capital - 1) * 100, 2),
        "trades": int(len(trades_df)),
        "win_rate_pct": round(float((trades_df["net_pnl"] > 0).mean() * 100), 2),
        "profit_factor": round(profit_factor(trades_df), 3),
        "max_drawdown_pct": round(max_drawdown(equity_df["equity"]) * 100, 2) if not equity_df.empty else 0.0,
        "avg_r": round(float(trades_df["r_multiple"].mean()), 3),
        "net_pnl": round(float(trades_df["net_pnl"].sum()), 2),
        "fees": round(float(trades_df["fee"].sum()), 2),
        "slippage_cost": round(float(trades_df["slippage_cost"].sum()), 2),
    }


def backtest_experiment_detailed(records, exp: dict, config: dict, start: str, end: str) -> tuple[list[dict], list[dict]]:
    """Same as backtest_experiment but returns (trades, equity_curve) instead of summary.

    Returns:
        (trades, equity_curve) where equity_curve has [{"time": ..., "equity": ...}, ...]
    """
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    n = len(records)

    fee_rate = float(config["fees"]["taker"]) * float(exp.get("fee_mult", 1.0))
    symbol = exp["symbol"]
    base_slippage = float(config["slippage"].get(symbol, config["slippage"]["default"]))
    slippage_rate = base_slippage * float(exp.get("slippage_mult", 1.0))

    initial_capital = float(config["backtest"]["initial_capital"])
    equity = initial_capital

    risk_per_trade = float(config["risk"]["risk_per_trade"])
    max_leverage = float(exp.get("max_leverage", config["risk"]["max_leverage"]))
    max_trades_per_day = int(exp.get("max_trades_per_day", config["risk"]["max_trades_per_day"]))
    consecutive_loss_limit = int(exp.get("consecutive_loss_limit", 999))
    entry_delay = int(exp.get("entry_delay", 0))
    conservative_stop = bool(exp.get("conservative_stop", False))

    trades = []
    equity_curve = []
    trades_by_day = {}
    in_position_until = -1
    consecutive_losses = 0

    min_i = 350

    for i in range(min_i, n - 2 - entry_delay):
        row = records[i]
        now = row["open_time"]

        if now < start_ts or now >= end_ts:
            continue

        day_key = str(pd.Timestamp(now).date())

        # Record equity at each bar
        equity_curve.append({"time": str(now), "equity": round(equity, 2)})

        if i <= in_position_until:
            continue

        if consecutive_losses >= consecutive_loss_limit:
            continue

        if trades_by_day.get(day_key, 0) >= max_trades_per_day:
            continue

        if not confirmation_ok(row, exp):
            continue

        chosen_side = None
        for side in ["LONG", "SHORT"]:
            if not direction_allowed(row, side, exp.get("direction_filter", "none")):
                continue
            if not funding_ok(row, side, exp):
                continue
            if entry_trigger(records, i, side, exp):
                chosen_side = side
                break

        if chosen_side is None:
            continue

        entry_i = i + 1 + entry_delay
        if entry_i >= len(records) - 1:
            continue
        raw_entry = safe_float(records[entry_i]["open"])

        if chosen_side == "LONG":
            entry = raw_entry * (1 + slippage_rate)
        else:
            entry = raw_entry * (1 - slippage_rate)

        stop, take = make_stop_take(row, chosen_side, entry, exp)

        if conservative_stop and stop is not None:
            if chosen_side == "LONG":
                stop = stop * (1 - slippage_rate)
            else:
                stop = stop * (1 + slippage_rate)

        if stop is None or take is None:
            continue

        stop_distance_pct = abs(entry - stop) / entry
        if stop_distance_pct <= 0:
            continue

        max_loss = equity * risk_per_trade
        notional = max_loss / stop_distance_pct
        notional = min(notional, equity * max_leverage)
        if notional <= 0:
            continue

        fills = simulate_fills(records, entry_i, chosen_side, entry, stop, take, exp)
        exit_i = fills[-1][3]
        reason = fills[-1][2]

        funding_cost = 0.0
        if exp.get("funding_include"):
            for b in range(entry_i, exit_i + 1):
                if b < len(records):
                    fr = safe_float(records[b].get("funding_rate"), 0)
                    funding_cost += notional * abs(fr)

        qty = notional / entry
        gross_pnl = 0.0
        exit_price = 0.0
        for frac, raw_price, _reason, _bar in fills:
            if chosen_side == "LONG":
                exit_price = raw_price * (1 - slippage_rate)
                gross_pnl += (exit_price - entry) * qty * frac
            else:
                exit_price = raw_price * (1 + slippage_rate)
                gross_pnl += (entry - exit_price) * qty * frac

        fee = notional * fee_rate * 2
        slippage_cost = notional * slippage_rate * 2
        net_pnl = gross_pnl - fee - funding_cost

        prev_equity = equity
        equity += net_pnl

        if net_pnl < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0

        trades.append({
            "entry_time": str(records[entry_i]["open_time"]),
            "exit_time": str(records[exit_i]["open_time"]),
            "side": chosen_side,
            "entry_price": round(entry, 2),
            "exit_price": round(exit_price, 2),
            "stop_loss": round(stop, 2),
            "take_profit": round(take, 2),
            "reason": reason,
            "notional": round(notional, 2),
            "leverage": round(notional / prev_equity, 2) if prev_equity > 0 else 0,
            "gross_pnl": round(gross_pnl, 2),
            "fee": round(fee, 2),
            "slippage_cost": round(slippage_cost, 2),
            "funding_cost": round(funding_cost, 2),
            "net_pnl": round(net_pnl, 2),
            "equity_after": round(equity, 2),
            "r_multiple": round(net_pnl / (notional * stop_distance_pct) if stop_distance_pct > 0 else 0, 3),
            "holding_bars": exit_i - entry_i,
        })

        trades_by_day[day_key] = trades_by_day.get(day_key, 0) + 1
        in_position_until = exit_i

    return trades, equity_curve


ALL_FAMILIES = [
    "trend_pullback",
    "vwap_pullback",
    "donchian_breakout",
    "range_breakout",
    "bollinger_reversion",
    "vwap_reversion",
    "rsi_momentum",
    "macd_momentum",
    "volatility_breakout",
    "squeeze_breakout",
    "supertrend_flip",
    "keltner_breakout",
    "keltner_reversion",
]

DEFAULT_AXES = {
    "families": list(ALL_FAMILIES),
    "direction_filters": ["none", "ema200", "ema_stack", "ema_fast_stack", "price_ema100", "supertrend", "mtf_trend"],
    "lookbacks": [20, 48, 96, 144],
    "volume_mins": [0.0, 0.7, 1.0, 1.2, 1.5],
    "atr_stop_mults": [1.0, 1.2, 1.5, 2.0, 2.5],
    "take_profit_rs": [1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
    "max_holding_bars": [12, 24, 48, 72, 96, 144],
    "stop_rules": ["atr", "swing", "hybrid"],
    "adx_mins": [0, 20, 25],
    "regimes": ["any", "low_vol", "high_vol"],
    "trailing_atr_mults": [None, 2.0, 3.0],
    "breakeven_rs": [None, 1.0],
    "partial_tp_rs": [None, 1.0],
}

# focus key (singular axis) -> DEFAULT_AXES key
FOCUS_ALIASES = {
    "families": "families",
    "family": "families",
    "direction_filters": "direction_filters",
    "lookbacks": "lookbacks",
    "volume_mins": "volume_mins",
    "atr_stop_mults": "atr_stop_mults",
    "take_profit_rs": "take_profit_rs",
    "max_holding_bars": "max_holding_bars",
    "stop_rules": "stop_rules",
    "adx_mins": "adx_mins",
    "regimes": "regimes",
    "trailing_atr_mults": "trailing_atr_mults",
    "breakeven_rs": "breakeven_rs",
    "partial_tp_rs": "partial_tp_rs",
}


def _as_list(v):
    if v is None:
        return [None]
    if isinstance(v, list):
        return v
    return [v]


def build_experiments(
    symbol: str,
    interval: str,
    max_experiments: int | None = None,
    focus: dict | None = None,
    seed: int = 42,
) -> list[dict]:
    """Random-sample distinct experiments from the (possibly focused) search space.

    `focus` may override any axis (e.g. {"families": ["volatility_breakout"],
    "lookbacks": [10, 20, 30], "trailing_atr_mults": [2.0]}) plus scalar
    family params (rsi_low, rsi_high, vwap_dev_atr, squeeze_pct,
    atr_breakout_min, tolerance_pct, partial_tp_frac, hours_allowed, weekday_only).
    """
    focus = focus or {}

    axes = {k: list(v) for k, v in DEFAULT_AXES.items()}
    for fkey, value in focus.items():
        target = FOCUS_ALIASES.get(fkey)
        if target is not None:
            vals = _as_list(value)
            if vals:
                axes[target] = vals

    # scalar / extra family params (override defaults)
    rsi_low = focus.get("rsi_low", 35)
    rsi_high = focus.get("rsi_high", 65)
    vwap_dev_atr = focus.get("vwap_dev_atr", 1.5)
    squeeze_pct = focus.get("squeeze_pct", 25)
    atr_breakout_min = focus.get("atr_breakout_min", 70)
    tolerance_pct = focus.get("tolerance_pct", 0.006)
    partial_tp_frac = float(focus.get("partial_tp_frac", 0.5))
    hours_allowed = focus.get("hours_allowed")  # list[int] or None
    weekday_only = bool(focus.get("weekday_only", False))
    # Funding gate is opt-in (requires funding data download). Off by default.
    funding_max_zs = _as_list(focus.get("funding_max_zs", [None]))

    # Custom DeepSeek-invented strategies (interpreted predicate rules).
    valid_rules = sanitize_custom_rules(focus.get("custom_rules"))
    if valid_rules and "custom" not in axes["families"]:
        axes["families"] = list(axes["families"]) + ["custom"]

    if max_experiments is None:
        max_experiments = 1000

    rng = random.Random(seed)

    def draw_one():
        family = rng.choice(axes["families"])
        exp = {
            "symbol": symbol,
            "interval": interval,
            "family": family,
            "direction_filter": rng.choice(axes["direction_filters"]),
            "lookback": rng.choice(axes["lookbacks"]),
            "volume_min": rng.choice(axes["volume_mins"]),
            "atr_stop_mult": rng.choice(axes["atr_stop_mults"]),
            "take_profit_r": rng.choice(axes["take_profit_rs"]),
            "max_holding_bars": rng.choice(axes["max_holding_bars"]),
            "stop_rule": rng.choice(axes["stop_rules"]),
            "adx_min": rng.choice(axes["adx_mins"]),
            "regime": rng.choice(axes["regimes"]),
            "trailing_atr_mult": rng.choice(axes["trailing_atr_mults"]),
            "breakeven_r": rng.choice(axes["breakeven_rs"]),
            "partial_tp_r": rng.choice(axes["partial_tp_rs"]),
            "partial_tp_frac": partial_tp_frac,
            "funding_max_z": rng.choice(funding_max_zs),
            "atr_pct_min": 5,
            "atr_pct_max": 95,
        }

        if family in ["trend_pullback", "vwap_pullback"]:
            exp["tolerance_pct"] = tolerance_pct
            exp["pullback_ref"] = "ema20"
        if family in ["bollinger_reversion", "keltner_reversion"]:
            exp["rsi_low"] = rsi_low
            exp["rsi_high"] = rsi_high
        if family == "vwap_reversion":
            exp["vwap_dev_atr"] = vwap_dev_atr
        if family == "rsi_momentum":
            exp["rsi_mid"] = 50
        if family == "volatility_breakout":
            exp["atr_breakout_min"] = atr_breakout_min
        if family == "squeeze_breakout":
            exp["squeeze_pct"] = squeeze_pct

        if family == "custom" and valid_rules:
            rule = rng.choice(valid_rules)
            exp["custom_long"] = rule["long"]
            exp["custom_short"] = rule["short"]
            exp["custom_name"] = rule["name"]

        if hours_allowed:
            exp["hours_allowed"] = list(hours_allowed)
        if weekday_only:
            exp["weekday_only"] = True

        return exp

    seen = set()
    experiments = []
    attempts = 0
    max_attempts = max_experiments * 60 + 200

    while len(experiments) < max_experiments and attempts < max_attempts:
        attempts += 1
        exp = draw_one()
        key = tuple(sorted(
            (k, str(v)) for k, v in exp.items() if k not in ("symbol", "interval")
        ))
        if key in seen:
            continue
        seen.add(key)
        experiments.append(exp)

    return experiments


def make_wf_folds(k: int) -> list[tuple[str, str]]:
    start = pd.Timestamp(TRAIN_START, tz="UTC")
    end = pd.Timestamp(TEST_END, tz="UTC")
    edges = pd.date_range(start, end, periods=k + 1)
    return [(str(edges[i].date()), str(edges[i + 1].date())) for i in range(k)]


def format_progress_message(prefix: str, done: int, total: int, unit: str) -> str:
    if total <= 0:
        pct_done = 100.0
    else:
        pct_done = max(0.0, min(100.0, (done / total) * 100.0))
    pct_remaining = max(0.0, 100.0 - pct_done)
    return (
        f"{prefix} {done}/{total} {unit} "
        f"({pct_done:.1f}% done, {pct_remaining:.1f}% remaining)"
    )


def robust_score(row: dict) -> float:
    # 안정성 중심. 단순 수익률이 아니라 OOS 일관성을 본다.
    train = float(row["train_return_pct"])
    val = float(row["val_return_pct"])
    test = float(row["test_return_pct"])

    train_trades = int(row["train_trades"])
    val_trades = int(row["val_trades"])
    test_trades = int(row["test_trades"])

    test_mdd = abs(float(row["test_mdd_pct"]))
    val_mdd = abs(float(row["val_mdd_pct"]))

    pf = float(row["test_pf"])

    if test_trades < 15:
        return -9999.0
    if val_trades < 20:
        return -9999.0
    if train_trades < 30:
        return -9999.0

    if not math.isfinite(pf):
        pf = 2.0
    pf = min(pf, 3.0)

    if test_trades > 500:
        trade_penalty = 20
    elif test_trades > 250:
        trade_penalty = 10
    else:
        trade_penalty = 0

    consistency_penalty = abs(train - val) * 0.35 + abs(val - test) * 0.7
    val_penalty = 25 if val <= 0 else 0
    test_penalty = 50 if test <= 0 else 0

    score = (
        test * 1.5
        + val * 1.2
        + train * 0.25
        + max(0, pf - 1) * 15
        - test_mdd * 1.2
        - val_mdd * 0.6
        - consistency_penalty
        - trade_penalty
        - val_penalty
        - test_penalty
    )

    # Walk-forward consistency (rewards strategies positive across many time folds)
    wf_folds = int(row.get("wf_folds", 0)) or 0
    if wf_folds > 0:
        wf_pos = int(row.get("wf_pos_folds", 0))
        wf_mean = float(row.get("wf_mean_return", 0.0))
        wf_min = float(row.get("wf_min_return", 0.0))
        score += (wf_pos / wf_folds) * 20.0
        score += wf_mean * 0.5
        score -= abs(min(0.0, wf_min)) * 0.8

    return round(float(score), 3)


_WORKER_RECORDS = None
_WORKER_CONFIG = None
_WORKER_WF = []


def _init_worker(df, config, wf_folds):
    global _WORKER_RECORDS, _WORKER_CONFIG, _WORKER_WF
    # Convert to a plain list of dicts ONCE per process. Row access on this list
    # is ~10x faster than df.iloc inside the bar-by-bar backtest loop.
    _WORKER_RECORDS = df.to_dict("records")
    _WORKER_CONFIG = config
    _WORKER_WF = wf_folds


def _run_one_experiment_worker(item):
    idx, exp = item

    train = backtest_experiment(_WORKER_RECORDS, exp, _WORKER_CONFIG, TRAIN_START, TRAIN_END)
    val = backtest_experiment(_WORKER_RECORDS, exp, _WORKER_CONFIG, VAL_START, VAL_END)
    test = backtest_experiment(_WORKER_RECORDS, exp, _WORKER_CONFIG, TEST_START, TEST_END)

    wf_returns = []
    wf_trades = []
    for (s, e) in _WORKER_WF:
        r = backtest_experiment(_WORKER_RECORDS, exp, _WORKER_CONFIG, s, e)
        wf_returns.append(r["return_pct"])
        wf_trades.append(r["trades"])

    wf_pos = sum(1 for x in wf_returns if x > 0)
    wf_mean = float(np.mean(wf_returns)) if wf_returns else 0.0
    wf_min = float(np.min(wf_returns)) if wf_returns else 0.0

    row = {
        "experiment_id": idx,
        "symbol": exp["symbol"],
        "interval": exp["interval"],
        "family": exp["family"],
        "params_json": json.dumps(exp, ensure_ascii=False),
        "train_return_pct": train["return_pct"],
        "train_mdd_pct": train["max_drawdown_pct"],
        "train_pf": train["profit_factor"],
        "train_trades": train["trades"],
        "val_return_pct": val["return_pct"],
        "val_mdd_pct": val["max_drawdown_pct"],
        "val_pf": val["profit_factor"],
        "val_trades": val["trades"],
        "test_return_pct": test["return_pct"],
        "test_mdd_pct": test["max_drawdown_pct"],
        "test_pf": test["profit_factor"],
        "test_trades": test["trades"],
        "test_win_rate_pct": test["win_rate_pct"],
        "test_fees": test["fees"],
        "test_slippage_cost": test["slippage_cost"],
        "wf_folds": len(wf_returns),
        "wf_pos_folds": wf_pos,
        "wf_mean_return": round(wf_mean, 3),
        "wf_min_return": round(wf_min, 3),
        "wf_total_trades": int(sum(wf_trades)),
    }

    row["robust_score"] = robust_score(row)
    return row


def run_research(
    symbol: str,
    interval: str,
    max_experiments: int = 300,
    config_path: str = "config.yaml",
    max_workers: int | None = None,
    focus: dict | None = None,
    seed: int = 42,
    wf_folds: int = 4,
) -> dict:
    config = load_config(config_path)

    start_date = config["backtest"]["start_date"]
    end_date = config["backtest"]["end_date"]

    experiments = build_experiments(
        symbol, interval, max_experiments=max_experiments, focus=focus, seed=seed
    )

    # Extract cost multipliers from focus, apply to all experiments
    fee_mult = float((focus or {}).get("fee_mult", 1.0))
    slippage_mult = float((focus or {}).get("slippage_mult", 1.0))
    for exp in experiments:
        exp["fee_mult"] = fee_mult
        exp["slippage_mult"] = slippage_mult

    lookbacks_used = {int(e.get("lookback", 20)) for e in experiments}
    lookbacks_used |= custom_rule_donchian_lookbacks((focus or {}).get("custom_rules"))
    lookbacks_used = sorted(lookbacks_used)

    df = load_or_download_klines(symbol, interval, start_date, end_date)
    df = enrich_features(df, interval, lookbacks=lookbacks_used)

    # Opt-in funding factor (only when requested, since it needs a download).
    needs_funding = bool((focus or {}).get("use_funding")) or any(
        e.get("funding_max_z") is not None for e in experiments
    )
    if needs_funding:
        df = add_funding(df, symbol, start_date, end_date)

    wf = make_wf_folds(wf_folds) if wf_folds and wf_folds > 0 else []

    if max_workers is None:
        cpu = os.cpu_count() or 2
        max_workers = max(1, min(cpu - 1, 8))

    print(
        f"Running research: experiments={len(experiments)}, workers={max_workers}, "
        f"wf_folds={len(wf)}, focus={'yes' if focus else 'no'}"
    )

    rows = []

    if max_workers <= 1:
        _init_worker(df, config, wf)
        for idx, exp in enumerate(experiments, start=1):
            rows.append(_run_one_experiment_worker((idx, exp)))
            if idx % 50 == 0 or idx == len(experiments):
                print(format_progress_message("researched", idx, len(experiments), "experiments"))
    else:
        items = list(enumerate(experiments, start=1))
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(df, config, wf),
        ) as executor:
            futures = [executor.submit(_run_one_experiment_worker, item) for item in items]
            for done_count, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if done_count % 50 == 0 or done_count == len(experiments):
                    print(format_progress_message("researched", done_count, len(experiments), "experiments"))

    results = pd.DataFrame(rows)
    results = results.sort_values("robust_score", ascending=False).reset_index(drop=True)

    out_dir = Path("results/research")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{symbol}_{interval}_research.csv"
    results.to_csv(out_path, index=False)

    top = results.head(10).copy()

    return {
        "symbol": symbol,
        "interval": interval,
        "max_experiments": max_experiments,
        "total_experiments": int(len(results)),
        "workers": int(max_workers),
        "wf_folds": len(wf),
        "focus": focus or {},
        "path": str(out_path),
        "top": top.to_dict(orient="records"),
    }


def format_research_report(report: dict) -> str:
    lines = [
        "[investmentsystem Research]",
        f"Symbol: {report['symbol']}",
        f"Interval: {report['interval']}",
        f"Experiments: {report['total_experiments']}",
        f"Walk-forward folds: {report.get('wf_folds', 0)}",
    ]

    if report.get("focus"):
        lines.append(f"Focus: {json.dumps(report['focus'], ensure_ascii=False)}")

    lines += [
        f"Saved: {report['path']}",
        "",
        "[Top 10 Robust Candidates]",
    ]

    if not report["top"]:
        lines.append("No result.")
        return "\n".join(lines)

    for i, row in enumerate(report["top"], start=1):
        params = json.loads(row["params_json"])
        lines.extend([
            "",
            f"{i}. {row['family']} | score={row['robust_score']}",
            f"Train: {row['train_return_pct']}%, PF={row['train_pf']}, MDD={row['train_mdd_pct']}%, trades={row['train_trades']}",
            f"Val: {row['val_return_pct']}%, PF={row['val_pf']}, MDD={row['val_mdd_pct']}%, trades={row['val_trades']}",
            f"Test: {row['test_return_pct']}%, PF={row['test_pf']}, MDD={row['test_mdd_pct']}%, trades={row['test_trades']}",
            (
                f"WF: pos={row.get('wf_pos_folds')}/{row.get('wf_folds')}, "
                f"mean={row.get('wf_mean_return')}%, min={row.get('wf_min_return')}%"
            ),
            (
                "Params: "
                f"dir={params.get('direction_filter')}, "
                f"lb={params.get('lookback')}, "
                f"vol={params.get('volume_min')}, "
                f"ATRstop={params.get('atr_stop_mult')}, "
                f"TP={params.get('take_profit_r')}, "
                f"hold={params.get('max_holding_bars')}, "
                f"stop={params.get('stop_rule')}, "
                f"adx={params.get('adx_min')}, "
                f"regime={params.get('regime')}, "
                f"trail={params.get('trailing_atr_mult')}, "
                f"be={params.get('breakeven_r')}, "
                f"ptp={params.get('partial_tp_r')}"
            )
        ])

    return "\n".join(lines)
