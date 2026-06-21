"""
GitHub Actions Paper Trader — $330 3-Coin Portfolio
DOGE macd_momentum/8h + SUI macd_momentum/4h + AVAX trend_pullback/8h
Each run: fetch latest data → check open positions → check new signals → report.
State persisted as paper_state.json committed back to repo.
"""
import json, os, sys, time, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ── Config ──────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
BINANCE_BASE = "https://fapi.binance.com"
STATE_FILE = "paper_state.json"
TOTAL_ALLOC = 330.0
RISK_PER_TRADE = 0.05  # 5% of allocation

# ── Verified Strategy Parameters ────────────────────────────────────
STRATEGIES = [
    {
        "name": "DOGE macd_momentum/8h",
        "symbol": "DOGEUSDT", "interval": "8h", "alloc": 110.0,
        "family": "macd_momentum",
        "direction_filter": "price_ema100",
        "lookback": 48, "volume_min": 1.2, "atr_stop_mult": 2.0,
        "take_profit_r": 3.0, "max_holding_bars": 12,
        "stop_rule": "swing", "adx_min": 20, "regime": "low_vol",
        "breakeven_r": 1.0, "partial_tp_r": 1.0, "partial_tp_frac": 0.5,
    },
    {
        "name": "SUI macd_momentum/4h",
        "symbol": "SUIUSDT", "interval": "4h", "alloc": 110.0,
        "family": "macd_momentum",
        "direction_filter": "ema_fast_stack",
        "lookback": 48, "volume_min": 2.0, "atr_stop_mult": 3.0,
        "take_profit_r": 4.0, "max_holding_bars": 24,
        "stop_rule": "swing", "adx_min": 20, "regime": "any",
        "partial_tp_r": 1.0, "partial_tp_frac": 0.5,
    },
    {
        "name": "AVAX trend_pullback/8h",
        "symbol": "AVAXUSDT", "interval": "8h", "alloc": 110.0,
        "family": "trend_pullback",
        "direction_filter": "price_ema100",
        "lookback": 48, "volume_min": 1.2, "atr_stop_mult": 2.0,
        "take_profit_r": 5.0, "max_holding_bars": 12,
        "stop_rule": "swing", "adx_min": 0, "regime": "low_vol",
        "partial_tp_frac": 0.5, "pullback_ref": "ema20",
    },
]

# ── Binance Data ────────────────────────────────────────────────────
def fetch_klines(symbol, interval, limit=300):
    url = f"{BINANCE_BASE}/fapi/v1/klines"
    resp = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError(f"No data for {symbol} ({interval})")
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_vol",
        "taker_buy_quote_vol","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    return df

# ── Indicators ──────────────────────────────────────────────────────
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()

def compute_atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def compute_adx(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    pdm = h.diff().clip(lower=0); ndm = (-l.diff()).clip(lower=0)
    atr = compute_atr(df, period)
    pdi = 100 * ema(pdm, period) / atr; ndi = 100 * ema(ndm, period) / atr
    dx = (pdi - ndi).abs() / (pdi + ndi) * 100
    return ema(dx, period), pdi, ndi

def compute_macd(close, fast=12, slow=26, signal=9):
    e_f = ema(close, fast); e_s = ema(close, slow)
    macd_line = e_f - e_s
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# ── Regime Detection ────────────────────────────────────────────────
def detect_regime(df, atr):
    """Simplified regime: compare current ATR to 100-period baseline."""
    atr_base = atr.rolling(100).mean()
    ratio = atr / atr_base
    if ratio.iloc[-1] > 1.3:
        return "high_vol"
    elif ratio.iloc[-1] < 0.7:
        return "low_vol"
    return "normal"

# ── MACD Momentum Signal ────────────────────────────────────────────
def macd_momentum_signal(df, params):
    """MACD histogram momentum: histogram reversing from below/above zero line."""
    close, high, low = df["close"], df["high"], df["low"]
    macd_line, signal_line, hist = compute_macd(close)
    atr = compute_atr(df, 14)
    ema100 = ema(close, 100)
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    adx, pdi, ndi = compute_adx(df, 14)

    regime = detect_regime(df, atr)
    if params["regime"] != "any" and regime != params["regime"]:
        return []

    adx_ok = adx.iloc[-1] >= params.get("adx_min", 0)
    if not adx_ok:
        return []

    # Volume filter
    vol_ma = sma(df["volume"], 20)
    vol_ok = df["volume"].iloc[-1] > vol_ma.iloc[-1] * params["volume_min"]
    if not vol_ok:
        return []

    price = close.iloc[-1]
    atr_val = atr.iloc[-1]
    stop_mult = params["atr_stop_mult"]
    tp_mult = params["take_profit_r"]

    signals = []

    # LONG: histogram was negative, now turning positive (momentum reversal)
    hist_neg_before = hist.iloc[-3] < 0 and hist.iloc[-2] < 0
    hist_turning = hist.iloc[-1] > hist.iloc[-2] and hist.iloc[-1] < 0  # still neg but improving
    hist_cross = hist.iloc[-2] < 0 and hist.iloc[-1] > 0  # crossed above zero

    # Direction filter: price above EMA100 for long
    dir_filter = params.get("direction_filter", "")
    trend_ok_long = True
    if dir_filter == "price_ema100":
        trend_ok_long = close.iloc[-1] > ema100.iloc[-1]
    elif dir_filter == "ema_fast_stack":
        trend_ok_long = ema20.iloc[-1] > ema50.iloc[-1]

    if (hist_turning or hist_cross) and trend_ok_long:
        stop = price - atr_val * stop_mult
        take = price + atr_val * stop_mult * tp_mult
        if stop > 0:
            signals.append({"side": "LONG", "entry": round(float(price), 4),
                           "stop": round(float(stop), 4), "take": round(float(take), 4)})

    # SHORT: histogram turning down
    hist_pos_before = hist.iloc[-3] > 0 and hist.iloc[-2] > 0
    hist_turning_short = hist.iloc[-1] < hist.iloc[-2] and hist.iloc[-1] > 0
    hist_cross_short = hist.iloc[-2] > 0 and hist.iloc[-1] < 0

    trend_ok_short = True
    if dir_filter == "price_ema100":
        trend_ok_short = close.iloc[-1] < ema100.iloc[-1]
    elif dir_filter == "ema_fast_stack":
        trend_ok_short = ema20.iloc[-1] < ema50.iloc[-1]

    if (hist_turning_short or hist_cross_short) and trend_ok_short:
        stop = price + atr_val * stop_mult
        take = price - atr_val * stop_mult * tp_mult
        if stop > 0:
            signals.append({"side": "SHORT", "entry": round(float(price), 4),
                           "stop": round(float(stop), 4), "take": round(float(take), 4)})

    return signals

# ── Trend Pullback Signal ───────────────────────────────────────────
def trend_pullback_signal(df, params):
    """Trend following: price pulls back to reference EMA in direction of trend."""
    close, high, low = df["close"], df["high"], df["low"]
    atr = compute_atr(df, 14)
    ema100 = ema(close, 100)
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    adx, pdi, ndi = compute_adx(df, 14)

    regime = detect_regime(df, atr)
    if params["regime"] != "any" and regime != params["regime"]:
        return []

    adx_ok = adx.iloc[-1] >= params.get("adx_min", 0)
    if not adx_ok:
        return []

    vol_ma = sma(df["volume"], 20)
    vol_ok = df["volume"].iloc[-1] > vol_ma.iloc[-1] * params["volume_min"]
    if not vol_ok:
        return []

    price = close.iloc[-1]
    atr_val = atr.iloc[-1]
    ref = params.get("pullback_ref", "ema20")
    ref_line = ema20 if ref == "ema20" else ema50
    stop_mult = params["atr_stop_mult"]
    tp_mult = params["take_profit_r"]

    signals = []

    # LONG: uptrend (ema20 > ema50, price > ema100) + price pulled back near ema20
    uptrend = ema20.iloc[-1] > ema50.iloc[-1]
    above_ema100 = close.iloc[-1] > ema100.iloc[-1]
    near_ref = abs(close.iloc[-1] - ref_line.iloc[-1]) < atr_val * 1.5
    # Price was above ref, now close to it (pullback)
    pulled_back = close.iloc[-3] > ref_line.iloc[-3] and near_ref

    if uptrend and above_ema100 and pulled_back:
        stop = price - atr_val * stop_mult
        take = price + atr_val * stop_mult * tp_mult
        if stop > 0:
            signals.append({"side": "LONG", "entry": round(float(price), 4),
                           "stop": round(float(stop), 4), "take": round(float(take), 4)})

    # SHORT: downtrend + pullback up to reference
    downtrend = ema20.iloc[-1] < ema50.iloc[-1]
    below_ema100 = close.iloc[-1] < ema100.iloc[-1]
    near_ref_short = abs(close.iloc[-1] - ref_line.iloc[-1]) < atr_val * 1.5
    pulled_back_short = close.iloc[-3] < ref_line.iloc[-3] and near_ref_short

    if downtrend and below_ema100 and pulled_back_short:
        stop = price + atr_val * stop_mult
        take = price - atr_val * stop_mult * tp_mult
        if stop > 0:
            signals.append({"side": "SHORT", "entry": round(float(price), 4),
                           "stop": round(float(stop), 4), "take": round(float(take), 4)})

    return signals

# ── Signal Dispatch ─────────────────────────────────────────────────
def get_signals(df, s):
    family = s["family"]
    if family == "macd_momentum":
        return macd_momentum_signal(df, s)
    elif family == "trend_pullback":
        return trend_pullback_signal(df, s)
    return []

# ── State Management ────────────────────────────────────────────────
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "version": 1,
        "balance": TOTAL_ALLOC,
        "initial_balance": TOTAL_ALLOC,
        "positions": [],
        "closed_trades": [],
        "last_run": None,
    }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def commit_state():
    """Commit and push state file back to repo using GitHub Actions token."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("No GITHUB_TOKEN — skipping state commit")
        return
    repo = os.environ.get("GITHUB_REPOSITORY", "movingredstone/crypto-signals")
    actor = os.environ.get("GITHUB_ACTOR", "paper-trader")
    
    # Configure git
    os.system(f"git config user.name '{actor}'")
    os.system(f"git config user.email '{actor}@users.noreply.github.com'")
    os.system(f"git remote set-url origin https://x-access-token:{token}@github.com/{repo}.git")
    
    os.system(f"git add {STATE_FILE}")
    r = os.system("git diff --cached --quiet")
    if r != 0:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        os.system(f"git commit -m 'paper: update state [{ts}]'")
        os.system("git push")

# ── Position Sizing ─────────────────────────────────────────────────
def size_position(alloc, risk_pct, entry, stop):
    stop_pct = abs(entry - stop) / entry
    if stop_pct == 0:
        return 0
    risk_dollar = alloc * risk_pct
    position = risk_dollar / stop_pct
    return round(position, 2)

# ── Check Open Positions ────────────────────────────────────────────
def check_positions(state, kline_data):
    """Check each open position against latest price data. Close if SL/TP/expiry hit."""
    closed = []
    now = datetime.now(timezone.utc)

    for pos in state["positions"]:
        sym = pos["symbol"]
        if sym not in kline_data:
            continue
        df = kline_data[sym]
        if len(df) == 0:
            continue

        current_price = float(df["close"].iloc[-1])
        current_high = float(df["high"].iloc[-1])
        current_low = float(df["low"].iloc[-1])

        exit_price = None
        exit_reason = None

        # Check TP (using high for LONG, low for SHORT)
        if pos["side"] == "LONG" and current_high >= pos["take"]:
            exit_price = pos["take"]
            exit_reason = "TP"
        elif pos["side"] == "SHORT" and current_low <= pos["take"]:
            exit_price = pos["take"]
            exit_reason = "TP"
        # Check SL
        elif pos["side"] == "LONG" and current_low <= pos["stop"]:
            exit_price = pos["stop"]
            exit_reason = "SL"
        elif pos["side"] == "SHORT" and current_high >= pos["stop"]:
            exit_price = pos["stop"]
            exit_reason = "SL"

        # Check breakeven (move stop to entry after breakeven_r)
        if exit_price is None and pos.get("breakeven_activated") == False:
            be_r = pos.get("breakeven_r")
            if be_r:
                entry = pos["entry"]
                if pos["side"] == "LONG" and current_price >= entry * (1 + be_r * pos["stop_pct"]):
                    pos["stop"] = entry
                    pos["breakeven_activated"] = True
                elif pos["side"] == "SHORT" and current_price <= entry * (1 - be_r * pos["stop_pct"]):
                    pos["stop"] = entry
                    pos["breakeven_activated"] = True

        # Check max holding bars
        pos["bars_held"] = pos.get("bars_held", 0) + 1
        if exit_price is None and pos["bars_held"] >= pos.get("max_bars", 999):
            exit_price = current_price
            exit_reason = "EXPIRY"

        if exit_price is not None:
            # Calculate P&L
            if pos["side"] == "LONG":
                pnl_pct = (exit_price - pos["entry"]) / pos["entry"]
            else:
                pnl_pct = (pos["entry"] - exit_price) / pos["entry"]
            pnl_dollar = pnl_pct * pos.get("notional", pos["alloc"])

            state["balance"] += pnl_dollar

            closed.append({
                "symbol": sym, "side": pos["side"],
                "entry": pos["entry"], "exit": exit_price,
                "exit_reason": exit_reason,
                "pnl": round(pnl_dollar, 2),
                "pnl_pct": round(pnl_pct * 100, 2),
                "entry_time": pos["entry_time"],
                "exit_time": now.isoformat(),
                "strategy": pos["strategy"],
            })

    # Remove closed positions
    state["positions"] = [p for p in state["positions"]
                          if not any(c["entry_time"] == p["entry_time"] and c["symbol"] == p["symbol"]
                                    for c in closed)]
    state["closed_trades"].extend(closed)
    return closed

# ── Telegram ────────────────────────────────────────────────────────
def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)

# ── Main ────────────────────────────────────────────────────────────
def main():
    now = datetime.now(timezone.utc)
    state = load_state()
    
    # Prevent duplicate runs within same bar (allow manual retest after 5 min)
    if state.get("last_run"):
        last = datetime.fromisoformat(state["last_run"])
        if (now - last).total_seconds() < 300:
            print("Skipping — last run too recent")
            return

    # Fetch latest data for each strategy
    kline_data = {}
    errors = []
    for s in STRATEGIES:
        try:
            kline_data[s["symbol"]] = fetch_klines(s["symbol"], s["interval"])
        except Exception as e:
            errors.append(f"{s['name']}: {e}")

    # Check open positions
    closed = check_positions(state, kline_data)

    # Check for new signals
    new_signals = []
    for s in STRATEGIES:
        if s["symbol"] not in kline_data:
            continue
        # Skip if already in a position for this strategy
        already_in = any(p["strategy"] == s["name"] for p in state["positions"])
        if already_in:
            continue
        try:
            signals = get_signals(kline_data[s["symbol"]], s)
            for sig in signals:
                sig["strategy"] = s
                new_signals.append(sig)
        except Exception as e:
            errors.append(f"Signal {s['name']}: {e}")

    # Build report
    lines = [f"<b>📊 Paper Portfolio — {now.strftime('%Y-%m-%d %H:%M UTC')}</b>", ""]

    # Balance
    total_value = state["balance"]
    for pos in state["positions"]:
        # Add unrealized P&L
        sym = pos["symbol"]
        if sym in kline_data and len(kline_data[sym]) > 0:
            cp = float(kline_data[sym]["close"].iloc[-1])
            upnl = (cp - pos["entry"]) / pos["entry"] * pos.get("notional", pos["alloc"])
            if pos["side"] == "SHORT":
                upnl = (pos["entry"] - cp) / pos["entry"] * pos.get("notional", pos["alloc"])
            total_value += upnl

    pnl_total = total_value - state["initial_balance"]
    pnl_pct = (pnl_total / state["initial_balance"]) * 100
    emoji = "🟢" if pnl_total >= 0 else "🔴"
    lines.append(f"{emoji} Balance: <b>${total_value:.2f}</b> (initial $330)")
    lines.append(f"   P&L: ${pnl_total:+.2f} ({pnl_pct:+.1f}%)")
    lines.append("")

    # Closed trades this run
    if closed:
        lines.append("<b>🔔 Closed Trades:</b>")
        for t in closed:
            e = "✅" if t["pnl"] >= 0 else "❌"
            lines.append(f"  {e} {t['symbol']} {t['side']} — {t['exit_reason']}")
            lines.append(f"     Entry ${t['entry']:.4f} → Exit ${t['exit']:.4f}")
            lines.append(f"     P&L: ${t['pnl']:+.2f} ({t['pnl_pct']:+.1f}%)")
        lines.append("")

    # Open positions
    if state["positions"]:
        lines.append("<b>📌 Open Positions:</b>")
        for pos in state["positions"]:
            sym = pos["symbol"]
            cp = float(kline_data[sym]["close"].iloc[-1]) if sym in kline_data else pos["entry"]
            upnl_pct = (cp - pos["entry"]) / pos["entry"] * 100
            if pos["side"] == "SHORT":
                upnl_pct = (pos["entry"] - cp) / pos["entry"] * 100
            upnl_e = "🟢" if upnl_pct >= 0 else "🔴"
            lines.append(f"  {pos['strategy']}")
            lines.append(f"  {pos['side']} @ ${pos['entry']:.4f} | Now ${cp:.4f} {upnl_e} {upnl_pct:+.1f}%")
            lines.append(f"  SL: ${pos['stop']:.4f} | TP: ${pos['take']:.4f} | Bars: {pos['bars_held']}/{pos['max_bars']}")
        lines.append("")

    # New signals
    if new_signals:
        lines.append("<b>🎯 NEW SIGNALS:</b>")
        for sig in new_signals:
            s = sig["strategy"]
            alloc = s["alloc"]
            risk = alloc * RISK_PER_TRADE
            entry = sig["entry"]
            stop = sig["stop"]
            pos_size = size_position(alloc, RISK_PER_TRADE, entry, stop)
            lines.append(f"  🚀 {s['name']} {sig['side']} @ ${entry}")
            lines.append(f"     SL: ${stop} | TP: ${sig['take']}")
            lines.append(f"     Size: ${pos_size:.0f} notional | Risk: ${risk:.2f}")
    elif not state["positions"]:
        lines.append("💤 No signals, no positions — waiting.")
        lines.append("")

    # Trade history summary
    if state["closed_trades"]:
        total_trades = len(state["closed_trades"])
        wins = sum(1 for t in state["closed_trades"] if t["pnl"] > 0)
        wr = wins / total_trades * 100 if total_trades > 0 else 0
        total_pnl = sum(t["pnl"] for t in state["closed_trades"])
        lines.append(f"<b>📈 All-Time:</b> {total_trades} trades | WR {wr:.0f}% | P&L ${total_pnl:+.2f}")

    msg = "\n".join(lines)
    print(msg)
    send_telegram(msg)

    # Update state
    state["last_run"] = now.isoformat()
    save_state(state)
    commit_state()

if __name__ == "__main__":
    # Initialize state file if missing
    if not os.path.exists(STATE_FILE):
        save_state({
            "version": 1,
            "balance": TOTAL_ALLOC,
            "initial_balance": TOTAL_ALLOC,
            "positions": [],
            "closed_trades": [],
            "last_run": None,
        })
        print("Initialized paper state")
    main()
