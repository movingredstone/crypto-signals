"""
GitHub Actions Paper Trader v3 — $330 3-Coin Portfolio
Signal logic matched to backtest engine (research_engine.py).
Records: trade history, signal log, equity curve, market context per entry.
"""
import json, os, sys, time, requests, csv, math
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
BINANCE_BASE = "https://fapi.binance.com"
STATE_FILE = "paper_state.json"
TRADE_LOG_CSV = "paper_trades.csv"
TOTAL_ALLOC = 330.0
RISK_PER_TRADE = 0.05

STRATEGIES = [
    {"name":"DOGE macd_momentum/8h","symbol":"DOGEUSDT","interval":"8h","alloc":110.0,
     "family":"macd_momentum","direction_filter":"price_ema100","lookback":48,"volume_min":1.2,
     "atr_stop_mult":2.0,"take_profit_r":3.0,"max_holding_bars":12,"stop_rule":"swing",
     "adx_min":20,"regime":"low_vol","breakeven_r":1.0,"partial_tp_r":1.0,"partial_tp_frac":0.5,
     "tolerance_pct":0.006},
    {"name":"SUI macd_momentum/4h","symbol":"SUIUSDT","interval":"4h","alloc":110.0,
     "family":"macd_momentum","direction_filter":"ema_fast_stack","lookback":48,"volume_min":2.0,
     "atr_stop_mult":3.0,"take_profit_r":4.0,"max_holding_bars":24,"stop_rule":"swing",
     "adx_min":20,"regime":"any","partial_tp_r":1.0,"partial_tp_frac":0.5,
     "tolerance_pct":0.006},
    {"name":"AVAX trend_pullback/8h","symbol":"AVAXUSDT","interval":"8h","alloc":110.0,
     "family":"trend_pullback","direction_filter":"price_ema100","lookback":48,"volume_min":1.2,
     "atr_stop_mult":2.0,"take_profit_r":5.0,"max_holding_bars":12,"stop_rule":"swing",
     "adx_min":0,"regime":"low_vol","partial_tp_frac":0.5,"pullback_ref":"ema20",
     "tolerance_pct":0.006},
]

# ══════════════════════════════════════════════════════════════════════
# Data Fetching
# ══════════════════════════════════════════════════════════════════════
def fetch_klines(symbol, interval, limit=300):
    url = f"{BINANCE_BASE}/fapi/v1/klines"
    resp = requests.get(url, params={"symbol":symbol,"interval":interval,"limit":limit}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError(f"No data for {symbol}")
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_vol","taker_buy_quote_vol","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    return df

# ══════════════════════════════════════════════════════════════════════
# Indicators (match backtest engine naming: ema20, atr14, etc.)
# ══════════════════════════════════════════════════════════════════════
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _sma(s, n): return s.rolling(n).mean()

def compute_indicators(df):
    """Compute all indicators the backtest engine uses, stored as columns."""
    c, h, l = df["close"], df["high"], df["low"]
    # EMAs
    df["ema20"] = _ema(c, 20)
    df["ema50"] = _ema(c, 50)
    df["ema100"] = _ema(c, 100)
    df["ema200"] = _ema(c, 200)
    # ATR
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(span=14, adjust=False).mean()
    # MACD
    e_f, e_s = _ema(c, 12), _ema(c, 26)
    macd = e_f - e_s
    sig = _ema(macd, 9)
    df["macd_hist"] = macd - sig
    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi14"] = 100 - (100 / (1 + rs))
    # ADX
    pdm = h.diff().clip(lower=0); ndm = (-l.diff()).clip(lower=0)
    atr14 = df["atr14"]
    pdi = 100 * _ema(pdm, 14) / atr14; ndi = 100 * _ema(ndm, 14) / atr14
    df["adx14"] = _ema((pdi - ndi).abs() / (pdi + ndi) * 100, 14)
    # Volume ratio
    df["volume_ratio"] = df["volume"] / _sma(df["volume"], 20)
    # ATR percentile (for regime)
    df["atr_pct"] = df["atr14"].rolling(100).apply(lambda x: (x.iloc[-1] < x).mean() * 100, raw=False)
    # Swing highs/lows (lookback window)
    lookback = 48
    df["swing_high"] = h.rolling(lookback, min_periods=1).max()
    df["swing_low"]  = l.rolling(lookback, min_periods=1).min()
    return df

# ══════════════════════════════════════════════════════════════════════
# Backtest-Matched Signal Logic
# ══════════════════════════════════════════════════════════════════════

def safe_float(val, default=0.0):
    """Match backtest engine's safe_float."""
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (ValueError, TypeError):
        return default

def near(value, target, tolerance_pct):
    """Match backtest engine's near()."""
    if pd.isna(value) or pd.isna(target) or value == 0:
        return False
    return abs(value - target) / abs(value) <= tolerance_pct

def direction_allowed(row, side, direction_filter):
    """Match backtest engine's direction_allowed()."""
    if direction_filter == "none":
        return True
    if direction_filter == "ema200":
        return (side == "LONG" and row["close"] > safe_float(row["ema200"])) or \
               (side == "SHORT" and row["close"] < safe_float(row["ema200"]))
    if direction_filter == "ema_stack":
        return (side == "LONG" and safe_float(row["ema20"]) > safe_float(row["ema50"]) > safe_float(row["ema200"])) or \
               (side == "SHORT" and safe_float(row["ema20"]) < safe_float(row["ema50"]) < safe_float(row["ema200"]))
    if direction_filter == "ema_fast_stack":
        return (side == "LONG" and safe_float(row["ema20"]) > safe_float(row["ema50"])) or \
               (side == "SHORT" and safe_float(row["ema20"]) < safe_float(row["ema50"]))
    if direction_filter == "price_ema100":
        return (side == "LONG" and row["close"] > safe_float(row["ema100"])) or \
               (side == "SHORT" and row["close"] < safe_float(row["ema100"]))
    if direction_filter == "mtf_trend":
        # Multi-timeframe trend: use EMA20/EMA50/EMA100 alignment
        ema20_v = safe_float(row["ema20"]); ema50_v = safe_float(row["ema50"])
        ema100_v = safe_float(row["ema100"])
        trend_up = ema20_v > ema50_v > ema100_v
        trend_dn = ema20_v < ema50_v < ema100_v
        return (side == "LONG" and trend_up) or (side == "SHORT" and trend_dn)
    return True

def confirmation_ok(row, exp):
    """Match backtest engine's confirmation_ok() — volume, ATR%, ADX, regime."""
    vr = safe_float(row["volume_ratio"], 0)
    if vr < exp.get("volume_min", 0.0):
        return False

    atr_pct = safe_float(row.get("atr_pct", 50), 50)
    if atr_pct < exp.get("atr_pct_min", 0):
        return False
    if atr_pct > exp.get("atr_pct_max", 100):
        return False

    adx_min = exp.get("adx_min", 0)
    if adx_min and adx_min > 0:
        adx = safe_float(row.get("adx14"), 0)
        if not math.isfinite(adx) or adx < adx_min:
            return False

    regime = exp.get("regime", "any")
    if regime and regime != "any":
        rv_pct = safe_float(row.get("atr_pct", 50), 50)
        if not math.isfinite(rv_pct):
            return False
        if regime == "low_vol" and rv_pct > 50:
            return False
        if regime == "high_vol" and rv_pct < 50:
            return False

    return True

def check_entry(row, prev_row, side, exp, df=None):
    """Match backtest engine's check_entry(). Returns True/False."""
    family = exp["family"]
    
    if family == "macd_momentum":
        # Backtest: LONG = hist > 0 and hist > prev_hist
        #          SHORT = hist < 0 and hist < prev_hist
        hist = safe_float(row["macd_hist"])
        prev_hist = safe_float(prev_row["macd_hist"])
        if side == "LONG":
            return hist > 0 and hist > prev_hist
        return hist < 0 and hist < prev_hist

    if family == "trend_pullback":
        target_col = exp.get("pullback_ref", "ema20")
        target = safe_float(row[target_col])
        tol = exp.get("tolerance_pct", 0.006)
        return near(row["close"], target, tol)

    return False

def make_stop_take(row, side, entry, exp):
    """Match backtest engine's make_stop_take()."""
    atr = safe_float(row["atr14"])
    if not math.isfinite(atr) or atr <= 0:
        return None, None

    atr_mult = exp.get("atr_stop_mult", 1.5)
    tp_r = exp.get("take_profit_r", 2.0)
    stop_rule = exp.get("stop_rule", "atr")

    if side == "LONG":
        atr_stop = entry - atr_mult * atr
        swing_stop = safe_float(row.get("swing_low"), entry)

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
        return round(stop, 4), round(take, 4)

    # SHORT
    atr_stop = entry + atr_mult * atr
    swing_stop = safe_float(row.get("swing_high"), entry)

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
    return round(stop, 4), round(take, 4)

def get_signals(df, exp):
    """Generate signals using backtest-matched logic. Returns (signals, snapshot)."""
    # Ensure indicators are computed
    if "ema20" not in df.columns:
        df = compute_indicators(df)

    if len(df) < 3:
        return [], {}

    row = df.iloc[-1]
    prev_row = df.iloc[-2]

    # Market snapshot for logging
    snap = {
        "regime": "low_vol" if safe_float(row.get("atr_pct",50)) <= 50 else "high_vol",
        "adx": round(safe_float(row.get("adx14")), 1),
        "atr": round(safe_float(row["atr14"]), 4),
        "atr_ratio": 0,  # unused now, keeping for compat
        "price": round(float(row["close"]), 4),
        "volume_ratio": round(safe_float(row.get("volume_ratio", 1)), 2),
    }

    # Pre-checks (confirmation)
    if not confirmation_ok(row, exp):
        return [], snap

    dfilter = exp.get("direction_filter", "none")
    signals = []

    for side in ["LONG", "SHORT"]:
        if not direction_allowed(row, side, dfilter):
            continue
        if not check_entry(row, prev_row, side, exp, df):
            continue

        entry = round(float(row["close"]), 4)
        stop, take = make_stop_take(row, side, entry, exp)
        if stop is not None and take is not None:
            signals.append({"side": side, "entry": entry, "stop": stop, "take": take})

    return signals, snap

# ══════════════════════════════════════════════════════════════════════
# State (v2 — richer data)
# ══════════════════════════════════════════════════════════════════════
def empty_state():
    return {
        "version": 2,
        "balance": TOTAL_ALLOC,
        "initial_balance": TOTAL_ALLOC,
        "peak_balance": TOTAL_ALLOC,
        "positions": [],
        "closed_trades": [],
        "signal_log": [],
        "equity_curve": [],
        "last_run": None,
    }

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            s = json.load(f)
        if s.get("version",1) < 2:
            s["peak_balance"] = s.get("balance", s.get("initial_balance", TOTAL_ALLOC))
            s["signal_log"] = []
            s["equity_curve"] = []
            s["version"] = 2
        return s
    return empty_state()

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def commit_state():
    token = os.environ.get("GITHUB_TOKEN","")
    if not token: return
    repo = os.environ.get("GITHUB_REPOSITORY","movingredstone/crypto-signals")
    actor = os.environ.get("GITHUB_ACTOR","paper-trader")
    os.system(f"git config user.name '{actor}'")
    os.system(f"git config user.email '{actor}@users.noreply.github.com'")
    os.system(f"git remote set-url origin https://x-access-token:***@github.com/{repo}.git")
    os.system(f"git add {STATE_FILE} {TRADE_LOG_CSV}")
    if os.system("git diff --cached --quiet") != 0:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        os.system(f"git commit -m 'paper: update [{ts}]'")
        os.system("git push")

# ══════════════════════════════════════════════════════════════════════
# Position Sizing
# ══════════════════════════════════════════════════════════════════════
def size_position(alloc, risk_pct, entry, stop):
    sp = abs(entry-stop)/entry
    return round(alloc*risk_pct/sp, 2) if sp > 0 else 0

# ══════════════════════════════════════════════════════════════════════
# Trade Log CSV
# ══════════════════════════════════════════════════════════════════════
TRADE_CSV_FIELDS = [
    "exit_time","strategy","symbol","side","entry","exit","exit_reason",
    "pnl","pnl_pct","bars_held","entry_time","regime","adx_entry","atr_entry",
    "volume_ratio","alloc","notional","risk_dollar",
]

def append_trade_csv(trade):
    exists = os.path.exists(TRADE_LOG_CSV)
    with open(TRADE_LOG_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: trade.get(k,"") for k in TRADE_CSV_FIELDS})

# ══════════════════════════════════════════════════════════════════════
# Check Open Positions
# ══════════════════════════════════════════════════════════════════════
def check_positions(state, kline_data):
    closed = []; now = datetime.now(timezone.utc)
    for pos in state["positions"]:
        sym = pos["symbol"]
        if sym not in kline_data: continue
        df = kline_data[sym]
        if len(df) == 0: continue
        cp = float(df["close"].iloc[-1]); ch = float(df["high"].iloc[-1]); cl = float(df["low"].iloc[-1])
        exit_price = None; exit_reason = None

        if   pos["side"]=="LONG"  and ch >= pos["take"]:  exit_price, exit_reason = pos["take"], "TP"
        elif pos["side"]=="SHORT" and cl <= pos["take"]:  exit_price, exit_reason = pos["take"], "TP"
        elif pos["side"]=="LONG"  and cl <= pos["stop"]:  exit_price, exit_reason = pos["stop"], "SL"
        elif pos["side"]=="SHORT" and ch >= pos["stop"]:  exit_price, exit_reason = pos["stop"], "SL"

        if exit_price is None and not pos.get("breakeven_activated"):
            be = pos.get("breakeven_r")
            if be:
                entry = pos["entry"]; sp = pos.get("stop_pct",0.02)
                if pos["side"]=="LONG" and cp >= entry*(1+be*sp):
                    pos["stop"] = entry; pos["breakeven_activated"] = True
                elif pos["side"]=="SHORT" and cp <= entry*(1-be*sp):
                    pos["stop"] = entry; pos["breakeven_activated"] = True

        pos["bars_held"] = pos.get("bars_held",0) + 1
        if exit_price is None and pos["bars_held"] >= pos.get("max_bars",999):
            exit_price, exit_reason = cp, "EXPIRY"

        if exit_price is not None:
            pnl_pct = (exit_price-pos["entry"])/pos["entry"] if pos["side"]=="LONG" else (pos["entry"]-exit_price)/pos["entry"]
            pnl_dollar = pnl_pct * pos.get("notional", pos["alloc"])
            state["balance"] += pnl_dollar
            if state["balance"] > state.get("peak_balance", TOTAL_ALLOC):
                state["peak_balance"] = state["balance"]

            trade = {
                "exit_time": now.isoformat(), "strategy": pos["strategy"],
                "symbol": sym, "side": pos["side"], "entry": pos["entry"],
                "exit": exit_price, "exit_reason": exit_reason,
                "pnl": round(pnl_dollar,2), "pnl_pct": round(pnl_pct*100,2),
                "bars_held": pos["bars_held"], "entry_time": pos["entry_time"],
                "regime": pos.get("entry_regime",""), "adx_entry": pos.get("entry_adx",""),
                "atr_entry": pos.get("entry_atr",""), "volume_ratio": pos.get("entry_vol_ratio",""),
                "alloc": pos["alloc"], "notional": pos.get("notional",""),
                "risk_dollar": pos.get("risk_dollar",""),
            }
            closed.append(trade)
            append_trade_csv(trade)

    state["positions"] = [p for p in state["positions"]
                          if not any(c["entry_time"]==p["entry_time"] and c["symbol"]==p["symbol"] for c in closed)]
    state["closed_trades"].extend(closed)
    return closed

# ══════════════════════════════════════════════════════════════════════
# Telegram
# ══════════════════════════════════════════════════════════════════════
def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id":TELEGRAM_CHAT_ID,"text":text}, timeout=10)
        if r.status_code != 200:
            print(f"Telegram error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"Telegram exception: {e}")

# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    now = datetime.now(timezone.utc)
    state = load_state()

    if state.get("last_run"):
        last = datetime.fromisoformat(state["last_run"])
        if (now - last).total_seconds() < 300:
            print("Skipping — last run too recent"); return

    # Fetch data
    kline_data = {}; errors = []
    for s in STRATEGIES:
        try:
            df = fetch_klines(s["symbol"], s["interval"])
            df = compute_indicators(df)
            kline_data[s["symbol"]] = df
        except Exception as e:
            errors.append(f"{s['name']}: {e}")

    # Check open positions
    closed = check_positions(state, kline_data)

    # Check new signals + log them
    new_signals = []; signal_entries = []
    for s in STRATEGIES:
        if s["symbol"] not in kline_data: continue
        already_in = any(p["strategy"] == s["name"] for p in state["positions"])
        try:
            signals, snap = get_signals(kline_data[s["symbol"]], s)
            for sig in signals:
                entry = {
                    "time": now.isoformat(), "strategy": s["name"],
                    "symbol": s["symbol"], "side": sig["side"],
                    "entry": sig["entry"], "stop": sig["stop"], "take": sig["take"],
                    "taken": not already_in,
                    "regime": snap.get("regime",""), "adx": snap.get("adx",""),
                    "atr": snap.get("atr",""), "atr_ratio": snap.get("atr_ratio",""),
                    "price": snap.get("price",""), "volume_ratio": snap.get("volume_ratio",""),
                }
                signal_entries.append(entry)
                if not already_in:
                    sig["_strategy_name"] = s["name"]
                    new_signals.append(sig)
        except Exception as e:
            errors.append(f"Signal {s['name']}: {e}")

    state["signal_log"].extend(signal_entries)

    # Equity curve snapshot (daily)
    today = now.strftime("%Y-%m-%d")
    if not state["equity_curve"] or state["equity_curve"][-1]["date"] != today:
        total_value = state["balance"]
        for pos in state["positions"]:
            sym = pos["symbol"]
            if sym in kline_data and len(kline_data[sym]) > 0:
                cp = float(kline_data[sym]["close"].iloc[-1])
                upnl = (cp-pos["entry"])/pos["entry"]*pos.get("notional",pos["alloc"])
                if pos["side"]=="SHORT": upnl = (pos["entry"]-cp)/pos["entry"]*pos.get("notional",pos["alloc"])
                total_value += upnl
        state["equity_curve"].append({"date":today,"balance":round(total_value,2),"positions":len(state["positions"])})

    # Enter new positions
    for sig in new_signals:
        sname = sig.pop("_strategy_name")
        s = next(x for x in STRATEGIES if x["name"] == sname)
        alloc = s["alloc"]; risk = alloc * RISK_PER_TRADE
        pos_size = size_position(alloc, RISK_PER_TRADE, sig["entry"], sig["stop"])
        df = kline_data[s["symbol"]]
        row = df.iloc[-1]
        state["positions"].append({
            "strategy": s["name"], "symbol": s["symbol"], "side": sig["side"],
            "entry": sig["entry"], "stop": sig["stop"], "take": sig["take"],
            "alloc": alloc, "notional": pos_size, "risk_dollar": round(risk,2),
            "stop_pct": abs(sig["entry"]-sig["stop"])/sig["entry"],
            "max_bars": s["max_holding_bars"], "bars_held": 0,
            "breakeven_activated": False, "breakeven_r": s.get("breakeven_r"),
            "entry_time": now.isoformat(),
            "entry_bar_time": str(df.index[-1]),
            "entry_regime": "low_vol" if safe_float(row.get("atr_pct",50)) <= 50 else "high_vol",
            "entry_adx": round(safe_float(row.get("adx14")), 1),
            "entry_atr": round(safe_float(row["atr14"]), 4),
            "entry_vol_ratio": round(safe_float(row.get("volume_ratio",1)), 2),
        })

    # ── Build Telegram Report ──
    lines = [f"📊 Paper Portfolio — {now.strftime('%Y-%m-%d %H:%M UTC')}", ""]

    total_value = state["balance"]
    for pos in state["positions"]:
        sym = pos["symbol"]
        if sym in kline_data and len(kline_data[sym]) > 0:
            cp = float(kline_data[sym]["close"].iloc[-1])
            upnl = (cp-pos["entry"])/pos["entry"]*pos.get("notional",pos["alloc"])
            if pos["side"]=="SHORT": upnl = (pos["entry"]-cp)/pos["entry"]*pos.get("notional",pos["alloc"])
            total_value += upnl

    pnl_total = total_value - state["initial_balance"]
    pnl_pct = pnl_total/state["initial_balance"]*100
    dd = (total_value - state["peak_balance"])/state["peak_balance"]*100 if state.get("peak_balance") else 0
    e = "🟢" if pnl_total >= 0 else "🔴"
    lines.append(f"{e} Balance: ${total_value:.2f} | P&L: ${pnl_total:+.2f} ({pnl_pct:+.1f}%) | DD: {dd:.1f}%")
    lines.append("")

    if closed:
        lines.append("🔔 Closed:")
        for t in closed:
            icon = "✅" if t["pnl"] >= 0 else "❌"
            lines.append(f"  {icon} {t['symbol']} {t['side']} {t['exit_reason']} | ${t['pnl']:+.2f} ({t['pnl_pct']:+.1f}%) | {t.get('regime','')} ADX{t.get('adx_entry','')}")
        lines.append("")

    if state["positions"]:
        lines.append("📌 Open:")
        for pos in state["positions"]:
            sym = pos["symbol"]
            cp = float(kline_data[sym]["close"].iloc[-1]) if sym in kline_data else pos["entry"]
            up = (cp-pos["entry"])/pos["entry"]*100
            if pos["side"]=="SHORT": up = (pos["entry"]-cp)/pos["entry"]*100
            ie = "🟢" if up >=0 else "🔴"
            lines.append(f"  {pos['strategy']} {pos['side']} @{pos['entry']:.4f} → {ie} {up:+.1f}% [{pos['bars_held']}/{pos['max_bars']}]")
        lines.append("")

    if new_signals:
        lines.append("🎯 Signals:")
        for sig in new_signals:
            lines.append(f"  🚀 {sig.get('_strategy_name','')} {sig['side']} @{sig['entry']} | SL:{sig['stop']} TP:{sig['take']}")
    elif not state["positions"]:
        lines.append("💤 No signals. Normal.")
        lines.append("")

    if state["closed_trades"]:
        total = len(state["closed_trades"])
        wins = sum(1 for t in state["closed_trades"] if t["pnl"] > 0)
        wr = wins/total*100
        gross = sum(t["pnl"] for t in state["closed_trades"])
        avg_r = gross/total if total else 0
        lines.append(f"📈 All-Time: {total} trades | WR {wr:.0f}% | P&L ${gross:+.2f} | Avg ${avg_r:+.2f}/trade")
        by_strat = {}
        for t in state["closed_trades"]:
            k = t["strategy"]
            if k not in by_strat: by_strat[k] = {"t":0,"w":0,"pnl":0.0}
            by_strat[k]["t"] += 1
            if t["pnl"] > 0: by_strat[k]["w"] += 1
            by_strat[k]["pnl"] += t["pnl"]
        lines.append("  By strategy:")
        for name, st in by_strat.items():
            swr = st["w"]/st["t"]*100 if st["t"] else 0
            lines.append(f"    {name}: {st['t']} trades, WR {swr:.0f}%, P&L ${st['pnl']:+.2f}")
        by_regime = {}
        for t in state["closed_trades"]:
            r = t.get("regime","unknown")
            if r not in by_regime: by_regime[r] = {"t":0,"w":0,"pnl":0.0}
            by_regime[r]["t"] += 1
            if t["pnl"] > 0: by_regime[r]["w"] += 1
            by_regime[r]["pnl"] += t["pnl"]
        if len(by_regime) > 1:
            lines.append("  By regime:")
            for r, st in sorted(by_regime.items()):
                rwr = st["w"]/st["t"]*100 if st["t"] else 0
                lines.append(f"    {r}: {st['t']} trades, WR {rwr:.0f}%, P&L ${st['pnl']:+.2f}")

    msg = "\n".join(lines)
    print(msg)
    send_telegram(msg)

    state["last_run"] = now.isoformat()
    save_state(state)
    commit_state()

if __name__ == "__main__":
    if not os.path.exists(STATE_FILE):
        save_state(empty_state())
    main()
