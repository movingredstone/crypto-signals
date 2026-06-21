"""
GitHub Actions Paper Trader v2 — $330 3-Coin Portfolio
Records: trade history, signal log, equity curve, market context per entry.
"""
import json, os, sys, time, requests, csv
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
     "adx_min":20,"regime":"low_vol","breakeven_r":1.0,"partial_tp_r":1.0,"partial_tp_frac":0.5},
    {"name":"SUI macd_momentum/4h","symbol":"SUIUSDT","interval":"4h","alloc":110.0,
     "family":"macd_momentum","direction_filter":"ema_fast_stack","lookback":48,"volume_min":2.0,
     "atr_stop_mult":3.0,"take_profit_r":4.0,"max_holding_bars":24,"stop_rule":"swing",
     "adx_min":20,"regime":"any","partial_tp_r":1.0,"partial_tp_frac":0.5},
    {"name":"AVAX trend_pullback/8h","symbol":"AVAXUSDT","interval":"8h","alloc":110.0,
     "family":"trend_pullback","direction_filter":"price_ema100","lookback":48,"volume_min":1.2,
     "atr_stop_mult":2.0,"take_profit_r":5.0,"max_holding_bars":12,"stop_rule":"swing",
     "adx_min":0,"regime":"low_vol","partial_tp_frac":0.5,"pullback_ref":"ema20"},
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
# Indicators
# ══════════════════════════════════════════════════════════════════════
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _sma(s, n): return s.rolling(n).mean()

def compute_atr(df, period=14):
    h,l,c = df["high"],df["low"],df["close"]
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def compute_adx(df, period=14):
    h,l = df["high"],df["low"]
    pdm = h.diff().clip(lower=0); ndm = (-l.diff()).clip(lower=0)
    atr = compute_atr(df, period)
    pdi = 100*_ema(pdm,period)/atr; ndi = 100*_ema(ndm,period)/atr
    return _ema((pdi-ndi).abs()/(pdi+ndi)*100, period), pdi, ndi

def compute_macd(close, fast=12, slow=26, signal=9):
    macd = _ema(close,fast) - _ema(close,slow)
    return macd, _ema(macd,signal), macd - _ema(macd,signal)

def market_snapshot(df, atr):
    """Return regime + key metrics at current bar for data recording."""
    atr_base = atr.rolling(100).mean()
    ratio = atr / atr_base
    r = ratio.iloc[-1]
    regime = "high_vol" if r > 1.3 else ("low_vol" if r < 0.7 else "normal")
    adx,_,_ = compute_adx(df, 14)
    return {
        "regime": regime,
        "adx": round(float(adx.iloc[-1]), 1),
        "atr": round(float(atr.iloc[-1]), 4),
        "atr_ratio": round(float(r), 2),
        "price": round(float(df["close"].iloc[-1]), 4),
        "volume_ratio": round(float(df["volume"].iloc[-1] / _sma(df["volume"],20).iloc[-1]), 2),
    }

# ══════════════════════════════════════════════════════════════════════
# Strategy Signal Logic
# ══════════════════════════════════════════════════════════════════════
def macd_momentum_signal(df, params):
    close, high, low = df["close"], df["high"], df["low"]
    macd_line, signal_line, hist = compute_macd(close)
    atr = compute_atr(df, 14)
    ema100, ema20, ema50 = _ema(close,100), _ema(close,20), _ema(close,50)
    adx, pdi, ndi = compute_adx(df, 14)
    atr_base = atr.rolling(100).mean()
    regime_ratio = atr / atr_base
    rv = regime_ratio.iloc[-1]

    reg = "high_vol" if rv > 1.3 else ("low_vol" if rv < 0.7 else "normal")
    if params["regime"] != "any" and reg != params["regime"]:
        return [], market_snapshot(df, atr)
    if adx.iloc[-1] < params.get("adx_min", 0):
        return [], market_snapshot(df, atr)

    vol_ma = _sma(df["volume"], 20)
    if df["volume"].iloc[-1] <= vol_ma.iloc[-1] * params["volume_min"]:
        return [], market_snapshot(df, atr)

    price = close.iloc[-1]; atr_val = atr.iloc[-1]
    sm = params["atr_stop_mult"]; tm = params["take_profit_r"]
    signals = []
    hist_cross_up   = hist.iloc[-2] < 0 and hist.iloc[-1] > 0
    hist_turning_up = hist.iloc[-1] > hist.iloc[-2] and hist.iloc[-1] < 0
    dfilter = params.get("direction_filter","")
    trend_long = True
    if dfilter == "price_ema100":   trend_long = close.iloc[-1] > ema100.iloc[-1]
    elif dfilter == "ema_fast_stack": trend_long = ema20.iloc[-1] > ema50.iloc[-1]

    if (hist_cross_up or hist_turning_up) and trend_long:
        stop = price - atr_val*sm; take = price + atr_val*sm*tm
        if stop > 0:
            signals.append({"side":"LONG","entry":round(float(price),4),
                           "stop":round(float(stop),4),"take":round(float(take),4)})

    hist_cross_dn   = hist.iloc[-2] > 0 and hist.iloc[-1] < 0
    hist_turning_dn = hist.iloc[-1] < hist.iloc[-2] and hist.iloc[-1] > 0
    trend_short = True
    if dfilter == "price_ema100":   trend_short = close.iloc[-1] < ema100.iloc[-1]
    elif dfilter == "ema_fast_stack": trend_short = ema20.iloc[-1] < ema50.iloc[-1]

    if (hist_cross_dn or hist_turning_dn) and trend_short:
        stop = price + atr_val*sm; take = price - atr_val*sm*tm
        if stop > 0:
            signals.append({"side":"SHORT","entry":round(float(price),4),
                           "stop":round(float(stop),4),"take":round(float(take),4)})
    return signals, market_snapshot(df, atr)

def trend_pullback_signal(df, params):
    close = df["close"]
    atr = compute_atr(df, 14)
    ema100, ema20, ema50 = _ema(close,100), _ema(close,20), _ema(close,50)
    adx,_,_ = compute_adx(df, 14)
    atr_base = atr.rolling(100).mean()
    rv = (atr/atr_base).iloc[-1]
    reg = "high_vol" if rv > 1.3 else ("low_vol" if rv < 0.7 else "normal")
    if params["regime"] != "any" and reg != params["regime"]:
        return [], market_snapshot(df, atr)
    if adx.iloc[-1] < params.get("adx_min",0):
        return [], market_snapshot(df, atr)
    vol_ma = _sma(df["volume"],20)
    if df["volume"].iloc[-1] <= vol_ma.iloc[-1] * params["volume_min"]:
        return [], market_snapshot(df, atr)

    price = close.iloc[-1]; atr_val = atr.iloc[-1]
    ref = ema20 if params.get("pullback_ref","ema20") == "ema20" else ema50
    sm = params["atr_stop_mult"]; tm = params["take_profit_r"]
    signals = []

    uptrend = ema20.iloc[-1] > ema50.iloc[-1] and close.iloc[-1] > ema100.iloc[-1]
    near_ref = abs(close.iloc[-1] - ref.iloc[-1]) < atr_val * 1.5
    pulled_back = close.iloc[-3] > ref.iloc[-3] and near_ref
    if uptrend and pulled_back:
        stop = price - atr_val*sm; take = price + atr_val*sm*tm
        if stop > 0:
            signals.append({"side":"LONG","entry":round(float(price),4),
                           "stop":round(float(stop),4),"take":round(float(take),4)})

    downtrend = ema20.iloc[-1] < ema50.iloc[-1] and close.iloc[-1] < ema100.iloc[-1]
    near_ref_s = abs(close.iloc[-1] - ref.iloc[-1]) < atr_val * 1.5
    pulled_back_s = close.iloc[-3] < ref.iloc[-3] and near_ref_s
    if downtrend and pulled_back_s:
        stop = price + atr_val*sm; take = price - atr_val*sm*tm
        if stop > 0:
            signals.append({"side":"SHORT","entry":round(float(price),4),
                           "stop":round(float(stop),4),"take":round(float(take),4)})
    return signals, market_snapshot(df, atr)

def get_signals(df, s):
    f = s["family"]
    if f == "macd_momentum":    return macd_momentum_signal(df, s)
    elif f == "trend_pullback": return trend_pullback_signal(df, s)
    return [], {}

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
        # migrate v1 → v2
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
    os.system(f"git remote set-url origin https://x-access-token:{token}@github.com/{repo}.git")
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
# Trade Log CSV (append-only, human-readable)
# ══════════════════════════════════════════════════════════════════════
TRADE_CSV_FIELDS = [
    "exit_time","strategy","symbol","side","entry","exit","exit_reason",
    "pnl","pnl_pct","bars_held","entry_time","regime","adx_entry","atr_entry",
    "volume_ratio","alloc","notional","risk_dollar",
]

def append_trade_csv(trade):
    """Append one closed trade to CSV file."""
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
        cp = float(df["close"].iloc[-1])
        ch = float(df["high"].iloc[-1])
        cl = float(df["low"].iloc[-1])
        exit_price = None; exit_reason = None

        if   pos["side"]=="LONG"  and ch >= pos["take"]:  exit_price, exit_reason = pos["take"], "TP"
        elif pos["side"]=="SHORT" and cl <= pos["take"]:  exit_price, exit_reason = pos["take"], "TP"
        elif pos["side"]=="LONG"  and cl <= pos["stop"]:  exit_price, exit_reason = pos["stop"], "SL"
        elif pos["side"]=="SHORT" and ch >= pos["stop"]:  exit_price, exit_reason = pos["stop"], "SL"

        # breakeven
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
                "exit_time": now.isoformat(),
                "strategy": pos["strategy"], "symbol": sym, "side": pos["side"],
                "entry": pos["entry"], "exit": exit_price, "exit_reason": exit_reason,
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
            kline_data[s["symbol"]] = fetch_klines(s["symbol"], s["interval"])
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
                    "time": now.isoformat(),
                    "strategy": s["name"], "symbol": s["symbol"],
                    "side": sig["side"], "entry": sig["entry"],
                    "stop": sig["stop"], "take": sig["take"],
                    "taken": not already_in,
                    "regime": snap.get("regime",""), "adx": snap.get("adx",""),
                    "atr": snap.get("atr",""), "atr_ratio": snap.get("atr_ratio",""),
                    "price": snap.get("price",""), "volume_ratio": snap.get("volume_ratio",""),
                }
                signal_entries.append(entry)
                if not already_in:
                    sig["strategy_obj"] = s
                    sig["snapshot"] = snap
                    sig["_strategy_name"] = s["name"]  # for report display
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
        state["equity_curve"].append({"date": today, "balance": round(total_value,2), "positions": len(state["positions"])})

    # Enter new positions
    for sig in new_signals:
        s = sig.pop("strategy_obj")
        snap = sig.pop("snapshot")
        alloc = s["alloc"]; risk = alloc * RISK_PER_TRADE
        pos_size = size_position(alloc, RISK_PER_TRADE, sig["entry"], sig["stop"])
        state["positions"].append({
            "strategy": s["name"], "symbol": s["symbol"], "side": sig["side"],
            "entry": sig["entry"], "stop": sig["stop"], "take": sig["take"],
            "alloc": alloc, "notional": pos_size, "risk_dollar": round(risk,2),
            "stop_pct": abs(sig["entry"]-sig["stop"])/sig["entry"],
            "max_bars": s["max_holding_bars"], "bars_held": 0,
            "breakeven_activated": False, "breakeven_r": s.get("breakeven_r"),
            "entry_time": now.isoformat(), "entry_bar_time": str(kline_data[s["symbol"]].index[-1]),
            "entry_regime": snap.get("regime",""), "entry_adx": snap.get("adx",""),
            "entry_atr": snap.get("atr",""), "entry_vol_ratio": snap.get("volume_ratio",""),
        })

    # ── Build Telegram Report ──
    lines = [f"📊 Paper Portfolio — {now.strftime('%Y-%m-%d %H:%M UTC')}", ""]

    # Balance
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

    # Closed trades this run
    if closed:
        lines.append("🔔 Closed:")
        for t in closed:
            icon = "✅" if t["pnl"] >= 0 else "❌"
            lines.append(f"  {icon} {t['symbol']} {t['side']} {t['exit_reason']} | ${t['pnl']:+.2f} ({t['pnl_pct']:+.1f}%) | {t.get('regime','')} ADX{t.get('adx_entry','')}")
        lines.append("")

    # Open positions
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

    # New signals (fix broken name display)
    if new_signals:
        lines.append("🎯 Signals:")
        for sig in new_signals:
            lines.append(f"  🚀 {sig.get('_strategy_name','')} {sig['side']} @{sig['entry']} | SL:{sig['stop']} TP:{sig['take']}")
    elif not state["positions"]:
        lines.append("💤 No signals. Normal.")
        lines.append("")

    # Strategy stats (from all-time closed trades)
    if state["closed_trades"]:
        total = len(state["closed_trades"])
        wins = sum(1 for t in state["closed_trades"] if t["pnl"] > 0)
        wr = wins/total*100
        gross = sum(t["pnl"] for t in state["closed_trades"])
        avg_r = gross/total if total else 0
        lines.append(f"📈 All-Time: {total} trades | WR {wr:.0f}% | P&L ${gross:+.2f} | Avg ${avg_r:+.2f}/trade")

        # Per-strategy breakdown
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

        # Regime breakdown
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

    # ── Persist ──
    state["last_run"] = now.isoformat()
    save_state(state)
    commit_state()

if __name__ == "__main__":
    if not os.path.exists(STATE_FILE):
        save_state(empty_state())
    main()
