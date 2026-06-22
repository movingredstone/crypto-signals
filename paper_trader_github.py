"""
GitHub Actions Paper Trader v3 — $330 Best Aggressive 3-Coin Portfolio
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
FEE_RATE = 0.0005        # config.yaml fees.taker
DEFAULT_SLIPPAGE = 0.0004 # config.yaml slippage.default
SLIPPAGE_BY_SYMBOL = {"BTCUSDT": 0.0002, "ETHUSDT": 0.0003}

def slippage_rate(symbol):
    return SLIPPAGE_BY_SYMBOL.get(symbol, DEFAULT_SLIPPAGE)

STRATEGIES = [
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
    {"name":"SOL macd_momentum/4h","symbol":"SOLUSDT","interval":"4h","alloc":110.0,
     "family":"macd_momentum","direction_filter":"ema200","lookback":96,"volume_min":0.0,
     "atr_stop_mult":2.5,"take_profit_r":3.0,"max_holding_bars":24,"stop_rule":"atr",
     "adx_min":20,"regime":"any","partial_tp_frac":0.5,
     "tolerance_pct":0.006},
]

# ══════════════════════════════════════════════════════════════════════
# Data Fetching
# ══════════════════════════════════════════════════════════════════════
def fetch_klines(symbol, interval, limit=500):
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
def _ema(s, n): return s.ewm(span=n, adjust=False, min_periods=n).mean()
def _sma(s, n): return s.rolling(n, min_periods=n).mean()
def _wilder(s, n): return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

def compute_indicators(df):
    """Compute columns to match src.indicators.add_indicators() + research_engine.enrich_features()."""
    c, h, l = df["close"], df["high"], df["low"]
    # EMAs — add_indicators + enrich_features
    df["ema10"] = _ema(c, 10)
    df["ema20"] = _ema(c, 20)
    df["ema50"] = _ema(c, 50)
    df["ema100"] = _ema(c, 100)
    df["ema200"] = _ema(c, 200)
    # ATR 14 — Wilder smoothing, same as src.indicators
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["atr14"] = _wilder(tr, 14)
    # ATR percentile — same window/min_periods as src.indicators
    df["atr_pct"] = df["atr14"].rolling(200, min_periods=50).rank(pct=True) * 100
    # Realized volatility percentile — regime gate uses rv_pct, not atr_pct
    ret = c.pct_change()
    df["rv"] = ret.rolling(24, min_periods=24).std()
    df["rv_pct"] = df["rv"].rolling(300, min_periods=100).rank(pct=True) * 100
    # MACD — same min_periods as research_engine.enrich_features
    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    # RSI — Wilder smoothing, same as src.indicators
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _wilder(gain, 14)
    avg_loss = _wilder(loss, 14)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))
    # ADX — same Wilder method as src.indicators.add_adx
    up = h.diff(); down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_di = 100 * _wilder(pd.Series(plus_dm, index=df.index), 14) / df["atr14"]
    minus_di = 100 * _wilder(pd.Series(minus_dm, index=df.index), 14) / df["atr14"]
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx14"] = _wilder(dx, 14)
    # Volume ratio
    df["vol_ma20"] = _sma(df["volume"], 20)
    df["volume_ratio"] = df["volume"] / df["vol_ma20"]
    # Recent swing levels — shifted to avoid lookahead, exactly like src.indicators
    df["recent_swing_high"] = h.rolling(20, min_periods=20).max().shift(1)
    df["recent_swing_low"] = l.rolling(20, min_periods=20).min().shift(1)
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
        return (side == "LONG" and safe_float(row["ema10"]) > safe_float(row["ema20"]) > safe_float(row["ema50"])) or \
               (side == "SHORT" and safe_float(row["ema10"]) < safe_float(row["ema20"]) < safe_float(row["ema50"]))
    if direction_filter == "price_ema100":
        return (side == "LONG" and row["close"] > safe_float(row["ema100"])) or \
               (side == "SHORT" and row["close"] < safe_float(row["ema100"]))
    if direction_filter == "supertrend":
        d = safe_float(row.get("supertrend_dir"), 0)
        return d > 0 if side == "LONG" else d < 0
    if direction_filter == "mtf_trend":
        t = safe_float(row.get("htf_trend"), 0)
        return t > 0 if side == "LONG" else t < 0
    return False

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
        rv_pct = safe_float(row.get("rv_pct"), 50)
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
        swing_stop = safe_float(row.get("recent_swing_low"), entry)

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
    swing_stop = safe_float(row.get("recent_swing_high"), entry)

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

    # Need previous closed bar for signal and current bar open for entry.
    # Backtest: signal at i, entry at i+1 open. Binance latest candle is treated as i+1.
    if len(df) < 4:
        return [], {}

    row = df.iloc[-2]       # signal bar (last closed / previous bar)
    prev_row = df.iloc[-3]
    entry_row = df.iloc[-1] # entry bar

    # Market snapshot for logging — based on signal bar, matching backtest decision time.
    snap = {
        "regime": "low_vol" if safe_float(row.get("rv_pct",50)) <= 50 else "high_vol",
        "adx": round(safe_float(row.get("adx14")), 1),
        "atr": round(safe_float(row["atr14"]), 4),
        "atr_ratio": 0,  # unused now, keeping for compat
        "price": round(float(row["close"]), 4),
        "entry_bar_time": str(entry_row.name),
        "signal_bar_time": str(row.name),
        "volume_ratio": round(safe_float(row.get("volume_ratio", 1)), 2),
    }

    # Pre-checks (confirmation)
    if not confirmation_ok(row, exp):
        return [], snap

    dfilter = exp.get("direction_filter", "none")
    signals = []
    slip = slippage_rate(exp["symbol"])
    raw_entry = safe_float(entry_row["open"])

    for side in ["LONG", "SHORT"]:
        if not direction_allowed(row, side, dfilter):
            continue
        if not check_entry(row, prev_row, side, exp, df):
            continue

        entry = raw_entry * (1 + slip) if side == "LONG" else raw_entry * (1 - slip)
        stop, take = make_stop_take(row, side, entry, exp)
        if stop is not None and take is not None:
            signals.append({
                "side": side, "entry": round(entry, 8), "raw_entry": round(raw_entry, 8),
                "stop": round(stop, 8), "take": round(take, 8),
                "entry_bar_time": str(entry_row.name), "signal_bar_time": str(row.name),
            })

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
    "pnl","gross_pnl","fee","slippage_cost","pnl_pct","bars_held","entry_time","regime","adx_entry","atr_entry",
    "volume_ratio","alloc","notional","risk_dollar","fills",
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
def _records_from_df(df):
    d = df.reset_index().copy()
    if "open_time" not in d.columns:
        d = d.rename(columns={d.columns[0]: "open_time"})
    return d.to_dict("records")

def _entry_index(records, entry_bar_time):
    target = pd.Timestamp(entry_bar_time)
    for i, rec in enumerate(records):
        if pd.Timestamp(rec["open_time"]) == target:
            return i
    return None

def simulate_live_fills(records, entry_i, side, entry, stop, take, exp, current_i):
    """Prefix-equivalent of research_engine.simulate_fills().
    Same intrabar order: stop -> partial TP -> final TP -> breakeven -> trailing.
    Unlike backtest, no time_exit until max_holding_bars is actually reached.
    """
    max_holding = exp.get("max_holding_bars", 72)
    end_i = min(current_i, entry_i + max_holding)
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
        atr_j = safe_float(rec.get("atr14"))

        if long and low <= cur_stop:
            fills.append((remaining, cur_stop, "stop_loss", j)); remaining = 0.0; break
        if (not long) and high >= cur_stop:
            fills.append((remaining, cur_stop, "stop_loss", j)); remaining = 0.0; break

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

        tp_hit = high >= take if long else low <= take
        if tp_hit:
            fills.append((remaining, take, "take_profit", j)); remaining = 0.0; break

        if be_r is not None:
            be_level = entry + be_r * risk if long else entry - be_r * risk
            reached = high >= be_level if long else low <= be_level
            if reached:
                cur_stop = max(cur_stop, entry) if long else min(cur_stop, entry)

        if trail_mult is not None and math.isfinite(atr_j) and atr_j > 0:
            cur_stop = max(cur_stop, high - trail_mult * atr_j) if long else min(cur_stop, low + trail_mult * atr_j)

    if remaining > 1e-9 and current_i >= entry_i + max_holding:
        fills.append((remaining, safe_float(records[end_i]["close"]), "time_exit", end_i))
    return fills

def _net_pnl_from_fills(pos, fills):
    entry = float(pos["entry"]); notional = float(pos.get("notional", pos["alloc"]))
    qty = notional / entry
    slip = float(pos.get("slippage_rate", slippage_rate(pos["symbol"])))
    gross = 0.0
    for frac, raw_price, _reason, _bar in fills:
        if pos["side"] == "LONG":
            exit_price = raw_price * (1 - slip)
            gross += (exit_price - entry) * qty * frac
        else:
            exit_price = raw_price * (1 + slip)
            gross += (entry - exit_price) * qty * frac
    fee = notional * float(pos.get("fee_rate", FEE_RATE)) * 2
    return gross - fee, gross, fee, notional * slip * 2

def check_positions(state, kline_data):
    closed = []; now = datetime.now(timezone.utc)
    for pos in state["positions"]:
        sym = pos["symbol"]
        if sym not in kline_data: continue
        df = kline_data[sym]
        if len(df) == 0: continue
        records = _records_from_df(df)
        entry_i = _entry_index(records, pos.get("entry_bar_time"))
        if entry_i is None: continue
        current_i = len(records) - 1
        strategy = next((s for s in STRATEGIES if s["name"] == pos["strategy"]), {})
        fills = simulate_live_fills(records, entry_i, pos["side"], pos["entry"], pos["stop"], pos["take"], strategy, current_i)
        pos["bars_held"] = max(0, current_i - entry_i + 1)
        pos["fills_seen"] = [{"frac": f, "price": p, "reason": r, "bar": b} for f,p,r,b in fills]

        # Only close and book P&L once all fractions are filled, matching backtest's final trade accounting.
        if not fills or sum(f[0] for f in fills) < 1.0 - 1e-9:
            continue

        pnl_dollar, gross_pnl, fee, slip_cost = _net_pnl_from_fills(pos, fills)
        exit_raw = fills[-1][1]
        exit_reason = fills[-1][2]
        exit_price = exit_raw * (1 - pos.get("slippage_rate", slippage_rate(sym))) if pos["side"] == "LONG" else exit_raw * (1 + pos.get("slippage_rate", slippage_rate(sym)))
        pnl_pct = pnl_dollar / float(pos.get("notional", pos["alloc"])) * 100
        state["balance"] += pnl_dollar
        if state["balance"] > state.get("peak_balance", TOTAL_ALLOC):
            state["peak_balance"] = state["balance"]

        trade = {
            "exit_time": now.isoformat(), "strategy": pos["strategy"],
            "symbol": sym, "side": pos["side"], "entry": pos["entry"],
            "exit": round(exit_price,8), "exit_reason": exit_reason,
            "pnl": round(pnl_dollar,2), "gross_pnl": round(gross_pnl,2),
            "fee": round(fee,4), "slippage_cost": round(slip_cost,4),
            "pnl_pct": round(pnl_pct,2), "bars_held": pos["bars_held"], "entry_time": pos["entry_time"],
            "regime": pos.get("entry_regime",""), "adx_entry": pos.get("entry_adx",""),
            "atr_entry": pos.get("entry_atr",""), "volume_ratio": pos.get("entry_vol_ratio",""),
            "alloc": pos["alloc"], "notional": pos.get("notional",""),
            "risk_dollar": pos.get("risk_dollar",""),
            "fills": json.dumps(pos["fills_seen"]),
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
        sname = sig.get("_strategy_name")
        s = next(x for x in STRATEGIES if x["name"] == sname)
        alloc = s["alloc"]; risk = alloc * RISK_PER_TRADE
        pos_size = size_position(alloc, RISK_PER_TRADE, sig["entry"], sig["stop"])
        df = kline_data[s["symbol"]]
        row = df.iloc[-2]  # signal bar, matching get_signals()/backtest decision bar
        state["positions"].append({
            "strategy": s["name"], "symbol": s["symbol"], "side": sig["side"],
            "entry": sig["entry"], "raw_entry": sig.get("raw_entry"), "stop": sig["stop"], "take": sig["take"],
            "alloc": alloc, "notional": pos_size, "risk_dollar": round(risk,2),
            "fee_rate": FEE_RATE, "slippage_rate": slippage_rate(s["symbol"]),
            "stop_pct": abs(sig["entry"]-sig["stop"])/sig["entry"],
            "max_bars": s["max_holding_bars"], "bars_held": 0,
            "entry_time": now.isoformat(),
            "signal_bar_time": sig.get("signal_bar_time"),
            "entry_bar_time": sig.get("entry_bar_time", str(df.index[-1])),
            "entry_regime": "low_vol" if safe_float(row.get("rv_pct",50)) <= 50 else "high_vol",
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
