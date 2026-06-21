"""
Standalone Crypto Signal Checker — GitHub Actions
No local dependencies. Only pandas, numpy, requests.
Runs every 4 hours, sends results to Telegram.
"""
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

# ── Config ──────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
BINANCE_BASE = "https://fapi.binance.com"

# ── Strategy Parameters ─────────────────────────────────────────────
STRATEGIES = [
    {
        "name": "DOGE macd/8h",
        "symbol": "DOGEUSDT", "interval": "8h", "alloc": 110,
        "risk_pct": 0.05,
        "params": {"lookback": 48, "atr_stop_mult": 2.0, "take_profit_r": 3.0,
                   "max_holding_bars": 12, "regime": "low_vol", "adx_min": 20,
                   "vol_min": 1.2},
    },
    {
        "name": "AVAX trend/8h",
        "symbol": "AVAXUSDT", "interval": "8h", "alloc": 110,
        "risk_pct": 0.05,
        "params": {"lookback": 48, "atr_stop_mult": 1.5, "take_profit_r": 3.0,
                   "max_holding_bars": 24, "regime": "any", "adx_min": 20,
                   "vol_min": 0.7},
    },
]

# ── Binance Data ────────────────────────────────────────────────────
def fetch_klines(symbol, interval, limit=300):
    url = f"{BINANCE_BASE}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_vol",
        "taker_buy_quote_vol","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df

# ── Indicators ──────────────────────────────────────────────────────
def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def compute_adx(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    atr = compute_atr(df, period)
    plus_di = 100 * compute_ema(plus_dm, period) / atr
    minus_di = 100 * compute_ema(minus_dm, period) / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx = compute_ema(dx, period)
    return adx, plus_di, minus_di

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    return macd_line, signal_line

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ── Signal Check ────────────────────────────────────────────────────
def check_signals(df, params):
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    lookback = params["lookback"]
    
    # Indicators
    ema20 = compute_ema(close, 20)
    ema50 = compute_ema(close, 50)
    atr = compute_atr(df, 14)
    adx, plus_di, minus_di = compute_adx(df, 14)
    macd_line, signal_line = compute_macd(close)
    rsi = compute_rsi(close)
    
    # Volume filter
    vol_ma = volume.rolling(20).mean()
    vol_ok = volume.iloc[-1] > vol_ma.iloc[-1] * params.get("vol_min", 1.0)
    
    # Regime (simplified)
    is_high_vol = atr.iloc[-1] > atr.rolling(100).mean().iloc[-1] * 1.3
    is_low_vol = atr.iloc[-1] < atr.rolling(100).mean().iloc[-1] * 0.7
    regime_ok = True
    if params["regime"] == "high_vol":
        regime_ok = is_high_vol
    elif params["regime"] == "low_vol":
        regime_ok = is_low_vol
    
    adx_val = adx.iloc[-1]
    adx_ok = adx_val >= params.get("adx_min", 0)
    
    current_price = close.iloc[-1]
    stop_mult = params["atr_stop_mult"]
    tp_mult = params["take_profit_r"]
    current_atr = atr.iloc[-1]
    
    signals = []
    
    # Simplified entry: MACD crossover + EMA alignment
    # LONG
    macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]
    trend_up = ema20.iloc[-1] > ema50.iloc[-1]
    
    if macd_bullish and trend_up and vol_ok and regime_ok and adx_ok:
        stop = current_price - current_atr * stop_mult
        take = current_price + current_atr * stop_mult * tp_mult
        if stop > 0 and take > current_price:
            signals.append({
                "side": "LONG",
                "entry": round(float(current_price), 4),
                "stop": round(float(stop), 4),
                "take": round(float(take), 4),
            })
    
    # SHORT
    macd_bearish = macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]
    trend_down = ema20.iloc[-1] < ema50.iloc[-1]
    
    if macd_bearish and trend_down and vol_ok and regime_ok and adx_ok:
        stop = current_price + current_atr * stop_mult
        take = current_price - current_atr * stop_mult * tp_mult
        if stop > 0 and take < current_price:
            signals.append({
                "side": "SHORT",
                "entry": round(float(current_price), 4),
                "stop": round(float(stop), 4),
                "take": round(float(take), 4),
            })
    
    return signals, {
        "price": round(float(current_price), 4),
        "atr": round(float(current_atr), 4),
        "adx": round(float(adx_val), 1),
    }

# ── Telegram ────────────────────────────────────────────────────────
def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)

# ── Main ────────────────────────────────────────────────────────────
def main():
    now = datetime.now(timezone.utc)
    lines = [f"📡 Signal Check — {now.strftime('%Y-%m-%d %H:%M UTC')}", ""]
    total_signals = 0
    
    for s in STRATEGIES:
        df = fetch_klines(s["symbol"], s["interval"])
        signals, info = check_signals(df, s["params"])
        
        alloc = s["alloc"]
        risk = alloc * s["risk_pct"]
        
        emoji = "🟢" if signals else "⚪"
        lines.append(f"{emoji} {s['name']}: ${info['price']} (ATR={info['atr']}, ADX={info['adx']})")
        
        for sig in signals:
            stop_pct = abs(sig["entry"] - sig["stop"]) / sig["entry"] * 100
            pos_size = risk / (stop_pct / 100)
            leverage = pos_size / alloc
            
            lines.append(f"  🎯 {sig['side']} @ ${sig['entry']}")
            lines.append(f"     SL: ${sig['stop']} | TP: ${sig['take']}")
            lines.append(f"     ${alloc} alloc: \${pos_size:.0f} ({leverage:.1f}x) risk=\${risk:.2f}")
            total_signals += 1
        
        if not signals:
            lines.append(f"  waiting...")
    
    lines.append("")
    if total_signals:
        lines.append(f"🚨 {total_signals} SIGNAL(S) FOUND")
    else:
        lines.append("💤 No signals — normal")
    
    send_telegram("\n".join(lines))

if __name__ == "__main__":
    main()
