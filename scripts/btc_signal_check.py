#!/usr/bin/env python3
"""
BTC keltner_breakout/1h — DAILY SIGNAL CHECKER
Checks last 24 hours of 1h candles for trade signals.
Reports: entry signals, current positions, P&L.
"""
import sys
sys.path.insert(0, ".")
from src.binance_data import load_or_download_klines
from src.research_engine import enrich_features, load_config
from src.research_engine import direction_allowed, confirmation_ok, entry_trigger
from src.research_engine import make_stop_take, safe_float
import pandas as pd
from datetime import datetime, timezone

# Strategy params (BTC keltner_breakout/1h, verified)
STRATEGY = {
    "symbol": "BTCUSDT", "interval": "1h", "family": "keltner_breakout",
    "direction_filter": "mtf_trend", "lookback": 96, "volume_min": 0.7,
    "atr_stop_mult": 1.2, "take_profit_r": 3.0, "max_holding_bars": 96,
    "stop_rule": "swing", "adx_min": 0, "regime": "high_vol",
    "trailing_atr_mult": 3.0, "breakeven_r": None, "partial_tp_r": None,
}

def check_signal():
    config = load_config("config.yaml")
    
    # Load recent data (need 200 bars for indicators)
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df = load_or_download_klines("BTCUSDT", "1h", "2026-06-15", today_str)
    df = enrich_features(df, "1h", lookbacks=[96])
    records = df.to_dict("records")
    
    # Check last 24 bars for signals
    signals = []
    n = len(records)
    now = datetime.now(timezone.utc)
    
    for i in range(max(0, n-24), n-2):
        row = records[i]
        candle_time = row["open_time"]
        
        for side in ["LONG", "SHORT"]:
            if not direction_allowed(row, side, STRATEGY.get("direction_filter", "none")):
                continue
            if not confirmation_ok(row, STRATEGY):
                continue
            if entry_trigger(records, i, side, STRATEGY):
                entry_price = safe_float(records[i+1]["open"])
                stop, take = make_stop_take(row, side, entry_price, STRATEGY)
                
                if stop is None or take is None:
                    continue
                    
                signals.append({
                    "time": str(candle_time),
                    "side": side,
                    "entry": round(entry_price, 2),
                    "stop": round(stop, 2),
                    "take": round(take, 2),
                    "rr": round(abs(take-entry_price)/abs(entry_price-stop), 2),
                })
    
    # Report
    print(f"BTC keltner_breakout/1h Signal Check — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Last price: ${safe_float(records[-1]['close']):,.2f}")
    print(f"Last 24h candles: {min(24, n)}")
    print()
    
    if not signals:
        print("❌ No signals in last 24 hours.")
        # Show current regime info
        last = records[-1]
        adx = safe_float(last.get("adx14", 0))
        atr_rank = "check"  # simplified
        print(f"ADX: {adx:.1f} | Waiting for high_vol regime + keltner breakout...")
    else:
        print(f"🚨 {len(signals)} SIGNAL(S) FOUND:")
        for s in signals:
            emoji = "🟢" if s["side"] == "LONG" else "🔴"
            print(f"  {emoji} {s['side']} @ ${s['entry']:,.2f}")
            print(f"     SL: ${s['stop']:,.2f} | TP: ${s['take']:,.2f} | R:R = 1:{s['rr']}")
            print(f"     Time: {s['time']}")
            # Position sizing for $330 at 5% risk
            risk = 330 * 0.05  # $16.50
            stop_pct = abs(s['entry'] - s['stop']) / s['entry'] * 100
            position_size = risk / (stop_pct / 100)
            leverage = position_size / 330
            print(f"     $330: ${position_size:.0f} notional (~{leverage:.1f}x), "
                  f"risk=${risk:.2f} ({stop_pct:.2f}% stop)")
        print()

if __name__ == "__main__":
    check_signal()
