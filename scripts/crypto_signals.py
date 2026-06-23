#!/usr/bin/env python3
"""
CRYPTO PORTFOLIO SIGNAL CHECKER
3 coins: BTC keltner/1h + AVAX trend/8h + DOGE macd/8h
$330 allocation: BTC 40% + AVAX 30% + DOGE 30%
Risk: 5% per trade per coin
"""
import sys
sys.path.insert(0, ".")
from src.binance_data import load_or_download_klines
from src.research_engine import enrich_features, load_config
from src.research_engine import direction_allowed, confirmation_ok, entry_trigger
from src.research_engine import make_stop_take, safe_float
import pandas as pd
from datetime import datetime, timezone

PORTFOLIO = {
    "BTCUSDT": {
        "strategy": {
            "symbol": "BTCUSDT", "interval": "1h", "family": "keltner_breakout",
            "direction_filter": "mtf_trend", "lookback": 96, "volume_min": 0.7,
            "atr_stop_mult": 1.2, "take_profit_r": 3.0, "max_holding_bars": 96,
            "stop_rule": "swing", "adx_min": 0, "regime": "high_vol",
            "trailing_atr_mult": 3.0, "breakeven_r": None, "partial_tp_r": None,
        },
        "allocation": 130,  # $130
        "risk_pct": 0.05,
        "data_start": "2026-06-15",
    },
    "AVAXUSDT": {
        "strategy": {
            "symbol": "AVAXUSDT", "interval": "8h", "family": "trend_pullback",
            "direction_filter": "ema_fast_stack", "lookback": 48,
            "volume_min": 2.0, "atr_stop_mult": 2.0, "take_profit_r": 3.0,
            "max_holding_bars": 12, "stop_rule": "swing", "adx_min": 0,
            "regime": "any", "trailing_atr_mult": None,
            "breakeven_r": None, "partial_tp_r": None,
            "tolerance_pct": 0.006, "pullback_ref": "ema20",
        },
        "allocation": 100,
        "risk_pct": 0.05,
        "data_start": "2026-06-01",
    },
    "DOGEUSDT": {
        "strategy": {
            "symbol": "DOGEUSDT", "interval": "8h", "family": "macd_momentum",
            "direction_filter": "price_ema100", "lookback": 48,
            "volume_min": 1.2, "atr_stop_mult": 2.0, "take_profit_r": 3.0,
            "max_holding_bars": 12, "stop_rule": "swing", "adx_min": 20,
            "regime": "low_vol", "trailing_atr_mult": None,
            "breakeven_r": 1.0, "partial_tp_r": 1.0,
            "rsi_mid": 50,
        },
        "allocation": 100,
        "risk_pct": 0.05,
        "data_start": "2026-06-01",
    },
}

def check_coin(symbol, config_info):
    config = load_config("config.yaml")
    strat = config_info["strategy"]
    interval = strat["interval"]
    
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df = load_or_download_klines(symbol, interval, config_info["data_start"], today_str)
    df = enrich_features(df, interval, lookbacks=[strat.get("lookback", 20)])
    records = df.to_dict("records")
    
    # Check last bars for signals (last 12 for 1h, last 3 for 8h)
    lookback_bars = 12 if interval == "1h" else 3
    n = len(records)
    signals = []
    
    for i in range(max(0, n-lookback_bars), n-2):
        row = records[i]
        
        for side in ["LONG", "SHORT"]:
            if not direction_allowed(row, side, strat.get("direction_filter", "none")):
                continue
            if not confirmation_ok(row, strat):
                continue
            if entry_trigger(records, i, side, strat):
                entry_price = safe_float(records[i+1]["open"])
                stop, take = make_stop_take(row, side, entry_price, strat)
                if stop is None or take is None:
                    continue
                
                # Position sizing
                alloc = config_info["allocation"]
                risk_pct = config_info["risk_pct"]
                risk_dollar = alloc * risk_pct
                stop_pct = abs(entry_price - stop) / entry_price
                position_size = risk_dollar / stop_pct
                leverage = position_size / alloc
                
                signals.append({
                    "side": side,
                    "entry": round(entry_price, 4),
                    "stop": round(stop, 4),
                    "take": round(take, 4),
                    "alloc": alloc,
                    "risk": round(risk_dollar, 2),
                    "position": round(position_size, 2),
                    "leverage": round(leverage, 1),
                })
    
    return signals, records

if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    print(f"╔{'═'*58}╗")
    print(f"║  CRYPTO PORTFOLIO SIGNALS — {now.strftime('%Y-%m-%d %H:%M UTC'):<30}║")
    print(f"╠{'═'*58}╣")
    
    total_signals = 0
    prices = {}
    
    for symbol, info in PORTFOLIO.items():
        signals, records = check_coin(symbol, info)
        last_price = safe_float(records[-1]["close"])
        prices[symbol] = last_price
        
        emoji = "🟢" if signals else "⚪"
        print(f"║ {emoji} {symbol:<10} ${last_price:,.2f}  |  "
              f"alloc ${info['allocation']}  |  risk ${info['allocation']*info['risk_pct']:.2f}/trade")
        
        if signals:
            for s in signals:
                dir_emoji = "🟢 LONG " if s["side"] == "LONG" else "🔴 SHORT"
                print(f"║   {dir_emoji} @ ${s['entry']:,.4f}")
                print(f"║   SL: ${s['stop']:,.4f} | TP: ${s['take']:,.4f}")
                print(f"║   Pos: ${s['position']:.0f} (~{s['leverage']:.1f}x) | "
                      f"Risk: ${s['risk']:.2f}")
                total_signals += 1
        else:
            # Show current regime hint
            last = records[-1]
            adx = safe_float(last.get("adx14", 0))
            print(f"║   Waiting... ADX={adx:.1f}")
        print(f"║")
    
    print(f"╠{'═'*58}╣")
    print(f"║  TOTAL: {total_signals} signal(s) | Portfolio: $330")
    print(f"║  BTC ${prices.get('BTCUSDT', 0):,.0f} | "
          f"AVAX ${prices.get('AVAXUSDT', 0):,.2f} | "
          f"DOGE ${prices.get('DOGEUSDT', 0):,.4f}")
    print(f"╚{'═'*58}╝")
