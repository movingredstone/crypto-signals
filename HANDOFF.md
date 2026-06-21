# Crypto Paper Trading System — Handoff for Verification

## What This Is
A $330 paper trading system running on GitHub Actions, executing 3 backtest-verified crypto futures strategies. Every 4 hours: fetch Binance data → check positions → check signals → Telegram report → commit state.

**Repo:** https://github.com/movingredstone/crypto-signals
**Local:** ~/Desktop/hermes/investmentsystem

## Current State (2026-06-21)
- Deployed and operational. Telegram reports working.
- No trades yet. No open positions. Balance: $330.00
- **v3**: Signal logic now matched to backtest engine (research_engine.py)

## Architecture: How Signals Match the Backtest Engine
The standalone `paper_trader_github.py` implements the SAME functions as the backtest engine (`src/research_engine.py`):

| Function | Backtest Engine | Standalone |
|----------|----------------|------------|
| `near()` | `abs(value-target)/abs(value) <= tolerance_pct` | Same |
| `direction_allowed()` | Checks ema200/ema_stack/ema_fast_stack/price_ema100/mtf_trend | Same |
| `confirmation_ok()` | Volume min, ATR% range, ADX min, regime (rv_pct percentile) | Same |
| `check_entry()` | Family-specific entry condition | Same |
| `make_stop_take()` | ATR-based or swing-based stop + TP=R×risk | Same |

### Key Design: Separated Concerns
The backtest engine separates 4 independent checks per signal:
1. **confirmation_ok()** — volume, ATR%, ADX strength, regime filter
2. **direction_allowed()** — trend direction via EMA relationships
3. **check_entry()** — the actual entry trigger (family-specific)
4. **make_stop_take()** — stop loss placement (ATR or swing-based)

The standalone implements all 4 identically.

## 3 Strategies (Backtest-Verified Parameters)

### DOGE macd_momentum/8h
- Allocation: $110 | Risk: 5% ($5.50) | Price: ~$0.08
- Backtest: PF=1.99, WR=67%, WF=3/3, 60 trades, MDD=-0.90%
- Params: family=macd_momentum, direction_filter=price_ema100, lookback=48, volume_min=1.2, atr_stop_mult=2.0, take_profit_r=3.0, max_holding_bars=12, stop_rule=swing, adx_min=20, regime=low_vol, breakeven_r=1.0, tolerance_pct=0.006

### SUI macd_momentum/4h
- Allocation: $110 | Risk: 5% ($5.50) | Price: ~$0.90
- Backtest: PF=2.54, WR=48%, WF=2/3, 52 trades, MDD=-2.59%
- Params: family=macd_momentum, direction_filter=ema_fast_stack, lookback=48, volume_min=2.0, atr_stop_mult=3.0, take_profit_r=4.0, max_holding_bars=24, stop_rule=swing, adx_min=20, regime=any, tolerance_pct=0.006

### AVAX trend_pullback/8h
- Allocation: $110 | Risk: 5% ($5.50) | Price: ~$6.23
- Backtest: PF=1.90, WR=45%, WF=3/3, 7/7 stress folds, 40 trades
- Params: family=trend_pullback, direction_filter=price_ema100, lookback=48, volume_min=1.2, atr_stop_mult=2.0, take_profit_r=5.0, max_holding_bars=12, stop_rule=swing, adx_min=0, regime=low_vol, pullback_ref=ema20, tolerance_pct=0.006

## MACD Momentum Entry Logic (exact backtest match)
```python
# Backtest engine (research_engine.py line 378-381):
if family == "macd_momentum":
    if side == "LONG":
        return row["macd_hist"] > 0 and row["macd_hist"] > prev["macd_hist"]
    return row["macd_hist"] < 0 and row["macd_hist"] < prev["macd_hist"]
```
LONG = histogram is positive AND increasing (continued momentum, not reversal).
SHORT = histogram is negative AND decreasing.

## Trend Pullback Entry Logic (exact backtest match)
```python
# Backtest engine (research_engine.py line 342-344):
if family == "trend_pullback":
    target = safe_float(row[exp.get("pullback_ref", "ema20")])
    return near(close, target, tol)
```
Just checks: is price within `tolerance_pct` (0.6%) of the reference EMA?
Direction filter (ema100, ema_stack, etc.) and confirmation (volume, ADX, regime) are checked separately.

## Stop Placement (exact backtest match)
```python
# stop_rule="swing": uses swing_low/swing_high (lowest/highest over lookback window)
# stop_rule="atr": uses entry ± atr_stop_mult × ATR
# All 3 strategies use stop_rule="swing"
```

## Position Sizing
```
position_notional = allocation × risk_pct / (stop_distance / entry)
```
Where stop_distance = |entry - stop|. Risk is ALWAYS exactly risk_pct × allocation in dollars.

## Data Recorded
- `paper_trades.csv`: Every closed trade with entry/exit, regime, ADX, ATR, volume context
- `paper_state.json`: Full state (balance, positions, signal_log, equity_curve, closed_trades)
- `signal_log`: ALL signals (taken + skipped) with market context
- `equity_curve`: Daily balance snapshots

## Files
- `paper_trader_github.py` — Main trader (v3, backtest-matched)
- `.github/workflows/signal.yml` — 4-hourly schedule
- `paper_state.json` — Portfolio state (auto-committed)
- `paper_trades.csv` — Trade log (auto-committed)
- `requirements_paper.txt` — pandas, numpy, requests
- `HANDOFF.md` — This file

## How to Test
```bash
cd ~/Desktop/hermes/investmentsystem
git pull
TELEGRAM_TOKEN=*** TELEGRAM_CHAT_ID=test python3 paper_trader_github.py
```

## Verification: Known Checked Differences (v2→v3)
v2 had standalone implementations that differed from backtest. v3 fixes all:
- ✅ MACD: was "crossover" logic, now matches "hist > 0 and increasing"
- ✅ Stops: was ATR-only, now uses swing_low/swing_high (stop_rule=swing)
- ✅ Regime: was ATR-ratio, now uses ATR percentile (atr_pct)
- ✅ Trend pullback entry: was trend+pullback combined, now matches near() check
- ✅ Direction filter: was simplified, now matches all 6 filter types

## Verification Task for New Model
1. Verify strategy PARAMETERS match the backtest templates exactly
2. Verify signal LOGIC in check_entry(), direction_allowed(), confirmation_ok(), make_stop_take()
3. Verify the GITHUB ACTIONS WORKFLOW has correct permissions and schedule
4. Check for any unreported discrepancies
5. Report findings and recommended fixes
