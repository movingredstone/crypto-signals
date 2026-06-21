# Crypto Paper Trading System — Handoff for Verification

## What This Is
A $330 paper trading system running on GitHub Actions, executing 3 backtest-selected crypto futures strategies. Every 4 hours: fetch Binance data → compute indicators → check positions → check signals → Telegram report → commit state.

**Repo:** https://github.com/movingredstone/crypto-signals
**Local:** ~/Desktop/hermes/investmentsystem

## Current State
- Deployed and operational. Telegram reports working.
- No trades yet. No open positions. Balance: $330.00
- Current implementation: `paper_trader_github.py`
- Legacy simple checker: `paper_signal.py` is NOT used.

## Important Clarification
The trained/backtested DATA is not copied into the algorithm. What must be copied is:
1. The selected strategy family
2. The selected interval
3. The selected parameter set
4. The exact entry/confirmation/direction/stop/take-profit logic from the backtest engine

So the verification question is: “Did the live paper trader implement the selected strategies and the backtest engine logic exactly?”

## Source of Truth
- Strategy templates:
  - DOGE: `btc-optimization-pipeline/references/doge-strategy-template.md`
  - SUI: `btc-optimization-pipeline/references/sui-strategy-template.md`
  - AVAX: `btc-optimization-pipeline/references/avax-strategy-template.md`
- Backtest logic: `src/research_engine.py`
- Indicator logic: `src/indicators.py`

If original optimization CSV/stress CSV is available, extract `params_json` directly from that CSV and compare against `STRATEGIES`. If CSV is not available, use the templates above as current source of truth.

## 3 Strategy Parameters

### DOGEUSDT macd_momentum/8h
- allocation: 110
- risk: 5%
- direction_filter: price_ema100
- lookback: 48
- volume_min: 1.2
- atr_stop_mult: 2.0
- take_profit_r: 3.0
- max_holding_bars: 12
- stop_rule: swing
- adx_min: 20
- regime: low_vol
- breakeven_r: 1.0
- partial_tp_r: 1.0
- partial_tp_frac: 0.5

### SUIUSDT macd_momentum/4h
- allocation: 110
- risk: 5%
- direction_filter: ema_fast_stack
- lookback: 48
- volume_min: 2.0
- atr_stop_mult: 3.0
- take_profit_r: 4.0
- max_holding_bars: 24
- stop_rule: swing
- adx_min: 20
- regime: any
- partial_tp_r: 1.0
- partial_tp_frac: 0.5

### AVAXUSDT trend_pullback/8h
- allocation: 110
- risk: 5%
- direction_filter: price_ema100
- lookback: 48
- volume_min: 1.2
- atr_stop_mult: 2.0
- take_profit_r: 5.0
- max_holding_bars: 12
- stop_rule: swing
- adx_min: 0
- regime: low_vol
- pullback_ref: ema20
- partial_tp_frac: 0.5
- tolerance_pct: 0.006

## Backtest Logic That Must Match

### Direction Filter: `research_engine.direction_allowed()`
Relevant lines: `src/research_engine.py:292-326`
- `ema_fast_stack` means:
  - LONG: `ema10 > ema20 > ema50`
  - SHORT: `ema10 < ema20 < ema50`
- `price_ema100` means:
  - LONG: `close > ema100`
  - SHORT: `close < ema100`

### Entry Trigger: `research_engine.entry_trigger()`
Relevant lines: `src/research_engine.py:329-422`
- `macd_momentum`:
  - LONG: `macd_hist > 0 and macd_hist > prev_macd_hist`
  - SHORT: `macd_hist < 0 and macd_hist < prev_macd_hist`
- `trend_pullback`:
  - `near(close, row[pullback_ref], tolerance_pct)`

### Confirmation: `research_engine.confirmation_ok()`
Relevant lines: `src/research_engine.py:441-481`
- `volume_ratio >= volume_min`
- `atr_pct` within optional min/max
- `adx14 >= adx_min` if adx_min > 0
- Regime uses `rv_pct`, not `atr_pct`:
  - low_vol: `rv_pct <= 50`
  - high_vol: `rv_pct >= 50`

### Stop/Take: `research_engine.make_stop_take()`
Relevant lines: `src/research_engine.py:484-530`
- `stop_rule='swing'`:
  - LONG stop: `recent_swing_low`
  - SHORT stop: `recent_swing_high`
- TP uses actual risk distance:
  - LONG: `entry + take_profit_r * (entry - stop)`
  - SHORT: `entry - take_profit_r * (stop - entry)`

## Indicator Columns That Must Exist
From `src/indicators.py` and `research_engine.enrich_features()`:
- ema10, ema20, ema50, ema100, ema200
- atr14 with Wilder smoothing
- atr_pct
- rv_pct with rolling realized volatility percentile
- macd_hist with min_periods
- rsi14 with Wilder smoothing
- adx14 with Wilder smoothing
- volume_ratio
- recent_swing_high / recent_swing_low shifted by one bar to avoid lookahead

## Data Recorded
- `paper_state.json`: balance, positions, closed_trades, signal_log, equity_curve
- `paper_trades.csv`: closed trades, one row per trade
- Telegram report each run

## Known Verification Notes
- The old v2 implementation had real mismatches: MACD crossover, ATR-only stop, ATR-ratio regime, simplified ema_fast_stack.
- Current code was patched to use:
  - `ema10 > ema20 > ema50` for ema_fast_stack
  - `rv_pct` for regime
  - `recent_swing_low/high` shifted by one bar
  - Wilder ATR/RSI/ADX style smoothing
- Still verify independently.

## Known Missing/Needs Review
- Partial take-profit parameters exist, but verify whether current live fill simulation fully implements partial exits. If not, mark as missing.
- Slippage/fees may not be fully modeled in the GitHub paper trader. If absent, paper returns may be optimistic.
- Original stress CSV/result CSV is not present in repo; if needed, request it and compare `params_json` directly.

## Independent Verification Prompt
See `VERIFY_PROMPT.md` in this repo.
