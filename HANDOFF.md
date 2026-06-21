# Crypto Paper Trading System — Handoff for Verification

## What This Is
A $330 paper trading system running on GitHub Actions, executing 3 backtest-verified crypto futures strategies. It runs every 4 hours, sends Telegram reports, and auto-commits trade data back to the repo.

**Repo:** https://github.com/movingredstone/crypto-signals
**Local:** ~/Desktop/hermes/investmentsystem

## Current State (2026-06-21)
- System deployed and operational
- Telegram reports working
- No trades yet — waiting for first signal
- No open positions
- Balance: $330.00

## 3 Strategies (All Backtest-Verified)

### DOGE macd_momentum/8h
- Allocation: $110 | Risk: 5% ($5.50) | Price: ~$0.08
- Backtest: PF=1.99, WR=67%, WF=3/3 passes, 60 trades, MDD=-0.90%
- Params: lookback=48, atr_stop_mult=2.0, take_profit_r=3.0, max_holding_bars=12, adx_min=20, regime=low_vol, volume_min=1.2, direction_filter=price_ema100, breakeven_r=1.0, stop_rule=swing

### SUI macd_momentum/4h
- Allocation: $110 | Risk: 5% ($5.50) | Price: ~$0.90
- Backtest: PF=2.54, WR=48%, WF=2/3 passes, 52 trades, MDD=-2.59%
- Params: lookback=48, atr_stop_mult=3.0, take_profit_r=4.0, max_holding_bars=24, adx_min=20, regime=any, volume_min=2.0, direction_filter=ema_fast_stack, stop_rule=swing, partial_tp_r=1.0, partial_tp_frac=0.5

### AVAX trend_pullback/8h
- Allocation: $110 | Risk: 5% ($5.50) | Price: ~$6.23
- Backtest: PF=1.90, WR=45%, WF=3/3 passes, 7/7 stress folds, 40 trades
- Params: lookback=48, atr_stop_mult=2.0, take_profit_r=5.0, max_holding_bars=12, adx_min=0, regime=low_vol, volume_min=1.2, direction_filter=price_ema100, stop_rule=swing, pullback_ref=ema20

## Files
- `paper_trader_github.py` — Main paper trading script (standalone, no local deps)
- `.github/workflows/signal.yml` — GitHub Actions workflow (4-hourly schedule)
- `paper_state.json` — Portfolio state (balance, positions, trade history, signal log, equity curve)
- `paper_trades.csv` — Closed trade log (appends one row per trade)
- `requirements_paper.txt` — pandas, numpy, requests
- `paper_signal.py` — Old simple signal checker (NOT used anymore, ignore)

## How The System Works
1. GitHub Actions triggers every 4 hours (cron: `0 */4 * * *`)
2. Ubuntu runner checks out the repo
3. Installs pandas, numpy, requests
4. Runs `python paper_trader_github.py`
5. Script fetches klines from Binance public API (fapi/v1/klines) — no API key needed
6. Calculates indicators (EMA, ATR, ADX, MACD, RSI)
7. Checks open positions for SL/TP/expiry → closes if triggered
8. Checks for new entry signals using strategy-specific logic
9. Sends Telegram report with balance, positions, stats
10. Commits updated `paper_state.json` and `paper_trades.csv` back to repo

## Your Verification Task

Verify that the strategy implementation in `paper_trader_github.py` correctly matches the backtest-verified parameters above. Specifically check:

1. **Signal Logic Fidelity**
   - `macd_momentum_signal()` — Does it use MACD histogram momentum (not simple MACD line crossover)? Does it apply the correct direction_filter? Does it check regime, ADX, and volume filters correctly?
   - `trend_pullback_signal()` — Does it check for pullback to reference EMA within a trend? Does it use price_ema100 direction filter? Are the entry conditions matching the backtest engine?

2. **Parameter Accuracy**
   - Are all strategy parameters hardcoded correctly? Cross-reference: lookback, atr_stop_mult, take_profit_r, max_holding_bars, adx_min, regime, volume_min, direction_filter, stop_rule, breakeven_r, partial_tp_r, partial_tp_frac
   - Do any parameters differ from what was verified in backtests?

3. **Position Sizing**
   - Formula: position = allocation * risk_pct / (stop_distance / entry)
   - Is this correct? Does it match the backtest engine's sizing?

4. **Risk Management**
   - SL/TP calculation: SL = entry ± ATR * atr_stop_mult, TP = entry ± ATR * atr_stop_mult * take_profit_r
   - Breakeven logic: move stop to entry after price moves breakeven_r * stop_distance in favor
   - Max holding bars enforced correctly?

5. **Data Recording**
   - Are trades being recorded with sufficient context (regime, ADX, ATR, volume ratio at entry)?
   - Is the equity curve being tracked daily?
   - Is the signal log capturing all signals (taken + skipped)?

6. **GitHub Actions Workflow**
   - Does it have correct permissions (contents: write)?
   - Is the schedule correct?
   - Are secrets properly referenced?

## How To Verify
```bash
# Clone and test
cd ~/Desktop/hermes/investmentsystem
git pull

# Run locally (won't send Telegram without real token)
TELEGRAM_TOKEN=test TELEGRAM_CHAT_ID=test python3 paper_trader_github.py

# Check the current state
cat paper_state.json | python3 -m json.tool
```

## Known Issues / Concerns
- The script implements a STANDALONE version of the strategies (must work without the full backtest engine's src/ imports). Verify that the standalone implementations are logically equivalent to the backtest engine's strategy logic.
- AVAX trend_pullback had a history of wrong params (direction_filter was ema_fast_stack instead of price_ema100, stop_rule was atr instead of swing). Verify current params are the CORRECTED ones.
- SUI has shorter price history than DOGE/AVAX (listed ~2024). The strategy was tested on available data only.

## What To Report Back
1. Any discrepancies between implementation and verified backtest parameters
2. Any bugs in signal logic
3. Any missing features that would affect paper trading accuracy
4. Risk assessment: is this system safe to run as-is for paper trading?
5. Recommended fixes (if any)
