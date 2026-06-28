# AUDIT CHANGES — 2026-06-27 (read this before touching the paper trader)

Independent quant audit of `paper_trader_github.py`. This file is the
inter-agent changelog: what was wrong, what changed, and what is still open.
If you are another agent picking this up, **start here**, then read
`HANDOFF.md`.

## Verdict (honest)
- At the **research 0.5% risk** baseline the portfolio does **NOT** beat the
  S&P500 quarterly hurdle (+1.865% avg vs +2.41% needed).
- The **5% risk** "win" (~+18.65%/qtr) was a **leverage illusion**: 5% is just
  10x the 0.5% baseline. Linear scaling leaves Sharpe unchanged — you took 10x
  the risk to clear the hurdle, which is not alpha. Anyone can lever the S&P
  10x too.
- Conclusion: the strategy does **not beat the benchmark on a risk-adjusted
  basis at any risk level**. To genuinely beat it you must raise Sharpe
  (decorrelate, kill overfit, keep fees in) — NOT raise leverage.

## Corrections to the prior internal review
- Fees/slippage **ARE modeled** (entry slippage, exit slippage, taker
  round-trip `notional*0.0005*2`). The earlier "fee omission" suspicion was
  wrong.
- Entry timing is **next-bar open** (`signal=df.iloc[-2]`, `entry=df.iloc[-1].open`).
  **No lookahead** on entry. Earlier suspicion wrong.

## Real defects found in code (the actual problem)
1. **No portfolio-level risk cap.** Each strategy sized independently at the
   hardcoded risk. `already_in` is keyed per strategy NAME, so the two SUI
   strategies could hold **two simultaneous SUIUSDT positions** = doubled
   directional exposure.
2. **No leverage cap.** `size_position = alloc*risk/stop_pct`. With a 2% stop
   and 5% risk, notional = 2.5x alloc. `config.max_leverage:2` was **ignored**.
3. **config.yaml risk controls were dead code** for the paper trader
   (`daily_loss_limit`, `weekly_loss_limit`, `max_trades_per_day`,
   `max_open_positions`). The trader hardcodes its own constants and enforced
   none of them.
4. **Overfit surface:** two SUI configs on the same symbol = curve-fit
   ensemble; LINK PF 1.26 over 32 trades = marginal/noise.

## What changed in `paper_trader_github.py`
| Item | Before | After |
|------|--------|-------|
| RISK_PER_TRADE | 0.05 | **0.01** |
| Strategies | 5 (SUI, SUI#2, LINK, XRP, DOGE) | **3 (SUI, XRP, DOGE)** |
| Allocations | 350/200/150/150/150 | **SUI 400 / XRP 300 / DOGE 300** |
| Leverage cap | none | **MAX_LEVERAGE = 2.0** (notional ≤ alloc×2) |
| Concurrent cap | none (≤5) | **MAX_CONCURRENT_POSITIONS = 3** |
| Portfolio risk cap | none | **MAX_PORTFOLIO_RISK = 0.04** (Σ open risk ≤ 4%) |
| Drawdown pause | none | **PAUSE_DRAWDOWN = 0.15** (halt new entries at −15%) |
| Telegram | — | shows risk/concurrent/cap + ⏸PAUSED + skipped reasons |

Removed: **SUI #2** (≈1.0 correlation to SUI #1) and **LINK** (marginal PF,
high_vol gate rarely fires). Strategy params themselves are unchanged — only
count, allocation, and risk plumbing changed.

## Why these numbers
- 1% risk trades the leverage "win" for **survivability**. Expected ~+3.7%/qtr,
  roughly benchmark-parity, MDD ~−10% instead of −30%+.
- Caps make the linear-scaling assumption actually hold: correlated alts can no
  longer stack into 25% aggregate risk.

## Honest probability of beating S&P after this change
| risk | beats S&P (absolute) | risk |
|------|------|------|
| 0.5% | no | very low |
| **1% (now)** | borderline/parity | MDD ~−10% |
| 2% | likely on avg | MDD ~−20% |
| 5% | yes (illusion) | ruin risk, not investible at $1k |
Risk-adjusted (Sharpe): does **not clearly beat** S&P at any level given
overfit + fees. Leverage ≠ edge.

## Still OPEN — do not mark done
1. **0 closed trades.** No live evidence yet. Gates may be too tight (price
   below EMA filters). Verify against `src/research_engine.py` whether the live
   signal rate matches backtest signal rate.
2. ~~Overfit not retired. Need trade-level bootstrap PF CI.~~ **DONE — see
   `BOOTSTRAP_PF.md`.** Result: only **SUI #1** has a robust PF>1 (95% CI
   [1.024, 2.367], barely); **XRP and DOGE are fragile** (CIs include ≤1; XRP
   has ~26% chance of being net-losing). The claimed forward PFs (1.919 /
   1.332 / 1.66) sit at only 13–21% bootstrap probability — they were an
   optimistic window, not the expected value. TVT split still pending.
3. **config.yaml is decoupled** from the trader (BTC/ETH research config vs
   SUI/XRP/DOGE live). Decide: drive the trader from config, or document the
   split. `config_5pct_tmp.yaml` is a dangerous leftover — delete it.
4. **commit_state() race:** `git pull --rebase` after commit with no
   error-check; fine at 4h cadence but fragile. Not changed.
5. To actually beat the benchmark: add a **decorrelated engine/timeframe**
   (e.g. keltner_breakout, or 8h) rather than more alt-momentum clones.

## Deploy note
These edits are in a local clone only. To go live: push to
`movingredstone/crypto-signals` main. The 4h GitHub Action picks up the new
`STRATEGIES`/constants automatically. Existing `paper_state.json` is
compatible (normalize_state backfills).
