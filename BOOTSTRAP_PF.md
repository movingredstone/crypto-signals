# Bootstrap Profit Factor — deployed portfolio (audit OPEN #2)

Resolves OPEN #2 from `AUDIT_CHANGES.md`. Script: `scripts/bootstrap_pf_deployed.py`.

## Method
- Exact **deployed** params (SUI #1 / XRP / DOGE, post-audit portfolio).
- Backtest via `src.research_engine.backtest_experiment_detailed` over
  2023-01-01 .. 2026-06-01 → per-trade `net_pnl`.
- **Profit Factor** = Σ(gains) / |Σ(losses)|. Trade-level bootstrap, 10,000
  resamples with replacement, seed 42.
- PF is **scale-invariant** → identical at 0.5% or 5% risk. This is the honest,
  risk-agnostic edge metric.
- Data: **spot** OHLCV from `data-api.binance.vision` (Binance fapi is
  geo-blocked from CI/local, HTTP 451). Spot vs perp 4h candles differ
  marginally; this validates PF robustness, not exact P&L.

## Results (full sample, 10k resamples)

| Strategy | trades | WR | full PF | bootstrap median | 95% CI | P(PF>1) | P(PF ≥ claimed) | verdict |
|----------|-------:|----:|--------:|-----------------:|--------|--------:|----------------:|---------|
| SUI #1 | 186 | 39% | 1.590 | 1.579 | **[1.024, 2.367]** | 98.1% | 17.9% (≥1.919) | ROBUST (marginal) |
| XRP | 252 | 37% | 1.144 | 1.138 | [0.754, 1.636] | 73.9% | 20.8% (≥1.332) | **FRAGILE** |
| DOGE | 252 | 38% | 1.360 | 1.357 | [0.927, 1.914] | 94.7% | 13.0% (≥1.66) | **FRAGILE** |

## Honest interpretation
1. **The claimed forward PFs were optimistic.** SUI 1.919 / XRP 1.332 / DOGE
   1.660 (the 2025-07..2026-06 window) each sit at only **13–21% probability**
   under the full-sample bootstrap. The good forward window was a favorable
   slice, not the expected value. Do not size on those numbers.
2. **Only SUI #1 clears PF>1 robustly** — and its CI lower bound is **1.024**,
   i.e. *barely*. The edge is real but thin.
3. **XRP and DOGE are statistically fragile.** Both 95% CIs include ≤1; XRP has
   a **26% chance of being a net-losing strategy**. They are low-confidence.
4. Low win rate (37–39%) is expected for a 10R / 4R momentum design — survival
   depends on the long right tail holding up, which is exactly what bootstrap
   stresses. SUI's tail survives; XRP/DOGE's is shakier.

## Implication for the portfolio
- SUI #1 is the only statistically supported alpha → it correctly carries the
  largest weight (400/1000).
- XRP and DOGE should be treated as **low-confidence satellites**, not relied
  on to beat the benchmark. Candidates for reduced weight or replacement by a
  decorrelated engine (see OPEN #4).
- Combined with the Sharpe-invariance finding (`AUDIT_CHANGES.md`), this
  confirms: the portfolio does **not** have a robust, benchmark-beating edge at
  honest (risk-adjusted, full-sample) terms. SUI alone is marginally positive.

## Caveats
- Spot data, not perp; early SUI history is thin/illiquid (listed 2023-05).
- Full-period single backtest, not walk-forward — overstates in-sample edge if
  params were tuned on this period. Treat these CIs as an **upper bound** on
  confidence. Rolling WF (`scripts/*_wf.py`) remains the stricter test.
- `breakeven_r` / `partial_tp_r` are unset in deployed params (no partial TP),
  matching `STRATEGIES`.

## Reproduce
```
pip install pandas numpy pyyaml requests
python scripts/bootstrap_pf_deployed.py
```
