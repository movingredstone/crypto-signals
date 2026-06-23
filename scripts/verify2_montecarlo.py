#!/usr/bin/env python3
"""
VERIFICATION #2: Bootstrap Confidence Interval
Resample trade PnLs 10,000x with replacement.
If 95% CI of mean PnL doesn't include zero → real edge.
Also: t-test, skewness check, profit stability.
"""
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.research_engine import (
        load_config, load_or_download_klines, enrich_features,
        backtest_experiment_detailed,
    )
    import numpy as np

    # Simple t-test without scipy
    def simple_ttest(data, popmean=0):
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        se = std / np.sqrt(n)
        t = (mean - popmean) / se
        # Approximate p-value using normal distribution (valid for n > 30)
        from math import erf, sqrt
        def norm_cdf(x):
            return 0.5 * (1 + erf(x / sqrt(2)))
        p = 2 * (1 - norm_cdf(abs(t)))
        return t, p

    config = load_config("config.yaml")
    
    strategy = {
        "symbol": "BTCUSDT", "interval": "1h", "family": "keltner_breakout",
        "direction_filter": "mtf_trend", "lookback": 96, "volume_min": 0.7,
        "atr_stop_mult": 1.2, "take_profit_r": 3.0, "max_holding_bars": 96,
        "stop_rule": "swing", "adx_min": 0, "regime": "high_vol",
        "trailing_atr_mult": 3.0, "breakeven_r": None, "partial_tp_r": None,
    }

    print("=" * 60)
    print("VERIFICATION #2: BOOTSTRAP + STATISTICAL TESTS")
    print("Resample trades 10,000x. Check if edge is real.")
    print("=" * 60)

    df = load_or_download_klines("BTCUSDT", "1h", "2023-01-01", "2026-06-01")
    df = enrich_features(df, "1h", lookbacks=[96])
    records = df.to_dict("records")

    trades, _ = backtest_experiment_detailed(
        records, strategy, config, "2023-01-01", "2026-06-01"
    )

    pnls = np.array([t["net_pnl"] for t in trades])
    n = len(pnls)
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls, ddof=1)
    win_rate = np.mean(pnls > 0)
    
    print(f"\nTrade stats: {n} trades")
    print(f"  Mean PnL: ${mean_pnl:.3f} ± ${std_pnl:.3f}")
    print(f"  Win rate: {win_rate:.1%}")
    print(f"  Total: ${pnls.sum():.2f}")
    print(f"  Best: ${pnls.max():.2f}, Worst: ${pnls.min():.2f}")
    print(f"  Skewness: {np.mean(((pnls - mean_pnl)/std_pnl)**3):.3f} (positive=good)")

    # Test 1: One-sample t-test (H0: mean = 0)
    t_stat, t_pval = simple_ttest(pnls, 0)
    print(f"\n1. T-test: t={t_stat:.3f}, p={t_pval:.4f}")

    # Test 2: Bootstrap 95% CI
    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(10000):
        sample = rng.choice(pnls, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)
    pct_positive = np.mean(boot_means > 0) * 100
    
    print(f"2. Bootstrap 95% CI: [${ci_low:.3f}, ${ci_high:.3f}]")
    print(f"   % of bootstraps positive: {pct_positive:.1f}%")

    # Test 3: Profit factor bootstrap
    real_pf = pnls[pnls>0].sum() / abs(pnls[pnls<0].sum()) if (pnls<0).sum() != 0 else float('inf')
    boot_pfs = []
    for _ in range(10000):
        sample = rng.choice(pnls, size=n, replace=True)
        wins = sample[sample>0].sum()
        losses = abs(sample[sample<0].sum())
        boot_pfs.append(wins/losses if losses > 0 else 999)
    pf_ci_low = np.percentile(boot_pfs, 2.5)
    pf_ci_high = np.percentile(boot_pfs, 97.5)
    
    print(f"3. Profit Factor: {real_pf:.3f}")
    print(f"   Bootstrap 95% CI: [{pf_ci_low:.3f}, {pf_ci_high:.3f}]")

    # Test 4: Runs test for streak randomness
    # (are wins/losses randomly distributed?)
    signs = (pnls > 0).astype(int)
    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1]:
            runs += 1
    expected_runs = 1 + 2 * win_rate * (1 - win_rate) * n
    print(f"4. Runs test: {runs} runs (expected {expected_runs:.0f})")
    
    # Final verdict
    checks = [
        t_pval < 0.05,
        ci_low > 0,
        pf_ci_low > 1.0,
        pct_positive > 95,
    ]
    passed = sum(checks)
    
    print(f"\n{'='*60}")
    print(f"VERDICT: {passed}/4 checks passed")
    
    if ci_low > 0:
        print(f"✅ Bootstrap CI entirely above zero → REAL EDGE")
    elif t_pval < 0.05:
        print(f"✅ T-test significant → likely edge")
    else:
        print(f"❌ Cannot reject 'no edge' hypothesis")
