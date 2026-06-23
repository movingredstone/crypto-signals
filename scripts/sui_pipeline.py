#!/usr/bin/env python3
"""SUI full pipeline: broad → stress → walk-forward"""
import sys
sys.path.insert(0, ".")
from src.fold_evaluator import evaluate_folds
from src.optimizer import _run_stress
from src.research_engine import load_config, load_or_download_klines, enrich_features, backtest_experiment, backtest_experiment_detailed
from src.reports import profit_factor, max_drawdown
import pandas as pd
from pathlib import Path

HIGH_VOL_FOCUS = {
    "atr_stop_mults": [2.0, 2.5, 3.0, 3.5, 4.0],
    "take_profit_rs": [3.0, 4.0, 5.0],
    "lookbacks": [48, 96, 144],
    "max_holding_bars": [12, 24, 48, 72],
    "volume_mins": [1.5, 2.0],
}

if __name__ == "__main__":
    print("=" * 60)
    print("SUI FULL PIPELINE")
    print("=" * 60)
    
    # 1. Broad fold eval
    print("\n[1/3] Broad fold evaluation...")
    result = evaluate_folds(
        symbol="SUIUSDT", intervals=["4h", "8h", "1d"],
        experiments=1500, workers=9, top_n=50, seed=42,
        config_path="config.yaml", output_dir="results/optimization",
        mode="baseline", focus_override=HIGH_VOL_FOCUS,
    )
    df = result.get("df", pd.DataFrame())
    csv_path = result.get("csv_path", "")
    surv7 = int((df["pos_folds"] >= 7).sum()) if not df.empty else 0
    surv5 = int((df["pos_folds"] >= 5).sum()) if not df.empty else 0
    print(f"  Results: {len(df)}, 7/7={surv7}, 5/7+={surv5}")
    
    if not csv_path or not Path(csv_path).exists():
        print("  ERROR: No baseline CSV generated")
        sys.exit(1)
    
    # 2. Stress test
    print("\n[2/3] Stress test on top 50...")
    stress = _run_stress(
        symbol="SUIUSDT", intervals=["4h", "8h", "1d"],
        experiments=50, workers=9, output_dir=Path("results/optimization"),
        top_n=50, seed=42, config_path="config.yaml",
        source_top_path=csv_path,
    )
    
    stress_csv = stress.get("top_candidates_path", "")
    sdf = pd.read_csv(stress_csv) if Path(stress_csv).exists() else pd.DataFrame()
    surv6_stress = int((sdf["pos_folds"] >= 6).sum()) if not sdf.empty else 0
    
    # Best SUI candidate from stress
    clean = sdf[(sdf["total_trades"] >= 10) & (sdf["mean_pf"] < 50)] if not sdf.empty else sdf
    best = clean.sort_values("mean_return", ascending=False).iloc[0] if not clean.empty else None
    
    if best is not None:
        import json
        best_params = json.loads(best["params_json"])
        best_params["symbol"] = "SUIUSDT"
        interval = best_params.get("interval", "8h")
        
        print(f"\n  Best SUI: {best['family']}/{interval} "
              f"{int(best['pos_folds'])}/7 +{best['mean_return']:.2f}% PF={best['mean_pf']:.2f} {int(best['total_trades'])}t")
        
        # 3. Walk-forward
        print("\n[3/3] Walk-forward verification...")
        config = load_config("config.yaml")
        df_wf = load_or_download_klines("SUIUSDT", interval, "2023-01-01", "2026-06-01")
        df_wf = enrich_features(df_wf, interval, lookbacks=[best_params.get("lookback", 20)])
        records = df_wf.to_dict("records")
        
        # Full stats
        trades, eq = backtest_experiment_detailed(records, best_params, config, "2023-01-01", "2026-06-01")
        tdf = pd.DataFrame(trades)
        eqdf = pd.DataFrame(eq)
        ret = (eqdf["equity"].iloc[-1]/10000-1)*100
        pf = profit_factor(tdf)
        mdd = max_drawdown(eqdf["equity"])*100
        wr = (tdf["net_pnl"]>0).mean()*100
        
        print(f"\n  Full: {len(tdf)}t, WR={wr:.0f}%, PF={pf:.3f}, MDD={mdd:.2f}%, Ret={ret:+.2f}%")
        
        # Walk-forward folds
        folds = [
            ("Fold1","2023-01-01","2024-01-01","2024-01-01","2025-01-01"),
            ("Fold2","2024-01-01","2025-01-01","2025-01-01","2026-01-01"),
            ("Fold3","2023-07-01","2024-07-01","2025-01-01","2026-06-01"),
        ]
        wf_pos = 0
        for name,t1,t2,v1,v2 in folds:
            vr = backtest_experiment(records, best_params, config, v1, v2)
            status = "✅" if vr["return_pct"]>0 else "❌"
            if vr["return_pct"]>0: wf_pos += 1
            print(f"  {name}: Test {vr['return_pct']:+.2f}% PF={vr['profit_factor']:.2f} {vr['trades']}t {status}")
        
        print(f"\n  Walk-forward: {wf_pos}/3 positive")
        
        # Position sizing for $110
        stop_pct = 3.5  # estimate
        risk = 110 * 0.05
        pos = risk / (stop_pct/100)
        print(f"\n  $110 allocation, 5% risk:")
        print(f"    Risk: ${risk:.2f}, Stop: {stop_pct}%, Position: ${pos:.0f}, Leverage: {pos/110:.1f}x")
    
    print(f"\n{'='*60}")
    print("SUI PIPELINE COMPLETE")
    print(f"{'='*60}")
