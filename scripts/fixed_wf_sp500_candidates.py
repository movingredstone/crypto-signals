#!/usr/bin/env python3
"""Fixed-parameter rolling walk-forward validation for current S&P500-beater candidates.
This is NOT re-optimizing inside each train window; it validates selected params across rolling train/test windows.
"""
import sys
sys.path.insert(0,'.')
from src.research_engine import load_config, load_or_download_klines, enrich_features, backtest_experiment

WINDOWS = [
    ("WF1", "2023-01-01", "2024-01-01", "2024-01-01", "2024-04-01"),
    ("WF2", "2023-04-01", "2024-04-01", "2024-04-01", "2024-07-01"),
    ("WF3", "2023-07-01", "2024-07-01", "2024-07-01", "2024-10-01"),
    ("WF4", "2023-10-01", "2024-10-01", "2024-10-01", "2025-01-01"),
    ("WF5", "2024-01-01", "2025-01-01", "2025-01-01", "2025-04-01"),
    ("WF6", "2024-04-01", "2025-04-01", "2025-04-01", "2025-07-01"),
    ("WF7", "2024-07-01", "2025-07-01", "2025-07-01", "2025-10-01"),
    ("WF8", "2024-10-01", "2025-10-01", "2025-10-01", "2026-01-01"),
    ("WF9", "2025-01-01", "2026-01-01", "2026-01-01", "2026-04-01"),
]
STRATEGIES = [
    {"name":"HBAR macd/8h","symbol":"HBARUSDT","interval":"8h","family":"macd_momentum","direction_filter":"ema200","lookback":20,"volume_min":1.0,"atr_stop_mult":2.0,"take_profit_r":2.0,"max_holding_bars":12,"stop_rule":"atr","adx_min":0,"regime":"low_vol","partial_tp_frac":0.5},
    {"name":"AVAX trend/8h","symbol":"AVAXUSDT","interval":"8h","family":"trend_pullback","direction_filter":"mtf_trend","lookback":48,"volume_min":0.7,"atr_stop_mult":1.5,"take_profit_r":3.0,"max_holding_bars":24,"stop_rule":"atr","adx_min":20,"regime":"any","trailing_atr_mult":2.0,"partial_tp_frac":0.5,"tolerance_pct":0.006,"pullback_ref":"ema20"},
    {"name":"DOT trend/4h","symbol":"DOTUSDT","interval":"4h","family":"trend_pullback","direction_filter":"ema200","lookback":144,"volume_min":1.0,"atr_stop_mult":4.0,"take_profit_r":3.0,"max_holding_bars":72,"stop_rule":"swing","adx_min":20,"regime":"low_vol","trailing_atr_mult":3.0,"partial_tp_frac":0.5,"tolerance_pct":0.006,"pullback_ref":"ema20"},
]

def main():
    cfg=load_config('config.yaml')
    print('Fixed-param rolling WF: train=12mo, test=3mo, windows=', len(WINDOWS))
    portfolio_tests=[]
    for s in STRATEGIES:
        df=load_or_download_klines(s['symbol'],s['interval'],'2023-01-01','2026-06-01')
        df=enrich_features(df,s['interval'],lookbacks=[int(s.get('lookback',20))])
        records=df.to_dict('records')
        print('\n'+s['name'])
        pos=0; test_sum=0; trades_sum=0
        for name,tr_s,tr_e,te_s,te_e in WINDOWS:
            train=backtest_experiment(records,s,cfg,tr_s,tr_e)
            test=backtest_experiment(records,s,cfg,te_s,te_e)
            tr_ret=train['return_pct']; te_ret=test['return_pct']; te_tr=test['trades']; te_pf=test.get('pf', test.get('profit_factor', 0))
            pos += te_ret>0
            test_sum += te_ret
            trades_sum += te_tr
            portfolio_tests.append((name,s['name'],te_ret,te_tr,te_pf))
            print(f" {name}: train {tr_s}~{tr_e} ret={tr_ret:+.2f}% | TEST {te_s}~{te_e} ret={te_ret:+.2f}% PF={te_pf:.2f} trades={te_tr}")
        print(f" summary: positive_tests={pos}/{len(WINDOWS)} test_sum={test_sum:+.2f}% avg_test={test_sum/len(WINDOWS):+.2f}% trades={trades_sum}")
    print('\nPORTFOLIO equal-weight by test window')
    by={}
    for w,n,r,t,pf in portfolio_tests:
        by.setdefault(w,[]).append((r,t))
    pos=0; total=0
    for w,vals in by.items():
        avg=sum(r for r,t in vals)/len(vals)
        tr=sum(t for r,t in vals)
        total+=avg; pos+=avg>0
        print(f" {w}: avg_test={avg:+.2f}% trades={tr}")
    print(f" portfolio summary: positive_tests={pos}/{len(by)} avg_3mo={total/len(by):+.2f}% summed_test={total:+.2f}%")
if __name__=='__main__': main()
