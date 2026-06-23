#!/usr/bin/env python3
"""Fixed-parameter walk-forward for ARBUSDT supported candidates."""
import sys, glob, json
sys.path.insert(0,'.')
from pathlib import Path
import pandas as pd
from src.research_engine import load_config, load_or_download_klines, enrich_features, backtest_experiment

WINDOWS=[
 ('WF1','2023-01-01','2024-01-01','2024-01-01','2024-04-01'),
 ('WF2','2023-04-01','2024-04-01','2024-04-01','2024-07-01'),
 ('WF3','2023-07-01','2024-07-01','2024-07-01','2024-10-01'),
 ('WF4','2023-10-01','2024-10-01','2024-10-01','2025-01-01'),
 ('WF5','2024-01-01','2025-01-01','2025-01-01','2025-04-01'),
 ('WF6','2024-04-01','2025-04-01','2025-04-01','2025-07-01'),
 ('WF7','2024-07-01','2025-07-01','2025-07-01','2025-10-01'),
 ('WF8','2024-10-01','2025-10-01','2025-10-01','2026-01-01'),
 ('WF9','2025-01-01','2026-01-01','2026-01-01','2026-04-01'),
]

def main():
    cfg=load_config('config.yaml')
    path=sorted(glob.glob('results/optimization/ARBUSDT_stress_*_fold_flat.csv'))[-1]
    df=pd.read_csv(path)
    cands=df[(df.family=='macd_momentum')&(df.interval.isin(['4h','8h']))&(df.pos_folds>=5)&(df.total_trades>=40)].copy()
    cands=cands.sort_values(['mean_return','pos_folds','mean_pf'], ascending=[False,False,False]).head(3)
    rows=[]
    print(f'ARBUSDT fixed-param WF from {path}')
    for ci,r in cands.iterrows():
        exp=json.loads(r.params_json); exp['symbol']='ARBUSDT'; exp['family']=r.family; exp['interval']=r.interval
        dfk=load_or_download_klines('ARBUSDT', exp['interval'], '2023-01-01','2026-06-01')
        dfk=enrich_features(dfk, exp['interval'], lookbacks=[int(exp.get('lookback',20))])
        rec=dfk.to_dict('records')
        print(f"\nCAND idx={ci} {exp['family']}/{exp['interval']} stress_ret={r.mean_return:+.2f}% folds={int(r.pos_folds)}/7 PF={r.mean_pf:.2f} trades={int(r.total_trades)}")
        total=0; pos=0; trades=0
        for w,tr_s,tr_e,te_s,te_e in WINDOWS:
            test=backtest_experiment(rec, exp, cfg, te_s, te_e)
            pf=test.get('profit_factor', test.get('pf',0))
            total+=test['return_pct']; pos+=test['return_pct']>0; trades+=test['trades']
            rows.append(dict(candidate_idx=int(ci), family=exp['family'], interval=exp['interval'], window=w, test_ret=test['return_pct'], test_pf=pf, trades=test['trades'], params=json.dumps(exp)))
            print(f" {w}: TEST={test['return_pct']:+.2f}% PF={pf:.2f} trades={test['trades']}")
        print(f" summary: pos={pos}/9 avg={total/9:+.2f}% sum={total:+.2f}% trades={trades}")
    out='results/optimization/arbusdt_fixed_param_wf.csv'
    pd.DataFrame(rows).to_csv(out,index=False)
    print('SAVED '+out)
if __name__=='__main__': main()
