#!/usr/bin/env python3
"""Candidate-pool rolling walk-forward hunt.
For each 12mo train / 3mo test window, select the best candidate using TRAIN ONLY, then evaluate the selected candidate on the next TEST window.
"""
import sys, glob, json, math
from pathlib import Path
sys.path.insert(0,'.')
import pandas as pd
from src.research_engine import load_config, load_or_download_klines, enrich_features, backtest_experiment

SYMBOLS=['BTCUSDT','DOGEUSDT','SUIUSDT','AVAXUSDT','SOLUSDT','BNBUSDT','ETHUSDT','XRPUSDT','ADAUSDT','LINKUSDT','NEARUSDT','DOTUSDT','HBARUSDT']
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
SUPPORTED={'macd_momentum','trend_pullback'}

def honest_df(df):
    fold_pf_cols=[c for c in df.columns if c.endswith('_pf') and c[:4].isdigit()]
    r=df[(df.pos_folds>=5)&(df.total_trades>=40)].copy()
    def honest(row):
        vals=[]
        for c in fold_pf_cols:
            try: vals.append(float(row[c]))
            except Exception: pass
        return vals and all(math.isfinite(v) and v<=20 for v in vals) and math.isfinite(float(row.mean_pf)) and float(row.mean_pf)<10
    return r[r.apply(honest,axis=1)]

def load_candidates(symbol):
    paths=sorted(glob.glob(f'results/optimization/{symbol}_stress_*_fold_flat.csv'))
    if not paths: return []
    df=honest_df(pd.read_csv(paths[-1]))
    out=[]
    for _,r in df.iterrows():
        try: exp=json.loads(r.params_json)
        except Exception: continue
        exp['symbol']=symbol; exp['interval']=str(r.interval); exp['family']=str(r.family)
        out.append(exp)
    return out

def score_train(m):
    # train-only selection score: require enough trades, reward return/PF, penalize too sparse
    trades=m['trades']; ret=m['return_pct']; pf=m.get('profit_factor', m.get('pf',0))
    if trades < 12 or ret <= 0 or not math.isfinite(pf) or pf < 1.15: return -999
    return ret*0.45 + min(pf,5)*0.35 + min(trades,80)/80*0.6

def main():
    cfg=load_config('config.yaml')
    all_results=[]
    print('Candidate-pool WF: 12mo train -> 3mo test, train selects candidate, windows', len(WINDOWS), flush=True)
    for symbol in SYMBOLS:
        cands=load_candidates(symbol)
        if not cands: continue
        by_interval={}
        for exp in cands: by_interval.setdefault(exp['interval'],[]).append(exp)
        records_by_interval={}
        for interval, exps in by_interval.items():
            lookbacks=sorted({int(e.get('lookback',20)) for e in exps})
            df=load_or_download_klines(symbol, interval, '2023-01-01','2026-06-01')
            df=enrich_features(df, interval, lookbacks=lookbacks)
            records_by_interval[interval]=df.to_dict('records')
        print('\n'+symbol+' candidates='+str(len(cands)), flush=True)
        pos=0; total=0; tr_total=0
        for w,tr_s,tr_e,te_s,te_e in WINDOWS:
            best=None
            for exp in cands:
                rec=records_by_interval[exp['interval']]
                train=backtest_experiment(rec, exp, cfg, tr_s, tr_e)
                sc=score_train(train)
                if best is None or sc>best[0]: best=(sc,exp,train)
            if best is None or best[0] <= -999:
                print(f' {w}: no train-qualified candidate')
                continue
            sc,exp,train=best
            test=backtest_experiment(records_by_interval[exp['interval']], exp, cfg, te_s, te_e)
            pf=test.get('profit_factor', test.get('pf',0))
            total += test['return_pct']; tr_total += test['trades']; pos += test['return_pct']>0
            all_results.append(dict(symbol=symbol,window=w,family=exp['family'],interval=exp['interval'],supported=exp['family'] in SUPPORTED,train_ret=train['return_pct'],train_trades=train['trades'],test_ret=test['return_pct'],test_pf=pf,test_trades=test['trades'],params=json.dumps(exp)))
            print(f" {w}: pick {exp['family']}/{exp['interval']} train={train['return_pct']:+.2f}%/{train['trades']}t -> TEST={test['return_pct']:+.2f}% PF={pf:.2f} {test['trades']}t", flush=True)
        n=sum(1 for r in all_results if r['symbol']==symbol)
        if n:
            print(f' summary {symbol}: pos={pos}/{n} avg_test={total/n:+.2f}% summed={total:+.2f}% trades={tr_total}', flush=True)
    out=pd.DataFrame(all_results)
    Path('results/optimization').mkdir(parents=True,exist_ok=True)
    out.to_csv('results/optimization/walk_forward_hunt_results.csv', index=False)
    print('\nOVERALL BY SYMBOL')
    if not out.empty:
        g=out.groupby('symbol').agg(avg_test=('test_ret','mean'),sum_test=('test_ret','sum'),pos=('test_ret',lambda s:int((s>0).sum())),n=('test_ret','count'),trades=('test_trades','sum'),supported_rate=('supported','mean')).sort_values('avg_test',ascending=False)
        print(g.to_string())
        print('\nTOP supported-rate>=0.8')
        print(g[g.supported_rate>=0.8].head(10).to_string())
    print('SAVED results/optimization/walk_forward_hunt_results.csv')
if __name__=='__main__': main()
