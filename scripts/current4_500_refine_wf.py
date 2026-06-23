
import glob, json, math, sys
from pathlib import Path
sys.path.insert(0, ".")
import pandas as pd
from src.research_engine import load_config, load_or_download_klines, enrich_features, backtest_experiment

OUTPUT=Path('results/optimization')
SYMBOLS=['UNIUSDT','NEARUSDT','SOLUSDT','ADAUSDT']
SUPPORTED={'macd_momentum','trend_pullback'}
SP500_Q=2.41
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

def honest_df(df):
    pf_cols=[c for c in df.columns if c.endswith('_pf') and c[:4].isdigit()]
    r=df[(df.pos_folds>=5)&(df.total_trades>=40)&(df.family.isin(list(SUPPORTED)))].copy()
    def ok(row):
        vals=[]
        for c in pf_cols:
            try: vals.append(float(row[c]))
            except Exception: pass
        try: mpf=float(row.mean_pf)
        except Exception: return False
        return bool(vals) and all(math.isfinite(v) and v<=20 for v in vals) and math.isfinite(mpf) and mpf<10
    return r[r.apply(ok, axis=1)].copy() if not r.empty else r

def load_candidates(symbol):
    paths=sorted(glob.glob(str(OUTPUT/f'{symbol}_stress_*_fold_flat.csv')))
    if not paths: return []
    path=paths[-1]
    df=honest_df(pd.read_csv(path))
    out=[]
    for _,r in df.iterrows():
        try: exp=json.loads(str(r.params_json))
        except Exception: continue
        exp['symbol']=symbol; exp['interval']=str(r.interval); exp['family']=str(r.family)
        exp['_stress_path']=path; exp['_stress_mean_return']=float(r.mean_return); exp['_stress_pos_folds']=int(r.pos_folds); exp['_stress_total_trades']=int(r.total_trades); exp['_stress_mean_pf']=float(r.mean_pf)
        out.append(exp)
    return out

def score_train(m):
    trades=int(m.get('trades',0)); ret=float(m.get('return_pct',0)); pf=float(m.get('profit_factor',m.get('pf',0)) or 0)
    if trades<12 or ret<=0 or not math.isfinite(pf) or pf<1.15: return -999
    return ret*0.45 + min(pf,5)*0.35 + min(trades,80)/80*0.6

cfg=load_config('config.yaml')
rows=[]
for si,symbol in enumerate(SYMBOLS,1):
    cands=load_candidates(symbol)
    print(f'[{si}/{len(SYMBOLS)}] {symbol}: candidates={len(cands)}')
    if not cands: continue
    by_interval={}
    for e in cands: by_interval.setdefault(e['interval'],[]).append(e)
    recs={}
    for interval, exps in by_interval.items():
        lookbacks=sorted({int(e.get('lookback',20)) for e in exps})
        df=load_or_download_klines(symbol, interval, '2023-01-01', '2026-06-01')
        df=enrich_features(df, interval, lookbacks=lookbacks)
        recs[interval]=df.to_dict('records')
    for w,tr_s,tr_e,te_s,te_e in WINDOWS:
        best=None
        for exp in cands:
            train=backtest_experiment(recs[exp['interval']], exp, cfg, tr_s, tr_e)
            sc=score_train(train)
            if best is None or sc>best[0]: best=(sc,exp,train)
        if best is None or best[0]<=-999:
            print(f'  {w}: no train-qualified candidate')
            continue
        sc,exp,train=best
        test=backtest_experiment(recs[exp['interval']], exp, cfg, te_s, te_e)
        pf=test.get('profit_factor', test.get('pf', 0))
        print(f"  {w}: {exp['family']}/{exp['interval']} train={train['return_pct']:+.2f}%/{train['trades']}t -> TEST={test['return_pct']:+.2f}% PF={pf:.2f} {test['trades']}t")
        rows.append({'symbol':symbol,'window':w,'family':exp['family'],'interval':exp['interval'],'train_ret':train['return_pct'],'train_trades':train['trades'],'test_ret':test['return_pct'],'test_pf':pf,'test_trades':test['trades'],'stress_mean_return':exp['_stress_mean_return'],'stress_pos_folds':exp['_stress_pos_folds'],'stress_mean_pf':exp['_stress_mean_pf'],'stress_total_trades':exp['_stress_total_trades'],'stress_path':exp['_stress_path'],'params':json.dumps(exp)})

out=pd.DataFrame(rows)
out_path=OUTPUT/'current4_500_refine_walk_forward.csv'
out.to_csv(out_path,index=False)
print('\nSAVED',out_path)
if not out.empty:
    g=out.groupby('symbol').agg(avg_test=('test_ret','mean'),sum_test=('test_ret','sum'),pos=('test_ret',lambda s:int((s>0).sum())),n=('test_ret','count'),trades=('test_trades','sum')).sort_values('avg_test',ascending=False)
    g['beats_sp500_q']=g.avg_test>=SP500_Q
    print('\nBY SYMBOL')
    print(g.to_string())
    p=out.pivot_table(index='window', columns='symbol', values='test_ret', aggfunc='mean')
    p['portfolio_avg']=p.mean(axis=1)
    print('\nPORTFOLIO WINDOWS')
    print(p.to_string())
    avg=p['portfolio_avg'].mean(); pos=int((p['portfolio_avg']>0).sum()); n=len(p); total=p['portfolio_avg'].sum(); annual=(1+avg/100)**4-1
    print(f"\nPORTFOLIO avgQ={avg:+.3f}% sumQ={total:+.3f}% pos={pos}/{n} annualized={annual*100:+.2f}% beats_sp500_q={avg>=SP500_Q} vs_sp500_q={SP500_Q:+.2f}%")
    print('\nFAMILY PICKS')
    print(out.groupby(['family','interval']).agg(picks=('symbol','count'),avg_test=('test_ret','mean'),pos=('test_ret',lambda s:int((s>0).sum())),trades=('test_trades','sum')).sort_values(['picks','avg_test'],ascending=[False,False]).to_string())
