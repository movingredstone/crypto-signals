#!/usr/bin/env python3
"""Volatility-adaptive strategy scout.

Idea: classify volatility first, then choose investment/trading style per regime.
For now this evaluates existing honest stress candidates by entry-time volatility buckets:
- low_vol: quiet market -> breakout setup may work after compression
- mid_vol: normal waves -> momentum/trend following
- high_vol: storm -> either reduce size or only take strongest directional setup

Outputs which symbols/families actually make money in each volatility state.
"""
import sys, glob, json, math
from pathlib import Path
sys.path.insert(0,'.')
import pandas as pd
from src.research_engine import load_config, load_or_download_klines, enrich_features, backtest_experiment_detailed

SYMBOLS=[
 'BTCUSDT','DOGEUSDT','SUIUSDT','AVAXUSDT','SOLUSDT','BNBUSDT','ETHUSDT','XRPUSDT','ADAUSDT','LINKUSDT','NEARUSDT','DOTUSDT','HBARUSDT',
 'APTUSDT','ARBUSDT','OPUSDT','FILUSDT','INJUSDT','ATOMUSDT','TRXUSDT','SEIUSDT','FETUSDT','WIFUSDT','TONUSDT','AAVEUSDT'
]
SUPPORTED={'macd_momentum','trend_pullback'}

def honest_df(df):
    pf_cols=[c for c in df.columns if c.endswith('_pf') and c[:4].isdigit()]
    r=df[(df.pos_folds>=5)&(df.total_trades>=40)].copy()
    if r.empty: return r
    def honest(row):
        vals=[]
        for c in pf_cols:
            try: vals.append(float(row[c]))
            except Exception: pass
        try: mpf=float(row.mean_pf)
        except Exception: return False
        return vals and all(math.isfinite(v) and v<=20 for v in vals) and math.isfinite(mpf) and mpf<10
    return r[r.apply(honest,axis=1)]

def load_candidates(symbol, max_per_symbol=12):
    paths=sorted(glob.glob(f'results/optimization/{symbol}_stress_*_fold_flat.csv'))
    if not paths: return []
    df=honest_df(pd.read_csv(paths[-1]))
    if df.empty: return []
    df=df.sort_values(['mean_return','pos_folds','mean_pf','total_trades'], ascending=[False,False,False,False]).head(max_per_symbol)
    out=[]
    for _,r in df.iterrows():
        try: exp=json.loads(r.params_json)
        except Exception: continue
        exp['symbol']=symbol; exp['family']=str(r.family); exp['interval']=str(r.interval)
        exp['_stress_mean_return']=float(r.mean_return); exp['_stress_pos_folds']=int(r.pos_folds); exp['_stress_trades']=int(r.total_trades); exp['_stress_pf']=float(r.mean_pf)
        out.append(exp)
    return out

def summarize_trades_by_vol(df, trades):
    if not trades:
        return []
    t=pd.DataFrame(trades)
    t['entry_ts']=pd.to_datetime(t.entry_time, utc=True).astype('datetime64[ns, UTC]')
    feat=df[['open_time','rv_pct','atr14','close']].copy()
    feat['open_time']=pd.to_datetime(feat.open_time, utc=True).astype('datetime64[ns, UTC]')
    feat['vol_bucket']=pd.qcut(feat['rv_pct'].rank(method='first'), 3, labels=['low_vol','mid_vol','high_vol'])
    merged=pd.merge_asof(t.sort_values('entry_ts'), feat.sort_values('open_time'), left_on='entry_ts', right_on='open_time', direction='backward')
    rows=[]
    for bucket, sub in merged.groupby('vol_bucket', observed=True):
        if len(sub)<5: continue
        gross_pos=sub[sub.net_pnl>0].net_pnl.sum(); gross_neg=-sub[sub.net_pnl<0].net_pnl.sum()
        pf=gross_pos/gross_neg if gross_neg>0 else float('inf')
        rows.append(dict(vol_bucket=str(bucket), trades=int(len(sub)), net=float(sub.net_pnl.sum()), avg_r=float(sub.r_multiple.mean()), win_rate=float((sub.net_pnl>0).mean()*100), pf=float(pf) if math.isfinite(pf) else 99.0))
    return rows

def main():
    cfg=load_config('config.yaml')
    rows=[]
    for si,symbol in enumerate(SYMBOLS,1):
        cands=load_candidates(symbol)
        if not cands:
            continue
        by_interval={}
        for e in cands: by_interval.setdefault(e['interval'],[]).append(e)
        data={}
        for interval, exps in by_interval.items():
            lookbacks=sorted({int(e.get('lookback',20)) for e in exps})
            df=load_or_download_klines(symbol, interval, '2023-01-01','2026-06-01')
            df=enrich_features(df, interval, lookbacks=lookbacks)
            data[interval]=df
        print(f'[{si}/{len(SYMBOLS)}] {symbol} candidates={len(cands)}', flush=True)
        for exp in cands:
            df=data[exp['interval']]
            trades, eq=backtest_experiment_detailed(df.to_dict('records'), exp, cfg, '2023-01-01','2026-06-01')
            for s in summarize_trades_by_vol(df, trades):
                rows.append({
                    'symbol':symbol, 'family':exp['family'], 'interval':exp['interval'], 'deployable':exp['family'] in SUPPORTED,
                    'stress_mean_return':exp['_stress_mean_return'], 'stress_folds':exp['_stress_pos_folds'], 'stress_trades':exp['_stress_trades'], 'stress_pf':exp['_stress_pf'],
                    **s,
                    'params':json.dumps(exp),
                })
    out=pd.DataFrame(rows)
    Path('results/optimization').mkdir(parents=True, exist_ok=True)
    out_path='results/optimization/volatility_adaptive_scout.csv'
    out.to_csv(out_path,index=False)
    print('\nSAVED '+out_path)
    if not out.empty:
        filt=out[(out.trades>=10)&(out.net>0)&(out.pf>=1.5)].copy()
        print('\nTOP VOL BUCKET EDGES')
        print(filt.sort_values(['net','pf','trades'], ascending=[False,False,False]).head(30)[['symbol','family','interval','deployable','vol_bucket','trades','net','avg_r','win_rate','pf','stress_folds','stress_mean_return']].to_string(index=False))
        print('\nDEPLOYABLE TOP')
        d=filt[filt.deployable]
        print(d.sort_values(['net','pf','trades'], ascending=[False,False,False]).head(20)[['symbol','family','interval','vol_bucket','trades','net','avg_r','win_rate','pf','stress_folds','stress_mean_return']].to_string(index=False))
if __name__=='__main__': main()
