#!/usr/bin/env python3
"""Automatic volatility-adaptive meta strategy walk-forward.

Loop:
1. Build trade library from honest, deployable candidates only (macd_momentum/trend_pullback).
2. Tag every historical trade with entry-time volatility bucket: low/mid/high.
3. For each 12mo train -> 3mo test window:
   - Within each vol bucket, rank candidates on TRAIN only.
   - Select the best candidate per bucket if it passes gates.
   - Evaluate only those selected candidate+bucket trades in TEST.
4. Report whether the adaptive rule beats S&P500 quarter hurdle.

No future leakage: test choices are based only on train window trades.
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
SP500_Q=2.41
INITIAL=10000.0

def honest_df(df):
    pf_cols=[c for c in df.columns if c.endswith('_pf') and c[:4].isdigit()]
    r=df[(df.pos_folds>=5)&(df.total_trades>=40)&(df.family.isin(SUPPORTED))].copy()
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

def candidate_rows(symbol, top_n=8):
    paths=sorted(glob.glob(f'results/optimization/{symbol}_stress_*_fold_flat.csv'))
    if not paths: return []
    df=honest_df(pd.read_csv(paths[-1]))
    if df.empty: return []
    df=df.sort_values(['mean_return','pos_folds','mean_pf','total_trades'], ascending=[False,False,False,False]).head(top_n)
    out=[]
    for idx,r in df.iterrows():
        try: exp=json.loads(r.params_json)
        except Exception: continue
        exp['symbol']=symbol; exp['family']=str(r.family); exp['interval']=str(r.interval)
        cid=f"{symbol}:{idx}:{exp['family']}:{exp['interval']}"
        out.append((cid, exp, dict(stress_return=float(r.mean_return), stress_pf=float(r.mean_pf), stress_folds=int(r.pos_folds), stress_trades=int(r.total_trades))))
    return out

def tag_trades(df, trades):
    if not trades: return pd.DataFrame()
    t=pd.DataFrame(trades)
    t['entry_ts']=pd.to_datetime(t.entry_time, utc=True).astype('datetime64[ns, UTC]')
    feat=df[['open_time','rv_pct']].copy()
    feat['open_time']=pd.to_datetime(feat.open_time, utc=True).astype('datetime64[ns, UTC]')
    # Causal volatility bucket: compare current rv_pct with prior rolling quantiles only.
    # This avoids future leakage from whole-sample ranking.
    rv=feat['rv_pct'].astype(float)
    low_q=rv.rolling(500, min_periods=100).quantile(0.33).shift(1)
    high_q=rv.rolling(500, min_periods=100).quantile(0.67).shift(1)
    feat['vol_bucket']='mid_vol'
    feat.loc[rv <= low_q, 'vol_bucket']='low_vol'
    feat.loc[rv >= high_q, 'vol_bucket']='high_vol'
    m=pd.merge_asof(t.sort_values('entry_ts'), feat.sort_values('open_time'), left_on='entry_ts', right_on='open_time', direction='backward')
    return m

def pf_of(s):
    pos=s[s>0].sum(); neg=-s[s<0].sum()
    if neg<=0: return 99.0 if pos>0 else 0.0
    return float(pos/neg)

def score_group(g):
    trades=len(g); net=float(g.net_pnl.sum()); pf=pf_of(g.net_pnl); wr=float((g.net_pnl>0).mean()*100) if trades else 0
    if trades < 8 or net <= 0 or pf < 1.2:
        return -999, dict(trades=trades, net=net, pf=pf, wr=wr, ret=net/INITIAL*100)
    # reward return, PF, sample size; cap PF artifacts
    score=(net/INITIAL*100)*0.55 + min(pf,5)*0.35 + min(trades,40)/40*0.7
    return score, dict(trades=trades, net=net, pf=pf, wr=wr, ret=net/INITIAL*100)

def build_library():
    cfg=load_config('config.yaml')
    all_trades=[]
    meta=[]
    total=sum(1 for s in SYMBOLS for _ in [s])
    for si,symbol in enumerate(SYMBOLS,1):
        cands=candidate_rows(symbol)
        if not cands: continue
        by_interval={}
        for cid,exp,st in cands: by_interval.setdefault(exp['interval'],[]).append((cid,exp,st))
        data={}
        for interval, items in by_interval.items():
            lookbacks=sorted({int(exp.get('lookback',20)) for _,exp,_ in items})
            df=load_or_download_klines(symbol, interval, '2023-01-01','2026-06-01')
            df=enrich_features(df, interval, lookbacks=lookbacks)
            data[interval]=df
        print(f'library [{si}/{total}] {symbol} candidates={len(cands)}', flush=True)
        for cid,exp,st in cands:
            df=data[exp['interval']]
            trades,_=backtest_experiment_detailed(df.to_dict('records'), exp, cfg, '2023-01-01','2026-06-01')
            tt=tag_trades(df,trades)
            if tt.empty: continue
            tt['candidate_id']=cid; tt['symbol']=symbol; tt['family']=exp['family']; tt['interval']=exp['interval']
            for k,v in st.items(): tt[k]=v
            all_trades.append(tt)
            meta.append({'candidate_id':cid,'symbol':symbol,'family':exp['family'],'interval':exp['interval'],**st,'params':json.dumps(exp)})
    if not all_trades:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(all_trades, ignore_index=True), pd.DataFrame(meta)

def run_wf(lib, meta, top_per_bucket=1):
    lib=lib.copy(); lib['entry_ts']=pd.to_datetime(lib.entry_ts, utc=True)
    rows=[]; picks=[]
    for w,tr_s,tr_e,te_s,te_e in WINDOWS:
        tr_s=pd.Timestamp(tr_s,tz='UTC'); tr_e=pd.Timestamp(tr_e,tz='UTC'); te_s=pd.Timestamp(te_s,tz='UTC'); te_e=pd.Timestamp(te_e,tz='UTC')
        train=lib[(lib.entry_ts>=tr_s)&(lib.entry_ts<tr_e)]
        test=lib[(lib.entry_ts>=te_s)&(lib.entry_ts<te_e)]
        selected=[]
        for bucket in ['low_vol','mid_vol','high_vol']:
            tb=train[train.vol_bucket.astype(str)==bucket]
            scored=[]
            for cid,g in tb.groupby('candidate_id'):
                sc,stats=score_group(g)
                if sc>-999:
                    sym=str(g.symbol.iloc[0])
                    scored.append((sc,cid,sym,stats))
            raw_scored=sorted(scored, reverse=True, key=lambda x:x[0])
            scored=[]
            used_symbols=set()
            for item in raw_scored:
                _sc,_cid,sym,_stats=item
                if sym in used_symbols:
                    continue
                used_symbols.add(sym)
                scored.append(item)
                if len(scored)>=top_per_bucket:
                    break
            for sc,cid,sym,stats in scored:
                selected.append((bucket,cid,sc,stats))
                m=meta[meta.candidate_id==cid].iloc[0].to_dict()
                picks.append({'window':w,'bucket':bucket,'candidate_id':cid,'score':sc,**stats,**{f'meta_{k}':v for k,v in m.items() if k!='params'}})
        # Evaluate selected on test: per bucket only candidate chosen for that bucket
        test_parts=[]
        for bucket,cid,sc,stats in selected:
            part=test[(test.vol_bucket.astype(str)==bucket)&(test.candidate_id==cid)]
            if not part.empty:
                test_parts.append(part)
        if test_parts:
            pt=pd.concat(test_parts, ignore_index=True)
            net=float(pt.net_pnl.sum()); trades=len(pt); ret=net/INITIAL*100; pf=pf_of(pt.net_pnl); wr=float((pt.net_pnl>0).mean()*100)
        else:
            net=0.0; trades=0; ret=0.0; pf=0.0; wr=0.0
        rows.append({'window':w,'test_ret':ret,'test_net':net,'test_trades':trades,'test_pf':pf,'test_wr':wr,'n_selected':len(selected),'selected':';'.join([f'{b}:{cid}' for b,cid,_,_ in selected])})
        print(f"{w}: ret={ret:+.2f}% net={net:+.2f} trades={trades} PF={pf:.2f} WR={wr:.1f}% selected={len(selected)}", flush=True)
        for bucket,cid,_,_ in selected:
            m=meta[meta.candidate_id==cid].iloc[0]
            print(f"  {bucket}: {m.symbol} {m.family}/{m.interval}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(picks)

def main():
    lib_path=Path('results/optimization/vol_meta_trade_library_causal.csv')
    meta_path=Path('results/optimization/vol_meta_candidate_meta_causal.csv')
    if lib_path.exists() and meta_path.exists():
        lib=pd.read_csv(lib_path); meta=pd.read_csv(meta_path)
        print(f'loaded cached library trades={len(lib)} candidates={len(meta)}')
    else:
        lib,meta=build_library()
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        lib.to_csv(lib_path,index=False); meta.to_csv(meta_path,index=False)
        print(f'saved library trades={len(lib)} candidates={len(meta)}')
    for k in [1,2,3]:
        print(f"\n=== VOL META WF top_per_bucket={k} ===")
        res,picks=run_wf(lib,meta,top_per_bucket=k)
        out=f'results/optimization/vol_meta_wf_top{k}.csv'; pout=f'results/optimization/vol_meta_wf_top{k}_picks.csv'
        res.to_csv(out,index=False); picks.to_csv(pout,index=False)
        avg=res.test_ret.mean(); pos=int((res.test_ret>0).sum()); trades=int(res.test_trades.sum()); annual=(1+avg/100)**4-1
        print(f"SUMMARY top{k}: avgQ={avg:+.2f}% annual={annual*100:+.1f}% pos={pos}/{len(res)} trades={trades} beats_sp500={avg>=SP500_Q}")
        print(f"SAVED {out} {pout}")
if __name__=='__main__': main()
