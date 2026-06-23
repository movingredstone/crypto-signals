
import os, sys, json
sys.path.insert(0,'.')
os.environ.setdefault('TELEGRAM_TOKEN','dummy')
os.environ.setdefault('TELEGRAM_CHAT_ID','dummy')
import pandas as pd
from pathlib import Path
import paper_trader_github as pt
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

def run(config_path,outname):
    cfg=load_config(config_path)
    rows=[]
    for s in pt.STRATEGIES:
        exp={k:v for k,v in s.items() if k not in ['name','alloc']}
        interval=exp['interval']; symbol=exp['symbol']
        lookback=int(exp.get('lookback',20))
        df=load_or_download_klines(symbol, interval, '2023-01-01', '2026-06-01')
        df=enrich_features(df, interval, lookbacks=[lookback])
        rec=df.to_dict('records')
        for w,tr_s,tr_e,te_s,te_e in WINDOWS:
            test=backtest_experiment(rec, exp, cfg, te_s, te_e)
            rows.append({'config':config_path,'strategy':s['name'],'symbol':symbol,'window':w,'test_ret':test['return_pct'],'test_trades':test['trades'],'test_pf':test.get('profit_factor',test.get('pf',0))})
    out=pd.DataFrame(rows)
    p=Path('results/optimization')/outname
    out.to_csv(p,index=False)
    print('\nCONFIG',config_path,'SAVED',p)
    g=out.groupby('symbol').agg(avg_test=('test_ret','mean'),sum_test=('test_ret','sum'),pos=('test_ret',lambda x:int((x>0).sum())),n=('test_ret','count'),trades=('test_trades','sum')).sort_values('avg_test',ascending=False)
    print(g.to_string())
    pivot=out.pivot_table(index='window',columns='symbol',values='test_ret',aggfunc='mean')
    pivot['portfolio_avg']=pivot.mean(axis=1)
    print('\nWINDOWS')
    print(pivot.to_string())
    avg=pivot.portfolio_avg.mean(); pos=int((pivot.portfolio_avg>0).sum()); n=len(pivot); annual=(1+avg/100)**4-1
    print(f'PORTFOLIO avgQ={avg:+.3f}% pos={pos}/{n} annualized={annual*100:+.2f}% beats_sp500={avg>=2.41}')

run('config.yaml','deployed4_fixed_wf_0p5pct.csv')
run('config_5pct_tmp.yaml','deployed4_fixed_wf_5pct.csv')
