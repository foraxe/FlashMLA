import json
from pathlib import Path
from statistics import median
root=Path(__file__).parent
runs={label:json.loads((root/f'mrv2-{label}.json').read_text()) for label in ('a1','b1','b2','a2')}
ref=runs['a1']
summary={'runner':'MRV2','runs':{},'all_tokens_exact':True,'matched_steps':True}
for label,run in runs.items():
    assert len(run['results'])==8
    assert run['mixed_outputs']==ref['mixed_outputs'],label
    for row,control in zip(run['results'],ref['results']):
        assert row['token_ids']==control['token_ids'],(label,row['trial'])
        assert row['steps']['prefill']==1 and row['steps']['decode']==63,(label,row['steps'])
    for worker in run['diagnostic']:
        assert worker['projection_calls']['eager']==(40 if label.startswith('a') else 0),(label,worker)
        assert worker['replays']['run_pw_graph']==1,(label,worker)
        assert sum(v for k,v in worker['replays'].items() if k.startswith('run_fullgraph'))==2
    measured=[x for x in run['results'] if not x['warmup']]
    summary['runs'][label]={k:median(x[k] for x in measured) for k in ('ttft_ms','prefill_ms','tpot_ms')}
for arm in ('a','b'):
    rows=[x for label,run in runs.items() if label.startswith(arm) for x in run['results'] if not x['warmup']]
    summary[arm]={k:median(x[k] for x in rows) for k in ('ttft_ms','prefill_ms','tpot_ms')}
summary['change_pct']={k:(summary['b'][k]/summary['a'][k]-1)*100 for k in summary['a']}
summary['both_orders_favor_candidate']=all(summary['runs']['b'+i]['ttft_ms']<summary['runs']['a'+i]['ttft_ms'] for i in ('1','2'))
summary['no_material_decode_regression']=summary['change_pct']['tpot_ms']<2
assert summary['both_orders_favor_candidate'],summary
assert summary['no_material_decode_regression'],summary
(root/'mrv2-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
