#!/usr/bin/env python3
"""Report frozen two-layer evaluation, without modifying or selecting weights."""
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent

def main():
    results=json.loads((HERE/'deep-holdout.json').read_text())['results']
    labels=['Authored two hidden layers','Authored one hidden layer','Shipped v4']
    lines=['# Frozen two-layer holdout results','',
           '56 new seeds per policy (830000–830055), seven scenarios, 1,200-tick limit, '
           'heuristic opponents. Both authored artifacts were frozen before this suite. '
           'The two-layer initial and final candidate are identical; zero revisions.', '',
           '| Policy | Survived | Mean return | Mean food | Mean ticks | Kills | Circles | Boost spent |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,r in zip(labels,results):
        lines.append(f"| {label} | {sum(x['alive'] for x in r['runs'])}/56 | {r['mean_return']:.3f} | "
                     f"{r['mean_food']:.3f} | {r['mean_survival_ticks']:.2f} | {r['kills']} | {r['circles']} | "
                     f"{sum(x['boost_spent'] for x in r['runs']):.2f} |")
    lines+=['','Approximate paired 95% return intervals (two-layer minus baseline, pooled '
            'over fixed scenarios, 55 df and critical value 2.004):','']
    for label,r in zip(labels[1:],results[1:]):
        assert [(x['seed'],x['scenario']) for x in results[0]['runs']]==[(x['seed'],x['scenario']) for x in r['runs']]
        d=[a['return']-b['return'] for a,b in zip(results[0]['runs'],r['runs'])]
        mean=statistics.mean(d);margin=2.004*statistics.stdev(d)/math.sqrt(56)
        lines.append(f'* Versus {label}: {mean:+.3f}; [{mean-margin:+.3f}, {mean+margin:+.3f}].')
    lines+=['','| Scenario | Two-layer survivors / return | One-layer survivors / return | v4 survivors / return |',
            '|---|---:|---:|---:|']
    for scenario in ('arena','cutoff','escape','border','forage','chase','production'):
        values=[]
        for r in results:
            runs=[x for x in r['runs'] if x['scenario']==scenario]
            values.append(f"{sum(x['alive'] for x in runs)}/8 / {statistics.mean(x['return'] for x in runs):.3f}")
        lines.append('| '+scenario+' | '+' | '.join(values)+' |')
    lines+=['','The extra layer successfully expresses the conditional food and boost circuits, '
            'but additional representational capacity alone does not establish better performance. '
            'Use the measured differences and uncertainty above; this is a finite evaluation, '
            'not proof of superiority over either baseline. No weights were revised after these outcomes.', '',
            'The two-layer candidate still died in six episodes, had more detected circles '
            'than the single-layer candidate (12 versus 8), and had slightly lower mean '
            'survival ticks despite one additional full-horizon survivor. Its arena and '
            'border survival were lower, while forage and production survival were higher. '
            'It does not uniformly improve the single-layer policy.', '',
            'Remaining limitations: fixed ray width and angular sampling; smooth danger averaging; '
            'no opponent velocity prediction or tail identity; no planning; unused temporal inputs. '
            'The extra layer does not recover missing observations. Boost can still be unsafe '
            'against an opponent that moves into a previously clear direction. Short diagnostics '
            'validate the implemented circuit conditions, not universal safety.', '',
            'Personality handling follows the existing evaluator (aggressive for compact '
            'three-opponent scenarios, conservative for pairs, afraid for production). '
            'The authored circuits ignore personality; the suite does not independently cover '
            'greedy behavior or personality-specific GPU rewards.', '',
            'Raw data: [deep-holdout.json](deep-holdout.json). '
            'Circuit, reproduction, and evaluation commands: [DEEP.md](DEEP.md). '
            'Evaluation usage: [deep-evaluation-log.json](deep-evaluation-log.json).','']
    (HERE/'DEEP-RESULTS.md').write_text('\n'.join(lines))
    rows=[]
    for i,(label,r) in enumerate(zip(labels,results)):
        count=sum(x['alive'] for x in r['runs'])
        rows.append(f'<div class="row"><span>{label}</span><div class="track"><div class="bar" '
                    f'style="width:{100*count/56:.4f}%;background:var(--chart-{i+1})"></div></div><b>{count}/56</b></div>')
    html='''<!doctype html><html><head><meta charset="utf-8"><style>
body{font:14px var(--font-sans,system-ui);color:var(--foreground);margin:0}p{line-height:1.5}
.plot{height:200px;display:flex;flex-direction:column;justify-content:space-around}
.row{display:grid;grid-template-columns:minmax(100px,1.4fr) 2fr 44px;gap:12px;align-items:center}
.track,.bar{height:20px;border-radius:3px}.track{background:var(--secondary)}
small{color:var(--muted-foreground);line-height:1.5}</style></head><body>
<p>Survival to 1,200 ticks · 56 new held-out seeds per policy</p><div class="plot">'''+''.join(rows)+'''</div>
<small>Seeds 830000–830055; seven scenarios; heuristic opponents. The two-layer candidate was frozen before this evaluation. No revisions.</small></body></html>'''
    (HERE/'deep-holdout.html').write_text(html)

if __name__=='__main__':
    main()
