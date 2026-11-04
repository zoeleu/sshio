#!/usr/bin/env python3
"""Summarize the additional frozen v4 comparison without selecting weights."""
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent

def main():
    authored = json.loads((HERE/'holdout.json').read_text())['results'][0]
    baselines = json.loads((HERE/'v4-holdout.json').read_text())['results']
    results = [authored] + baselines
    labels = ['Authored', 'v4', 'v4 personalities.best (Docker default)']
    lines = ['# Additional v4 comparison', '',
             'Same 56 seeds (730000–730055), seven scenarios and 1,200-tick horizon as '
             'the original frozen holdout. The authored results are reused, not rerun. '
             'No weights were changed. Each learner faces heuristic opponents.', '',
             '| Policy | Survived | Mean return | Mean food | Mean ticks | Kills | Circles |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for label,r in zip(labels,results):
        lines.append(f"| {label} | {sum(x['alive'] for x in r['runs'])}/56 | {r['mean_return']:.3f} | "
                     f"{r['mean_food']:.3f} | {r['mean_survival_ticks']:.2f} | {r['kills']} | {r['circles']} |")
    lines += ['', 'Paired return differences (authored minus baseline); approximate pooled '
              '95% t intervals with 55 df, critical value 2.004:','']
    for label,r in zip(labels[1:],baselines):
        assert [(x['seed'],x['scenario']) for x in authored['runs']] == [(x['seed'],x['scenario']) for x in r['runs']]
        differences = [a['return']-b['return'] for a,b in zip(authored['runs'],r['runs'])]
        mean=statistics.mean(differences); margin=2.004*statistics.stdev(differences)/math.sqrt(56)
        lines.append(f'* Versus {label}: {mean:+.3f}, interval [{mean-margin:+.3f}, {mean+margin:+.3f}].')
    lines += ['', '| Scenario | Authored survivors / return | v4 survivors / return | Docker v4 survivors / return |',
              '|---|---:|---:|---:|']
    for scenario in ('arena','cutoff','escape','border','forage','chase','production'):
        values=[]
        for r in results:
            runs=[x for x in r['runs'] if x['scenario']==scenario]
            values.append(f"{sum(x['alive'] for x in runs)}/8 / {statistics.mean(x['return'] for x in runs):.3f}")
        lines.append('| '+scenario+' | '+' | '.join(values)+' |')
    lines += ['', 'This finite suite is not a live all-neural tournament. Existing TrainingEnv '
              'personality assignment is unchanged: the learner is aggressive in compact '
              'three-opponent scenarios, conservative in paired scenarios, and afraid in production. '
              'It does not cover the greedy personality. The authored policy ignores personality; '
              'v4 consumes it. This comparison measures the existing evaluation interface, '
              'not every personality-specific training objective.', '',
              'Reproduce:', '', '```sh',
              'go run ./cmd/evalpolicy -policies models/wriggle-v4.json,models/wriggle-v4-personalities.best.json \\',
              '  -seed 730000 -episodes 56 -horizon 1200 -output authored/v4-holdout.json',
              'python3 authored/summarize_v4.py', '```', '',
              'Raw episode results and artifact hashes: [v4-holdout.json](v4-holdout.json).', '']
    (HERE/'V4-RESULTS.md').write_text('\n'.join(lines))
    rows=[]
    for i,(label,r) in enumerate(zip(labels,results)):
        count=sum(x['alive'] for x in r['runs'])
        rows.append(f'<div class="row"><span>{label}</span><div class="track"><div class="bar" '
                    f'style="width:{count/56*100:.4f}%;background:var(--chart-{i+1})"></div></div><b>{count}/56</b></div>')
    html='''<!doctype html><html><head><meta charset="utf-8"><style>
body{font:14px var(--font-sans,system-ui);color:var(--foreground);margin:0}p{line-height:1.5}
.plot{height:200px;display:flex;flex-direction:column;justify-content:space-around}
.row{display:grid;grid-template-columns:minmax(100px,1.4fr) 2fr 44px;gap:12px;align-items:center}
.track,.bar{height:20px;border-radius:3px}.track{background:var(--secondary)}
small{color:var(--muted-foreground);line-height:1.5}</style></head><body>
<p>Survival to 1,200 ticks · same 56 held-out seeds</p><div class="plot">'''+''.join(rows)+'''</div>
<small>Seven scenarios; heuristic opponents; frozen weights. v4 personalities.best is the Docker default. No tuning on these results.</small></body></html>'''
    (HERE/'v4-holdout.html').write_text(html)

if __name__=='__main__':
    main()
