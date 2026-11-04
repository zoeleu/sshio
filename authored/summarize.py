#!/usr/bin/env python3
"""Render saved evaluation data; never changes or selects weights."""
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent

def main():
    development = json.loads((HERE / 'initial-development.json').read_text())['results'][0]
    results = json.loads((HERE / 'holdout.json').read_text())['results']
    labels = ['Authored (initial = final)', 'Existing heuristic', 'Shipped wriggle-v6']
    lines = ['# Measured results', '',
             'Zero revisions. Initial and final weights have SHA-256 '
             '`d2409101346528deb8b458d6dbf47d8fc90c938cd1daf6eb5ace294509f5df9a`.', '',
             '## Development (initial candidate)', '',
             f"14 episodes, seeds 310000–310013, 600-tick limit: all 14 survived; "
             f"mean return {development['mean_return']:.3f}, mean raw food {development['mean_food']:.3f}, "
             f"{development['kills']} kills and {development['circles']} circles. "
             'These seeds were available during construction review, so are not holdout.', '',
             '## Frozen holdout', '',
             '56 episodes per policy, seeds 730000–730055, 1,200-tick limit, eight episodes '
             'per scenario. Opponents use the existing heuristic. All three policies use the '
             'same initial seeds/scenarios, but their actions lead to different later worlds. '
             'No weights were changed or selected using these outcomes.', '',
             '| Policy | Survived | Mean return | Mean raw food | Mean final score | Mean ticks | Kills | Circles |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, r in zip(labels, results):
        alive = sum(x['alive'] for x in r['runs'])
        lines.append(f"| {label} | {alive}/56 | {r['mean_return']:.3f} | {r['mean_food']:.3f} | "
                     f"{r['mean_score']:.3f} | {r['mean_survival_ticks']:.2f} | {r['kills']} | {r['circles']} |")
    differences = [a['return'] - b['return'] for a,b in zip(results[0]['runs'], results[1]['runs'])]
    delta = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(len(differences))
    lines += ['', f'Authored minus heuristic mean paired return is {delta:.3f}. An approximate '
              f'paired t interval (55 df, critical value 2.004) is [{delta-2.004*se:.3f}, '
              f'{delta+2.004*se:.3f}]. This pooled, fixed-scenario suite does not establish a '
              'general return improvement over the heuristic. The authored policy has higher '
              'observed survival here but less food and fewer kills. It performs substantially '
              'better than the shipped v6 file in this specific Go evaluation, which is not the '
              'GPU multi-personality training objective.', '',
              '| Scenario | Authored survivors / return | Heuristic survivors / return | v6 survivors / return |',
              '|---|---:|---:|---:|']
    for scenario in ('arena','cutoff','escape','border','forage','chase','production'):
        values=[]
        for r in results:
            runs=[x for x in r['runs'] if x['scenario']==scenario]
            values.append(f"{sum(x['alive'] for x in runs)}/8 / {statistics.mean(x['return'] for x in runs):.3f}")
        lines.append('| '+scenario+' | '+' | '.join(values)+' |')
    lines += ['', '## Failures and scope', '',
              'The authored candidate died in 10 holdout episodes, including 3/8 escape episodes. '
              'The heuristic had better escape, forage, and chase mean returns. The authored '
              'policy registered 15 circles, so it does not eliminate looping. It never boosted '
              '(zero measured boost spend); all its 13 kills were incidental to steering, not '
              'a programmed attack objective. None of the three policies recorded a mutual kill.', '',
              'There is no global route/occupancy computation or opponent velocity extrapolation. '
              'The encoder misses some width-dependent collisions and does not label tails; '
              'the network ignores older frames and personality inputs. Its smooth, averaged '
              'danger field is not a guarantee against a moving snake or an unavoidable collision. '
              'Short local diagnostic success must not be read as long-horizon safety.', '',
              'The evaluator measures one controlled learner against heuristic opponents. '
              'It does not measure a full population of these networks, human opponents, '
              'all possible personalities/objectives, or generalization beyond this finite suite.', '',
              '## Checks', '',
              '* Initial diagnostics: loader/round trip/live dispatch, ten action/physics fixtures, '
              'twelve rotated equivalents, and two tail/growth variants passed. The outward '
              'border fixture was expected to die and did.',
              '* Python: three generator/reproduction/shape/finiteness/bound tests passed.',
              '* `go test ./...`, `go vet ./...`, and `git diff --check` passed.',
              '* Seed 730000 was repeated for all three policies, with exact per-episode metric '
              'matches. This is a three-episode repeatability check, not additional independent evidence.', '',
              'Full raw episodes: [initial-development.json](initial-development.json), '
              '[holdout.json](holdout.json), [repeatability.json](repeatability.json). '
              'Evaluation usage: [evaluation-log.json](evaluation-log.json).', '']
    (HERE/'RESULTS.md').write_text('\n'.join(lines))
    # Small self-contained comparison chart, suitable both as a file and inline.
    rows=[]
    for label,r in zip(labels,results):
        alive=sum(x['alive'] for x in r['runs'])
        rows.append(f'<div class="row"><span>{label}</span><div class="track"><div class="bar" '
                    f'style="width:{100*alive/56:.4f}%"></div></div><b>{alive}/56</b></div>')
    html='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:14px var(--font-sans,system-ui);color:var(--foreground);margin:0}
p{line-height:1.5}.plot{height:180px;display:flex;flex-direction:column;justify-content:space-around}
.row{display:grid;grid-template-columns:minmax(100px,1.4fr) 2fr 44px;gap:12px;align-items:center}
.track{height:20px;background:var(--secondary,#ddd);border-radius:3px}.bar{height:20px;background:var(--chart-1,#3985ba);border-radius:3px}
.row:nth-child(2) .bar{background:var(--chart-2,#b18b2e)}.row:nth-child(3) .bar{background:var(--chart-3,#9e63b5)}
small{color:var(--muted-foreground);line-height:1.5}</style></head><body>
<p>Survival to 1,200 ticks · 56 held-out episodes per policy</p><div class="plot" role="img" aria-label="Survivors: authored 46 of 56, heuristic 40 of 56, shipped v6 20 of 56">'''+''.join(rows)+'''</div>
<small>Seeds 730000–730055; seven scenarios; heuristic opponents. Initial and final authored weights are identical. This finite suite does not establish general superiority.</small></body></html>'''
    (HERE/'holdout.html').write_text(html)

if __name__=='__main__':
    main()
