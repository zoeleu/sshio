# Additional v4 comparison

Same 56 seeds (730000–730055), seven scenarios and 1,200-tick horizon as the original frozen holdout. The authored results are reused, not rerun. No weights were changed. Each learner faces heuristic opponents.

| Policy | Survived | Mean return | Mean food | Mean ticks | Kills | Circles |
|---|---:|---:|---:|---:|---:|---:|
| Authored | 46/56 | 15.487 | 65.925 | 1094.05 | 13 | 15 |
| v4 | 45/56 | 17.344 | 62.482 | 1116.62 | 43 | 37 |
| v4 personalities.best (Docker default) | 46/56 | 14.697 | 59.332 | 1101.11 | 22 | 73 |

Paired return differences (authored minus baseline); approximate pooled 95% t intervals with 55 df, critical value 2.004:

* Versus v4: -1.857, interval [-5.928, +2.214].
* Versus v4 personalities.best (Docker default): +0.789, interval [-2.587, +4.165].

| Scenario | Authored survivors / return | v4 survivors / return | Docker v4 survivors / return |
|---|---:|---:|---:|
| arena | 6/8 / 11.410 | 6/8 / 16.407 | 5/8 / 8.250 |
| cutoff | 8/8 / 15.357 | 7/8 / 22.141 | 8/8 / 12.419 |
| escape | 5/8 / 9.326 | 7/8 / 11.086 | 6/8 / 9.328 |
| border | 6/8 / 15.883 | 6/8 / 17.872 | 6/8 / 19.971 |
| forage | 7/8 / 21.420 | 6/8 / 23.743 | 6/8 / 22.188 |
| chase | 7/8 / 15.622 | 7/8 / 14.245 | 8/8 / 12.477 |
| production | 7/8 / 19.390 | 6/8 / 15.912 | 7/8 / 18.249 |

This finite suite is not a live all-neural tournament. Existing TrainingEnv personality assignment is unchanged: the learner is aggressive in compact three-opponent scenarios, conservative in paired scenarios, and afraid in production. It does not cover the greedy personality. The authored policy ignores personality; v4 consumes it. This comparison measures the existing evaluation interface, not every personality-specific training objective.

Reproduce:

```sh
go run ./cmd/evalpolicy -policies models/wriggle-v4.json,models/wriggle-v4-personalities.best.json \
  -seed 730000 -episodes 56 -horizon 1200 -output authored/v4-holdout.json
python3 authored/summarize_v4.py
```

Raw episode results and artifact hashes: [v4-holdout.json](v4-holdout.json).
