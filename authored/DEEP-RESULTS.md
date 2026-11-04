# Frozen two-layer holdout results

56 new seeds per policy (830000–830055), seven scenarios, 1,200-tick limit, heuristic opponents. Both authored artifacts were frozen before this suite. The two-layer initial and final candidate are identical; zero revisions.

| Policy | Survived | Mean return | Mean food | Mean ticks | Kills | Circles | Boost spent |
|---|---:|---:|---:|---:|---:|---:|---:|
| Authored two hidden layers | 50/56 | 18.497 | 71.707 | 1131.25 | 23 | 12 | 28.00 |
| Authored one hidden layer | 49/56 | 18.199 | 71.011 | 1137.84 | 22 | 8 | 0.00 |
| Shipped v4 | 40/56 | 11.771 | 50.718 | 1036.66 | 26 | 46 | 0.00 |

Approximate paired 95% return intervals (two-layer minus baseline, pooled over fixed scenarios, 55 df and critical value 2.004):

* Versus Authored one hidden layer: +0.298; [-1.442, +2.038].
* Versus Shipped v4: +6.726; [+2.690, +10.762].

| Scenario | Two-layer survivors / return | One-layer survivors / return | v4 survivors / return |
|---|---:|---:|---:|
| arena | 6/8 / 17.503 | 7/8 / 20.577 | 4/8 / 9.238 |
| cutoff | 8/8 / 20.484 | 8/8 / 21.689 | 7/8 / 15.543 |
| escape | 8/8 / 16.684 | 8/8 / 17.424 | 7/8 / 9.667 |
| border | 7/8 / 15.134 | 8/8 / 15.859 | 4/8 / 7.058 |
| forage | 6/8 / 21.786 | 4/8 / 15.268 | 5/8 / 21.042 |
| chase | 8/8 / 16.211 | 8/8 / 16.290 | 8/8 / 12.916 |
| production | 7/8 / 21.678 | 6/8 / 20.288 | 5/8 / 6.934 |

The extra layer successfully expresses the conditional food and boost circuits, but additional representational capacity alone does not establish better performance. Use the measured differences and uncertainty above; this is a finite evaluation, not proof of superiority over either baseline. No weights were revised after these outcomes.

The two-layer candidate still died in six episodes, had more detected circles than the single-layer candidate (12 versus 8), and had slightly lower mean survival ticks despite one additional full-horizon survivor. Its arena and border survival were lower, while forage and production survival were higher. It does not uniformly improve the single-layer policy.

Remaining limitations: fixed ray width and angular sampling; smooth danger averaging; no opponent velocity prediction or tail identity; no planning; unused temporal inputs. The extra layer does not recover missing observations. Boost can still be unsafe against an opponent that moves into a previously clear direction. Short diagnostics validate the implemented circuit conditions, not universal safety.

Personality handling follows the existing evaluator (aggressive for compact three-opponent scenarios, conservative for pairs, afraid for production). The authored circuits ignore personality; the suite does not independently cover greedy behavior or personality-specific GPU rewards.

Raw data: [deep-holdout.json](deep-holdout.json). Circuit, reproduction, and evaluation commands: [DEEP.md](DEEP.md). Evaluation usage: [deep-evaluation-log.json](deep-evaluation-log.json).
