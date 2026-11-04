# Measured results

Zero revisions. Initial and final weights have SHA-256 `d2409101346528deb8b458d6dbf47d8fc90c938cd1daf6eb5ace294509f5df9a`.

## Development (initial candidate)

14 episodes, seeds 310000–310013, 600-tick limit: all 14 survived; mean return 10.762, mean raw food 38.400, 2 kills and 0 circles. These seeds were available during construction review, so are not holdout.

## Frozen holdout

56 episodes per policy, seeds 730000–730055, 1,200-tick limit, eight episodes per scenario. Opponents use the existing heuristic. All three policies use the same initial seeds/scenarios, but their actions lead to different later worlds. No weights were changed or selected using these outcomes.

| Policy | Survived | Mean return | Mean raw food | Mean final score | Mean ticks | Kills | Circles |
|---|---:|---:|---:|---:|---:|---:|---:|
| Authored (initial = final) | 46/56 | 15.487 | 65.925 | 83.623 | 1094.05 | 13 | 15 |
| Existing heuristic | 40/56 | 14.547 | 72.468 | 81.020 | 1071.68 | 21 | 65 |
| Shipped wriggle-v6 | 20/56 | -4.440 | 11.407 | 32.141 | 643.27 | 4 | 275 |

Authored minus heuristic mean paired return is 0.940. An approximate paired t interval (55 df, critical value 2.004) is [-2.777, 4.656]. This pooled, fixed-scenario suite does not establish a general return improvement over the heuristic. The authored policy has higher observed survival here but less food and fewer kills. It performs substantially better than the shipped v6 file in this specific Go evaluation, which is not the GPU multi-personality training objective.

| Scenario | Authored survivors / return | Heuristic survivors / return | v6 survivors / return |
|---|---:|---:|---:|
| arena | 6/8 / 11.410 | 3/8 / 2.933 | 0/8 / -9.014 |
| cutoff | 8/8 / 15.357 | 6/8 / 13.322 | 3/8 / -4.948 |
| escape | 5/8 / 9.326 | 7/8 / 21.435 | 2/8 / -6.082 |
| border | 6/8 / 15.883 | 4/8 / 5.994 | 0/8 / -7.662 |
| forage | 7/8 / 21.420 | 6/8 / 24.111 | 1/8 / -5.229 |
| chase | 7/8 / 15.622 | 7/8 / 17.112 | 8/8 / 6.168 |
| production | 7/8 / 19.390 | 7/8 / 16.923 | 6/8 / -4.316 |

## Failures and scope

The authored candidate died in 10 holdout episodes, including 3/8 escape episodes. The heuristic had better escape, forage, and chase mean returns. The authored policy registered 15 circles, so it does not eliminate looping. It never boosted (zero measured boost spend); all its 13 kills were incidental to steering, not a programmed attack objective. None of the three policies recorded a mutual kill.

There is no global route/occupancy computation or opponent velocity extrapolation. The encoder misses some width-dependent collisions and does not label tails; the network ignores older frames and personality inputs. Its smooth, averaged danger field is not a guarantee against a moving snake or an unavoidable collision. Short local diagnostic success must not be read as long-horizon safety.

The evaluator measures one controlled learner against heuristic opponents. It does not measure a full population of these networks, human opponents, all possible personalities/objectives, or generalization beyond this finite suite.

## Checks

* Initial diagnostics: loader/round trip/live dispatch, ten action/physics fixtures, twelve rotated equivalents, and two tail/growth variants passed. The outward border fixture was expected to die and did.
* Python: three generator/reproduction/shape/finiteness/bound tests passed.
* `go test ./...`, `go vet ./...`, and `git diff --check` passed.
* Seed 730000 was repeated for all three policies, with exact per-episode metric matches. This is a three-episode repeatability check, not additional independent evidence.

Full raw episodes: [initial-development.json](initial-development.json), [holdout.json](holdout.json), [repeatability.json](repeatability.json). Evaluation usage: [evaluation-log.json](evaluation-log.json).
