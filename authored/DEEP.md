# Directly authored two-hidden-layer policy

The new artifact is `models/authored-deep.json`. It uses the existing optional
second-layer inference path: **692 → 64 tanh → 64 tanh → 14 affine**, with
**49,422 parameters**. This is 4,160 more parameters than the frozen single-hidden-
layer bot. No game, encoder, loader, action selection, or live controller was
changed. The user explicitly requested this two-layer experiment; the earlier
single-layer artifact remains unchanged.

```sh
python3 authored/generate_deep.py
python3 authored/generate_deep.py --check
go run . -bot-policy models/authored-deep.json
```

`generate_deep.py` starts with zero-filled tensors. It reuses **code** for the
previous analytically specified feature layer, then explicitly assigns the new
connections. It never reads a trained model, evaluates a simulation, or fits
parameters. Only Python's standard library is required.

The initial artifact, `models/authored-deep-initial.json`, was saved before any
behavioral evaluation. **Zero revision rounds** were used. Initial and final
files have the same SHA-256:

```
37942aede8996a543e65653335f9c197a56325da1927b08ee1ce3c14eeaf4f74
```

## Interface and scope

See [the original interface specification](README.md#exact-interface) for all
692 input indices, ranges, coordinate conventions, physics, scoring, and action
semantics. None changed. Version 3 is still used; personality columns and the
three older frames are still ignored. Both hidden layers use tanh; `wh` is
row-major `[64][64]`, `bh` has 64 entries. Output weights remain `[14][64]`.
The first-layer shapes are `[64][692]` and 64 biases. Argmax and its lowest-index
exact tie-break remain unchanged. This is two **hidden** layers, not two total
weight matrices; there are now three affine transformations.

## Neural circuit

Let `h` be the first hidden layer and `H` the second. For action a with relative
target angle theta, normalize the same angular tent as the single-layer bot:

```
q_s ∝ max(0, 1 - |s*pi/12 - theta|/(pi/3)), s=-8..8
R_a = sum_channels sum_s q_s * (1+h[3*(s+8)+channel])/2
```

The six outermost ray neurons have q=0 for every action, so they can be replaced
without changing R. R is the sum of three normalized body/head/border danger
averages, in [0,3]. The original distance thresholds remain (8,12,14) world
units and temperature widths (2,3,3). This is an angular warning field, not a
computed swept trajectory or collision probability.

### First hidden layer

* Units **3..47** retain body/head/border tanh danger detectors for rays -7..7.
* **51,52** retain `tanh(.5*best_food_x/y)`.
* **53,54** retain `tanh(.5*center_x/y)`.
* **55..63** retain `tanh(2*body_clearance)` for rays -4..4.
* The previously inactive units **0,1,2,48,49,50** become the six boost
  predicates below. Indices refer to the newest frame, offset 516:

| Unit | Exact pre-tanh computation | A positive value requires |
|---|---|---|
| 0 | `8*(obs[14]-.36)` | score above 30 |
| 1 | `12*(obs[16]-.10)` | food x greater than 4 world units |
| 2 | `-12*(obs[18]-.45)` | food distance below 27 world units |
| 48 | `6*(obs[19]-.12)` | augmented food value above 2.16 |
| 49 | `60*(obs[17]+.05)` | food y greater than -2 world units |
| 50 | `-60*(obs[17]-.05)` | food y less than +2 world units |

These are necessary conditions for boost, not sufficient thresholds. They must
jointly have high tanh confidence, and R must be extremely small. The food
encoder's existing inward proxy and clipped coordinates still apply.

### Second hidden layer and output

| Units | Computation | Output contribution |
|---|---|---|
| 0..6 | `H[a]=tanh(.01*R_a)` | `-12000*H[a]` to action a |
| `7+2*a`, `8+2*a` | `tanh(z_a+10*R_a)`, `tanh(z_a-10*R_a)` | 16 times their sum to action a |
| 21,22 | `tanh(.1*h[53/54])` | `20*(cos(theta)*H21+sin(theta)*H22)` |
| 23..31 | `tanh(.1*h[55..63])` | normalized 90-degree clearance tent, coefficient 5 |
| 32 | `tanh(4*(h0+h1+h2+h48+h49+h50-5.5)-80*R_3)` | +2 to straight boost, action 10 |
| 33..63 | zero | unused; retained 64-wide layer |

Here `z_a=.25*(cos(theta)*h51+sin(theta)*h52)`. Normal output bias is
`-.03*abs(theta)`. Boosted actions duplicate their corresponding non-boost row
and bias minus 1; only action 10 also gets `2*H32`. Thus turning boosts cannot
win, while straight boost beats straight non-boost iff H32>.5. It must still
beat the other six non-boost actions.

**Risk signal propagation.** `tanh(.01*R)` stays near-linear since its argument
is in [0,.03]. `-12000*tanh(.01*R)` differs from the original `-120*R` by at most
`12000*(.03-tanh(.03)) < .108` logit units. This uses the true tanh transfer,
not an assumption that tanh is an identity. Bias .015 and the half-weighted
ray inputs exactly assemble .01*R. No output clipping is applied.

**Conditional food attraction.** If R=0, the paired units give `32*tanh(z)`,
approximating the previous `8*(cos(theta)*h51+sin(theta)*h52)`. As R rises, the
opposite shifts saturate the two units toward +1 and -1, cancelling food's
contribution. The signal keeps its sign and never exceeds its R=0 magnitude.
For all legal food signals and R>=.3 its magnitude is below .06 logits. This
is an approximate multiplicative gate made from neural computations; it is
not an extra feature or runtime multiplier.

**Conditional boost.** Each predicate is <=1. If any predicate is <=0, their
sum is <=5, H32<=tanh(-2), and boost is impossible. Even six perfect predicates
cannot make boost win when R_3 >= `(2-atanh(.5))/80 ≈ .01813`. Maximum boost
advantage is +1 logit. The necessary score condition is stricter than the
engine's score>12 eligibility, so the circuit cannot select an ineligible boost
from valid observations. It does not predict a rival's next move; unexpectedly
entering hazards can still make a permitted boost dangerous.

**Safety dominance.** Food gating is now action-dependent. A conservative bound
on all non-danger action-score differences, including center, clearance,
turn preference, the maximum winning boost bonus, and the risk-relay error,
is **less than 14 logits**. Hence a represented R advantage >14/120 (~.11667)
dominates these other incentives. The Python tests verify this analytic bound.
This is weaker than a physical-safety guarantee: ray averaging, width mismatch,
moving opponents, and finite steering authority remain limitations. All
distance inputs used by the policy are existing encoder features; no exact
future occupancy, tail tracking, flood fill, or global planning is claimed.

The center/clearance relays also operate in their near-linear regions. Their
inputs have magnitude <=.04622 and <=.09641 respectively, so signs are preserved
and the clearance contribution remains bounded by `5*tanh(.1*tanh(2))<.481`.
All 64 first-layer units and 33 of the 64 second-layer units affect outputs.

## Diagnostics and evaluation usage

```sh
python3 -m unittest discover -s authored -p 'test_*.py' -v
go test ./game -run '^TestAuthoredDeep' -count=1 -v
AUTHORED_POLICY=../models/authored-deep.json go test ./game \
  -run '^TestAuthored(Diagnostics|Rotation|TailMovementAndGrowth)$' -count=1 -v
go test ./...
go vet ./...

# Development: only these seeds were available before final freeze.
go run ./cmd/evalpolicy -policies models/authored-deep-initial.json,models/authored.json \
  -seed 410000 -episodes 14 -horizon 1200 -output authored/deep-development.json

# New holdout, disjoint from all earlier development and evaluation seeds.
go run ./cmd/evalpolicy \
  -policies models/authored-deep.json,models/authored.json,models/wriggle-v4.json \
  -seed 830000 -episodes 56 -horizon 1200 -output authored/deep-holdout.json
```

The new candidate passed the same ten local action/physics fixtures, twelve
rotated equivalents, and tail-vacating/growth pair as the single-layer candidate.
As before, the outward-facing .05-unit border fixture is expected to die.
See [initial diagnostic output](deep-initial-diagnostics.txt).

New real-Go-forward-path tests verify dimensions/finiteness via the loader,
JSON round trip, the live staggered bot dispatch, food suppression, and eight
boost cases. The rich/aligned/clear case chooses action 10 and spends .8 score
over four ticks; insufficient score, distant/behind/lateral/poor food and a
rival ahead all decline boost and survive their four-tick fixture. Holding the
food fixed while setting body-clearance rays to zero reduces its straight-action
contribution from 1.190531 to approximately zero. See
[deep-circuit-diagnostics.txt](deep-circuit-diagnostics.txt).

Development: the initial deep candidate survived 12/14 episodes, mean return
15.277648, food 66.157143, two kills, one circle, boost spend 3.2. The frozen
single-layer baseline survived 11/14, mean return 15.262018, food 68.814286,
two kills, one circle, no boost. This is essentially tied reward, not evidence
of a general improvement. No parameter changes followed these results.

Final held-out results and limitations are in [DEEP-RESULTS.md](DEEP-RESULTS.md).
All episode data, hashes, and evaluation usage are saved. Existing evaluation
personality assignments and heuristic opponents are unchanged. This is not a
live all-neural tournament, and does not cover every personality independently.
