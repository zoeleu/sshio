# Authored Snake network

`models/authored.json` is a directly constructed **692 → 64 tanh → 14 linear**
policy (version 3), exactly the architecture of `models/wriggle-v6.json`. All
45,262 parameters start at zero; `generate.py` assigns the circuit below. It
uses no training, fitting, random weights, downloaded policies, or controller
outside the network. Existing game rules, encoders, inference, and action
selection are unchanged. The existing `-bot-policy` flag selects it:

```sh
python3 authored/generate.py
python3 authored/generate.py --check
# Server; connect with ssh -p 23234 localhost.
go run . -bot-policy models/authored.json
```

`models/authored-initial.json` was saved **before any behavioral evaluation**.
There were **zero revision rounds**. The final and initial artifacts are identical:

```
d2409101346528deb8b458d6dbf47d8fc90c938cd1daf6eb5ace294509f5df9a
```

The approach is inspired by [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/):
write the policy directly as neural computations. The numbers and circuits here
were constructed for this repository, not copied from that article or a trained model.

## Exact interface

Source of truth: `game/policy.go`, `game/world.go`, `game/snake.go`,
`game/bot.go`, `game/training.go`, and `game/vec.go`. Some training README
physics descriptions and command references are stale. In particular, the
current Go width law is linear, boost costs 0.2/tick, and `cmd/trainenv` and
`training/env.py` are absent. The new `cmd/evalpolicy` uses the available Go
API directly. Runtime default without `-bot-policy` remains the heuristic.

### Every input

A frame has 172 features. Input index is `172*f+i`, `f=0,1,2,3`, oldest
first. Newest offset is **516**. First observation repeats the current frame in
all four slots. Live bot observations are collected at decisions, every four
world ticks; TrainingEnv collects at the end of each four-tick action (or an
earlier terminal tick). Values are float64 in Go. `clip` below means clamp to
[-1,1]; all stated nonnegative ranges assume a live, valid state inside the arena.

Coordinates: head `h=(X,Y)`, heading in radians, forward `F=(cos(hdg),sin(hdg))`,
side `S=(-sin(hdg),cos(hdg))`. Local `(x,y)=(delta·F,delta·S)`; positive side is
positive angular rotation (clockwise when positive world Y is rendered downward).
No absolute heading or absolute X/Y is supplied.

| Within-frame index i | Meaning and exact normalization | Range |
|---|---|---|
| 0 | `clip(score/100)` | [0.12,1] |
| 1 | `clip(log(max(1,score/12))/3)` | [0,1] |
| 2 | `score>12` boost eligibility | {0,1} |
| 3 | `clip((radius-|h|)/40)` | [0,1], negative if outside |
| 4,5 | Local unit direction to center, multiplied by `clamp01((|h|/radius-.60)/.40)` | [-1,1]; zero at center and inner 60% |
| 6 | `clip(speed/1.6)`; fallback speed .8 before recorded motion | .5 or 1 in ordinary play |
| 7 | Current boosting, including pending input if score>12 | {0,1} |
| 8 | `clip(wrapped heading change in last tick/.18)` | [-1,1] |
| 9 | `clip(turnRate*.18/.9)` | [-.2,.2] in current physics |
| 10 | Previous boost flag | {0,1} |
| 11 | `clip(segment count/120)` | [0,1] |
| 12 | Nearest living rival head distance/60, capped at 1; 1 when absent | [0,1] |
| 13 | `score-floor(score)` | [0,1) |
| 14 | `clip((score-12)/50)` | [0,1] |
| 15 | `clip(|h|/radius)` | [0,1] |
| `16+4*k+0`, k=0..3 | Food target local x/40, clipped | [-1,1] |
| `16+4*k+1` | Food target local y/40, clipped | [-1,1] |
| `16+4*k+2` | Food target distance/60, clipped | [0,1] |
| `16+4*k+3` | Food target augmented value/18, clipped | [0,1] |
| `32+11*k+0`, k=0..3 | Rival head local x/40, clipped | [-1,1] |
| `32+11*k+1` | Rival head local y/40, clipped | [-1,1] |
| `32+11*k+2,3` | Rival velocity along F,S divided by 1.6, clipped | [-1,1] |
| `32+11*k+4,5` | cos,sin(rival heading - own heading) | [-1,1] |
| `32+11*k+6` | `min(1,rival score/(2*own score))` | [0,1] |
| `32+11*k+7` | `min(1,rival head distance/60)` | [0,1] |
| `32+11*k+8` | Rival boosting | {0,1} |
| `32+11*k+9` | Rival normalized turn rate | [-1,1] |
| `32+11*k+10` | Rival present | {0,1} |
| `76+4*r+0`, r=0..23 | Body clearance ray, distance/30 | [0,1] |
| `76+4*r+1` | Head clearance ray, distance/30 | [0,1] |
| `76+4*r+2` | Ray-circle intersection distance/40, capped at 1 | [0,1] inside arena |
| `76+4*r+3` | `min(1,sum(.3*pellet value/(1+distance)))` for pellets within 40 assigned to this angular sector | [0,1] |

The trailing inputs **688,689,690,691** encode aggressive, greedy, conservative,
afraid respectively. They are a one-hot vector; balanced gets all zeros.
`PersonalityAction` appends it to a 688-vector, or accepts an existing 692-vector.
This network assigns all four columns zero, intentionally using the same behavior
for every personality. Version 3 has no v2 observation modulation or logit biases.

Missing food/rival slots are all zero. Rivals sort by distance, then ID. Food
slots use existing `foodTargets`: value is pellet value plus `.3*min(15, mass
in its axis-aligned 12-unit patch)`, and utility is `value/(8+distance+12*angle)`.
It excludes pellets outside radius-5 and targets closer than 9 behind the head;
equal utility preserves pellet order. Targets outside radius-10 have their
**position** replaced by `h/2`, while stored distance/value remain those of the
pellet. Thus food coordinates can be an inward proxy. Patch ranking can break
rotation invariance in multi-pellet scenes; rotated diagnostics use isolated
pellets so this ambiguity is absent.

Ray r points at `heading+r*pi/12`. For every other living snake's segment,
if forward projection >=0 and absolute perpendicular offset <1.6, clearance
is the minimum of 30 and `max(0,projection-.85)`. Segment 0 enters head rays;
all other segments enter body rays. **Own segments are excluded.** Border
range is `-h·dir+sqrt(max(0,(h·dir)^2+radius^2-|h|^2))`. Food sectors use rounded
angle/(pi/12), wrapped modulo 24. The encoder's .85 subtraction and 1.6 strip
are fixed even though physical collision width grows.

### Network, loader, actions

JSON uses `version,w1,b1,w2,b2`; matrices are row-major `[out][in]`. Hidden
units compute `tanh(b1[j]+sum(w1[j][i]*obs[i]))`, then logits use an affine
output with no activation. No second hidden layer is present in this artifact.
The existing loader also accepts v1 (49 inputs), v2 (688 inputs), and optional
`wh,bh` for a second tanh layer, but neither those versions nor that extra
layer is introduced here. Hidden widths are validated at 1..256, the second
layer at <=256. The strict JSON loader rejects unknown fields, extra JSON,
wrong dimensions, and non-finite parameters, and limits reads to 8 MiB.
`SetPolicy` validates and copies the policy.

Actions 0..6 set a target heading relative to current heading by
`[-.9,-.4,-.15,0,.15,.4,.9]` radians. Actions 7..13 have the same turns and
request boost. Go inference uses **argmax**, with **lowest index winning exact
ties**; no sampling or action masking. A wrong observation length returns 3.
Boost eligibility is enforced by physics, not an action mask. The authored
boost logits duplicate their corresponding non-boost logits minus 1, so never win.

The target is held between decisions. Live bots decide when `(tick+ID)%4==0`;
TrainingEnv also holds each action for four ticks. Actual heading changes by
at most .045 radians each substep, .18/tick, or .72 over a four-tick decision.
Actions +/- .9 are therefore target commands, not instantaneous direction changes.

### Physics and objective

* Continuous circular arena; live radius 300, target 15 snakes, 3500 pellet
  budget. 20 ticks/sec, four substeps per tick. Speed .8/tick, boost 1.6/tick.
* Each substep moves all heads, then follows the chain sequentially, with
  maximum segment spacing .65. Collision then checks updated positions of
  **all rival segments including heads and tails**. There is no self-collision.
* Width is `1+3*clamp01((score-12)/180)`. Hit distance is strictly less than
  `.425*(own width+rival width)` (between .85 and 3.4). The head dies if its
  distance from the center is strictly greater than radius; head width is not
  added to the border check. There are no arena corners or grid walls; the
  corner/corridor tests construct bends/channels from other snakes' bodies.
* Mutual deaths credited to each other are resolved by boost advantage, then
  score advantage; equal boost and score leaves mutual death. This rule is
  implemented for mutually attributed collisions, not an explicit head-only test.
  Bodies can also block escape. Dead snakes drop value-2 pellets every third
  segment; bots respawn to restore population.
* Food pickup occurs **after all four collision substeps**, within
  `1.3+.45*(width-1)`. Lower IDs get first claim on contested food.
  Score gain is `rawGain/(1+max(0,score-60)/120)`; `FoodEaten` counts raw gain.
  Score>120 then loses `.002*(score-120)` per tick. There is no starvation timer
  or starvation death.
* Boost spends up to .2 score per tick, never below 12, and drops .4-value
  trail pellets every fourth tick. Length is `int(score)+8`, adjusted after
  movement/collision/food; growth appends copies of the **already moved** tail.
  A growth tick does not hold the tail stationary as in grid Snake. Opponent
  tails are neither identified nor given next-step motion in the observation;
  treating a currently occupied ray as dangerous is conservative, not a tail
  predictor. Own-tail occupancy is harmless, including during growth.
* Live snapshots rank by score (ID breaks ties), with no single episode reward.
  Existing authoritative `TrainingEnv` reward, used here without alteration:
  `0.3*score_delta + 6*kill_delta - .01` each decision and `-10` on death.
  Evaluation also records raw food, score, survival, kills, circles and boost.
  The Python/GPU personality training objectives are distinct and are **not**
  the Go evaluation objective reported here.

## Circuit specification and bounds

Only newest-frame inputs have nonzero connections. There is no memory in this
construction despite the available older frames. The 64 units are assigned as follows:

| Hidden units | Computation | Output role |
|---|---|---|
| `3*j+c`, j=0..16, c=0..2 | For rays `s=-8..8` (mod 24): `tanh((D_c-distance)/T_c)` | Body/head/border danger |
| 51,52 | `tanh(.5*best_food_x/y)` | Bounded directional attraction |
| 53,54 | `tanh(.5*center_x/y)` | Weak outer-ring inward attraction |
| 55..63 | `tanh(2*body_clearance)` for rays -4..4 | Weak preference for broader clearance |

The six gate units for outermost rays -8 and +8 have zero output weights: these
rays fall outside all seven action tents. They do not affect decisions; 58 units
are functionally active within the unchanged 64-unit architecture.

Danger thresholds `(D,T)` in world units are body `(8,2)`, head `(12,3)`,
border `(14,3)`. W1 coefficients are `-30/T` for head/body or `-40/T` for
border, bias `D/T`. They explicitly undo encoder normalization. The body
horizon exceeds the 3.2 units traveled before the next decision plus the
maximum 2.55 difference between true collision radius and encoder subtraction.
Head horizon additionally allows an approaching boosted rival's movement;
border horizon allows room for the minimum turning radius (~4.44). These are
conservative warning distances, not sufficient collision-avoidance conditions.

For action target angle theta, normalize the angular tent
`q_s=max(0,1-|s*pi/12-theta|/(pi/3))` over the 17 rays. Let
`g=(1+tanh((D-distance)/T))/2`. Each channel contributes
`-120*sum(q_s*g_s)` to the action logit, via output weight `-60*q_s` and
bias -60. All three channels together use bias -180. The same angular weights
are used for each channel; obstacles near the prospective heading and its
neighbors cost more. This is **local angular clearance scoring**, not trajectory
integration. It does not compute the curved swept path to that heading.

Food contributes `8*(cos(theta)*h51+sin(theta)*h52)`; center contributes the
same form with coefficient 2. `tanh(.5*x)` is odd and monotone, bounded by
`±tanh(.5)=±.462118`; it approximates `.5*x` with at most .037883 absolute
error on [-1,1]. Coordinate clipping and this separate-component compression
can change the direction of distant targets; no multiplication or exact atan2
is claimed. Clearance contributes a normalized 90-degree angular tent average
of h55..h63 times .5, in `[0,.5*tanh(2)]`. Finally `-.03*|theta|` gently favors
straight motion. There is no ReLU or hidden layer propagation beyond this
single tanh layer.

**Dominance:** across any two actions, the total food+center+clearance+turn
contribution differs by less than **9.5**. A simple conservative bound is
`10*tanh(.5)*((1-cos(.9))+2*sin(.9)) + .5*tanh(2) + .027 < 9.5`.
Thus an action with summed normalized danger at least `9.5/120 ≈ .07917`
lower must win, regardless of food, center, and clearance. The tests also
check pairwise bounds from the compiled coefficients. This is dominance of
the **represented continuous danger score**, not lexicographic physical safety:
a small ray-weight difference, unobserved obstacle, or two large competing
hazards can defeat it. `g=.5` at D, >.9975 at D-3T and <.0025 at D+3T; gates
are deliberately smooth and have small nonzero tails. Averaging can dilute an
isolated ray. No exact Boolean safety gate is asserted.

## Verification and evaluation

```sh
python3 -m unittest discover -s authored -p 'test_*.py' -v
go test ./game -run TestAuthored -count=1 -v
go test ./...
go vet ./...

# Original development suite (not holdout).
go run ./cmd/evalpolicy -policies models/authored-initial.json \
  -seed 310000 -episodes 14 -horizon 600 -output authored/initial-development.json

# Frozen final versus the existing heuristic and shipped v6 neural model.
go run ./cmd/evalpolicy \
  -policies models/authored.json,heuristic,models/wriggle-v6.json \
  -seed 730000 -episodes 56 -horizon 1200 -output authored/holdout.json
```

To regenerate the Markdown summary and HTML comparison chart from saved results,
run `python3 authored/summarize.py`.

Generator uses Python standard library only; evaluator and diagnostics need Go
1.24+ and the project's existing Go dependencies. JSON output includes artifact
hashes and every episode's seed, scenario, and metrics. Generator has no dependency
on evaluation files. Reproduction is byte-identical in the tested Python/libm
environment; floating trig constants may differ in last bits on another libm.

The suite rotates arena, cutoff, escape, border, forage, chase, production, using
the existing `Scenario` implementation. The three paired scenarios use one
opponent; other compact scenarios use three; production resets to radius 300
and fourteen opponents. Every opponent uses the existing full-state heuristic.
The heuristic baseline is recomputed each tick; neural policies act every four
ticks, exactly as these APIs specify. Teacher-label calculation is disabled
for every policy to avoid mutating the heuristic's goal memory during observation.
This is not the unavailable Python RPC evaluator and does not measure an
all-neural live population or all GPU personality objectives.

`initial-diagnostics.txt` records the original tests. Expected food left/right/
straight actions were 1/5/3; hazard-hidden-food/head-on/corner actions 6/0/6;
corridor/own-tail stayed straight. All survived four ticks except the intentionally
unavoidable outward border state (.05 units from the wall). Four rotations of
three asymmetric states match their original observations to 1e-10 and select
the same action. The tail/growth pair moved the rival tail from y=.1 to .9
in both cases, grew only when fed, and left the learner alive. These are short
local fixtures, not long survival guarantees. The live bot dispatch, personality
handling, strict loader validation, 45,262-parameter shapes/finiteness, exact
serialization round trip, generator reproduction, and signal bounds are tested.

Evaluation usage and measured results are in [RESULTS.md](RESULTS.md) and
[evaluation-log.json](evaluation-log.json). The holdout was not used to revise
weights. Remaining limits: fixed-width rays miss some large-body collisions;
ray gaps and clipped coordinates discard geometry; no opponent motion prediction,
no tail identity, no flood fill, no global planning, no Hamiltonian cycle, no
boost-based escape or kill tactics, and no use of temporal observations. Even
perfect action selection cannot recover states beyond the turning envelope.

Additional comparison with plain v4 and the Docker-default v4 checkpoint: [V4-RESULTS.md](V4-RESULTS.md).

The subsequent two-hidden-layer experiment is documented in [DEEP.md](DEEP.md); it leaves this original policy unchanged.
