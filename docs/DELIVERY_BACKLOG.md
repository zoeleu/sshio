# Delivery backlog

This backlog implements [the project plan](SSHIO_PLAN.md). Numeric settings remain versioned experiments. Each phase produces a reviewable result and a decision gate; it is not an instruction to implement the entire platform before the first playtest.

**Dependency clarification:** initial learned bots require no human data. P4.1/P4.2 can begin after the canonical core, observations, and replay are ready. Model packaging is reusable independently of P3.1's human-capture work. Ship a zero-human-data self-play run before making human ingestion a research dependency; optional imitation can use heuristic-generated demonstrations.

## Current implementation

The first Rust/SSH/Python feasibility implementation is on `feat/sshio-rust-foundation`; see [actual capabilities and limits](IMPLEMENTATION.md). P0.1 and P0.2 have local smoke coverage but are not fully accepted: latency/readability playtests and managed-Colab wheel installation remain. P0.3 remains open. [Recovered Wriggle](WRIGGLE_REFERENCE.md) is an existing product and code reference; compare against it before extending or replacing mature behavior.

## Phase 0 — resolve the expensive uncertainties

### P0.1: terminal/SSH experience spike

Build a minimal Russh/Ratatui session with the world-anchored dot background, circular boundary, two differently sized striped snakes, three food classes, and name/color/spectate modal. Drive one snake with portable direction keys and toggle boost. Parse remote input independently per session.

**Accept:** two real SSH clients can use it simultaneously; resize and disconnect do not affect the other; 80×24 and 120×40 remain readable; boost has an evident but non-flashing effect; direction changes work without key-up events. Test representative RTT impairment and record bytes/s and render CPU.

**Decision:** retain the proposed Rust terminal stack if control feel and output cost are promising. If not, change control/render strategy or evaluate a Go Wish gateway while preserving the simulation boundary.

### P0.2: simulation/native-package spike

Create a pure Rust reset/step/snapshot loop, expose a batch call through PyO3, build a release wheel, and install it in a clean Python environment and managed Colab. No SSH dependencies in the wheel.

**Accept:** the native executable and Python binding produce the same state hashes for the same seed/actions on the supported target; Python can step multiple worlds in one call; release wheels install without a Rust compiler; a CPU-only smoke run works.

### P0.3: hardware and export spike

Run a small actor forward/backward pass on the available Colab GPU and on the actual RTX PRO 6000 controlled runtime when available. Export a tiny recurrent or feedforward actor to ONNX and execute it through the proposed production runtime.

**Accept:** hardware report and locked environment are saved; unsupported GPU kernels fail clearly; CPU deployment output matches PyTorch within declared tolerance; no assumption that managed Colab supplied Blackwell is required.

## Phase 1 — canonical simulation and heuristic arena

### P1.1: movement, geometry, and common actions

Implement fixed-step motion, bounded turn rate, boost eligibility, mass-dependent dimensions, arc-length bodies, circular boundary, and explicit head/body collision rules.

**Accept:** swept collision fixtures and brute-force/spatial-hash differential checks pass; simultaneous outcomes do not depend on iteration order; rendering has no influence on state.

### P1.2: food and fairness ledger

Implement the integer mass ledger, reserve/spawn allocation, multivalue food, boost shedding, upkeep, death partition, expiration, redistribution, hard cap, and persistent crown timer.

**Accept:** randomized action/lifecycle sequences conserve mass exactly; no negative balances; all allocations sum correctly; no spawn underpayment; death bursts respect entity limits; humans/bots follow identical rules; crown cannot be reset through threshold dipping.

### P1.3: observation and heuristic policies

Build visibility-limited observations, candidate-action safety scoring, a diverse heuristic pool, reaction delay, and deterministic per-life policy state.

**Accept:** no offscreen/world-private input leaks; bots complete basic navigation/feeding scenarios; large-body and boundary fixtures do not deadlock the server; decisions are reproducible from seed and observation.

### P1.4: headless recording and replay

Record all inputs/lifecycle events, periodic checkpoints, build/rules/RNG versions, and state hashes. Add replay CLI and basic event inspection.

**Accept:** random and heuristic runs replay on the pinned target, restore midway, and continue identically; corrupted/missing chunks are rejected or explicitly marked; numeric compatibility claims are documented.

## Phase 2 — enjoyable SSH closed alpha

### P2.1: complete player lifecycle

Finish lobby, validation, safe spawn, HUD, help, retry, delayed spectate, reconnect behavior, and terminal capability fallback.

**Accept:** a new player can connect and play without an account; safe spawn cannot be exploited; deaths have readable reasons; unsupported input capabilities do not break portable controls.

### P2.2: polish and accessibility

Add stable stripe coordinates, width growth, tapered tails, clear heads, boost highlights/trails, food classes, world dots, palette conversion, and reduced motion.

**Accept:** small bodies and collision gaps are distinguishable at supported sizes; palette choice does not change collision geometry or hide a snake; monochrome/basic mode still communicates important state.

### P2.3: session load and fault handling

Introduce bounded command/snapshot/output queues, coalescing before ANSI encoding, actor ownership, resource limits, and restricted SSH channels. Add app metrics and persistent host keys.

**Accept:** slow/stalled clients do not stall ticks or grow memory without bound; resize resynchronizes diffs; malformed input is contained; denied channel types cannot open a shell or forward traffic; load-test report names hardware and workload.

### P2.4: fairness playtests

Run adversarial population simulations and consented human sessions. Sweep upkeep/crown/redistribution parameters, starting with one change at a time.

**Accept:** mass invariants hold through soaks; observed exploits have regression fixtures; playtest feedback identifies controls/deaths as understandable; retain a baseline report and a versioned initial ruleset. If the game is not fun, extend this phase before investing in larger models.

## Phase 3 — trustworthy data and imitation

### P3.1: capture eligibility and recording pipeline

Implement explicit contribution choice, research-enabled arena admission, pseudonymous IDs, safe metadata fields, recording quality flags, local spool, object upload, and withdrawal/deletion lineage.

**Accept:** a full multiplayer recording is eligible only under the declared consent scope; ordinary replays cannot enter training accidentally; no raw terminal input/keys/IPs in training shards; recording loss never blocks play and always marks affected data.

### P3.2: offline dataset builder

Replay with the matching core; rebuild observations and event-derived rewards; align human actions with available frame history; emit sequence shards and manifests; split by participant/session/time.

**Accept:** final observations and life boundaries are preserved; no split shares an arena session; action timing uncertainty is represented; data summaries separate humans, heuristics, policies, and control modes.

### P3.3: imitation baseline and model packaging

Optionally train a small actor on heuristic-generated trajectories; add human subsets only when available and eligible. Scratch RL can bypass imitation and still reuse the packaging work. Implement normalization, masks, recurrent resets if used, CPU export, and model manifest.

**Accept:** an imitation baseline overfits a tiny known synthetic dataset, generalizes better than a trivial action baseline on held-out data, executes in the server, and passes output/action parity. Complete synthetic generation → dataset → train → export → play in a clean environment without human recordings. Human capture validation is a separate gate for the human-data ingestion feature.

## Phase 4 — RL and safe learned-bot deployment

### P4.1: environment APIs and batched collector

Build finite PettingZoo Parallel episodes, the focal-agent Gymnasium wrapper, and a native persistent collector with explicit slot/life/reset masks. Release the GIL during batch simulation.

**Accept:** API tests pass; terminal and truncated trajectories bootstrap correctly; no returns/hidden state cross life boundaries; benchmark distinguishes arena steps from live-agent transitions.

### P4.2: PPO and curriculum

Adapt an attributed reference trainer, add structured observations and optional GRU, event-derived reward, historical opponents, and scenarios. Support random-weight initialization as well as optional heuristic imitation. Freeze actor versions per rollout, use independent recurrent state per snake, and train on the learner's own on-policy transitions rather than mixing in frozen opponents' samples. Log named reward components and policy versions.

**Accept:** a `synthetic_only` run generates its own experience, learns small tasks, evaluates against frozen opponents, exports, and executes in the game with zero human trajectories and no external demonstration download. Fixed-budget experiments compare scratch and heuristic-warm-start self-play across multiple seeds. Resuming a clean checkpoint restores optimizer/schedule/RNG/config and opponent pool; self-feeding/passivity tests do not produce misleading high reward.

### P4.3: Colab release workflow

Deliver thin notebooks, compatible wheels, hardware detection, adaptive presets, local shard staging, checksummed periodic checkpoints, and controlled-runtime documentation.

**Accept:** a fresh managed Colab session can run smoke and resume a run; a controlled Blackwell session uses verified supported kernels; interrupted uploads do not replace good checkpoints; claimed hardware and throughput are measured.

### P4.4: evaluation and promotion

Freeze suites/opponents, run paired multi-seed evaluations, review fairness/product effects, export and verify the candidate, and deploy shadow/opt-in/canary stages with rollback.

**Accept:** candidate meets a predeclared improvement criterion and latency budget, no correctness regressions, no actor leakage, and no unexplained fairness degradation. Production weights stay frozen per life and fallback decisions are logged.

## Phase 5 — web adaptation

### P5.1: protocol and native browser renderer

Add Axum WebSocket ingress, generated schemas, snapshot/ack sequencing, and PixiJS rendering using the same visual palette/geometry rules. Use DOM for menus and accessibility labels.

**Accept:** SSH and web players join the same canonical arena; server rejects illegal/malicious inputs; resizing cannot expand visibility; browser animation does not alter simulation speed; the UI supports name, color, play, spectate, and contribution preferences.

### P5.2: network smoothing and dataset validation

Add interpolation, optional local prediction/reconciliation, snapshot display reporting, and browser input provenance. Profile WebGL compatibility before enabling WebGPU-specific extras.

**Accept:** authoritative collision results remain consistent; reconnect works; hold/toggle boost adapters remain bounded; client-type differences are measured and recorded before pooling new data blindly with terminal trajectories.

## Phase 6 — prove platform reuse

Implement a small checkers game with a declared rules variant, legal-action masks, turn scheduler, terminal board UI, and PettingZoo AEC adapter. Reuse session/lobby/data/policy services without importing snake geometry. Then prototype agar-like split/merge behavior on the real-time side.

**Accept:** turn-based recording/replay and one training/evaluation example work; no fake physics ticks are needed; each game owns actions, rewards, observations, and rendering; shared APIs become simpler through evidence rather than accumulating speculative options.

## Release checklist

- User-visible rules, controls, privacy/contribution behavior, and bot labels match the implementation.
- Core invariants, replay fixtures, session resilience, and selected load tests pass on named targets.
- Ruleset, schema, wheel, dataset, and model versions are traceable.
- Live service can run without Colab or a training GPU.
- A clean notebook environment can reproduce the smoke training/evaluation/export loop.
- The first self-play training release runs without human datasets or a live player server; human-data ingestion has its own eligibility and deletion gates.
- Model/rules rollback and recording-failure behavior have been exercised.
- Known limits—terminal timing uncertainty, cross-target floating-point reproducibility, latency differences, and dataset selection bias—are documented accurately.
