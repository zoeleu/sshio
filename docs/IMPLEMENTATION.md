# First feasibility implementation

This branch implements the beginning of Phase 0, with preliminary mechanics from Phase 1. The researched plan is a target architecture; this document describes what is actually present.

## Implemented boundaries

- `sshio-core`: seedable xorshift RNG, fixed 30 Hz reset/step, bounded bodies/food/population, bounded curvature, swept head/head and head/body contacts, simultaneous death resolution, integer reserve ledger, expiration/redistribution, contested pickup by distance, size cap, upkeep, crown expiry, deterministic state fingerprints, and visibility-limited heuristic steering.
- `sshio-server`: Russh gateway with persistent external host key, at most one session channel per connection, local bind default, 16 admitted connections, current-thread Tokio runtime, eight heuristic bots, per-client input/resize/lifecycle, and independent 10 Hz Ratatui output tasks. No process execution, subsystem, or forwarding handlers are enabled.
- Rendering: fixed world-space visibility, world-anchored dots, circular boundary, colorful food sizes, width/stripe variation, subdued boost stripe phase, and name/color/play/spectate dialog.
- `sshio-python`: PyO3 batch adapter calls the same Rust implementation, ten discrete relative actions, 76 local features, individual rewards and life boundaries. No alternate Python physics.

## Intentional spike limitations

This is not a production or ML release. It does not yet implement a spatial hash, snapshots/checkpoint replay, bot policy plug-ins, PettingZoo wrappers, historical leagues, PPO, ONNX, player data capture, reconnect, delayed spectators, camera smoothing, reduced-motion controls, palette fallback, or deployment packaging. The lightweight ASCII/CSI parser handles portable keys and fragmented arrows; it does not yet use the planned termwiz parser or implement enhanced keyboard negotiation.

Boost currently sends its entire charge to the reserve; the proposed 70/30 trail split is pending. Death loot is one temporary pellet containing 40% of mass rather than a bounded burst. Tail samples are spaced by tick distance and trimmed to length, rather than uniformly resampled arc-length geometry. There is no self-collision, as in Slither.io. The initial food allocation is generated subject to the total ledger, rather than fixed at exactly 8,000; subsequent replenishment targets 8,000. These differences need to be resolved and evaluated in P1 before claiming plan parity.

Terminal display names are restricted to 16 printable ASCII characters for now. Width/height are capped at 160×60; smaller than 40×16 is unsupported. Geometry and observations are the same regardless of terminal dimensions. A spectator follows the first living snake. Player tracking is direct; after death the camera holds the last observed position beneath a live UNRAVELED overlay.

The session output path renders after its previous send completes, has a bounded Russh event queue, and times out blocked sends. That is an initial bound, not evidence that every slow-consumer transport path is safe. Hostile connection handshakes, RTT loss, output bandwidth, stalled clients, and larger populations need measurement before exposing a public endpoint.

State hashes cover simulation-relevant state but do not serialize it. Floating point reproducibility is claimed only for the pinned executable/target; cross-CPU parity is not established. Xorshift is reproducible, not cryptographic. Changing geometry, RNG, or action semantics requires a rules version before recordings are introduced.

## Training interface

All actor slots in all worlds can use the current policy, trained from random initialization. The adapter automatically replaces dead lives after returning their zero terminal observation. Training code must use `observations()` for the next action and apply `done` to recurrent resets and returns. If a safe replacement spawn fails, the slot remains terminal and retries on the next step.

Rewards are an unvalidated baseline: log mass ratio minus 0.001 per decision, or −1 for death. Shared policy weights do not imply shared rewards. No training results or GPU speed claims are made.

## Validation record

The Rust unit checks and the initial native/Python parity and SSH smoke checks passed locally. Full details are recorded below after final verification. Colab/RTX PRO 6000 and ONNX acceptance remain unverified until the intended hardware/runtime is available. The hardware probe reports missing PyTorch clearly instead of assuming CUDA compatibility.

### Local evidence (2026-11-13)

- Rust workspace: four core regression tests, including a 900-tick paired deterministic run; single test worker. A Python-adapter regression across 16 seeds verifies that all returned transitions precede every respawn. Strict Clippy and formatting checks run on all targets.
- Native/Python parity: fingerprint `16002484208198803887` after the same seed, eight spawns, and three simulation ticks; another 100 paired decision steps check shapes, actions, conservation, and reproducibility.
- Built an ABI3 Python ≥3.10 Linux x86-64 wheel and installed it with no dependency downloads into a clean virtual environment on this host. This is a host-local wheel, not a verified manylinux/Colab artifact; portable release wheels remain a gate.
- Two real OpenSSH processes join and move; a split arrow sequence is accepted; a resize from 80×24 to 120×40 is checked by the resulting HUD row; one disconnect leaves the other active; remote exec fails promptly.
- Before the final Wriggle visual refinements, that short loopback SSH check recorded 10,276 KiB server peak RSS, one runtime thread, CPU affinity `0`, and 0.50 CPU seconds over 4.17 elapsed seconds. This debug-build observation is not a throughput target, frame-latency benchmark, or memory ceiling for larger populations.

No public deployment, player telemetry, GPU training, or external machine changes were performed. Managed Colab, actual Blackwell hardware, ONNX parity, impaired-network tests, human readability/fun, and production slow-consumer gates remain open.
