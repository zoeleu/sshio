# Neural Bot Training (NVIDIA CUDA & AMD ROCm)

Wriggle's reinforcement learning stack trains neural network snake policies via multi-agent self-play PPO on the GPU. Movement, sequential segment following, $1\times\text{–}4\times$ dynamic width scaling, head-to-head boost/size tie-breakers, closed-loop mass-budget food spawning, and 24-sector egocentric observations run directly on the GPU using PyTorch and custom **Triton kernels** (`training/gpu_kernels.py`, `training/gpu_sim.py`).

The GPU simulator is portable across **NVIDIA CUDA** and **AMD ROCm** GPUs. Meanwhile, the Go simulator (`cmd/trainenv`) serves as the authoritative validation and final-test environment, and exported JSON policies (`models/*.best.json`) execute in the live SSH server using pure Go with zero Python or GPU dependencies.

---

## 1. Installation

Requires Linux, Go 1.24+, Python 3.10–3.12, `uv` (or `pip`), and either an **NVIDIA GPU with CUDA** or an **AMD GPU with ROCm**.

### Option A: NVIDIA CUDA

```sh
uv venv --python 3.12 .venv-training
uv pip install --python .venv-training/bin/python torch numpy \
  --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA & Triton availability:
.venv-training/bin/python -c 'import torch, triton; print("CUDA:", torch.version.cuda, "Device:", torch.cuda.get_device_name(0))'
```

> **Note on CUDA Support (Untested / Community Port)**:
> `training/gpu_kernels.py` dynamically resolves `libdevice` across `triton.language.extra.cuda` and `triton.language.extra.hip`, and `training/train.py` supports `--device gpu` (auto-detect), `--device cuda`, and `--device rocm`.
> **Please note**: The development and CI environment uses an AMD GPU with ROCm (where all 13 unit, numerical parity, and integration tests have been executed and verified). The CUDA code path was ported to standard PyTorch/Triton CUDA conventions but has **not been tested or verified on physical NVIDIA hardware**. Community PRs or testing verification on NVIDIA GPUs are welcome!

### Option B: AMD ROCm

```sh
uv venv --python 3.12 .venv-training
uv pip install --python .venv-training/bin/python 'torch==2.14.1+rocm7.2' \
  --index-url https://download.pytorch.org/whl/rocm7.2
uv pip install --python .venv-training/bin/python 'numpy==2.5.4'

# Verify ROCm & Triton availability:
.venv-training/bin/python -c 'import torch, triton; print("ROCm:", torch.version.hip, "Device:", torch.cuda.get_device_name(0))'
```

---

## 2. Policy Architectures (v1, v2, and v3)

The Go server (`game/policy.go`) and Python trainer (`training/train.py`) support three policy generations:

| Policy Version | Input Dim | Hidden Layer (shipped) | Actions | Description |
| :--- | :--- | :--- | :--- | :--- |
| **v1** | `49` | `64` (`tanh`) | `14` | Legacy single-frame 16-sector observation model (`models/wriggle.json`). |
| **v2** | `688` | `64` (`tanh`) | `14` | 4-frame stack (`4 * 172` features) with 24 spatial sectors, 4 food targets, and 4 nearest rivals (`models/wriggle-v3.best.json`). |
| **v3** | `692` | `64` (`tanh`) | `14` | **Multi-Personality Policy**: `688` stacked observation features + `4`-dim one-hot personality vector `[AGG, GRD, CNS, AFR]` (`models/wriggle-v4-personalities.best.json`). |

### Per-Frame Observation Schema (`172` features $\times$ `4` frames = `688` inputs)
Each frame encodes normalized egocentric features in `[-1, 1]`:
- **Self State (`8` features)**: Linear score, log-scaled score, speed, boost state, turn rate, outer-ring center pull `(dx, dy)` (gated to `0` inside `70%` of arena radius and ramping linearly from `0.70R` to `1.00R`), and previous action's turn/boost.
- **Top 4 Food Targets (`16` features)**: Egocentric `(cos, sin)` direction, normalized distance, and caloric value for the 4 highest-utility food orbs in range.
- **Nearest 4 Rivals (`52` features)**: Relative head position, relative velocity, heading, width/score ratio, distance, boost state, turn rate, head-on win indicator, and presence flag for the 4 closest enemy snakes.
- **24 Egocentric Polar Sectors (`96` features)**: 24 angular bins around the snake head measuring body clearance, rival head clearance, border range, and sector food density.

### Action Space (`14` Discrete Actions)
7 relative target headings (`-0.9, -0.4, -0.15, 0.0, +0.15, +0.4, +0.9` radians per decision) $\times$ 2 boost states (`boost off`, `boost on`). Bots make a decision every 4 ticks (`200ms`).

### Optional second hidden layer (`--hidden2`)

`--hidden2 M` inserts a second `tanh` layer (`obs -> hidden -> hidden2 -> 14`). The JSON export then gains `wh`/`bh` keys, which `game/policy.go` reads when present; exports without them load unchanged. Both widths are capped at `256`.

```sh
.venv-training/bin/python -u training/train.py --device gpu --personalities \
  --hidden 256 --hidden2 128 --output models/wriggle-v5-deep.json \
  --simulator gpu --no-gpu-graphs --arenas 16 --agents 32 --updates 350
```

Notes:
- It is **opt-in and unproven here**: no deep model has been trained to convergence, so compare `validation mean_return` against a `--hidden 64`/`--hidden 256` baseline run with the same seeds before adopting it.
- It cannot warm start from `--init-v1`/`--init-v2` donors (the layer shapes differ), so it trains from scratch with the teacher-imitation warm start.
- Go inference cost per decision, measured on an i7-8086K (`go test ./game -bench Policy`): `64` ≈ 43 µs, `256` ≈ 171 µs, `256x128` ≈ 201 µs. Bots decide every 4 ticks, so this is a small share of one core at 15 bots.

---

## 3. Multi-Personality RL (`--personalities`, Policy v3)

When `--personalities` is enabled, each snake slot `i` is deterministically assigned one of 4 personalities (`p = i % 4`), fed to the network as a 4-element one-hot vector at inputs `[688:692]`, and trained with its own distinct PPO reward function:

| ID | Tag | Archetype | Score Gain Weight | Boost Spend Weight | Kill Reward | Death Penalty | Step Penalty | Proximity Shaping (`dist < 30`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0` | **`AGG`** | Aggressive | `+0.25` | `0.10` | **`+45.0 * (victim_score / 12.0) + 2.5 * corpse_eaten`** | `-8.0` | `-0.01` | **`+0.03`** (approaching rival heads) |
| `1` | **`GRD`** | Greedy | **`+0.60`** | `0.45` | `+4.0` | `-10.0` | `-0.01` | `0.00` (pure foraging focus) |
| `2` | **`CNS`** | Conservative | `+0.30` | `0.35` | `+5.0` | **`-14.0`** | `-0.005` | `-0.02` (avoids tight scrapes) |
| `3` | **`AFR`** | Afraid | `+0.25` | `0.25` | `+2.0` | **`-18.0`** | `-0.002` | **`-0.06`** (flees nearby rivals) |

> **Note on Aggressive (`AGG`) Kill & Feast Dynamics**:
> For aggressive worms, the kill reward is **proportional to the size of the victim killed** (scaled by `victim_score / 12.0`, giving `+45.0` for a starter worm, `+225.0` for a score-60 worm, and `+450.0` for a score-120 giant). In addition, corpse pellets dropped by the victim track the killer's ID, rewarding the aggressive worm with an extra **`+2.5` per unit of corpse mass it devours** (`+5.0` per 2-value corpse pellet), directly linking prey hunting with consuming the resulting spoils.

---

## 4. Running Training

### Train Policy v3 (Multi-Personality) on GPU (CUDA or ROCm)

You can warm-start Policy v3 (`692` inputs) from an existing Policy v2 (`688` inputs) JSON export using `--init-v2` (which copies the `688`-input weights and zero-initializes the 4 new personality input columns), or train from scratch:

```sh
.venv-training/bin/python -u training/train.py --device gpu \
  --personalities \
  --init-v2 models/wriggle-v3.best.json \
  --output models/wriggle-v4-personalities.json \
  --simulator gpu --no-gpu-graphs --max-segments 1024 --max-pellets 32768 \
  --arenas 16 --agents 32 --updates 350 --rollout 128 \
  --curriculum mixed --validate-every 25 \
  --eval-episodes 28 --test-episodes 56 --eval-horizon 1200 \
  --eval-seed 10000000 --test-seed 20000000
```

- `--hidden N` sets the first hidden width (default `64`, max `256`), so the shipped models are `--hidden 64`.
- Use `--device gpu` to automatically select CUDA or ROCm, or pass `--device cuda` / `--device rocm` explicitly.
- To resume an interrupted `.pt` checkpoint, replace `--init-v2 ...` with `--resume models/wriggle-v4-personalities.pt`.

### Train Policy v2 (Single-Reward, 688 Inputs)

Omit `--personalities` to train a standard `688 -> 64 -> 14` v2 network (width set by `--hidden`, default `64`):

```sh
.venv-training/bin/python -u training/train.py --device gpu \
  --output models/wriggle-v3.json \
  --simulator gpu --no-gpu-graphs --max-segments 1024 --max-pellets 32768 \
  --arenas 16 --agents 32 --updates 500 --rollout 128 \
  --curriculum mixed --validate-every 25 \
  --eval-episodes 28 --test-episodes 56 --eval-horizon 1200
```

### Quick Smoke Test (2 Updates)

```sh
.venv-training/bin/python training/train.py --device gpu \
  --personalities \
  --output .training/smoke.json \
  --arenas 4 --agents 8 --updates 2 --rollout 8 --epochs 1 \
  --stage-every 1 --validate-every 1 --eval-episodes 7 \
  --test-episodes 7 --eval-horizon 60
```

### Monitoring Progress & Checkpoints

For an output path `models/wriggle-v4-personalities.json`, the trainer atomically writes:
- `models/wriggle-v4-personalities.json`: Latest network weights after every update.
- `models/wriggle-v4-personalities.best.json`: Validation-selected best network weights (ready for `./wriggle -bot-policy`).
- `models/wriggle-v4-personalities.pt` & `.best.pt`: Full PyTorch training checkpoints (optimizer, league snapshots, RNG states, step counters).
- `models/wriggle-v4-personalities.report.json`: Full configuration, per-update learner telemetry, validation curve, and final holdout test results.
- `models/wriggle-v4-personalities.status.json`: Live process PID, phase, completed updates, and best validation return.

---

## 5. Curriculum, Opponent League & Physics Parity

### Curriculum Stages (`--curriculum mixed`)

| Stage | Updates | Arena Scenarios |
| :--- | :--- | :--- |
| **0** | `0–49` | Food trails, outward-facing border escapes, open arena foraging |
| **1** | `50–99` | Stage 0 + paired cutoffs, chases, and escapes from larger snakes |
| **2** | `100+` | Stage 1 + sparse, dense, and production-density arenas |

### Self-Play Opponent League
In a 512-worm run (`16` arenas $\times$ `32` snakes), `384` snakes are active PPO learners, `64` run the GPU scripted food/clearance controller, and `64` sample from a league of up to 8 historical policy snapshots (`--league-size 8`, `--snapshot-every 10`).

### Go $\leftrightarrow$ GPU Physics Parity
The Triton GPU simulator (`training/gpu_sim.py` & `training/gpu_kernels.py`) matches the Go world (`game/world.go` & `game/snake.go`) across:
1. **4 Substeps per Tick (`dt = 0.25`)** with sequential segment chain following.
2. **$1\times\text{–}4\times$ Dynamic Width Scaling**: Width multiplier `w = 1 + 3 * sqrt(clamp((score - 12) / 180, 0, 1))`, scaling head collision radius, body collision radius, and food pickup radius.
3. **Head-to-Head Boost & Size Tie-Breaker**: Boosting snake beats non-boosting snake; if equal boost state, higher-score snake wins; equal score and boost state results in mutual death.
4. **Closed-Loop Mass Economy**:
   - Boost spends `0.10` score per substep (`0.40`/tick) and drops `0.4`-value trail pellets.
   - Corpse drops spawn `2.0`-value orbs every 3 segments.
   - Diminishing food returns (`1 / sqrt(score / 60)`) above `Score = 60` and metabolic upkeep decay (`0.0005 * (score - 120)`/substep) above `Score = 120`.
   - Global mass budget caps `pellets + snake_mass` at `pellet_target` with a `25%` ambient floor.
5. **Outer-Ring Border Warning**: `obs[4:6]` is `0` inside `70%` of the arena radius and ramps linearly to `1` at the arena wall.

---

## 6. Verification & Unit Tests

Run all Go and Python tests (including Go $\leftrightarrow$ GPU Triton kernel parity fixtures):

```sh
go test ./...
go vet ./...
.venv-training/bin/python -m unittest discover -s training -p "test_*.py" -v
```

---

## 5. Pausing, resuming and open-ended training (`wtrain`)

`train.py` runs a fixed number of updates. `training/wtrain` wraps it in a detached supervisor that runs it as an endless series of short resumable chunks, so a run can go on indefinitely and be paused whenever you like:

```sh
# Start (everything after -- goes to train.py). --max-updates 0 (default) = never stop.
training/wtrain start deep --chunk 50 -- --personalities --hidden 64 \
  --simulator gpu --no-gpu-graphs --arenas 16 --agents 32 --curriculum mixed \
  --eval-episodes 28 --eval-horizon 1200 --eval-seed 10000000

training/wtrain status deep        # or: list   (updates, latest reward, best validation)
training/wtrain pause  deep        # checkpoints and frees the GPU; waits until safe
training/wtrain resume deep        # continues from the checkpoint
training/wtrain stop   deep        # pause + daemon exits; `resume` restarts it (e.g. after a reboot)
training/wtrain logs   deep -f
```

- **What pause does:** SIGTERM to the trainer, which saves model + optimizer + league and exits, so at most the update in flight is lost. The supervisor keeps no GPU memory while paused. State, logs and per-chunk reports live in `.training/runs/<name>/`.
- **Existing checkpoints:** `--output models/foo.json` continues from `models/foo.pt` if it exists, e.g. to finish the interrupted 256-wide run: `start h256 --output models/wriggle-v5-h256.json -- --personalities --hidden 256 <same flags as before>`.
- **Chunk size** (`--chunk`, default 50) trades overhead for responsiveness: each chunk re-validates once on start and runs a short end-of-chunk test pass (`--chunk-test-episodes`, default 7). The wrapper owns `--output`, `--updates`, `--resume` and `--test-episodes`; keep the other flags (and the `--eval-*` ones) identical across a run so best-checkpoint tracking carries over.
- **Warm starts** (`--init-v1/--init-v2`) are applied on the first chunk only. Pausing before the warm start finishes discards that unfinished checkpoint, so the next resume redoes it.
- **Stopping at a total:** `--max-updates N` ends the run at N updates (`resume --max-updates M` raises or removes the cap). Repeated trainer crashes (3 in a row) mark the run `failed`; `resume` retries.
- `start --foreground` runs in your terminal instead; Ctrl+C saves and exits. Tests: `python -m unittest test_runner` (uses a fake trainer, no GPU).
