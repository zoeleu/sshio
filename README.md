# Wriggle

**Wriggle** is a real-time multiplayer snake arena played directly over SSH, built with Go, [Charm Wish](https://github.com/charmbracelet/wish), [Bubble Tea](https://github.com/charmbracelet/bubbletea), and [Lip Gloss](https://github.com/charmbracelet/lipgloss), paired with a custom **NVIDIA CUDA / AMD ROCm Triton GPU reinforcement learning stack** that trains personality-conditioned neural network bots via self-play PPO.

One authoritative game world ticks at 20 Hz (`50ms` per tick, 4 physics substeps per tick) and keeps the circular arena populated up to 15 snakes with either heuristic bots or pure-Go neural policies exported from GPU training.

---

## Features

- **Zero-Client SSH Multiplayer**: Players join from any modern terminal with `ssh -p 23234 <host>` — no browser or client install required.
- **2.5D Slither.io Terminal Rendering**:
  - Unicode half-block (`▀` / `▄`) sub-cell vertical resolution (`2x` vertical pixel density).
  - Cylindrical 3D body shading with top-left specular highlights, core tones, bottom-right shadow edges, ground drop-shadows, ribbed scale banding, and alternating high-voltage boost flash.
  - Directional eyes with white sclera and dark pupils tracking each snake's live `TargetHeading`.
  - Multi-tiered glowing food orbs with out-of-phase twinkle, orbital sub-cell wobble, and cross-shaped light halos on high-value corpse loot.
- **Corner Radar Minimap & Live Death Overlay**:
  - Real-time radar in the bottom-right corner displaying the circular arena boundary, `(0, 0)` center crosshair, your position (`@`), rival snakes, and dense corpse food clusters.
  - When you die, the `"YOU UNRAVELED"` overlay floats above the live, ticking arena so you can watch the aftermath before pressing `Enter` to respawn.
- **Balanced Mass-Conservation Economy & Physics**:
  - **Dynamic Thickness (`1x` to `4x` Width)**: Snakes grow up to `4x` their starting width as their score increases (`12` to `192+`), widening both their rendered body and collision/pickup hitboxes.
  - **Head-to-Head Tie-Breaker**: In head-on collisions, a **boosting** snake defeats a non-boosting snake; if boost states match, the **larger** snake (`Score`) wins and devours the smaller snake; equal-score + equal-boost collisions remain mutual kills.
  - **Closed-Loop Mass Cap**: Ambient food replenishment counts both existing pellets and all living snake mass against the arena's mass budget (with a `25%` ambient floor), preventing runaway score inflation.
  - **Entropic Sinks & Metabolic Upkeep**: Boost trails (`0.4` value) and corpse drops (`2.0` value per 3 segments) recycle only a fraction of lost mass, while giant snakes experience diminishing food returns (`Score > 60`) and mild metabolic upkeep (`Score > 120`).
- **Multi-Personality Neural Bots**:
  - Zero-CGO, pure-Go MLP inference (`tanh` MLP; shipped models are `692 -> 64 -> 14`, with optional second-hidden-layer support) running inside the 20 TPS game loop.
  - Four distinct bot archetypes trained simultaneously with personality-conditioned rewards:
    - **`AGG` (Aggressive)**: Hunts rival heads and attempts boost cutoffs.
    - **`GRD` (Greedy)**: Prioritizes high-value corpse loot and rapid growth.
    - **`CNS` (Conservative)**: Balances steady foraging with wide safety margins.
    - **`AFR` (Afraid)**: Evades nearby rivals and flees danger.
- **Portable GPU Self-Play Trainer (CUDA & ROCm)**:
  - Custom Triton GPU kernels (`training/gpu_kernels.py`) and vectorized PyTorch simulator (`training/gpu_sim.py`) step **512 worms across 16 parallel arenas** entirely on the GPU, supporting both **NVIDIA CUDA** and **AMD ROCm**.

---

## Quickstart

### Option 1: Run with Docker (Recommended for Deployment)

Build and start the server with Docker Compose (bundles the pre-trained multi-personality neural policy `models/wriggle-v4-personalities.best.json` by default):

```sh
docker compose up --build -d
```

Or build and run the container directly:

```sh
docker build -t wriggle:latest .
docker run -d --name wriggle \
  -p 23234:23234 \
  -v wriggle-ssh-keys:/app/.ssh \
  wriggle:latest
```

Connect from any terminal:

```sh
ssh -p 23234 localhost
```

### Option 2: Build and Run Natively (Go 1.24+)

Requires **Go 1.24+** and an OpenSSH client:

```sh
# Build binary
go build -o wriggle .

# Run with the multi-personality neural bot policy (v3, 692 inputs)
./wriggle -port 23234 -bot-policy models/wriggle-v4-personalities.best.json

# Or run with the single-reward neural policy (v2, 688 inputs)
./wriggle -port 23234 -bot-policy models/wriggle-v3.best.json

# Or run with the built-in heuristic bots (no flags needed)
./wriggle
```

Then connect in another terminal:

```sh
ssh -p 23234 localhost
```

### Server CLI Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-port <int>` | `23234` | TCP port for the SSH server (`1`–`65535`). |
| `-bot-policy <path>` | *(empty)* | Optional path to an exported JSON neural policy (`models/wriggle-v4-personalities.best.json` or `models/wriggle-v3.best.json`). Uses heuristic bots when omitted. |

> **Security Note**: On first launch, Wriggle generates an Ed25519 host key at `.ssh/wriggle_ed25519` (ignored by Git; persisted in a Docker volume when using `docker-compose.yml`). The server accepts SSH connections without passwords or keys because it is a public game endpoint and never exposes a shell.

---

## Controls & Gameplay

| Input | Action |
| :--- | :--- |
| `Enter` | Confirm nickname on title screen, respawn after dying, or enter arena from Spectator mode |
| `Tab` | Enter Spectator mode from title screen, in-game, or death screen; or cycle to next bot while spectating |
| `Arrow Keys` / `WASD` | Steer in 8 directions (press two perpendicular keys within `130ms` for diagonals) |
| `Mouse Movement` | Smooth 360° cursor steering relative to the center of your viewport |
| `Space` / `Left Click` | Boost (`2x` speed, spends length, leaves small luminous trail pellets) |
| `q` / `Esc` / `Ctrl+C` | Quit session |

### Bot Spectator Mode (Ideal for Screen Recording)
To spectate bots without spawning a player snake:
- **Direct SSH Connection**: Connect with `ssh -p 23234 spectate@localhost` (or `watch@localhost`) to immediately start spectating a random bot.
- **In-Game Toggle**: Press `Tab` on the title screen, while playing, or on the death screen.
- **Spectator Controls**:
  - `Tab` / `n` / `Space` / `Right`: Cycle camera to next bot.
  - `p` / `Left`: Cycle camera to previous bot.
  - `r`: Switch to a random bot.
  - `Enter`: Spawn into the arena and play yourself.
  - Camera automatically tracks the spectated bot's head and auto-selects a new bot if the current one dies.
  - HUD displays `👁 SPECTATING <Name> [<Personality>]` with real-time length, rank, radar marker, and leaderboard highlighting.

### Terminal Requirements
- Use a terminal emulator with **256-color or TrueColor** support, **Unicode half-blocks**, and **mouse reporting** (e.g., Kitty, Alacritty, WezTerm, Ghostty, iTerm2, Windows Terminal, or GNOME Terminal).
- Minimum terminal size is `30x10`. The top-10 leaderboard and radar minimap appear automatically when the terminal is at least `65` columns wide.

---

## Architecture & Engineering Decisions

### 1. Repository Layout

```text
.
├── main.go                  # SSH listener (Charm Wish), session lifecycle, CLI flags
├── Dockerfile               # Multi-stage Alpine build bundling server + neural models
├── docker-compose.yml       # One-command container deployment with persistent SSH host key
├── cmd/
│   └── trainenv/            # Newline-delimited JSON RPC bridge for Go <-> Python evaluation
├── game/
│   ├── world.go             # Authoritative 20 TPS simulation, collisions, mass budget, bots
│   ├── snake.go             # Snake kinematics, 1x-4x width scaling, boost, segment chain
│   ├── policy.go            # Pure-Go MLP inference (v1/v2/v3), 688/692-dim observation builder
│   ├── spatial.go           # Spatial hash grid for O(1) neighborhood queries
│   └── *_test.go            # Unit, race, parity, and benchmark tests
├── ui/
│   ├── model.go             # Bubble Tea state machine, input handling, tick subscription
│   ├── render.go            # Half-block 2.5D renderer, cylindrical shading, food, radar HUD
│   └── render_test.go       # UI & rendering tests
├── models/
│   ├── wriggle-v4-personalities.best.json  # Policy v3 (692->64->14, 4 bot personalities)
│   └── wriggle-v3.best.json                # Policy v2 (688->64->14, single reward)
└── training/
    ├── train.py             # PPO trainer, opponent league, curriculum, Go validation
    ├── gpu_sim.py           # Vectorized PyTorch/Triton multi-arena GPU environment
    ├── gpu_kernels.py       # Custom Triton GPU kernels (portable across CUDA & ROCm)
    ├── env.py               # Python client for cmd/trainenv (authoritative Go evaluation)
    ├── README.md            # Complete CUDA & ROCm training documentation
    └── test_*.py            # Go <-> GPU numerical parity & trainer unit tests
```

### 2. Key Design Decisions

1. **Authoritative Single-World Lock-Free Tick Fanout (`game/world.go`)**:
   - The server runs a single authoritative simulation at `20 TPS` (`50ms` per tick) with `4` physics substeps per tick (`dt = 0.25`) to prevent high-speed tunneling during boosts.
   - Each tick takes a write lock to advance physics and snapshots state into immutable value slices (`Snapshot`), then fans out non-blocking notifications over buffered channels (`chan struct{}`) so a slow or lagging SSH client can never stall the simulation or other players.
2. **Closed-Loop Mass Economy (`game/world.go`, `game/snake.go`)**:
   - Early versions had a fixed ambient pellet floor (`3500` pellets) and `100%+` corpse mass conversion, causing exponential score inflation where bots grew infinitely large.
   - We redesigned the arena as a **bounded mass ecosystem**:
     - **Global Mass Cap**: `replenish()` counts both ground pellets and living snake mass (`totalSnakeMass`) against `pelletTarget`, maintaining a `25%` ambient floor so small snakes can always forage.
     - **Entropic Sinks**: Boosting costs `0.10` score/substep (`0.40`/tick) and drops `0.4`-value trail pellets (`50%` entropy sink). Corpse drops spawn one `2.0`-value orb every 3 segments (`~50%` entropy sink).
     - **Diminishing Returns & Metabolic Upkeep**: Snakes above `Score = 60` gain food mass as `1 / sqrt(score / 60)`, and snakes above `Score = 120` burn `0.0005 * (score - 120)` per substep, naturally capping equilibrium size around `250–400` score unless actively hunting rivals.
3. **Dynamic Width Scaling (`1x` to `4x`) & Head-to-Head Tie-Breaker (`game/snake.go`, `game/world.go`)**:
   - Width scales smoothly via `1.0 + 3.0 * sqrt((score - 12) / 180)` up to `4.0x` at `Score >= 192`. Both collision radii (`headHitRadius`, `bodyHitRadius`) and food pickup radii scale proportionally.
   - Head-on collisions reward aggressive play: a **boosting** snake wins against a non-boosting snake; if boost states match, the **higher-score** snake survives and kills the smaller rival.
4. **Outer-Ring Border Pull Gating (`game/policy.go`, `training/gpu_kernels.py`)**:
   - Providing a raw `(dx, dy)` vector to `(0, 0)` in `obs[4:6]` caused bots to treat the map origin as an attractor and cluster in the center.
   - We gated `obs[4:6]` so it is strictly `0` inside `70%` of the arena radius (`dist <= 210` on a `300`-radius map) and ramps linearly from `0` to `1` only in the outer `30%` danger zone (`R = 210..300`), giving bots full freedom to roam the arena while still warning them before the lethal border.
5. **Hybrid GPU Training + Pure-Go Inference (`training/`, `game/policy.go`)**:
   - Training requires millions of environment steps, which we accelerate on GPU via custom Triton kernels (`movement_kernel`, `collision_kernel`, `food_kernel`, `obs_sector_kernel`).
   - Production deployment in an SSH server should not require a GPU or Python runtime. Therefore, `training/train.py` exports compact JSON weight files (`PolicyVersion 2` or `3`) that `game/policy.go` executes in pure Go (`~20 microseconds` per bot per tick).
   - Every validation and final test episode during training runs inside the **actual Go simulator** (`cmd/trainenv`) to guarantee that policies selected as `.best.json` transfer cleanly from float32 Triton kernels to float64 Go physics.

---

## Training Neural Bots (NVIDIA CUDA & AMD ROCm)

The GPU training pipeline in [`training/`](training/README.md) supports both **NVIDIA CUDA** and **AMD ROCm** GPUs via PyTorch and Triton.

> [!NOTE]
> **CUDA Port Status**: The Triton kernels (`training/gpu_kernels.py`) and PyTorch device logic have been ported to support NVIDIA CUDA using standard PyTorch/Triton CUDA APIs. However, because our development and testing environment is an AMD Radeon system running ROCm (where all 13 unit, numerical parity, and integration tests have been executed and verified), the CUDA path has **not been tested on physical NVIDIA hardware**. It is provided as an unverified community port.

### Quick Training Example

```sh
# 1. Create virtualenv and install PyTorch (CUDA or ROCm) + numpy
uv venv --python 3.12 .venv-training
# For NVIDIA CUDA:
uv pip install --python .venv-training/bin/python torch numpy --index-url https://download.pytorch.org/whl/cu124
# For AMD ROCm:
uv pip install --python .venv-training/bin/python 'torch==2.14.1+rocm7.2' 'numpy==2.5.4' --index-url https://download.pytorch.org/whl/rocm7.2

# 2. Train a multi-personality Policy v3 network (512 concurrent worms across 16 GPU arenas)
.venv-training/bin/python -u training/train.py --device gpu \
  --personalities \
  --init-v2 models/wriggle-v3.best.json \
  --output models/wriggle-v4-personalities.json \
  --simulator gpu --no-gpu-graphs --max-segments 1024 --max-pellets 32768 \
  --arenas 16 --agents 32 --updates 350 --rollout 128 \
  --curriculum mixed --validate-every 25
```

See **[training/README.md](training/README.md)** for full details on:
- CUDA & ROCm environment setup
- Multi-reward personality training (`AGG`, `GRD`, `CNS`, `AFR`)
- Curriculum stages and self-play opponent league
- Go $\leftrightarrow$ GPU parity testing

---

## Testing & Verification

Run the Go test suite, static analysis, and race detector:

```sh
go test ./...
go vet ./...
go test -race ./...
```

Run the Python / GPU simulator parity and trainer unit tests (automatically runs GPU tests when CUDA or ROCm is available):

```sh
.venv-training/bin/python -m unittest discover -s training -p "test_*.py" -v
```

---

## License

Released under the [MIT License](LICENSE).
