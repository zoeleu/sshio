# sshio

A multiplayer snake arena played over SSH, built around a transport-free Rust simulation that also drives synthetic multi-agent learning. Browser play and learned bot serving follow the same authoritative rules.

**Status: first feasibility implementation.** The Rust core, SSH server, striped terminal renderer, heuristic population, and Python batch adapter are implemented. Balance is experimental. Production hardening, replay, PPO/league training, ONNX serving, and the browser client remain in the [delivery backlog](docs/DELIVERY_BACKLOG.md).

## Play locally

Requires Rust/Cargo, a C linker, and OpenSSH. The server binds to localhost by default; it never offers a shell or port forwarding. Use a terminal of at least 80×24 with true color and Unicode half blocks.

```sh
mkdir -p .sshio
ssh-keygen -t ed25519 -N '' -f .sshio/host_key
cargo run -p sshio-server
# In another terminal:
ssh -p 2222 localhost
```

Choose a name, press Tab for a color, then Enter to play or Ctrl-S to spectate. WASD/arrows set a persistent direction; Space toggles boost. `v` switches to spectating, Enter respawns, and `q` quits. The arena continues behind the entry dialog. Food expires and returns to the reserve, large snakes pay upkeep, and a persistent crown timer ends a long reign. Humans and bots use the same physics and economy.

`SSHIO_BIND` changes the listen address; `SSHIO_HOST_KEY` selects an existing persistent Ed25519 private key. This is a local spike: use a development host until connection admission, hostile traffic, slow consumers, latency, and resource soaks have been measured.

For a constrained Linux machine, `scripts/limited COMMAND ...` pins the command and its children to one available logical CPU and applies a 2 GiB address-space ceiling with one Cargo build job. It does not change system configuration. Example with Nix:

```sh
scripts/limited nix shell nixpkgs#cargo nixpkgs#rustc nixpkgs#gcc \
  --command cargo run -p sshio-server
```

## Headless simulation and Python

`sshio-core` has no dependencies on SSH, Python, graphics, wall clocks, or thread pools. The PyO3 adapter invokes that exact core sequentially for bounded batches (up to 64 worlds, 24 agents each). Start with one world on small machines.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install maturin
maturin develop
python tests/native_smoke.py
```

```python
from sshio import Batch

env = Batch(worlds=1, agents=8, seed=1)
obs = env.observations()                 # [worlds * agents, 76]
terminal_obs, rewards, done = env.step([2] * 8)  # actions 0..9
obs = env.observations()                 # fresh observations after any respawn
```

Ten actions combine five relative steering angles with boost off/on. Each action advances three simulation ticks (10 Hz controls). All agents may be controlled by the same sampled policy with individual life boundaries: **no human data is required**. The first observation is local and egocentric, using food/body/boundary sectors plus mass, boost, age, and crown status. Rewards currently use log mass growth, a small time cost, and a death penalty. These are a baseline to evaluate, not a validated objective for fun.

`step()` returns terminal observations before replacing dead lives; call `observations()` for the following action. `done` must reset recurrent state and stop advantage bootstrapping. Rollout versioning, historical opponents, and training are specified in the plan and are not yet implemented. No player recording or training capture is enabled.

Build a wheel with `maturin build --release`. The local Nix-built Linux wheel is verified only on this host; produce a compatible manylinux release wheel and validate its installation in Colab before training there. `scripts/hardware_probe.py` reports the actual PyTorch/CUDA/GPU environment and performs a tiny forward/backward check where a GPU exists. Managed Colab does not promise a particular GPU.

## Validate

```sh
cargo test --workspace -- --test-threads=1
cargo run -p sshio-core --example fingerprint
python tests/native_smoke.py NATIVE_FINGERPRINT
# Start and check an isolated temporary-key server:
cargo build -p sshio-server
python scripts/check_ssh.py
```

See [implementation and validation notes](docs/IMPLEMENTATION.md) for current evidence and remaining gates.

Wriggle is preserved on its existing GitHub branches. See the [recovery links and source reference](docs/WRIGGLE_REFERENCE.md) for its game, training stack, and the terminal design ideas carried into sshio.

## Plan and references

- [Complete project plan](docs/SSHIO_PLAN.md): stack, fairness, visuals, networking, ML, Colab, data collection, and reuse across games.
- [Delivery backlog](docs/DELIVERY_BACKLOG.md): acceptance criteria and phase gates.
- [Research sources](docs/RESEARCH.md): attributed primary sources and limitations.
- [Interactive architecture reference](docs/architecture.html).

The plan includes four attributed Slither.io visual references. Those third-party screenshots retain their original ownership and are excluded from this repository's MIT license. New game artwork should be original.
