# Research and evidence

Researched through the browser on 2026-11-12. These sources establish capabilities and constraints; they do not establish sshio's performance, balance, retention, or development duration. All game constants and capacity targets in the plan are proposals to validate.

## SSH and terminal implementation

**[S1] Russh shared Ratatui application example**
https://github.com/Eugeny/russh/blob/main/russh/examples/ratatui_shared_app.rs

Demonstrates a Rust SSH server with a Ratatui terminal per connection, a `CrosstermBackend` writing through a custom SSH output handle, shared application state, PTY dimensions, and resize handling. This supports the feasibility of the proposed stack. Its unbounded output channel, broad locking, and simple example lifecycle are not production patterns to copy; sshio needs bounded output and isolated arena ownership.

**[S2] Wish**
https://github.com/charmbracelet/wish

Go SSH application framework with Bubble Tea middleware, session input/output integration, and window resize handling. A strong alternative for an SSH-first product. The plan selects Rust because the same simulation can be linked into Python and eventually WebAssembly without a separate Go/native simulation boundary.

**[S3] termwiz input module**
https://docs.rs/termwiz/latest/termwiz/input/index.html

Provides an `InputParser` for terminal input bytes. This is useful because remote SSH session input must be parsed independently of the server process's local terminal. Full supported-protocol coverage, escape timeout behavior, and fragmented-input behavior still need verification in the feasibility spike.

**[S4] Kitty keyboard protocol**
https://sw.kovidgoyal.net/kitty/keyboard-protocol/

Explains the limitations of legacy terminal keyboard reporting, including lack of reliable press/repeat/release distinctions, and its opt-in progressive enhancement. This is why portable controls must not require key-release events. Support through multiplexers and specific client versions must be tested, not inferred from a terminal name.

**[S5] Axum WebSocket module**
https://docs.rs/axum/latest/axum/extract/ws/index.html

Documents WebSocket upgrades and separate read/write handling. Supports the later browser transport within the Rust server stack.

## Shared simulation and learning

**[S6] PyO3 user guide**
https://pyo3.rs/main/

Documents Rust native Python extensions and development/distribution through maturin. Release builds matter for performance. Pin a tested stable release rather than depending on the moving `main` documentation version. Wheel portability and NumPy buffer lifetimes remain project engineering responsibilities.

**[S7] PettingZoo Parallel API**
https://pettingzoo.farama.org/api/parallel/

Designed for simultaneous multi-agent actions. Separates observation, reward, termination, truncation, and information dictionaries; includes mutable active agents and a fixed possible-agent universe. Documents `parallel_api_test`. This fits arena games; respawning needs an explicit wrapper policy.

**[S8] PettingZoo AEC API**
https://pettingzoo.farama.org/api/aec/

Supports sequential action selection and legal-action masks. Appropriate for chess/checkers. The documented caveats around converting between AEC and Parallel reinforce the decision to maintain distinct scheduler contracts.

**[S9] CleanRL**
https://docs.cleanrl.dev/

Provides readable, benchmarked single-file online RL implementations, seeding, TensorBoard, and experiment tracking. It explicitly is not an importable modular framework and does not provide offline RL algorithms. Use a pinned, attributed implementation as a reference for sshio's maintained PPO trainer, not a promise of turnkey multi-agent training.

**[S10] PufferLib repository**
https://github.com/PufferAI/PufferLib

Research-oriented high-performance RL tooling. At research time the repository default branch was `5.0` and the listed latest release was experimental. Attractive optimization candidate, but its current native interfaces and release stability must be benchmarked before adoption. Do not assume older Gym/PettingZoo examples describe the current implementation or that its throughput transfers to sshio.

**[S11] OpenSpiel documentation**
https://openspiel.readthedocs.io/en/latest/

Provides game abstractions, algorithms, AlphaZero materials, and evaluation tooling. Useful for future turn-based game research and reference behavior. Not selected as the live real-time arena server or as a mandatory dependency for sshio.

**[S12] PyTorch ONNX export**
https://docs.pytorch.org/docs/stable/onnx.html

Documents the `torch.export`-based ONNX path and consumption by ONNX Runtime, including shape constraints and exporter verification. Export is not automatically behaviorally equivalent: test recurrent state, masks, dynamic batch sizes, numerical tolerances, and resulting actions against the trained PyTorch model.

## Colab and Blackwell

**[S13] Google Colab FAQ**
https://research.google.com/colaboratory/faq.html

GPU types, limits, availability, and session lifetime vary; managed Colab does not guarantee a requested GPU. Local VM files are ephemeral. Recommends reducing many small Drive reads by staging archives locally. Its restrictions also mean the production game server belongs elsewhere. The FAQ specifically lists chess training among activities restricted on free managed runtimes without a positive compute-unit balance; revisit the current applicable plan before the later chess project.

**[S14] Colab local runtimes**
https://research.google.com/colaboratory/local-runtimes.html

Documents connecting Colab to a controlled local runtime or a remote runtime through SSH port forwarding. This is the supported route to using a particular workstation GPU with the Colab frontend. The supplied Docker image is described as demo-only and tested with T4/L4/A100; it is not evidence of Blackwell readiness. Prefer a separately tested Jupyter/PyTorch environment for the RTX PRO 6000.

**[S15] NVIDIA CUDA compute capability table**
https://developer.nvidia.com/cuda/gpus

Lists RTX PRO 6000 Blackwell Server, Workstation, and Max-Q Workstation editions at compute capability **12.0**. B200/GB200 are listed separately at **10.0**. Do not treat every Blackwell target as having identical instruction compatibility.

**[S16] PyTorch 2.7 release**
https://pytorch.org/blog/pytorch-2-7/

Introduced Blackwell support and prebuilt CUDA 12.8 wheels, with Triton 3.3 support. This establishes a compatibility milestone, not a recommendation to freeze on PyTorch 2.7. Select and lock a currently supported build only after exercising it on the actual GPU and driver.

**[S17] NVIDIA RTX PRO 6000 Blackwell Workstation Edition**
https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/

Official product page specifies 96 GB GDDR7 ECC memory. This is physical hardware capacity, not a promise that a managed or partitioned runtime exposes all 96 GB. Read actual available memory at runtime. Marketing TOPS figures are not RL throughput forecasts.

## Browser rendering

**[S18] PixiJS architecture**
https://pixijs.com/8.x/guides/concepts/architecture

Documents a 2D scene graph, WebGL/WebGPU rendering, tickers, pointer events, and filters. Fits a custom 2D snake renderer without importing another physics engine. The client ticker must not become authoritative game time.

## Visual references and self-play follow-up

**[S19] Slither.io publisher screenshots — Google Play**
https://play.google.com/store/apps/details?id=air.com.hypah.io.slither&hl=en

The official Lowtech Studios listing supplies the entry and gameplay reference screenshots embedded in Section 7 of the plan. Four browser captures were saved locally with browser margins cropped, leaving the artwork unchanged. [Image provenance and original CDN URLs](assets/slitherio/sources.json) accompany the files. These are visual references, not assets licensed for inclusion in sshio. The current images include themed backgrounds; they do not change the chosen dark dot-field design. Static highlights alone do not establish boost animation behavior. The listing's “play against A.I.” UI is not evidence that Slither.io uses learned/self-play policies.

**[S20] DeepMind — AlphaZero: Shedding new light on chess, shogi, and Go**
https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/

The authors describe training from random play using self-play reinforcement learning without human demonstrations. This is a concrete precedent for human-data-free learning. AlphaZero combines neural networks with MCTS in perfect-information board games; its algorithm, compute requirements, convergence behavior, and reported training times cannot be assumed to transfer to sshio. The proposed sshio implementation remains PPO plus a diverse opponent league.

## Research conclusions and remaining uncertainty

- The Rust → SSH and Rust → Python integration paths are documented. Their combination into an enjoyable, scalable game remains unproven until implemented.
- Colab can be the notebook frontend for the specified GPU, but GPU allocation in managed Colab is a separate question.
- CPU simulation plus GPU learning is the lowest-risk initial design. A GPU simulator could be faster, but variable-length bodies and collision handling make its cost and benefit uncertain.
- PettingZoo can standardize research interfaces across the proposed games; it cannot eliminate their different rules, observations, rendering, or learning algorithms.
- No empirical sshio measurements have yet been made. The plan deliberately supplies tests and decision gates instead of invented benchmark results.
