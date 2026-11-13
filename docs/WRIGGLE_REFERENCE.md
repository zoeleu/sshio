# Recovered Wriggle reference

Wriggle survived the local wipe in this GitHub repository. Its code is preserved on these existing branches; sshio's Rust work does not replace them:

- [Game and training stack](https://github.com/zoeleu/sshio/tree/8001c95d81973a23a53859b3955cc6590bf5a062), commit `8001c95d81973a23a53859b3955cc6590bf5a062`, branch `feat/wriggle-game-and-training`.
- [Additional authored neural policy](https://github.com/zoeleu/sshio/tree/e6273a56132f66a625ad37b9c61d769f707a51e5), commit `e6273a56132f66a625ad37b9c61d769f707a51e5`, branch `t3/neural-snake-weights`. Its documentation identifies that artifact as directly authored weights, rather than a trained policy.

To recover a separate local copy without touching this checkout:

```sh
gh repo clone zoeleu/sshio wriggle-recovered -- --branch feat/wriggle-game-and-training
```

## What sshio takes from it

The user identified Wriggle as a successful product reference. Its source is stronger evidence for terminal interaction choices than hypothetical mockups.

| Wriggle source | Reference for sshio | Current status |
| --- | --- | --- |
| `ui/render.go` | Half-block output; top-left body lighting and dark rims; heading-oriented eyes | Implemented with original Rust raster code; color and geometry follow sshio's rules |
| `ui/model.go`, `ui/render.go` | A live board beneath entry and UNRAVELED overlays; preserve the death location; useful spectator camera | Entry/death overlays and death camera retention implemented; spectators currently follow the first living snake |
| `ui/render.go` | Tail-first painting, stripes, width variation, boost travel | Baseline implemented; further polish and low-motion controls pending |
| `game/training.go`, `game/training_batch.go` | Real simulation accessible without a wall clock, bounded native batches | Same boundary in `sshio-core` and PyO3 batch adapter |
| `training/league.py` | Bounded historical opponents with a permanent initial anchor | Reference for later league implementation; not yet in the Rust training path |
| `training/gpu_sim.py`, `training/gpu_kernels.py` | Prior GPU vectorization, profiling, and test ideas | Inspect for later optimization; no claim of parity with sshio's physics |

Before rewriting mature behavior, compare it against Wriggle and preserve what works. Recovering this implementation changes the Phase 0 stack decision: Go/Wish is now an existing alternative to benchmark, rather than only a hypothetical fallback. Retain Rust only if native simulation reuse, terminal behavior, and measured cost justify continuing the new implementation.

## Rules and provenance

The two games currently differ in head-to-head priority, food economy, observation/action sizes, and lifetime rules. Wriggle's 692-input/14-action model files cannot be loaded into sshio's initial 76-input/10-action adapter. A GPU simulator also needs differential tests against the canonical rules before its rollouts can be treated as sshio training data.

The current sshio renderer uses Wriggle as a design reference for shaded segments, eyes, and death UX. Wriggle's original MIT notice is retained below for attribution and any future ports:

> MIT License — Copyright (c) 2026 Wriggle Contributors.
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
