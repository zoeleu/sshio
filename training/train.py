#!/usr/bin/env python3
"""ROCm PPO self-play, using the Go simulation and a portable JSON policy."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import signal
import fcntl

from league import League, role_masks

import torch
from torch import nn
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
# V2: 172-feature single frame stacked 4 deep (oldest first); see game/policy.go.
SINGLE_OBS, FRAME_STACK = 172, 4
OBS, ACTIONS = SINGLE_OBS * FRAME_STACK, 14
PERSONALITY_COUNT = 4
PERSONALITY_OBS = OBS + PERSONALITY_COUNT
POLICY_VERSION = 2
PERSONALITY_POLICY_VERSION = 3

# V1 (49) -> V2 single-frame index map for warm starting; -1 drops a feature.
# The mapped weights land in the newest frame's slice of the stacked input.
V1_TO_V2 = [0, 2, 3, 4, 5, 16, 17, 32, 33, 36, 37, 38, 12] + [
    target for ray in range(12)
    for target in (76 + 2 * ray * 4, 76 + 2 * ray * 4 + 2, 76 + 2 * ray * 4 + 3)]


MAX_HIDDEN = 256  # game/policy.go maxHiddenWidth


class Policy(nn.Module):
    """tanh MLP: obs -> hidden -> [hidden2] -> actor/critic heads.

    hidden2=0 keeps the original single-hidden-layer network. With hidden2>0
    the exported JSON gains "wh"/"bh" (see game/policy.go); the critic shares
    the trunk, as before.
    """

    def __init__(self, hidden=64, obs_dim=OBS, version=POLICY_VERSION, hidden2=0):
        super().__init__()
        if not 1 <= hidden <= MAX_HIDDEN or not 0 <= hidden2 <= MAX_HIDDEN:
            raise ValueError(f"hidden widths must be 1..{MAX_HIDDEN} (hidden2 may be 0)")
        self.obs_dim = obs_dim
        self.version = version
        self.hidden = nn.Linear(obs_dim, hidden)
        self.hidden2 = nn.Linear(hidden, hidden2) if hidden2 else None
        trunk = hidden2 or hidden
        self.actor = nn.Linear(trunk, ACTIONS)
        self.critic = nn.Linear(trunk, 1)

    def forward(self, obs):
        h = torch.tanh(self.hidden(obs))
        if self.hidden2 is not None:
            h = torch.tanh(self.hidden2(h))
        return self.actor(h), self.critic(h).squeeze(-1)

    def export(self):
        data = {"version": self.version, "w1": self.hidden.weight.detach().cpu().tolist(),
                "b1": self.hidden.bias.detach().cpu().tolist(),
                "w2": self.actor.weight.detach().cpu().tolist(),
                "b2": self.actor.bias.detach().cpu().tolist()}
        if self.hidden2 is not None:
            data["wh"] = self.hidden2.weight.detach().cpu().tolist()
            data["bh"] = self.hidden2.bias.detach().cpu().tolist()
        return data


def reject_deep_warm_start(model, source):
    # A 256-unit tanh layer cannot be squeezed into a 128-unit one without
    # changing the function, so a donor's output head would be meaningless.
    if model.hidden2 is not None:
        raise ValueError(f"cannot warm start a two-hidden-layer model from {source}; "
                         "omit --init-v1/--init-v2 (teacher imitation warm start still runs)")


def init_from_v1(model, data):
    """Copy a v1 JSON export into a v2/v3 model where features line up.

    First-layer weights go to the newest stacked frame; earlier frames and
    unmatched features (velocity, extra rivals/sectors, critic) stay at
    their random init, so the v1 baseline behaviour is the starting point.
    """
    reject_deep_warm_start(model, "a v1 export")
    if data.get("version") != 1:
        raise ValueError("warm-start file is not a version 1 policy")
    w1 = torch.as_tensor(data["w1"], dtype=torch.float32)
    if w1.ndim != 2 or w1.shape[1] != 49:
        raise ValueError("warm-start w1 must have 49 inputs")
    rows = min(w1.shape[0], model.hidden.weight.shape[0])
    with torch.no_grad():
        model.hidden.weight.zero_()
        newest = (FRAME_STACK - 1) * SINGLE_OBS
        for old_index, new_index in enumerate(V1_TO_V2):
            if new_index >= 0:
                model.hidden.weight[:rows, newest + new_index] = w1[:rows, old_index]
        model.hidden.bias[:rows] = torch.as_tensor(data["b1"], dtype=torch.float32)[:rows]
        w2 = torch.as_tensor(data["w2"], dtype=torch.float32)
        model.actor.weight[:, :rows] = w2[:, :rows]
        model.actor.bias.copy_(torch.as_tensor(data["b2"], dtype=torch.float32))
    return model


def init_from_v2(model, data, pt_path=None):
    """Copy a v2 (688-input) JSON export into a v3 (692-input) personality-conditioned model.

    The 688 observation weights and biases are copied directly; the 4 trailing
    personality one-hot columns are initialized to zero so all 4 personalities
    start from the v2 donor behaviour and diverge under their reward weights.
    If a matching .pt checkpoint exists, the donor critic weights are also copied
    so PPO advantage estimates start calibrated.
    """
    reject_deep_warm_start(model, "a v2 export")
    if data.get("version") != 2:
        raise ValueError("warm-start file is not a version 2 policy")
    w1 = torch.as_tensor(data["w1"], dtype=torch.float32)
    if w1.ndim != 2 or w1.shape[1] != OBS:
        raise ValueError(f"warm-start w1 must have {OBS} inputs")
    rows = min(w1.shape[0], model.hidden.weight.shape[0])
    with torch.no_grad():
        model.hidden.weight.zero_()
        model.hidden.weight[:rows, :OBS] = w1[:rows, :OBS]
        model.hidden.bias[:rows] = torch.as_tensor(data["b1"], dtype=torch.float32)[:rows]
        w2 = torch.as_tensor(data["w2"], dtype=torch.float32)
        model.actor.weight[:, :rows] = w2[:, :rows]
        model.actor.bias.copy_(torch.as_tensor(data["b2"], dtype=torch.float32))
        if pt_path is not None and Path(pt_path).exists():
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
            state = ckpt.get("model", {})
            if "critic.weight" in state and "critic.bias" in state:
                cw = state["critic.weight"].to(model.critic.weight.device, dtype=torch.float32)
                cb = state["critic.bias"].to(model.critic.bias.device, dtype=torch.float32)
                model.critic.weight[:, :rows].copy_(cw[:, :rows])
                model.critic.bias.copy_(cb)
    return model


class Bridge:
    def __init__(self):
        binary = ROOT / ".training/trainenv"
        binary.parent.mkdir(exist_ok=True)
        subprocess.run(["go", "build", "-o", str(binary), "./cmd/trainenv"], cwd=ROOT, check=True)
        self.proc = subprocess.Popen([str(binary)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)

    def call(self, **request):
        self.proc.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("Go training environment exited")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def close(self):
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.proc.stdout.close()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def advantages(rewards, dones, values, last_value, gamma=0.99, lam=0.95):
    result = torch.zeros_like(rewards)
    carry = torch.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        alive = 1 - dones[t]
        delta = rewards[t] + gamma * next_value * alive - values[t]
        carry = delta + gamma * lam * alive * carry
        result[t] = carry
    return result, result + values


def evaluate(env, policy, seed, horizon, episodes):
    if policy is not None:
        env.call(op="policy", policy=policy)
    scenarios = ("arena", "cutoff", "escape", "border", "forage", "chase", "production")
    runs = []
    for i in range(episodes):
        scenario = scenarios[i % len(scenarios)]
        state = env.call(op="reset", seed=seed + i, opponents=1 if scenario in
                         ("cutoff", "escape", "chase") else 3,
                         horizon=horizon, scenario=scenario)
        reward = 0.0
        while not state["done"]:
            state = env.call(op="step", action=-1 if policy is None else -2)
            reward += state["reward"]
        runs.append({"seed": seed + i, "scenario": scenario, "return": reward,
                     **{k: state[k] for k in ("food", "kills", "ticks", "alive", "score",
                                              "mutual_kills", "circles", "boost_spent")}})
    ticks = sum(r["ticks"] for r in runs)
    kills = sum(r["kills"] for r in runs)
    return {"mean_return": sum(r["return"] for r in runs) / len(runs),
            "mean_food": sum(r["food"] for r in runs) / len(runs),
            "mean_survival_ticks": ticks / len(runs), "kills": kills,
            "kills_per_minute": kills * 1200 / max(1, ticks),
            "survival_fraction": sum(r["alive"] for r in runs) / len(runs),
            "mutual_kills": sum(r["mutual_kills"] for r in runs),
            "circles": sum(r["circles"] for r in runs), "runs": runs}


def warm_start(env, model, args, device):
    if not args.warmup_steps:
        return
    state = env.call(op="reset_batch", arenas=args.arenas, agents=args.agents,
                     horizon=args.horizon, seed=args.seed, teach=True,
                     curriculum=args.curriculum == "mixed", stage=0)
    observations, labels = [], []
    started = time.monotonic()
    for step in range(args.warmup_steps):
        observations.append(torch.as_tensor(state["observations"],dtype=torch.float32,device=device).clone())
        labels.append(torch.as_tensor(state["teachers"],dtype=torch.long,device=device).clone())
        state = env.call(op="step_batch", actions=state["teachers"], teach=True)
        if (step + 1) % 8 == 0:
            print(f"teacher rollout {step+1}/{args.warmup_steps}", flush=True)
    x = torch.cat(observations)
    y = torch.cat(labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    counts = torch.bincount(y, minlength=ACTIONS).float().clamp_min(1)
    weights = (len(y) / (ACTIONS * counts)).clamp(0.25, 4)
    for _ in range(args.warmup_epochs):
        for batch in torch.randperm(len(y), device=device).split(args.minibatch):
            logits, _ = model(x[batch])
            loss = nn.functional.cross_entropy(logits, y[batch], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        accuracy = (model(x)[0].argmax(-1) == y).float().mean().item()
    print(f"warm start: {len(y)} examples, accuracy={accuracy:.3f}, "
          f"seconds={time.monotonic()-started:.1f}", flush=True)


def save_checkpoint(path, model, optimizer, league, updates, transitions, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format_version": 2, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "updates": updates,
                "transitions": transitions, "league": league.states(),
                "rng": torch.get_rng_state(),
                "gpu_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "report": report}, temporary)
    temporary.replace(path)


def train(args):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(".lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f"another trainer is writing {output}") from None
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    try:
        train_locked(args, output)
    finally:
        lock.close()


def train_locked(args, output):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if args.device in ("rocm", "cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"GPU device '{args.device}' requested, but torch.cuda.is_available() is False. "
                               "CPU fallback is never automatic.")
        if args.device == "rocm" and not torch.version.hip:
            raise RuntimeError("ROCm requested (--device rocm), but PyTorch was not built with HIP. "
                               "Use --device cuda on NVIDIA GPUs or --device gpu for auto-detection.")
        if args.device == "cuda" and torch.version.hip and not torch.version.cuda:
            print("Note: --device cuda selected on a ROCm/HIP PyTorch build; running on ROCm device.", flush=True)
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        probe = torch.ones((32, 32), device=device, requires_grad=True)
        (probe @ probe).sum().backward()
        torch.cuda.synchronize()
        hardware = {"device": torch.cuda.get_device_name(),
                    "backend": "rocm" if torch.version.hip else "cuda",
                    "hip": torch.version.hip,
                    "cuda": torch.version.cuda,
                    "torch": str(torch.__version__)}
    else:
        device = torch.device("cpu")
        hardware = {"device": "cpu", "backend": "cpu", "torch": str(torch.__version__)}
    hardware["simulator"] = args.simulator
    print(json.dumps(hardware), flush=True)
    count = args.arenas * args.agents
    print(f"{args.arenas} parallel arenas × {args.agents} worms = {count} active worms", flush=True)
    obs_dim = PERSONALITY_OBS if args.personalities else OBS
    policy_version = PERSONALITY_POLICY_VERSION if args.personalities else POLICY_VERSION
    model = Policy(hidden=args.hidden, hidden2=args.hidden2,
                   obs_dim=obs_dim, version=policy_version).to(device)
    shape = "->".join(str(n) for n in (obs_dim, args.hidden, args.hidden2 or None, ACTIONS) if n)
    print(f"policy v{policy_version}: {shape} (tanh)", flush=True)
    if args.init_v1 and not args.resume:
        init_from_v1(model, json.loads(Path(args.init_v1).read_text()))
        print(f"warm started v{policy_version} policy from v1 export {args.init_v1}", flush=True)
    if args.init_v2 and not args.resume:
        v2_path = Path(args.init_v2)
        init_from_v2(model, json.loads(v2_path.read_text()), v2_path.with_suffix(".pt"))
        print(f"warm started v{policy_version} policy from v2 export {args.init_v2}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = {}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        if "rng" in checkpoint:
            torch.set_rng_state(checkpoint["rng"].cpu())
        if device.type == "cuda" and checkpoint.get("gpu_rng"):
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["gpu_rng"]])
    start_update = checkpoint.get("updates", 0)
    start_transitions = checkpoint.get("transitions", 0)
    if args.resume and "transitions" not in checkpoint:
        legacy_report = Path(args.resume).with_suffix(".report.json")
        if legacy_report.exists():
            previous = json.loads(legacy_report.read_text())
            if previous.get("updates"):
                start_transitions = previous["updates"][-1]["transitions"]
    league = League(model, args.league_size, checkpoint.get("league"))
    heuristic, historical, learner = role_masks(count, args.agents, device)
    learner_indices = learner.cpu().nonzero().flatten().tolist()
    learner_ids = torch.tensor(learner_indices, device=device, dtype=torch.long)
    env = Bridge()  # Authoritative Go validation; not used by GPU rollouts.
    if args.simulator == "gpu":
        if args.gpu_graphs:
            print("Experimental graph replay enabled (CUDAGraph/HIPGraph).", flush=True)
        if device.type != "cuda":
            env.close()
            raise RuntimeError("--simulator gpu requires --device gpu, --device cuda, or --device rocm")
        from gpu_sim import GPUBridge
        training_env = GPUBridge(device, args.max_segments, args.max_pellets, args.gpu_graphs, args.personalities)
    else:
        training_env = env
    report = {"hardware": hardware, "config": vars(args), "updates": [], "validation": [],
              "selection_metric": "validation mean_return", "start_update": start_update,
              "start_transitions": start_transitions, "learner_count": int(learner.sum()),
              "resume_best_update": checkpoint.get("report", {}).get("best_update"),
              "validation_spec": {"seed": args.eval_seed, "horizon": args.eval_horizon,
                                  "episodes": args.eval_episodes, "suite_version": 2},
              "best_policy_path": str(output.with_suffix(".best.json"))}
    started = time.monotonic()
    completed, transitions = start_update, start_transitions
    initial_policy = model.export()
    best_policy, best_score, best_update = initial_policy, float("-inf"), start_update
    prior = checkpoint.get("report", {})
    prior_best = Path(prior.get("best_policy_path", "/nonexistent-wriggle-policy"))
    if (prior.get("validation_spec") == report["validation_spec"] and prior_best.exists()
            and "best_validation_return" in prior):
        best_policy = json.loads(prior_best.read_text())
        best_score, best_update = prior["best_validation_return"], prior["best_update"]
        report["best_update"], report["best_validation_return"] = best_update, best_score
        atomic_json(output.with_suffix(".best.json"), best_policy)
        prior_pt = prior_best.with_suffix(".pt")
        destination = output.with_suffix(".best.pt")
        if prior_pt.exists() and prior_pt.resolve() != destination.resolve():
            import shutil
            shutil.copy2(prior_pt, destination)
    status_path = output.with_suffix(".status.json")
    old_handler = signal.getsignal(signal.SIGTERM)
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    phase = "initial_validation"
    def status(running=True):
        atomic_json(status_path, {"pid": os.getpid(), "running": running, "phase": phase,
                    "update": completed, "target_update": start_update + args.updates,
                    "transitions": transitions, "best_update": best_update,
                    "simulator": args.simulator, "learning_device": str(device),
                    "gpu_graphs": args.gpu_graphs if args.simulator=="gpu" else False,
                    "seconds": time.monotonic() - started,
                    "latest_policy": str(output), "best_policy": str(output.with_suffix(".best.json"))})
    def persist():
        atomic_json(output, model.export())
        save_checkpoint(output.with_suffix(".pt"), model, optimizer, league, completed, transitions, report)
        atomic_json(output.with_suffix(".report.json"), report)
        status()
    def validate():
        nonlocal phase, best_score, best_policy, best_update
        phase = "validation"
        status()
        result = evaluate(env, model.export(), args.eval_seed, args.eval_horizon, args.eval_episodes)
        entry = {"update": completed, **result}
        report["validation"].append(entry)
        if result["mean_return"] > best_score:
            best_score = result["mean_return"]
            best_policy = model.export()
            best_update = completed
            report["best_update"] = best_update
            report["best_validation_return"] = best_score
            atomic_json(output.with_suffix(".best.json"), best_policy)
            save_checkpoint(output.with_suffix(".best.pt"), model, optimizer, league,
                            completed, transitions, report)
        print(json.dumps({"validation_update": completed, "return": result["mean_return"],
                          "food": result["mean_food"], "kills": result["kills"],
                          "best_update": best_update}), flush=True)
        return result
    try:
        status()
        report["before"] = validate()
        if not args.resume and not args.init_v1 and not args.init_v2:
            phase = "warm_start"
            status()
            warm_start(training_env, model, args, device)
            league = League(model, args.league_size)
            validate()
        stage = min(2, completed // args.stage_every)
        state = training_env.call(op="reset_batch", arenas=args.arenas, agents=args.agents,
                         horizon=args.horizon, seed=args.seed + 100000 + start_update * 1009,
                         curriculum=args.curriculum == "mixed", stage=stage)
        obs = torch.as_tensor(state["observations"], dtype=torch.float32, device=device)
        for update in range(args.updates):
            iteration_started = time.monotonic()
            global_update = start_update + update + 1
            next_stage = min(2, global_update // args.stage_every)
            if args.curriculum == "mixed" and next_stage != stage:
                training_env.call(op="curriculum", stage=next_stage)
                stage = next_stage
            if update and completed % args.snapshot_every == 0:
                league.add(model)
            phase = "rollout"
            status()
            shape = (args.rollout, count)
            obs_buf = torch.empty((*shape, obs_dim), device=device)
            act_buf = torch.empty(shape, dtype=torch.long, device=device)
            log_buf, val_buf, rew_buf, done_buf = [torch.empty(shape, device=device) for _ in range(4)]
            metric_names = ("food", "kills", "deaths", "mutual_kills", "circles")
            totals = torch.zeros(5,device=device)
            for step in range(args.rollout):
                with torch.no_grad():
                    logits, values = model(obs)
                    distribution = Categorical(logits=logits, validate_args=False)
                    actions = distribution.sample()
                    league.play(obs, actions, historical, args.agents,
                                global_update // args.snapshot_every)
                    log_probs = distribution.log_prob(actions)
                    actions[heuristic] = -1
                obs_buf[step], act_buf[step] = obs, actions
                log_buf[step], val_buf[step] = log_probs, values
                state = training_env.call(op="step_batch", actions=actions if args.simulator=="gpu" else actions.cpu().tolist())
                rew_buf[step] = torch.as_tensor(state["rewards"], device=device)
                done_buf[step] = torch.as_tensor(state["dones"], dtype=torch.float32, device=device)
                obs = torch.as_tensor(state["observations"], dtype=torch.float32, device=device)
                totals += torch.stack([torch.as_tensor(state["agent_"+name],device=device,dtype=torch.float32)
                                       for name in metric_names]).index_select(1,learner_ids).sum(1)
            if args.simulator=="gpu":
                training_env.check_capacity()
            with torch.no_grad():
                adv, returns = advantages(rew_buf, done_buf, val_buf, model(obs)[1],
                                          args.gamma, args.gae_lambda)
            x, a, old_log, target, advantage = [
                buf.index_select(1, learner_ids).flatten(0, 1) for buf in (obs_buf, act_buf, log_buf, returns, adv)]
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
            phase = "optimization"
            kls, clips, entropies = [], [], []
            for _ in range(args.epochs):
                stop_early = False
                for batch in torch.randperm(len(a), device=device).split(args.minibatch):
                    logits, value = model(x[batch])
                    dist = Categorical(logits=logits, validate_args=False)
                    log_ratio = dist.log_prob(a[batch]) - old_log[batch]
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        kl = ((ratio - 1) - log_ratio).mean().item()
                        kls.append(kl)
                        clips.append(((ratio - 1).abs() > 0.2).float().mean().item())
                    if args.target_kl and kl > args.target_kl:
                        stop_early = True
                        break
                    policy_loss = -torch.minimum(ratio * advantage[batch],
                                                ratio.clamp(0.8, 1.2) * advantage[batch]).mean()
                    value_loss = (value - target[batch]).square().mean()
                    entropy = dist.entropy().mean()
                    entropies.append(entropy.detach().item())
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite loss; refusing optimizer step")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    if not torch.isfinite(norm):
                        raise RuntimeError("non-finite gradient; refusing optimizer step")
                    optimizer.step()
                if stop_early:
                    break
            completed = global_update
            transitions += args.rollout * count
            iteration_seconds = time.monotonic() - iteration_started
            metric = {"update": completed, "transitions": transitions,
                      "learner_transitions": args.rollout * len(learner_indices),
                      "mean_reward": rew_buf.index_select(1, learner_ids).mean().item(), **dict(zip(metric_names,totals.tolist())),
                      "stage": stage, "scenarios": sorted(set(state["scenarios"])),
                      "league_size": len(league.models), "approx_kl": sum(kls) / max(1, len(kls)),
                      "clip_fraction": sum(clips) / max(1, len(clips)),
                      "entropy": sum(entropies) / max(1, len(entropies)),
                      "iteration_seconds": iteration_seconds,
                      "worm_transitions_per_second": args.rollout * count / iteration_seconds,
                      "seconds": time.monotonic() - started}
            report["updates"].append(metric)
            print(json.dumps(metric), flush=True)
            if completed % args.validate_every == 0 or update == args.updates - 1:
                validate()
            phase = "checkpoint"
            persist()
        phase = "final_test"
        status()
        report["test"] = {"seed": args.test_seed,
            "best": evaluate(env, best_policy, args.test_seed, args.eval_horizon, args.test_episodes),
            "initial": evaluate(env, initial_policy, args.test_seed, args.eval_horizon, args.test_episodes),
            "heuristic": evaluate(env, None, args.test_seed, args.eval_horizon, args.test_episodes)}
        phase = "complete"
    except KeyboardInterrupt:
        report["interrupted"] = True
        phase = "interrupted"
        print("Stopping; saving current policy and optimizer", flush=True)
    except BaseException as exc:
        report["error"] = str(exc)
        phase = "failed"
        raise
    finally:
        try:
            persist()
            status(False)
        finally:
            signal.signal(signal.SIGTERM, old_handler)
            if training_env is not env:
                training_env.close()
            env.close()
    print(f"saved {output}; latest update {completed}, best validation update {best_update}", flush=True)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("gpu", "cuda", "rocm", "cpu"), default="gpu",
                   help="compute backend: 'gpu' (auto CUDA/ROCm), 'cuda' (NVIDIA), 'rocm' (AMD), or 'cpu'")
    p.add_argument("--simulator", choices=("gpu", "go"), default="gpu")
    p.add_argument("--gpu-graphs", action=argparse.BooleanOptionalAction, default=False,
                   help="experimental CUDA/HIP graph replay")
    p.add_argument("--max-segments", type=int, default=1024)
    p.add_argument("--max-pellets", type=int, default=32768)
    p.add_argument("--arenas", type=int, default=16)
    p.add_argument("--agents", type=int, default=32)
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--horizon", type=int, default=1200)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=32)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-seed", type=int, default=9000000)
    p.add_argument("--eval-horizon", type=int, default=600)
    p.add_argument("--eval-episodes", type=int, default=28)
    p.add_argument("--test-episodes", type=int, default=28)
    p.add_argument("--test-seed", type=int, default=19000000)
    p.add_argument("--validate-every", type=int, default=25)
    p.add_argument("--stage-every", type=int, default=50)
    p.add_argument("--league-size", type=int, default=8)
    p.add_argument("--curriculum", choices=("mixed", "none"), default="mixed")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--resume")
    p.add_argument("--init-v1", metavar="PATH",
                   help="warm start a fresh v2/v3 policy from a v1 JSON export (ignored with --resume)")
    p.add_argument("--init-v2", metavar="PATH",
                   help="warm start a fresh v3 personality policy from a v2 JSON export (ignored with --resume)")
    p.add_argument("--personalities", action=argparse.BooleanOptionalAction, default=False,
                   help="condition observation (692 inputs, policy v3) and PPO rewards on 4 bot personalities")
    p.add_argument("--hidden", type=int, default=64,
                   help="width of the first hidden layer (1..256); existing models use 64")
    p.add_argument("--hidden2", type=int, default=0,
                   help="width of an optional second hidden layer (0 = none, up to 256)")
    p.add_argument("--output", default="models/wriggle-gpu.json")
    args = p.parse_args()
    for name in ("arenas", "agents", "updates", "rollout", "horizon", "epochs", "minibatch",
                 "snapshot_every", "eval_episodes", "eval_horizon", "warmup_epochs",
                 "test_episodes", "validate_every", "stage_every", "league_size"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_steps < 0 or args.agents < 2 or args.arenas > 64 or args.agents > 64 or args.arenas * args.agents > 2048:
        p.error("need 1..64 arenas, 2..64 agents, <=2048 worms and nonnegative warmup steps")
    if args.lr <= 0 or not 4 <= args.horizon <= 20000 or not 4 <= args.eval_horizon <= 20000:
        p.error("learning rate must be positive and horizons must be 4..20000")
    if not 1 <= args.hidden <= MAX_HIDDEN or not 0 <= args.hidden2 <= MAX_HIDDEN:
        p.error(f"--hidden must be 1..{MAX_HIDDEN} and --hidden2 0..{MAX_HIDDEN}")
    if args.hidden2 and (args.init_v1 or args.init_v2):
        p.error("--hidden2 cannot warm start from --init-v1/--init-v2 (different architecture); "
                "it trains from scratch with the teacher-imitation warm start")
    if args.max_segments < 80 or args.max_pellets < 64:
        p.error("GPU capacities need >=80 segments and >=64 pellets")
    if args.simulator == "gpu" and args.device == "cpu":
        p.error("GPU simulation requires --device gpu, --device cuda, or --device rocm; use --simulator go for CPU debugging")
    if args.league_size < 2 or not 0 < args.gamma <= 1 or not 0 <= args.gae_lambda <= 1 or args.target_kl < 0:
        p.error("league size >=2, gamma in (0,1], GAE lambda in [0,1], target KL >=0 required")
    validation_range = range(args.eval_seed, args.eval_seed + args.eval_episodes)
    if any(seed in validation_range for seed in range(args.test_seed, args.test_seed + args.test_episodes)):
        p.error("validation and final-test seeds must not overlap")
    return args


if __name__ == "__main__":
    train(arguments())
