#!/usr/bin/env python3
"""Pause/resume supervisor for open-ended Wriggle training.

train.py already checkpoints every update and can --resume, but it runs for a
fixed number of updates. This wrapper runs it as an endless series of short
"chunks" (each one `train.py --resume ... --updates N`) inside a detached
daemon, so training can run for as long as you like and be paused at will:

    training/wtrain start  mybot --max-updates 0 -- --personalities --hidden 64
    training/wtrain status mybot
    training/wtrain pause  mybot      # saves a checkpoint, frees the GPU
    training/wtrain resume mybot      # continues from that checkpoint
    training/wtrain stop   mybot      # pause + daemon exits (resume restarts it)
    training/wtrain logs   mybot -f

Everything after `--` is passed to train.py. The wrapper owns --output,
--updates, --resume and --test-episodes. Pause/stop send SIGTERM, which
train.py turns into "save checkpoint and exit", so at most the update in
flight is lost. Pausing survives the terminal closing; after a reboot just run
`resume` again. State lives in .training/runs/<name>/.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("WRIGGLE_RUNS_DIR") or ROOT / ".training/runs")
TRAIN = Path(os.environ.get("WRIGGLE_TRAIN_SCRIPT") or ROOT / "training/train.py")
POLL = float(os.environ.get("WRIGGLE_RUNNER_POLL", "0.5"))        # control-file poll (s)
GRACE = float(os.environ.get("WRIGGLE_RUNNER_GRACE", "180"))      # SIGTERM -> SIGKILL (s)
BACKOFF = float(os.environ.get("WRIGGLE_RUNNER_BACKOFF", "10"))   # per consecutive failure (s)
MAX_FAILURES = 3
LOG_LIMIT = 32 << 20
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
RESERVED = ("--updates", "--resume", "--output", "--test-episodes")
INIT_FLAGS = ("--init-v1", "--init-v2")
# train.py phases that happen before the first update; a checkpoint saved
# while still in them has not finished warm-starting.
UNSEEDED_PHASES = ("initial_validation", "warm_start")


# ---------------------------------------------------------------- helpers
def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def has_flag(args, flag):
    return any(a == flag or a.startswith(flag + "=") for a in args)


def strip_flags(args, flags):
    """Drop `--flag value` and `--flag=value` for each flag."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a in flags:
            skip = True
        elif any(a.startswith(f + "=") for f in flags):
            continue
        else:
            out.append(a)
    return out


def run_dir(name):
    if not NAME_RE.match(name):
        raise SystemExit(f"invalid run name {name!r} (letters, digits, . _ - ; max 64)")
    return RUNS / name


def checkpoint_path(output):
    return Path(output).with_suffix(".pt")


def chunk_command(cfg, updates, resume):
    args = strip_flags(cfg["train_args"], INIT_FLAGS) if resume else list(cfg["train_args"])
    # Wrapper flags go last so they win over anything abbreviated in train_args.
    cmd = [sys.executable, "-u", str(TRAIN), *args, "--output", cfg["output"],
           "--updates", str(updates), "--test-episodes", str(cfg["chunk_test_episodes"])]
    if resume:
        cmd += ["--resume", str(checkpoint_path(cfg["output"]))]
    return cmd


def read_control(rd):
    try:
        value = (rd / "control").read_text().strip()
    except OSError:
        return "run"
    return value if value in ("run", "pause", "stop") else "run"


def write_control(rd, value):
    (rd / "control").write_text(value + "\n")


def daemon_alive(rd):
    lock = (rd / "daemon.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        lock.close()
    return False


def terminate(child):
    child.terminate()
    try:
        child.wait(timeout=GRACE)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


# ------------------------------------------------------------------ daemon
def daemon(name):
    rd = run_dir(name)
    lock = (rd / "daemon.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"{name}: daemon already running", file=sys.stderr)
        return 1
    cfg = json.loads((rd / "config.json").read_text())
    output = Path(cfg["output"])
    stop_signal = {"set": False}

    def on_signal(signum, frame):
        stop_signal["set"] = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    state = read_json(rd / "state.json")
    state.setdefault("started_at", time.time())
    state.setdefault("chunk", 0)
    state.setdefault("train_seconds", 0.0)
    if "completed_updates" not in state:
        state["completed_updates"] = read_json(output.with_suffix(".status.json")).get("update", 0)
    state.update(daemon_pid=os.getpid(), child_pid=None, failures=0)

    def publish(new_state, message="", **extra):
        state.update(state=new_state, message=message, updated_at=time.time(), **extra)
        atomic_json(rd / "state.json", state)

    def wanted():
        return "stop" if stop_signal["set"] else read_control(rd)

    def nap(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end and wanted() == "run":
            time.sleep(min(POLL, max(0.0, end - time.monotonic())))

    publish("starting")
    try:
        while True:
            want = wanted()
            if want == "stop":
                publish("stopped", "stopped by request")
                return 0
            if want == "pause":
                if state["state"] != "paused":
                    publish("paused", f"paused at update {state['completed_updates']}")
                time.sleep(POLL)
                continue
            cfg = json.loads((rd / "config.json").read_text())  # allow edits between chunks
            cap = cfg["max_updates"]
            if cap and state["completed_updates"] >= cap:
                publish("done", f"reached max_updates={cap}")
                return 0
            size = min(cfg["chunk"], cap - state["completed_updates"]) if cap else cfg["chunk"]
            outcome = run_chunk(rd, cfg, size, state, publish, wanted)
            if outcome == "failed":
                state["failures"] += 1
                if state["failures"] >= MAX_FAILURES:
                    publish("failed", f"trainer failed {MAX_FAILURES} times in a row; see train.log")
                    return 1
                publish("backoff", f"trainer failed ({state['failures']}/{MAX_FAILURES}); retrying")
                nap(BACKOFF * state["failures"])
            elif outcome == "ok":
                state["failures"] = 0
            elif outcome == "external":
                write_control(rd, "pause")  # someone else SIGTERMed train.py; don't spin
                publish("paused", "trainer was interrupted externally")
    finally:
        state["child_pid"] = None
        if state.get("state") in ("running", "starting", "backoff"):
            publish("stopped", "daemon exited")
        lock.close()


def run_chunk(rd, cfg, size, state, publish, wanted):
    """Run one train.py invocation. Returns ok | failed | requested | external."""
    output = Path(cfg["output"])
    resume = checkpoint_path(output).exists()
    cmd = chunk_command(cfg, size, resume)
    log_path = rd / "train.log"
    if log_path.exists() and log_path.stat().st_size > LOG_LIMIT:
        log_path.replace(rd / "train.log.1")
    started = time.monotonic()
    before = state["completed_updates"]
    with log_path.open("ab") as log:
        log.write(f"\n=== chunk {state['chunk'] + 1}: {'resume' if resume else 'fresh'}, "
                  f"{size} updates ===\n".encode())
        log.flush()
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True,
                                 env=dict(os.environ, PYTHONFAULTHANDLER="1"))
        publish("running", f"chunk {state['chunk'] + 1}: {size} updates", child_pid=child.pid,
                chunk_size=size)
        asked, unseeded = False, False
        while child.poll() is None:
            if wanted() != "run":
                asked = True
                # train.py overwrites the phase with "interrupted" on shutdown, so
                # sample it now. Phases only move forward: a stale read can only
                # make us discard too much, never too little.
                before_stop = read_json(output.with_suffix(".status.json"))
                unseeded = (before_stop.get("pid") != child.pid
                            or before_stop.get("phase") in UNSEEDED_PHASES)
                terminate(child)
                break
            time.sleep(POLL)
    state["train_seconds"] += time.monotonic() - started
    state["child_pid"] = None
    status = read_json(output.with_suffix(".status.json"))
    mine = status.get("pid") == child.pid
    if mine:
        state["completed_updates"] = status.get("update", state["completed_updates"])
    if asked:
        if not resume and unseeded:
            # Interrupted before warm start finished: a checkpoint here would
            # make the next chunk resume from an un-warm-started model.
            checkpoint_path(output).unlink(missing_ok=True)
            state["completed_updates"] = 0
        return "requested"
    code = child.returncode
    if code == 0 and mine and status.get("phase") == "complete":
        if state["completed_updates"] <= before:
            return "failed"  # chunk "finished" without progress
        state["chunk"] += 1
        archive_report(rd, output, state["chunk"])
        return "ok"
    if code == 0 and mine and status.get("phase") == "interrupted":
        return "external"
    return "failed"


def archive_report(rd, output, chunk):
    report = output.with_suffix(".report.json")
    if report.exists():
        folder = rd / "reports"
        folder.mkdir(exist_ok=True)
        shutil.copy2(report, folder / f"chunk-{chunk:05d}.json")


# --------------------------------------------------------------------- CLI
def spawn_daemon(name):
    rd = run_dir(name)
    with (rd / "daemon.log").open("ab") as log:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_daemon", name],
                         cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)


def wait_for(rd, states, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        current = read_json(rd / "state.json").get("state")
        if current in states:
            return current
        time.sleep(0.2)
    return None


def require_run(name):
    rd = run_dir(name)
    if not (rd / "config.json").exists():
        raise SystemExit(f"no run named {name!r} (see: list)")
    return rd


def cmd_start(a, train_args):
    rd = run_dir(a.name)
    if (rd / "config.json").exists():
        raise SystemExit(f"run {a.name!r} already exists; use resume (or pick a new name)")
    for flag in RESERVED:
        if has_flag(train_args, flag):
            raise SystemExit(f"{flag} is controlled by the wrapper"
                             + (" (use --chunk-test-episodes)" if flag == "--test-episodes" else ""))
    if a.chunk < 1 or a.max_updates < 0 or a.chunk_test_episodes < 1:
        raise SystemExit("--chunk and --chunk-test-episodes must be >= 1, --max-updates >= 0")
    output = Path(a.output or f"models/{a.name}.json")
    output = (output if output.is_absolute() else ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rd.mkdir(parents=True)
    atomic_json(rd / "config.json", {
        "name": a.name, "output": str(output), "chunk": a.chunk, "max_updates": a.max_updates,
        "chunk_test_episodes": a.chunk_test_episodes, "train_args": train_args,
        "created_at": time.time()})
    write_control(rd, "run")
    if a.foreground:
        return daemon(a.name)
    spawn_daemon(a.name)
    state = wait_for(rd, ("running", "failed", "done", "stopped"), 30)
    resumed = " (continuing existing checkpoint)" if checkpoint_path(output).exists() else ""
    print(f"{a.name}: {state or 'starting'}{resumed}; output {output}")
    print(f"  status: training/wtrain status {a.name}   pause: training/wtrain pause {a.name}")
    return 0


def cmd_pause(a, _):
    rd = require_run(a.name)
    if not daemon_alive(rd):
        print(f"{a.name}: not running (state: {read_json(rd / 'state.json').get('state')})")
        return 0
    write_control(rd, "pause")
    print(f"{a.name}: pausing, waiting for the checkpoint to be written...", flush=True)
    state = wait_for(rd, ("paused", "stopped", "failed", "done"), GRACE + 60)
    s = read_json(rd / "state.json")
    print(f"{a.name}: {state or 'still pausing'} at update {s.get('completed_updates')}")
    return 0 if state else 1


def cmd_resume(a, _):
    rd = require_run(a.name)
    cfg = read_json(rd / "config.json")
    if a.max_updates is not None or a.chunk is not None:
        if a.max_updates is not None:
            cfg["max_updates"] = a.max_updates
        if a.chunk is not None:
            cfg["chunk"] = a.chunk
        atomic_json(rd / "config.json", cfg)
    state = read_json(rd / "state.json")
    cap = cfg["max_updates"]
    if cap and state.get("completed_updates", 0) >= cap:
        raise SystemExit(f"{a.name}: already reached max_updates={cap}; "
                         "use resume --max-updates N (0 = unlimited)")
    write_control(rd, "run")
    if not daemon_alive(rd):
        spawn_daemon(a.name)
    result = wait_for(rd, ("running", "failed", "done"), 30)
    print(f"{a.name}: {result or 'starting'}")
    return 0


def cmd_stop(a, _):
    rd = require_run(a.name)
    if not daemon_alive(rd):
        print(f"{a.name}: not running")
        return 0
    write_control(rd, "stop")
    end = time.monotonic() + GRACE + 60
    while daemon_alive(rd) and time.monotonic() < end:
        time.sleep(0.2)
    s = read_json(rd / "state.json")
    print(f"{a.name}: {s.get('state')} at update {s.get('completed_updates')}")
    return 0


def tail_lines(path, count, window=256 << 10):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - window))
            return f.read().decode(errors="replace").splitlines()[-count:]
    except OSError:
        return []


def last_json(lines, key):
    for line in reversed(lines):
        if line.startswith("{") and f'"{key}"' in line:
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def age(seconds):
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m"
    return f"{seconds / 3600:.1f}h"


def describe(name):
    rd = run_dir(name)
    cfg, s = read_json(rd / "config.json"), read_json(rd / "state.json")
    alive = daemon_alive(rd)
    shown = s.get("state", "new") if alive else (
        "dead (daemon not running; was " + str(s.get("state")) + ")" if s.get("state") in
        ("running", "starting", "backoff") else s.get("state", "new"))
    lines = [f"{name}: {shown}" + (f" - {s['message']}" if s.get("message") else "")]
    cap = cfg.get("max_updates")
    lines.append(f"  updates: {s.get('completed_updates', 0)}"
                 f"{f' / {cap}' if cap else ' (unlimited)'}, chunk {s.get('chunk', 0) + 1}"
                 f" (size {cfg.get('chunk')}), trained {age(s.get('train_seconds', 0))}")
    log = tail_lines(rd / "train.log", 400)
    metric, validation = last_json(log, "mean_reward"), last_json(log, "validation_update")
    if metric:
        lines.append(f"  latest: update {metric['update']} reward {metric['mean_reward']:.3f} "
                     f"kills {metric['kills']:.0f} deaths {metric['deaths']:.0f}"
                     f" ({metric.get('iteration_seconds', 0):.1f}s/update)")
    if validation:
        lines.append(f"  last validation: update {validation['validation_update']} "
                     f"return {validation['return']:.2f}")
    output = Path(cfg.get("output", ""))
    best = read_json(output.with_suffix(".report.json")) if cfg else {}
    if best.get("best_validation_return") is not None:
        lines.append(f"  best validation: return {best['best_validation_return']:.2f} "
                     f"at update {best.get('best_update')} -> {output.with_suffix('.best.json')}")
    if alive:
        lines.append(f"  daemon pid {s.get('daemon_pid')}, trainer pid {s.get('child_pid')}")
    return "\n".join(lines)


def cmd_status(a, _):
    if a.name:
        require_run(a.name)
        print(describe(a.name))
        return 0
    return cmd_list(a, _)


def cmd_list(a, _):
    names = sorted(p.parent.name for p in RUNS.glob("*/config.json")) if RUNS.exists() else []
    if not names:
        print("no runs")
    for name in names:
        print(describe(name))
    return 0


def cmd_logs(a, _):
    rd = require_run(a.name)
    path = rd / "train.log"
    for line in tail_lines(path, a.lines):
        print(line)
    if a.follow:
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                while True:
                    chunk = f.read()
                    if chunk:
                        sys.stdout.write(chunk.decode(errors="replace"))
                        sys.stdout.flush()
                    else:
                        time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("start", help="create a run and start training (args after -- go to train.py)")
    s.add_argument("name")
    s.add_argument("--output", help="model JSON path (default models/<name>.json); an existing "
                                    "<output>.pt checkpoint is continued, not restarted")
    s.add_argument("--chunk", type=int, default=50, help="updates per train.py invocation (default 50)")
    s.add_argument("--max-updates", type=int, default=0, help="total updates to stop at; 0 = unlimited (default)")
    s.add_argument("--chunk-test-episodes", type=int, default=7,
                   help="episodes for the end-of-chunk test pass (default 7; keeps chunks cheap)")
    s.add_argument("--foreground", action="store_true", help="run in this terminal; Ctrl+C saves and exits")
    s.set_defaults(func=cmd_start)
    for cmd, fn, text in (("pause", cmd_pause, "checkpoint and pause training"),
                          ("resume", cmd_resume, "resume a paused/stopped run"),
                          ("stop", cmd_stop, "pause and exit the daemon")):
        q = sub.add_parser(cmd, help=text)
        q.add_argument("name")
        if cmd == "resume":
            q.add_argument("--max-updates", type=int, help="change the total-update cap (0 = unlimited)")
            q.add_argument("--chunk", type=int, help="change the chunk size")
        q.set_defaults(func=fn)
    q = sub.add_parser("status", help="show one run, or all runs")
    q.add_argument("name", nargs="?")
    q.set_defaults(func=cmd_status)
    q = sub.add_parser("list", help="show all runs")
    q.set_defaults(func=cmd_list)
    q = sub.add_parser("logs", help="show the trainer log")
    q.add_argument("name")
    q.add_argument("-n", "--lines", type=int, default=30)
    q.add_argument("-f", "--follow", action="store_true")
    q.set_defaults(func=cmd_logs)
    d = sub.add_parser("_daemon")
    d.add_argument("name")
    d.set_defaults(func=lambda a, _: daemon(a.name))
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    train_args = []
    if "--" in argv:
        index = argv.index("--")
        argv, train_args = argv[:index], argv[index + 1:]
    args = build_parser().parse_args(argv)
    if train_args and args.command != "start":
        raise SystemExit("arguments after -- are only used by start")
    return args.func(args, train_args)


if __name__ == "__main__":
    raise SystemExit(main())
