#!/usr/bin/env python3
"""Run a trainer in the foreground of a detached supervisor and record its exit.

Launch this script with an external log redirection / new session. Arguments
after -- are passed to train.py. It does not restart failed training jobs.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    args = sys.argv[1:]
    if args and args[0] == "--":
        args = args[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", default="models/wriggle-gpu.json")
    options, _ = parser.parse_known_args(args)
    output = Path(options.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONFAULTHANDLER="1")
    child = subprocess.Popen([sys.executable, "-u", str(ROOT / "training/train.py"), *args],
                             cwd=ROOT, env=environment)
    print(json.dumps({"trainer_pid": child.pid, "supervisor_pid": os.getpid()}), flush=True)

    def stop(signum, frame):
        if child.poll() is None:
            child.send_signal(signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    code = child.wait()
    exit_record = {"trainer_pid": child.pid, "exit_code": code, "ended_at": time.time()}
    status_path = output.with_suffix(".status.json")
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("pid") == child.pid and status.get("running"):
            status.update(running=False, phase="unexpected_exit", exit_code=code)
            temporary = status_path.with_suffix(".json.supervisor.tmp")
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(status_path)
    destination = output.with_suffix(".exit.json")
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(exit_record, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps(exit_record), flush=True)
    return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
