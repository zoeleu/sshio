"""Lifecycle tests for runner.py using a fake train.py (no torch/GPU needed)."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

import runner

RUNNER = Path(runner.__file__).resolve()

FAKE_TRAINER = textwrap.dedent('''
    import argparse, json, os, signal, sys, time
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--output"); p.add_argument("--updates", type=int)
    p.add_argument("--resume"); p.add_argument("--test-episodes")
    a, _ = p.parse_known_args()
    out = Path(a.output)
    with open(str(out) + ".argv.log", "a") as f:
        f.write(json.dumps(sys.argv[1:]) + "\\n")
    if os.environ.get("FAKE_FAIL"):
        sys.exit(1)
    ckpt, status = out.with_suffix(".pt"), out.with_suffix(".status.json")
    start = int(ckpt.read_text()) if a.resume else 0
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    def write(phase, update):
        status.write_text(json.dumps({"pid": os.getpid(), "running": phase not in
                                      ("complete", "interrupted"), "phase": phase, "update": update}))
    if os.environ.get("FAKE_SLOW_WARMUP") and not a.resume:
        write("warm_start", 0)
        while not stop:
            time.sleep(0.02)
        ckpt.write_text("0")  # like train.py: persist() runs on interrupt
        write("interrupted", 0)
        sys.exit(0)
    done = start
    while done < start + a.updates and not stop:
        write("rollout", done)
        time.sleep(0.05)
        done += 1
        ckpt.write_text(str(done))
        write("checkpoint", done)
    write("interrupted" if stop else "complete", done)
''')


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "fake_train.py").write_text(FAKE_TRAINER)
        self.env = dict(os.environ, WRIGGLE_RUNS_DIR=str(self.dir / "runs"),
                        WRIGGLE_TRAIN_SCRIPT=str(self.dir / "fake_train.py"),
                        WRIGGLE_RUNNER_POLL="0.05", WRIGGLE_RUNNER_BACKOFF="0.05",
                        WRIGGLE_RUNNER_GRACE="10")
        self.names = []

    def tearDown(self):
        for name in self.names:
            subprocess.run([sys.executable, str(RUNNER), "stop", name], env=self.env,
                           capture_output=True, timeout=30)
        self.tmp.cleanup()

    def cli(self, *args, check=True):
        result = subprocess.run([sys.executable, str(RUNNER), *args], env=self.env,
                                capture_output=True, text=True, timeout=60)
        if check:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def start(self, name, *extra, train=()):
        self.names.append(name)
        return self.cli("start", name, "--output", str(self.dir / f"{name}.json"), *extra, "--", *train)

    def state(self, name):
        return json.loads((self.dir / "runs" / name / "state.json").read_text())

    def wait(self, name, predicate, timeout=20):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                if predicate(self.state(name)):
                    return self.state(name)
            except (OSError, ValueError):
                pass
            time.sleep(0.05)
        self.fail(f"timed out; state={self.state(name)}")

    def argv(self, name):
        return [json.loads(l) for l in (self.dir / f"{name}.json.argv.log").read_text().splitlines()]

    def test_pause_resume_unlimited_and_stop(self):
        self.start("a", "--chunk", "4", "--chunk-test-episodes", "3",
                   train=["--personalities", "--init-v2", "donor.json"])
        self.wait("a", lambda s: s["completed_updates"] >= 6)  # spans a chunk boundary
        self.cli("pause", "a")
        paused = self.state("a")
        self.assertEqual(paused["state"], "paused")
        self.assertIsNone(paused["child_pid"])
        time.sleep(0.5)
        self.assertEqual(self.state("a")["completed_updates"], paused["completed_updates"])
        self.assertTrue(runner.daemon_alive(self.dir / "runs" / "a"))  # daemon stays for resume
        self.assertEqual(int((self.dir / "a.pt").read_text()), paused["completed_updates"])

        self.cli("resume", "a")
        self.wait("a", lambda s: s["state"] == "running"
                  and s["completed_updates"] > paused["completed_updates"] + 2)
        self.cli("stop", "a")
        self.assertFalse(runner.daemon_alive(self.dir / "runs" / "a"))
        self.assertEqual(self.state("a")["state"], "stopped")
        # stop -> resume restarts the dead daemon and keeps going from the checkpoint.
        stopped_at = self.state("a")["completed_updates"]
        self.cli("resume", "a")
        self.wait("a", lambda s: s["completed_updates"] > stopped_at)

        calls = self.argv("a")
        self.assertNotIn("--resume", calls[0])
        self.assertIn("--init-v2", calls[0])
        for later in calls[1:]:
            self.assertIn("--resume", later)
            self.assertNotIn("--init-v2", later)
            self.assertEqual(later[later.index("--test-episodes") + 1], "3")
            self.assertEqual(later[later.index("--output") + 1], str(self.dir / "a.json"))
        self.assertIn("running", self.cli("status", "a").stdout)

    def test_max_updates_finishes_with_exact_chunks(self):
        self.start("b", "--chunk", "3", "--max-updates", "8")
        state = self.wait("b", lambda s: s["state"] == "done")
        self.assertEqual(state["completed_updates"], 8)
        self.assertEqual([c[c.index("--updates") + 1] for c in self.argv("b")], ["3", "3", "2"])
        refused = self.cli("resume", "b", check=False)
        self.assertNotEqual(refused.returncode, 0)
        self.cli("resume", "b", "--max-updates", "10")
        self.wait("b", lambda s: s["state"] == "done" and s["completed_updates"] == 10)

    def test_repeated_failures_mark_run_failed(self):
        self.env["FAKE_FAIL"] = "1"
        self.start("c", "--chunk", "2")
        state = self.wait("c", lambda s: s["state"] == "failed")
        self.assertEqual(state["failures"], runner.MAX_FAILURES)
        self.assertEqual(len(self.argv("c")), runner.MAX_FAILURES)

    def test_pause_during_warm_start_discards_unseeded_checkpoint(self):
        self.env["FAKE_SLOW_WARMUP"] = "1"
        self.start("d", "--chunk", "2")
        self.wait("d", lambda s: s["state"] == "running")
        time.sleep(0.3)
        self.cli("pause", "d")
        self.assertFalse((self.dir / "d.pt").exists())  # next start redoes the warm start
        self.assertEqual(self.state("d")["completed_updates"], 0)

    def test_rejects_reserved_flags_duplicates_and_bad_names(self):
        self.assertNotEqual(self.cli("start", "e", "--", "--updates", "5", check=False).returncode, 0)
        self.assertFalse((self.dir / "runs" / "e").exists())
        self.assertNotEqual(self.cli("start", "../x", check=False).returncode, 0)
        self.start("f", "--max-updates", "1", "--chunk", "1")
        self.assertNotEqual(self.cli("start", "f", check=False).returncode, 0)

    def test_helpers(self):
        self.assertEqual(runner.strip_flags(["--a", "--init-v2", "x", "--init-v1=y", "--b", "1"],
                                            runner.INIT_FLAGS), ["--a", "--b", "1"])
        self.assertTrue(runner.has_flag(["--output=x"], "--output"))
        self.assertFalse(runner.has_flag(["--outputs"], "--output"))


if __name__ == "__main__":
    unittest.main()
