"""Start a temporary-key local server and run the real SSH smoke check."""
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

with tempfile.TemporaryDirectory(prefix="sshio-check-") as directory:
    key = str(Path(directory) / "host_key")
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", key], check=True)
    with socket.socket() as socket_probe:
        socket_probe.bind(("127.0.0.1", 0))
        port = socket_probe.getsockname()[1]
    env = dict(os.environ, SSHIO_BIND=f"127.0.0.1:{port}", SSHIO_HOST_KEY=key, SSHIO_TEST_PORT=str(port))
    with open(Path(directory) / "server.log", "w+") as log:
        server = subprocess.Popen(["target/debug/sshio-server"], env=env, stdout=log, stderr=log)
        try:
            for _ in range(100):
                if server.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(log.read())
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                raise RuntimeError("server did not become ready")
            subprocess.run(["python3", "tests/ssh_smoke.py"], env=env, check=True, timeout=30)
        finally:
            server.terminate()
            server.wait(timeout=3)
