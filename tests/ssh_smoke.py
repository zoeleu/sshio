"""Two real OpenSSH sessions, a PTY resize, disconnect isolation, and exec denial.
Run against SSHIO_BIND=127.0.0.1:2222; no third-party Python modules.
"""
import fcntl
import os
import pty
import select
import signal
import struct
import subprocess
import termios
import time

COMMAND = ["ssh", "-tt", "-p", os.environ.get("SSHIO_TEST_PORT", "2222"), "-o", "StrictHostKeyChecking=no",
           "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
           "-o", "PreferredAuthentications=none", "localhost"]

def connect():
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
    proc = subprocess.Popen(COMMAND, stdin=slave, stdout=slave, stderr=slave)
    os.close(slave)
    return proc, master

def read_for(fd, seconds):
    result = bytearray()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if select.select([fd], [], [], min(0.1, max(0, deadline-time.monotonic())))[0]:
            try:
                result.extend(os.read(fd, 65536))
            except OSError:
                break
    return bytes(result)

clients = []
try:
    clients = [connect(), connect()]
    for proc, fd in clients:
        data = read_for(fd, 1)
        assert proc.poll() is None and b"sshio" in data and b"Name:" in data, data[-500:]
        os.write(fd, b"tester\t\r")
    for proc, fd in clients:
        data = read_for(fd, 0.5)
        assert proc.poll() is None and b"mass" in data, data[-500:]
    os.write(clients[0][1], b"\x1b[")
    os.write(clients[0][1], b"A ")
    fcntl.ioctl(clients[0][1], termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    clients[0][0].send_signal(signal.SIGWINCH)
    resized = read_for(clients[0][1], 0.5)
    assert b"\x1b[39;" in resized, resized[-500:]
    os.write(clients[0][1], b"q")
    clients[0][0].wait(timeout=3)
    assert clients[1][0].poll() is None
    os.write(clients[1][1], b"d")
    assert read_for(clients[1][1], 0.5)
    denied = subprocess.run(COMMAND[:-1] + ["-T", "localhost", "echo forbidden"], capture_output=True, timeout=5)
    assert denied.returncode != 0 and b"forbidden" not in denied.stdout
    print("two SSH clients, split arrows, resize, disconnect isolation, and exec denial passed")
finally:
    for proc, fd in clients:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=3)
        os.close(fd)
