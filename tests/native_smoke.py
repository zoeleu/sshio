"""Run after installing the wheel; optional native fingerprint as argv[1]."""
import sys
from sshio import Batch

left, right = Batch(), Batch()
left.step([2] * 8)
right.step([2] * 8)
assert left.hashes() == right.hashes()
if len(sys.argv) > 1:
    assert left.hashes() == [int(sys.argv[1])], (left.hashes(), sys.argv[1])
for tick in range(100):
    actions = [(tick + slot) % 10 for slot in range(8)]
    obs, rewards, done = left.step(actions)
    right.step(actions)
    assert len(obs) == 8 and all(len(o) == 76 for o in obs)
    assert len(rewards) == len(done) == 8
    assert left.hashes() == right.hashes()
    assert left.mass_totals() == [20_000]
try:
    left.step([99] * 8)
except ValueError:
    pass
else:
    raise AssertionError("invalid action accepted")
print("native parity, deterministic stepping, shape, bounds, and mass checks passed")
