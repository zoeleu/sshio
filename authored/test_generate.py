"""Artifact reproducibility and analytic signal-bound checks (no optimization)."""
import hashlib
import json
import math
from pathlib import Path
import unittest

from generate import ROOT, compile_policy, serialized


class CircuitTests(unittest.TestCase):
    def test_reproduction(self):
        data = serialized()
        for name in ('authored-initial.json', 'authored.json'):
            self.assertEqual(data, (ROOT / 'models' / name).read_bytes())
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         'd2409101346528deb8b458d6dbf47d8fc90c938cd1daf6eb5ace294509f5df9a')
        self.assertEqual(json.loads(data), compile_policy())

    def test_parameters(self):
        p = compile_policy()
        self.assertEqual(len(p['w1']), 64)
        self.assertTrue(all(len(row) == 692 for row in p['w1']))
        self.assertEqual(len(p['w2']), 14)
        self.assertTrue(all(len(row) == 64 for row in p['w2']))
        flat = p['b1'] + p['b2'] + sum(p['w1'], []) + sum(p['w2'], [])
        self.assertEqual(len(flat), 45262)
        self.assertTrue(all(math.isfinite(x) for x in flat))
        # Histories and personalities deliberately ignored; no external modulation in v3.
        self.assertTrue(all(not any(row[:516]) and not any(row[688:]) for row in p['w1']))

    def test_risk_dominance_bound(self):
        p = compile_policy()
        # For each pair, bound all non-danger contributions using actual tanh
        # ranges. No sampled observations and no fitted parameters are involved.
        for a in range(7):
            for b in range(7):
                delta = [x-y for x,y in zip(p['w2'][a],p['w2'][b])]
                bound = abs(p['b2'][a]-p['b2'][b])
                bound += sum(abs(d)*math.tanh(.5) for d in delta[51:55])
                bound += sum(max(0,d)*math.tanh(2) for d in delta[55:])
                self.assertLess(bound, 9.5)
        # Every danger channel is a normalized angular average with coefficient -120.
        for a in range(7):
            for channel in range(3):
                self.assertAlmostEqual(sum(p['w2'][a][channel:51:3]), -60)


if __name__ == '__main__':
    unittest.main()
