#!/usr/bin/env python3
"""Compile an authored circuit; no training, simulation, or parameter fitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TURNS = (-0.9, -0.4, -0.15, 0.0, 0.15, 0.4, 0.9)
NEWEST = 3 * 172


def compile_policy():
    # Exactly the shipped wriggle-v6 architecture, including unused inputs.
    p = dict(version=3, w1=[[0.0] * 692 for _ in range(64)],
             b1=[0.0] * 64, w2=[[0.0] * 64 for _ in range(14)], b2=[0.0] * 14)
    # Smooth danger gates: g=(1+tanh((D-distance)/temperature))/2.
    # Clearance subtracts .85 already; border uses its own 40-unit scale.
    for j, sector in enumerate(range(-8, 9)):
        for channel, (distance, temperature, scale) in enumerate(
                ((8.0, 2.0, 30.0), (12.0, 3.0, 30.0), (14.0, 3.0, 40.0))):
            neuron = j * 3 + channel
            p['w1'][neuron][NEWEST + 76 + (sector % 24) * 4 + channel] = -scale / temperature
            p['b1'][neuron] = distance / temperature
    # Small-signal signed coordinate relays: tanh(.5*x), x in [-1,1].
    for neuron, feature in ((51, 16), (52, 17), (53, 4), (54, 5)):
        p['w1'][neuron][NEWEST + feature] = 0.5
    # Broader escape-space preference; not a free-space/pathfinding calculation.
    for j, sector in enumerate(range(-4, 5)):
        p['w1'][55+j][NEWEST + 76 + (sector % 24) * 4] = 2.0
    for action, theta in enumerate(TURNS):
        # A normalized angular tent over neighboring rays, 60 degrees each side.
        weights = [max(0.0, 1.0 - abs(s * math.pi / 12 - theta) / (math.pi / 3))
                   for s in range(-8, 9)]
        total = sum(weights)
        for j, weight in enumerate(weights):
            for channel in range(3):
                p['w2'][action][3*j+channel] = -60.0 * weight / total
        # Constants -60 per risk channel implement -(120*g) exactly.
        p['b2'][action] = -180.0 - 0.03 * abs(theta)
        p['w2'][action][51] = 8.0 * math.cos(theta)
        p['w2'][action][52] = 8.0 * math.sin(theta)
        p['w2'][action][53] = 2.0 * math.cos(theta)
        p['w2'][action][54] = 2.0 * math.sin(theta)
        escape = [max(0.0, 1.0 - abs(s * math.pi / 12 - theta) / (math.pi / 2))
                  for s in range(-4, 5)]
        for j, weight in enumerate(escape):
            p['w2'][action][55+j] = 0.5 * weight / sum(escape)
        # Exact pairwise suppression, no masking or runtime policy additions.
        p['w2'][action+7] = p['w2'][action].copy()
        p['b2'][action+7] = p['b2'][action] - 1.0
    return p


def serialized():
    return (json.dumps(compile_policy(), separators=(',', ':'), allow_nan=False) + '\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/authored.json')
    parser.add_argument('--check', action='store_true', help='Verify byte-for-byte reproduction without writing')
    args = parser.parse_args()
    data = serialized()
    if args.check:
        if args.output.read_bytes() != data:
            raise SystemExit('artifact differs from compiled circuit')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
    print(hashlib.sha256(data).hexdigest(), args.output)


if __name__ == '__main__':
    main()
