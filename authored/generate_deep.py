#!/usr/bin/env python3
"""Directly compile a 692 -> 64 tanh -> 64 tanh -> 14 policy, without fitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from generate import ROOT, TURNS, NEWEST, compile_policy as shallow_circuit


def compile_policy():
    # Reuse the analytic first-layer feature assignments, never a weight file.
    features = shallow_circuit()
    p = dict(version=3, w1=[[0.0]*692 for _ in range(64)], b1=[0.0]*64,
             wh=[[0.0]*64 for _ in range(64)], bh=[0.0]*64,
             w2=[[0.0]*64 for _ in range(14)], b2=[0.0]*14)
    for j in range(64):
        for i, weight in enumerate(features['w1'][j]):
            p['w1'][j][i] = weight
        p['b1'][j] = features['b1'][j]
    # The shallow circuit's six outermost ray units have no outgoing weight.
    # Replace them with soft predicates for conservative straight-line boosting.
    for unit, feature, slope, threshold in (
            (0,14,8.0,.36),       # sufficient score headroom
            (1,16,12.0,.10),      # food at least four units ahead
            (2,18,-12.0,.45),     # target within 27 units
            (48,19,6.0,.12),      # target augmented value above 2.16
            (49,17,60.0,-.05),    # within two units laterally
            (50,17,-60.0,.05)):
        p['w1'][unit] = [0.0]*692
        p['w1'][unit][NEWEST+feature] = slope
        p['b1'][unit] = -slope*threshold
    for a, theta in enumerate(TURNS):
        q = [max(0.0, 1-abs(s*math.pi/12-theta)/(math.pi/3)) for s in range(-8,9)]
        total = sum(q)
        # R = sum_channels sum_rays q*(1+h)/2, in [0,3].
        risk = [0.0]*64
        for j, weight in enumerate(q):
            for channel in range(3):
                risk[3*j+channel] = .5*weight/total
        # Almost-linear relay: -12000*tanh(.01*R) approximates -120*R.
        p['wh'][a] = [.01*w for w in risk]
        p['bh'][a] = .015
        p['w2'][a][a] = -12000.0
        # Two opposite shifts implement food gating, not a runtime multiplier:
        # 16*(tanh(z+10R)+tanh(z-10R)), z=.25*(cos*h51+sin*h52).
        for offset, sign in enumerate((1,-1)):
            unit = 7+2*a+offset
            p['wh'][unit] = [sign*10*w for w in risk]
            p['bh'][unit] = sign*15.0
            p['wh'][unit][51] = .25*math.cos(theta)
            p['wh'][unit][52] = .25*math.sin(theta)
            p['w2'][a][unit] = 16.0
        p['w2'][a][21] = 20*math.cos(theta)
        p['w2'][a][22] = 20*math.sin(theta)
        escape = [max(0.0,1-abs(s*math.pi/12-theta)/(math.pi/2)) for s in range(-4,5)]
        for j, weight in enumerate(escape):
            p['w2'][a][23+j] = 5*weight/sum(escape)
        p['b2'][a] = -.03*abs(theta)
        if a == 3:
            # AND-like confidence gate; each predicate <=1. Missing any one
            # (h<=0) caps sum at 5, hence gate <=tanh(-2), even with R=0.
            p['wh'][32] = [-80*w for w in risk]
            for unit in (0,1,2,48,49,50):
                p['wh'][32][unit] = 4.0
            p['bh'][32] = -4*5.5 - 80*1.5
    for j, source in enumerate((53,54)):
        p['wh'][21+j][source] = .1
    for j in range(9):
        p['wh'][23+j][55+j] = .1
    for a in range(7):
        p['w2'][a+7] = p['w2'][a].copy()
        p['b2'][a+7] = p['b2'][a]-1.0
    p['w2'][10][32] = 2.0  # Only straight boosting can beat its paired action.
    return p


def serialized():
    return (json.dumps(compile_policy(),separators=(',',':'),allow_nan=False)+'\n').encode()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'models/authored-deep.json')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(); data=serialized()
    if args.check:
        if args.output.read_bytes()!=data:
            raise SystemExit('artifact differs from compiled circuit')
    else:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_bytes(data)
    print(hashlib.sha256(data).hexdigest(),args.output)

if __name__=='__main__':
    main()
