"""Check the compiled deep circuit's reproduction and analytic bounds."""
import hashlib
import json
import math
import unittest
from generate import ROOT
from generate_deep import compile_policy, serialized


class DeepCircuitTests(unittest.TestCase):
    def test_artifacts_and_shapes(self):
        data=serialized()
        for name in ('authored-deep-initial.json','authored-deep.json'):
            self.assertEqual((ROOT/'models'/name).read_bytes(),data)
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         '37942aede8996a543e65653335f9c197a56325da1927b08ee1ce3c14eeaf4f74')
        p=json.loads(data)
        self.assertEqual(p,compile_policy())
        flat=[]
        for weights,bias,rows,cols in (('w1','b1',64,692),('wh','bh',64,64),('w2','b2',14,64)):
            self.assertEqual(len(p[weights]),rows)
            self.assertEqual(len(p[bias]),rows)
            self.assertTrue(all(len(row)==cols for row in p[weights]))
            flat+=p[bias]+sum(p[weights],[])
        self.assertEqual(len(flat),49422)
        self.assertTrue(all(math.isfinite(x) for x in flat))

    def test_risk_relay_error_and_dominance(self):
        # x-tanh(x) increases for x>=0. R lies in [0,3].
        error=12000*(.03-math.tanh(.03))
        self.assertLess(error,.108)
        # Food gating can vary by action; bound each independently.
        zmax=math.sqrt(2)*math.tanh(.5)/4
        food_spread=64*math.tanh(zmax)
        center_spread=20*math.tanh(.1*math.tanh(.5))*(1-math.cos(.9)+2*math.sin(.9))
        clearance_spread=5*math.tanh(.1*math.tanh(2))
        bound=food_spread+center_spread+clearance_spread+.027+1+error
        self.assertLess(bound,14)

    def test_conditional_boost_bound(self):
        # One nonpositive predicate and five predicates <=1 preclude boost.
        self.assertLess(2*math.tanh(4*(5-5.5))-1,0)
        # Even six perfect predicates cannot boost when R>=.025.
        self.assertLessEqual(2*math.tanh(4*(6-5.5)-80*.025)-1,0)
        p=compile_policy()
        for a in range(7):
            if a!=3:
                self.assertEqual(p['w2'][a],p['w2'][a+7])
            self.assertEqual(p['b2'][a+7],p['b2'][a]-1)
        difference=[x-y for x,y in zip(p['w2'][10],p['w2'][3])]
        self.assertEqual(difference,[2.0 if j==32 else 0.0 for j in range(64)])

    def test_food_gate_sign_and_suppression(self):
        # Deterministic diagnostic points, not parameter sampling or fitting.
        for z in (-.16,-.08,0,.08,.16):
            for risk in (0,.01,.1,.3,1,3):
                food=16*(math.tanh(z+10*risk)+math.tanh(z-10*risk))
                self.assertGreaterEqual(food*z,0)
                self.assertLessEqual(abs(food),32*abs(math.tanh(z))+1e-12)
                if risk>=.3:
                    self.assertLess(abs(food),.06)

if __name__=='__main__':
    unittest.main()
