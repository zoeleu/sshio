import unittest
import torch
from train import Bridge


@unittest.skipUnless(torch.cuda.is_available(), "CUDA or ROCm GPU required")
class GPUPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gpu_sim import GPUSim
        torch.set_num_threads(2)
        cls.bridge = Bridge()
        cls.sim = GPUSim(1, 2, 1200, 1, max_segments=128, max_pellets=1024, curriculum=False)

    @classmethod
    def tearDownClass(cls):
        cls.bridge.close()

    def assert_state(self, expected):
        sim = self.sim
        for i, snake in enumerate(sorted(expected["snakes"], key=lambda s: s["ID"])):
            self.assertEqual(bool(sim.alive[0, i]), snake["Alive"])
            self.assertAlmostEqual(float(sim.score[0, i]), snake["Score"], delta=2e-4)
            self.assertEqual(int(sim.length[0, i]), len(snake["Segments"]))
            for tensor, key in ((sim.kills, "Kills"), (sim.mutual, "MutualKills"),
                                (sim.food, "FoodEaten"), (sim.spent, "BoostSpent")):
                self.assertAlmostEqual(float(tensor[0, i]), snake[key], delta=2e-4, msg=key)
            body = torch.tensor([[p["X"], p["Y"]] for p in snake["Segments"]])
            torch.testing.assert_close(sim.body[0, :len(body), i].cpu(), body, atol=4e-4, rtol=1e-5)
        active = sim.active[0]
        actual = torch.cat((sim.pellet_pos[0, active], sim.pellet_value[0, active, None]), 1).cpu()
        pellets = expected["pellets"] or []
        wanted = torch.tensor([[p["Pos"]["X"], p["Pos"]["Y"], p["Value"]] for p in pellets], dtype=torch.float32).reshape(-1, 3)
        self.assertEqual(len(actual), len(wanted))
        if len(actual):
            distance = torch.cdist(actual, wanted, compute_mode="donot_use_mm_for_euclid_dist")
            self.assertLess(float(distance.min(0).values.max()), 5e-4)
            self.assertLess(float(distance.min(1).values.max()), 5e-4)
        torch.testing.assert_close(sim.observe().cpu(), torch.tensor(expected["observations"]), atol=5e-4, rtol=1e-4)
        sim.check_capacity()

    def test_go_physics_and_observation_parity(self):
        for name in ("straight", "turn", "growth", "boost", "head_on", "body", "border",
                     "minimum_boost", "own_body", "contested_food"):
            with self.subTest(fixture=name):
                actions = [[6, 0], [0, 6], [3, 3]] if name == "turn" else [[10 if "boost" in name else 3, 3]] * 3
                trace = self.bridge.call(op="physics_trace", fixture=name, action_sequence=actions)
                self.sim.load_fixture(trace["initial"])
                self.assert_state(trace["initial"])
                for action, expected in zip(actions, trace["steps"]):
                    state = self.sim.step(torch.tensor(action, device="cuda"), respawn=False)
                    self.assertEqual(state["observations"].device.type, "cuda")
                    self.assert_state(expected)

    def test_reproducible_reset_and_terminal(self):
        from gpu_sim import GPUSim
        a = GPUSim(2, 4, 4, 73, max_segments=128, max_pellets=2048)
        b = GPUSim(2, 4, 4, 73, max_segments=128, max_pellets=2048)
        torch.testing.assert_close(a.last_observation, b.last_observation)
        action = torch.full((8,), 3, device="cuda", dtype=torch.long)
        x, y = a.step(action), b.step(action)
        self.assertTrue(bool(x["dones"].all()))
        torch.testing.assert_close(x["observations"], y["observations"])
        a.check_capacity()

    def test_capacity_failure_is_explicit(self):
        trace = self.bridge.call(op="physics_trace", fixture="straight", action_sequence=[])
        self.sim.load_fixture(trace["initial"])
        self.sim.score[0, 0] = 200
        self.sim.step(torch.tensor([3, 3], device="cuda"), respawn=False)
        with self.assertRaisesRegex(RuntimeError, "capacity exceeded"):
            self.sim.check_capacity()
        self.sim.load_fixture(trace["initial"])
        self.sim.active.fill_(True)
        self.sim.free_count.zero_()
        self.sim.pellet_pos.fill_(5000)
        self.sim.step(torch.tensor([10, 3], device="cuda"), respawn=False)
        with self.assertRaisesRegex(RuntimeError, "capacity exceeded"):
            self.sim.check_capacity()
        self.sim.load_fixture(trace["initial"])


    def test_graph_matches_eager_through_death_and_partial_reset(self):
        from gpu_sim import GPUSim, GraphStep
        trace = self.bridge.call(op="physics_trace", fixture="head_on", action_sequence=[])
        eager = GPUSim(1, 2, 7, 51, max_segments=128, max_pellets=1024, curriculum=False)
        captured = GPUSim(1, 2, 7, 51, max_segments=128, max_pellets=1024, curriculum=False)
        eager.load_fixture(trace["initial"])
        captured.load_fixture(trace["initial"])
        replay = GraphStep(captured)
        actions = torch.tensor([3, 3], device="cuda")
        for _ in range(6):
            expected, actual = eager.step(actions), replay.step(actions)
            for name in ("observations", "rewards", "dones", "agent_food", "agent_kills", "agent_deaths"):
                torch.testing.assert_close(actual[name], expected[name], atol=5e-4, rtol=1e-4)
            torch.testing.assert_close(captured.ids, eager.ids)
            torch.testing.assert_close(captured.body, eager.body, atol=5e-4, rtol=1e-4)
        captured.check_capacity()


if __name__ == "__main__":
    unittest.main()
