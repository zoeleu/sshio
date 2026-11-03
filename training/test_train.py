import unittest
import torch
from train import OBS, Bridge, Policy, advantages, init_from_v1


class TrainingTests(unittest.TestCase):
    def test_terminal_resets_advantage(self):
        rewards = torch.tensor([[1.0], [100.0]])
        dones = torch.tensor([[1.0], [0.0]])
        values = torch.zeros_like(rewards)
        adv, returns = advantages(rewards, dones, values, torch.tensor([10.0]), gamma=1, lam=1)
        self.assertEqual(adv[0].item(), 1)
        self.assertEqual(returns[1].item(), 110)

    def test_go_inference_matches_pytorch(self):
        torch.manual_seed(73)
        model = Policy()
        bridge = Bridge()
        try:
            bridge.call(op="policy", policy=model.export())
            for _ in range(20):
                obs = torch.rand(OBS) * 2 - 1
                with torch.no_grad():
                    expected = model(obs)[0].argmax().item()
                actual = bridge.call(op="infer", observation=obs.tolist())["action"]
                self.assertEqual(expected, actual)
        finally:
            bridge.close()

    def test_two_hidden_layer_go_inference_matches_pytorch(self):
        torch.manual_seed(74)
        model = Policy(hidden=256, hidden2=128)
        exported = model.export()
        self.assertEqual((len(exported["wh"]), len(exported["wh"][0])), (128, 256))
        self.assertEqual(len(exported["w2"][0]), 128)
        bridge = Bridge()
        try:
            bridge.call(op="policy", policy=exported)
            for _ in range(20):
                obs = torch.rand(OBS) * 2 - 1
                with torch.no_grad():
                    expected = model(obs)[0].argmax().item()
                actual = bridge.call(op="infer", observation=obs.tolist())["action"]
                self.assertEqual(expected, actual)
        finally:
            bridge.close()

    def test_two_hidden_layer_shapes_league_and_single_layer_export(self):
        from league import League
        model = Policy(hidden=32, hidden2=16)
        logits, value = model(torch.zeros(5, OBS))
        self.assertEqual((logits.shape, value.shape), ((5, 14), (5,)))
        self.assertEqual(model.critic.in_features, 16)
        self.assertNotIn("wh", Policy().export())
        self.assertEqual(len(League(model).models), 1)
        with self.assertRaises(ValueError):
            Policy(hidden=257)
        with self.assertRaises(ValueError):
            Policy(hidden2=300)

    def test_deep_model_refuses_warm_start_from_shallow_donor(self):
        from train import init_from_v2
        deep = Policy(hidden=64, hidden2=32)
        with self.assertRaises(ValueError):
            init_from_v1(deep, {"version": 1})
        with self.assertRaises(ValueError):
            init_from_v2(deep, Policy().export())

    def test_v1_warm_start_maps_newest_frame(self):
        torch.manual_seed(3)
        donor = {"version": 1,
                 "w1": [[float(i * 100 + j) for j in range(49)] for i in range(4)],
                 "b1": [1.0, 2.0, 3.0, 4.0],
                 "w2": [[float(k)] * 64 for k in range(14)],
                 "b2": [float(k) for k in range(14)]}
        model = Policy(hidden=64)
        init_from_v1(model, donor)
        newest = 3 * 172
        # v1 feature 5 (food dx) lands on v2 food target 0 dx in newest frame.
        self.assertEqual(model.hidden.weight[2, newest + 16].item(), 205.0)
        # Unmapped inputs start at zero so initial behaviour is the donor's.
        self.assertEqual(model.hidden.weight[2, 0].item(), 0.0)
        self.assertEqual(model.hidden.weight[2, newest + 1].item(), 0.0)
        self.assertEqual(model.actor.bias[5].item(), 5.0)
        with self.assertRaises(ValueError):
            init_from_v1(Policy(), {"version": 2, "w1": [], "b1": [], "w2": [], "b2": []})

    def test_batch_protocol(self):
        bridge = Bridge()
        try:
            state = bridge.call(op="reset_batch", arenas=2, agents=4, horizon=8, seed=7, teach=True)
            self.assertEqual(len(state["observations"]), 8)
            state = bridge.call(op="step_batch", actions=state["teachers"], teach=True)
            self.assertEqual(len(state["teachers"]), 8)
            state = bridge.call(op="step_batch", actions=state["teachers"])
            self.assertTrue(all(state["dones"]))
            with self.assertRaises(RuntimeError):
                bridge.call(op="step_batch", actions=[999] * 8)
        finally:
            bridge.close()



class LeagueTests(unittest.TestCase):
    def test_roles_leave_learners_and_exclude_opponents(self):
        from league import role_masks
        for agents in (2, 3, 4, 8, 32, 64):
            heuristic, historical, learner = role_masks(agents * 4, agents, "cpu")
            self.assertTrue(learner.any())
            self.assertFalse((heuristic & historical).any())
            self.assertTrue((heuristic | historical | learner).all())
            self.assertFalse((learner & (heuristic | historical)).any())

    def test_league_retains_anchor_and_round_trips(self):
        from league import League
        model = Policy()
        league = League(model, capacity=3)
        anchor = league.models[0].actor.bias.detach().clone()
        for _ in range(5):
            with torch.no_grad():
                model.actor.bias.add_(1)
            league.add(model)
        self.assertEqual(len(league.models), 3)
        self.assertTrue(torch.equal(league.models[0].actor.bias, anchor))
        restored = League(model, capacity=3, states=league.states())
        self.assertTrue(torch.equal(restored.models[-1].actor.bias, model.actor.bias))
        self.assertFalse(restored.models[0].actor.weight.requires_grad)

    def test_checkpoint_is_loadable_with_weights_only(self):
        import tempfile
        from pathlib import Path
        from league import League
        from train import save_checkpoint
        model = Policy()
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, League(model), 12, 500,
                            {"hardware": {"torch": str(torch.__version__)}})
            checkpoint = torch.load(path, weights_only=True)
            self.assertEqual(checkpoint["updates"], 12)
            self.assertEqual(checkpoint["transitions"], 500)
            self.assertEqual(len(checkpoint["league"]), 1)

    def test_curriculum_protocol_and_heuristic_actions(self):
        bridge = Bridge()
        try:
            state = bridge.call(op="reset_batch", arenas=9, agents=4, horizon=8,
                                seed=17, curriculum=True, stage=2)
            self.assertIn("production", state["scenarios"])
            state = bridge.call(op="step_batch", actions=[-1, 3, 3, 3] * 9)
            self.assertEqual(len(state["agent_food"]), 36)
            bridge.call(op="curriculum", stage=0)
            state = bridge.call(op="step_batch", actions=[3] * 36)
            self.assertTrue(all(name in ("forage", "border", "arena") for name in state["scenarios"]))
        finally:
            bridge.close()

    def test_v2_to_v3_warm_start_and_personality_gpu_sim(self):
        from train import PERSONALITY_OBS, PERSONALITY_POLICY_VERSION, init_from_v2
        from gpu_sim import GPUSim
        donor_model = Policy()
        donor = donor_model.export()
        v3_model = Policy(obs_dim=PERSONALITY_OBS, version=PERSONALITY_POLICY_VERSION)
        init_from_v2(v3_model, donor)
        self.assertEqual(v3_model.export()["version"], PERSONALITY_POLICY_VERSION)
        self.assertTrue(torch.allclose(v3_model.hidden.weight[:, :OBS], donor_model.hidden.weight))
        self.assertTrue(torch.all(v3_model.hidden.weight[:, OBS:] == 0))
        if torch.cuda.is_available():
            sim = GPUSim(arenas=1, agents=4, horizon=16, seed=5, device="cuda", personalities=True)
            obs = sim.last_observation
            self.assertEqual(obs.shape, (4, PERSONALITY_OBS))
            # Each of the 4 worms in the arena gets a distinct one-hot personality vector in the last 4 dims.
            self.assertTrue(torch.equal(obs[:, OBS:].cpu(), torch.eye(4)))


if __name__ == "__main__":
    unittest.main()

