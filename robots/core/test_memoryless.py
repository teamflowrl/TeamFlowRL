"""Memoryless ablation: features at step t depend only on the observation at step t."""
import unittest
import torch
from policy import Config, Policy


class Memoryless(unittest.TestCase):
    def test_no_cross_step_information(self):
        torch.manual_seed(0)
        cfg = Config(obs_dim=5, num_agents=2, action_dim=3, state_dim=4, memoryless=True)
        p = Policy(cfg).eval()
        obs = torch.randn(2, 4, 2, 5)
        changed = obs.clone(); changed[:, :2] += 10.
        a, _ = p.encode_sequence(obs); b, _ = p.encode_sequence(changed)
        self.assertTrue(torch.allclose(a[:, 2:], b[:, 2:]))
        self.assertFalse(torch.allclose(a[:, :2], b[:, :2]))
        # Step-wise encoding equals sequence encoding; no carry is kept.
        f0, carry = p.encode_step(obs[:, 0]); self.assertIsNone(carry)
        self.assertTrue(torch.allclose(f0, a[:, 0], atol=1e-6))
        h, c = p.warmup(torch.randn(2, 3, 2, 5), torch.tensor([3, 2]))
        self.assertEqual(float(h.abs().sum() + c.abs().sum()), 0.)

    def test_default_config_unchanged(self):
        cfg = Config(obs_dim=5, num_agents=2, action_dim=3, state_dim=4)
        self.assertNotIn('memoryless', Policy(cfg).config_dict())


if __name__ == '__main__':
    unittest.main()
