"""CPU tests; run with torch installed, no ManiSkill required."""
import copy
import json
import math
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import h5py
import numpy as np
import torch
from obs_builder import build_observations, build_rgb_observations, TASKS, AGENT_IDS
from extract_states import extraction_observations
from env_adapter import BimanualEnv
from data import Demonstrations
from learner import MACFlowLearner, bellman_target
from policy import Config
from train import configure_bc, bc_step


def fake_raw(task, n=2):
    raw = dict(agent={key: dict(qpos=torch.randn(n, 9), qvel=torch.randn(n, 9)) for key in AGENT_IDS},
               extra=dict(left_arm_tcp=torch.randn(n, 7), right_arm_tcp=torch.randn(n, 7)))
    e = raw['extra']
    if task == TASKS[0]:
        e.update(cube_pose=torch.randn(n, 7), cube_to_goal_pos=torch.randn(n, 3))
        e['left_arm_tcp_to_cube_pos'] = e['cube_pose'][:, :3]-e['left_arm_tcp'][:, :3]
        e['right_arm_tcp_to_cube_pos'] = e['cube_pose'][:, :3]-e['right_arm_tcp'][:, :3]
    else:
        e.update(cubeA_pose=torch.randn(n, 7), cubeB_pose=torch.randn(n, 7), goal_region_pos=torch.randn(n, 3))
        e['left_arm_tcp_to_cubeA_pos'] = e['cubeA_pose'][:, :3]-e['left_arm_tcp'][:, :3]
        e['right_arm_tcp_to_cubeB_pos'] = e['cubeB_pose'][:, :3]-e['right_arm_tcp'][:, :3]
        e['cubeA_to_cubeB_pos'] = e['cubeB_pose'][:, :3]-e['cubeA_pose'][:, :3]
    return raw


def fake_rgb(raw):
    n = len(raw['agent'][AGENT_IDS[0]]['qpos'])
    raw['sensor_data'] = dict(base_camera=dict(rgb=torch.zeros(n, 128, 128, 3, dtype=torch.uint8)))
    target = SimpleNamespace(pose=SimpleNamespace(p=torch.randn(n, 3)))
    state_view = dict(agent=raw['agent'], extra=raw['extra'])
    base = SimpleNamespace(agent=SimpleNamespace(agents_dict=dict.fromkeys(AGENT_IDS)),
                           goal_site=target, goal_region=target, _obs_mode='rgb', get_info=lambda: {})
    # Mirrors ManiSkill: object state only appears when the obs mode contains "state".
    base._get_obs_state_dict = lambda info: (state_view if 'state' in base._obs_mode else None)
    def vision(rgb):
        return torch.ones(len(rgb), 1920)
    return raw, base, vision


def write_fixture(path):
    with h5py.File(path, 'w') as f:
        f.attrs['contract'] = json.dumps(dict(task=TASKS[0], validated=True, dino_weights_sha256='fixture'))
        split = f.create_group('split')
        for key, names in [('train', ['traj_0', 'traj_1', 'traj_2']), ('valid', ['traj_3'])]:
            split.create_dataset(key, data=names, dtype=h5py.string_dtype())
        data = f.create_group('data')
        for i, t in enumerate([20, 27, 100, 23]):
            g = data.create_group(f'traj_{i}')
            g.attrs['source_success'] = True
            raw = fake_raw(TASKS[0], t+1)
            built = build_observations(raw, TASKS[0], torch.arange(t+1))
            for key, v in built.items(): g.create_dataset(key, data=v.numpy())
            raw, base, vision = fake_rgb(raw)
            rgb = build_rgb_observations(raw, base, TASKS[0], torch.arange(t+1), vision)
            g.create_dataset('obs_rgb', data=rgb.numpy().astype(np.float16))
            # Encode transition index in rewards to make sampled chronology testable.
            rewards = np.repeat(np.arange(t, dtype=np.float32)[:, None], 2, axis=1)
            g.create_dataset('rewards', data=rewards)
            g.create_dataset('actions', data=np.zeros((t, 2, 8), np.float32))
            done = np.zeros(t, bool); done[-1] = True
            g.create_dataset('terminated', data=done)


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); torch.manual_seed(1)

    def test_dimensions_and_fields(self):
        for task, d, c in [(TASKS[0], 44, 69), (TASKS[1], 54, 79)]:
            raw = fake_raw(task)
            out, schema = build_observations(raw, task, torch.tensor([0, 100]), return_schema=True)
            self.assertEqual(out['obs_dec'].shape, (2, 2, d))
            self.assertEqual(out['obs_central'].shape, (2, 2, c))
            self.assertTrue(torch.equal(out['state'], out['obs_central']))
            self.assertTrue(torch.equal(out['obs_central'][:, 0, :-2], out['obs_central'][:, 1, :-2]))
            self.assertTrue(torch.equal(out['obs_dec'][:, 0, -3], torch.tensor([1., 0.])))
            self.assertTrue(torch.equal(out['obs_dec'][0, :, -2:], torch.eye(2)))
            self.assertEqual(sum(x['size'] for x in schema['obs_dec']), d)
            self.assertTrue(torch.equal(out['obs_dec'][:, 0, :9], raw['agent'][AGENT_IDS[0]]['qpos']))
            if task == TASKS[0]:
                self.assertTrue(torch.equal(out['obs_dec'][:, 0, 32:35],
                                           raw['extra']['cube_pose'][:, :3]+raw['extra']['cube_to_goal_pos']))

    def test_no_partner_proprio_leak(self):
        for task in TASKS:
            for partner in (0, 1):
                raw = fake_raw(task); original = build_observations(raw, task, 0)
                changed = copy.deepcopy(raw)
                changed['agent'][AGENT_IDS[partner]]['qpos'] += 100
                changed['agent'][AGENT_IDS[partner]]['qvel'] += 100
                arm = ('left', 'right')[partner]
                changed['extra'][arm+'_arm_tcp'] += 100
                result = build_observations(changed, task, 0)
                self.assertTrue(torch.equal(original['obs_dec'][:, 1-partner], result['obs_dec'][:, 1-partner]))
                self.assertFalse(torch.equal(original['obs_dec'][:, partner], result['obs_dec'][:, partner]))
                self.assertFalse(torch.equal(original['obs_central'], result['obs_central']))

    def test_rgb_actor_boundary_and_adapter(self):
        for task in TASKS:
            for partner in (0, 1):
                raw, base, vision = fake_rgb(fake_raw(task))
                original = build_rgb_observations(raw, base, task, 17, vision)
                self.assertEqual(original.shape, (2, 2, 1944))
                self.assertTrue(torch.equal(original[:, 0, 1920:1929], raw['agent'][AGENT_IDS[0]]['qpos']))
                self.assertTrue(torch.allclose(original[:, :, -3], torch.full((2, 2), .83)))
                self.assertTrue(torch.equal(original[0, :, -2:], torch.eye(2)))
                changed = copy.deepcopy(raw)
                changed['agent'][AGENT_IDS[partner]]['qpos'] += 100
                changed['agent'][AGENT_IDS[partner]]['qvel'] += 100
                # All privileged fields, including cube and TCP poses, are excluded.
                changed['extra'] = {k: v+100 for k, v in changed['extra'].items()}
                result = build_rgb_observations(changed, base, task, 17, vision)
                self.assertTrue(torch.equal(original[:, 1-partner], result[:, 1-partner]))
                self.assertFalse(torch.equal(original[:, partner], result[:, partner]))
                env = BimanualEnv.__new__(BimanualEnv)
                env.raw, env.base, env.vision = raw, base, vision
                env.task, env.elapsed, env.obs_mode, env.agent_ids = task, 17, 'rgb', AGENT_IDS
                obs, state = env.encode()
                self.assertTrue(torch.equal(obs, original))
                expected_state = build_observations(dict(agent=raw['agent'], extra=raw['extra']), task, 17, AGENT_IDS)['state']
                self.assertTrue(torch.equal(state, expected_state))
                self.assertEqual(base._obs_mode, 'rgb')
                raw['sensor_data']['base_camera']['rgb'] = torch.zeros(2, 64, 64, 3, dtype=torch.uint8)
                with self.assertRaises(ValueError):
                    build_rgb_observations(raw, base, task, 17, vision)

    def test_extraction_evaluation_paths_equal(self):
        for task in TASKS:
            raw = fake_raw(task)
            extracted, _ = extraction_observations(raw, task, 17, AGENT_IDS)
            for mode in ('dec', 'central'):
                # Exercise the real adapter encode path without constructing physics.
                env = BimanualEnv.__new__(BimanualEnv)
                env.raw, env.task, env.elapsed, env.agent_ids, env.obs_mode = raw, task, 17, AGENT_IDS, mode
                obs, state = env.encode()
                self.assertTrue(torch.equal(obs, extracted['obs_'+mode]))
                self.assertTrue(torch.equal(state, extracted['state']))

    def test_batch_shapes_chronology_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.h5'; write_fixture(path)
            for mode, dim in [('dec', 44), ('central', 69), ('rgb', 1944)]:
                data = Demonstrations(path, mode, 'cpu')
                batch = data.batch(np.random.default_rng(1), 32, 20, 'valid')
                self.assertEqual(batch['obs_sequence'].dtype, torch.float32)
                self.assertEqual(batch['prefix'].shape, (32, 100, 2, dim))
                self.assertEqual(batch['obs_sequence'].shape, (32, 21, 2, dim))
                self.assertEqual(batch['actions'].shape, (32, 20, 2, 8))
                self.assertEqual(batch['terminated'].dtype, torch.bool)
                self.assertEqual(batch['rewards'].shape, (32, 20, 2))
                for i, start in enumerate(batch['lengths'].tolist()):
                    ix = torch.arange(start, start+20)
                    coverage = torch.minimum(ix, torch.tensor(3)) - torch.maximum(torch.tensor(0), ix-19)+1
                    expected = 80/(23*coverage)
                    self.assertTrue(torch.allclose(batch['time_weights'][i], expected))
                    self.assertTrue(torch.equal(batch['rewards'][i, :, 0], ix.float()))
                    self.assertTrue(torch.equal(batch['terminated'][i, :, 0], ix == 22))
                    self.assertTrue(torch.equal(batch['obs_sequence'][i], data.tensors['obs'][3, start:start+21]))
                    self.assertTrue(torch.equal(batch['prefix'][i, :start], data.tensors['obs'][3, :start]))
                    self.assertEqual(float(batch['prefix'][i, start:].abs().sum()), 0.)
                data.split_indices['train'] = np.array([0])
                exact = data.batch(np.random.default_rng(1), 2, 20)
                self.assertTrue(torch.equal(exact['time_weights'], torch.ones(2, 20)))
                self.assertTrue(torch.equal(exact['lengths'], torch.zeros(2, dtype=torch.long)))
                with self.assertRaises(ValueError): data.batch(np.random.default_rng(1), 2, 21)
        # Enumerate every possible window: expected weighted mass is uniform.
        for t in (20, 23, 100):
            mass = torch.zeros(t)
            for start in range(t-20+1):
                ix = torch.arange(start, start+20)
                coverage = torch.minimum(ix, torch.tensor(t-20))-torch.maximum(torch.tensor(0), ix-19)+1
                mass[ix] += ((t-20+1)*20/(t*coverage))/(t-20+1)/20
            self.assertTrue(torch.allclose(mass, torch.full((t,), 1/t), atol=1e-6))

    def test_macflow_and_bc_update(self):
        cfg = Config(obs_dim=44, num_agents=2, action_dim=8, state_dim=69,
                     actor_hidden_dims=(32, 32), value_hidden_dims=(32, 32),
                     lstm_hidden_dim=16, lstm_pre_mlp_dims=(24,), flow_steps=3,
                     initial_std=.02, q_gradient_ratio=.25, balance_window_edges=True)
        batch = dict(obs_sequence=torch.randn(3, 5, 2, 44), prefix=torch.randn(3, 3, 2, 44),
                     lengths=torch.tensor([0, 1, 3]), actions=torch.rand(3, 4, 2, 8)*2-1,
                     rewards=torch.rand(3, 4, 2), terminated=torch.zeros(3, 4, 2, dtype=torch.bool),
                     time_weights=torch.ones(3, 4))
        batch['terminated'][:, -1] = True
        for method in ('macflow', 'bc'):
            learner = MACFlowLearner(cfg, 'cpu')
            if method == 'bc': configure_bc(learner)
            before = {k: v.clone() for k, v in learner.model.state_dict().items()}
            metrics = learner.train_step(batch) if method == 'macflow' else bc_step(learner, batch)
            self.assertTrue(np.isfinite(metrics['loss']))
            prefixes = ['encoder', 'actor_onestep_flow']
            if method == 'macflow': prefixes += ['actor_bc_flow', 'q', 'target_q']
            for prefix in prefixes:
                self.assertTrue(any(not torch.equal(before[k], v) for k, v in learner.model.state_dict().items()
                                    if k.startswith(prefix+'.')), prefix)
            if method == 'bc':
                for k, v in learner.model.state_dict().items():
                    if k.startswith(('actor_bc_flow.', 'q.', 'target_q.')): self.assertTrue(torch.equal(before[k], v), k)
            else:
                self.assertLessEqual(metrics['effective_q_action_gradient_norm'],
                                     .25001*metrics['distillation_action_gradient_norm']+1e-8)
        target = bellman_target(torch.ones(2), torch.tensor([False, True]), torch.tensor([10., 10.]), .9)
        self.assertTrue(torch.equal(target, torch.tensor([10., 1.])))


class MAPPOContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def make_models(algo="dfpo"):
        from policy import Policy, CentralValue
        from mappo import OnlinePolicy
        cfg = Config(obs_dim=44, num_agents=2, action_dim=8, state_dim=69,
                     actor_hidden_dims=(16, 16), value_hidden_dims=(16, 16),
                     lstm_hidden_dim=8, lstm_pre_mlp_dims=(12,), initial_std=.02,
                     student_latent_std=1., critic_detach_encoder=True)
        policy = OnlinePolicy(Policy(cfg), algo)
        ref = copy.deepcopy(policy).requires_grad_(False)
        critic = CentralValue(cfg.state_dim)
        actor_opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=2e-5, eps=1e-5)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4, eps=1e-5)
        return policy, ref, critic, actor_opt, critic_opt

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            self.assertTrue(np.array_equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_gae_finite(self):
        from mappo import gae_finite
        rewards = torch.tensor([1., 2., 3.])[:, None, None].expand(-1, 2, 2)
        values = torch.tensor([.5, 1., 1.5])[:, None, None].expand_as(rewards)
        adv, ret = gae_finite(rewards, values, gamma=.9, lam=.8)
        # delta = [1.4, 2.35, 1.5], backward recursion uses gamma*lambda=.72.
        expected = torch.tensor([3.8696, 3.43, 1.5])[:, None, None].expand_as(rewards)
        self.assertTrue(torch.allclose(adv, expected))
        self.assertTrue(torch.allclose(ret, expected + values))
        one, one_ret = gae_finite(rewards[-1:], values[-1:])
        self.assertTrue(torch.equal(one, rewards[-1:] - values[-1:]))
        self.assertTrue(torch.equal(one_ret, rewards[-1:]))

    def test_online_freeze(self):
        for algo in ('dfpo', 'fppo'):
            policy = self.make_models(algo)[0]
            self.assertFalse(policy.training)
            for name, param in policy.named_parameters():
                trainable = name.startswith('base.actor_onestep_flow.') or (
                    name.startswith('noise.') if algo == 'dfpo' else name == 'log_std')
                self.assertEqual(param.requires_grad, trainable, name)

    def test_execution_rules_and_latent_contract(self):
        from mappo import OnlinePolicy
        from policy import Policy
        for algo in ('dfpo', 'fppo'):
            with self.subTest(algo=algo):
                policy = self.make_models(algo)[0]
                features = torch.randn(5, 2, policy.cfg.lstm_hidden_dim)
                first = policy.act(features, stochastic=False, generator=torch.Generator().manual_seed(91))
                second = policy.act(features, stochastic=False, generator=torch.Generator().manual_seed(91))
                raw, z, lp = first
                self.assertTrue(torch.equal(raw, second[0]))
                self.assertTrue(torch.equal(z, second[1]))
                self.assertTrue(torch.isnan(lp).all())
                expected = policy.base.distribution(features, z).loc
                self.assertTrue(torch.equal(raw, expected))
                self.assertEqual(bool(torch.count_nonzero(z)), algo == 'dfpo')
                sampled, sample_z, sample_lp = policy.act(
                    features, stochastic=True, generator=torch.Generator().manual_seed(91))
                self.assertTrue(torch.equal(z, sample_z))
                self.assertFalse(torch.equal(raw, sampled))
                self.assertTrue(torch.equal(sample_lp, policy.distribution(features, sample_z).log_prob(sampled.double()).sum(-1)))
                cfg = policy.config_dict()
                cfg['student_latent_std'] = 0.
                with self.assertRaisesRegex(ValueError, 'student_latent_std'):
                    OnlinePolicy(Policy(cfg), algo)

    def test_fppo_std_constraints(self):
        policy = self.make_models('fppo')[0]
        features = torch.randn(3, 2, policy.cfg.lstm_hidden_dim)
        z = torch.zeros(3, 2, policy.cfg.action_dim)
        self.assertTrue(torch.allclose(policy.distribution(features, z).scale,
                                       torch.full_like(z.double(), .2)))
        for value, expected in ((10., 0.), (-10., -4.)):
            with torch.no_grad():
                policy.log_std.fill_(value)
            dist = policy.distribution(features, z)
            self.assertTrue(torch.allclose(dist.scale.log(), torch.full_like(z.double(), expected)))
            policy.project_exploration()
            self.assertTrue(torch.all(policy.log_std == expected))

    def test_dfpo_reference_density_and_kl(self):
        # Literal one-step PathPolicy equations: the reference package has no
        # __init__.py/ALGORITHMS export, so use its documented equivalent formula.
        from torch.distributions import Normal, kl_divergence
        policy = self.make_models('dfpo')[0]
        features = torch.randn(5, 2, policy.cfg.lstm_hidden_dim)
        z = torch.randn(5, 2, policy.cfg.action_dim)
        x = torch.randn_like(z) * 2  # Unclipped samples, including outside [-1,1].
        layers = list(policy.noise)
        self.assertIsInstance(layers[0], torch.nn.Linear)
        self.assertEqual(layers[0].in_features, features.shape[-1] + z.shape[-1] + 1)
        self.assertEqual(layers[0].out_features, 64)
        self.assertIsInstance(layers[1], torch.nn.Tanh)
        self.assertTrue(torch.equal(layers[2].weight, torch.zeros_like(layers[2].weight)))
        p = (.1 - .01) / (.5 - .01)
        self.assertTrue(torch.allclose(layers[2].bias, torch.full_like(layers[2].bias, math.log(p/(1-p)))))
        old = policy.distribution(features, z)
        with torch.no_grad():
            layers[2].weight.normal_(std=.03)
        mean = policy.base.distribution(features, z).loc
        time = torch.zeros_like(z[..., :1])  # k / steps = 0 / 1.
        std = .01 + (.5 - .01) * policy.noise(torch.cat((features, z, time), -1)).sigmoid()
        reference = Normal(mean.double(), std.double())
        actual = policy.distribution(features, z)
        self.assertEqual(actual.loc.dtype, torch.float64)
        self.assertTrue(torch.equal(actual.log_prob(x.double()).sum(-1), reference.log_prob(x.double()).sum(-1)))
        expected_kl = (torch.log(std.double()/old.scale)
                       + (old.scale.square() + (old.loc-mean.double()).square())/(2*std.double().square()) - .5).sum(-1)
        self.assertTrue(torch.allclose(kl_divergence(old, actual).sum(-1), expected_kl, atol=1e-12, rtol=1e-12))

    def test_rollout_update_resume(self):
        for algo in ('dfpo', 'fppo'):
            with self.subTest(algo=algo):
                self.check_rollout_update_resume(algo)

    def check_rollout_update_resume(self, algo):
        from mappo import rollout, update, save_latest, load_latest, assert_frozen, rng_state, restore_rng

        class StubEnv:
            def __init__(self):
                self.n = 2
                self.resets = 0
                self.t = 0

            def reset(self, seed):
                self.resets += 1
                self.seeds = seed
                self.t = 0
                return self.encode()

            def encode(self):
                return torch.randn(self.n, 2, 44), torch.randn(self.n, 2, 69)

            def step(self, actions):
                if actions.shape != (self.n, 2, 8):
                    raise ValueError('Invalid synchronized action shape')
                self.t += 1
                return None, torch.full((self.n,), .25), None, None, dict(success=torch.tensor([True, False]))

        torch.manual_seed(71)
        models = self.make_models(algo)
        policy, ref, critic, actor_opt, critic_opt = models
        pretrained = copy.deepcopy(ref.state_dict())
        env = StubEnv()
        # Exercise the full finite 100-step rollout, including shared rewards and terminal metrics.
        batch, metrics = rollout(env, policy, critic, iteration=3, envs=2)
        self.assertEqual(env.resets, 1)
        self.assertEqual(env.t, 100)
        self.assertEqual(env.seeds, [10_000_006, 10_000_007])
        self.assertEqual(batch['state'].shape, (200, 2, 69))
        self.assertEqual(metrics['mean_return'], 25.)
        self.assertEqual(metrics['native_training_final_success'], .5)
        # Only the first iteration warms up the critic.
        before = copy.deepcopy(policy.state_dict())
        for iteration in range(1):
            metrics = update(*models, batch, iteration, minibatch=400, epochs=1)
            self.assertEqual(metrics['actor_updates'], 0)
            self.assert_nested_equal(before, policy.state_dict())
        metrics = update(*models, batch, 1, minibatch=400, epochs=1)
        self.assertEqual(metrics['actor_updates'], 1)
        self.assertTrue(np.isfinite(metrics['value_loss']))
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in policy.state_dict().items()
                            if k.startswith('base.actor_onestep_flow.')))
        assert_frozen(policy, ref)
        corrupt = dict(batch, old_lp=batch['old_lp'] + .01)
        with self.assertRaisesRegex(RuntimeError, 'replay mismatch'):
            update(*models, corrupt, 4, minibatch=400, epochs=1)
        params = dict(algo=algo, envs=2, seed=71, steps=1000)
        progress = dict(steps=800, iteration=4, actor_updates=1, last_eval_step=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            save_latest(path, *models, progress, params, 'fixture-sha')
            saved_rng = rng_state()
            # Compare uninterrupted continuation with a reconstructed process.
            next_batch, _ = rollout(env, policy, critic, 4, 2)
            next_metrics = update(*models, next_batch, 4, minibatch=400, epochs=1)
            expected_rng = rng_state()
            resumed = self.make_models(algo)
            resumed[0].load_state_dict(pretrained)
            resumed[1].load_state_dict(pretrained)
            loaded = load_latest(path, *resumed, params, 'fixture-sha')
            self.assertEqual(loaded, progress)
            self.assert_nested_equal(saved_rng, rng_state())
            replay_batch, _ = rollout(StubEnv(), resumed[0], resumed[2], 4, 2)
            replay_metrics = update(*resumed, replay_batch, 4, minibatch=400, epochs=1)
            self.assert_nested_equal(next_batch, replay_batch)
            self.assertEqual(next_metrics, replay_metrics)
            self.assert_nested_equal(expected_rng, rng_state())
            for original, restored in zip(models, resumed):
                self.assert_nested_equal(original.state_dict(), restored.state_dict())
            with self.assertRaisesRegex(ValueError, 'parameters differ'):
                load_latest(path, *resumed, dict(params, envs=3), 'fixture-sha')
            with self.assertRaisesRegex(ValueError, 'parameters differ'):
                load_latest(path, *resumed, params, 'wrong-sha')
            restore_rng(saved_rng)

    def test_sigusr1_exit_75_and_resume(self):
        # Real subprocess exit status; only the physics/evaluation are stubbed.
        child = textwrap.dedent("""
            import os, signal, sys
            import torch
            import mappo
            class StubEnv:
                def __init__(self, task, n, obs_mode, vision=None):
                    assert obs_mode == 'dec'
                    self.n = n
                    self.t = 0
                def encode(self):
                    return torch.zeros(self.n, 2, 44), torch.zeros(self.n, 2, 69)
                def reset(self, seeds):
                    self.t = 0
                    return self.encode()
                def step(self, actions):
                    self.t += 1
                    if self.t == 1 and os.environ.get('TEST_SIGNAL_PHASE') == 'train':
                        os.kill(os.getpid(), int(os.environ['TEST_SIGNAL']))
                    return None, torch.ones(self.n), None, None, {'success': torch.ones(self.n, dtype=torch.bool)}
                def close(self):
                    pass
            def evaluation(*args, **kwargs):
                if os.environ.get('TEST_SIGNAL_PHASE') == 'eval':
                    os.kill(os.getpid(), int(os.environ['TEST_SIGNAL']))
                if kwargs['should_stop']():
                    raise mappo.EvaluationInterrupted()
                row = {'episodes': 1, 'native_final_success_rate': 1.}
                return {'deterministic': row, 'stochastic': row}
            mappo.BimanualEnv = StubEnv
            mappo.evaluate_pair = evaluation
            args = mappo.parser().parse_args()
            args.seed = 1 if args.seed is None else args.seed
            sys.exit(mappo.run(args))
        """)
        for algo, phase, sig in ((a, p, s) for a in ('dfpo', 'fppo')
                                 for p, s in (('train', signal.SIGUSR1), ('eval', signal.SIGTERM))):
            with self.subTest(algo=algo, phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                policy = self.make_models(algo)[0].base
                checkpoint = root / 'pretrained.pt'
                torch.save(dict(config=policy.config_dict(), model=policy.state_dict(),
                                method='macflow', obs_mode='dec', contract={'task': TASKS[0]},
                                state_mean=torch.zeros(69), state_std=torch.ones(69)), checkpoint)
                out = root / 'run'
                command = [sys.executable, '-c', child, '--checkpoint', str(checkpoint), '--out', str(out),
                           '--algo', algo, '--device', 'cpu', '--envs', '1', '--steps', '100',
                           '--eval-episodes', '1', '--epochs', '1']
                env = dict(os.environ, TEST_SIGNAL_PHASE=phase, TEST_SIGNAL=str(int(sig)), RESUME_RUN='0')
                first = subprocess.run(command, cwd=Path(__file__).resolve().parent, env=env,
                                       capture_output=True, text=True, timeout=120)
                self.assertEqual(first.returncode, 75, first.stdout + first.stderr)
                status = json.loads((out / 'status.json').read_text())
                self.assertEqual(status['steps'], 100 if phase == 'train' else 0)
                self.assertEqual(status['status'], 'checkpointed')
                self.assertEqual(status['signal'], sig)
                env.update(TEST_SIGNAL_PHASE='none', RESUME_RUN='1')
                second = subprocess.run(command, cwd=Path(__file__).resolve().parent, env=env,
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
                self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'complete')
                train = [json.loads(x) for x in (out / 'train.jsonl').read_text().splitlines()]
                evaluations = [json.loads(x) for x in (out / 'eval.jsonl').read_text().splitlines()]
                self.assertEqual([x['steps'] for x in train], [100])
                self.assertEqual([x['steps'] for x in evaluations], [0, 100])

    def test_pending_evaluation_and_log_recovery(self):
        from mappo import evaluation_due, reconcile_log
        self.assertTrue(evaluation_due(dict(steps=0, last_eval_step=-1), 1000, 5000))
        self.assertFalse(evaluation_due(dict(steps=0, last_eval_step=0), 1000, 5000))
        self.assertTrue(evaluation_due(dict(steps=1024, last_eval_step=0), 1000, 5000))
        self.assertFalse(evaluation_due(dict(steps=1024, last_eval_step=1024), 1000, 5000))
        self.assertTrue(evaluation_due(dict(steps=5120, last_eval_step=4096), 1000, 5000))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'eval.jsonl'
            path.write_text(' {"steps": 0}\n{"steps": 0}\n{"steps": 1024}\n{"steps":')
            reconcile_log(path, 0)
            self.assertEqual([json.loads(line) for line in path.read_text().splitlines()], [dict(steps=0)])


class ACContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def make_actor(algo):
        from ac_policy import from_pretrained, ACPolicy
        if algo == 'teamflow':
            base = MAPPOContracts.make_models()[0].base
            with torch.no_grad():
                base.obs_mean.fill_(.3)
                base.obs_std.fill_(2.)
            return from_pretrained(dict(method='macflow', obs_mode='dec', config=base.config_dict(),
                                        model=base.state_dict()), algo)
        return ACPolicy(dict(state_dim=69, generator_spec=dict(algorithm='C_MF_MAPPO',
                        obs_dim=44, action_dim=8, hidden=[16, 16], feature_dim=8)), algo)

    def test_teacher_and_path_density(self):
        actor = self.make_actor('teamflow')
        obs = torch.randn(3, 2, 44)
        f, _ = actor.encode_step(obs)
        x, t = torch.randn(3, 2, 8), torch.rand(3, 2, 1)
        self.assertTrue(torch.allclose(actor.path.generator.velocity(f, x, t),
            actor.base.actor_bc_flow(f, x, actor.base.time_sin_embed(t)), atol=1e-6))
        z = torch.randn_like(x)
        sample = actor.path.sample(f, z=z, exploration=False)
        self.assertTrue(torch.allclose(sample.actions, actor.base.teacher_actions(f, z), atol=1e-6))
        sample = actor.path.sample(f)
        manual = torch.zeros(3, 2, dtype=torch.float64)
        for k in range(10):
            d = actor.path.distribution(f, sample.chain[:, k], k)
            delta = (sample.chain[:, k+1].double()-d.loc)/d.scale
            manual += (-.5*delta.square()-d.scale.log()-.5*math.log(2*math.pi)).sum(-1)
        self.assertTrue(torch.allclose(manual, actor.path.log_prob(f, sample.chain), atol=1e-10))
        self.assertTrue(torch.allclose(manual, sample.log_prob, atol=1e-10))

    def test_ac_online_resume(self):
        from mappo import rollout, update, save_latest, load_latest
        from policy import CentralValue
        class Stub:
            def reset(self, seeds):
                return self.encode()
            def encode(self):
                return torch.randn(2, 2, 44), torch.randn(2, 2, 69)
            def step(self, actions):
                assert bool((actions.abs() <= 1).all())
                return None, torch.ones(2), None, None, dict(success=torch.ones(2, dtype=torch.bool))
        for algo in ('teamflow', 'maogpo'):
            policy = self.make_actor(algo)
            ref = copy.deepcopy(policy).requires_grad_(False)
            critic = CentralValue(69)
            ao = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=2e-5)
            co = torch.optim.Adam(critic.parameters(), lr=3e-4)
            models = (policy, ref, critic, ao, co)
            batch, _ = rollout(Stub(), policy, critic, 0, 2, horizon=3)
            self.assertEqual(update(*models, batch, 0, 6, 1)['actor_updates'], 0)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'latest.pt'
                progress = dict(steps=6, iteration=1, actor_updates=0, last_eval_step=0)
                save_latest(path, *models, progress, {}, 'fixture')
                expected = update(*models, batch, 1, 6, 1)
                expected_state = copy.deepcopy(policy.state_dict())
                self.assertEqual(load_latest(path, *models, {}, 'fixture'), progress)
                actual = update(*models, batch, 1, 6, 1)
                self.assertEqual(actual, expected)
                MAPPOContracts().assert_nested_equal(policy.state_dict(), expected_state)

    def test_meanflow_pretrain_resume(self):
        from unittest.mock import patch
        import pretrain_meanflow as mf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/'data.h5'
            write_fixture(data)
            args = mf.parser().parse_args(['--data', str(data), '--out', str(root/'run'),
                '--device', 'cpu', '--updates', '2', '--eval-every', '1', '--eval-episodes', '1',
                '--batch-size', '2', '--seed', '1'])
            def interrupted(*a, **kw):
                raise mf.EvaluationInterrupted()
            row = dict(deterministic={'episodes': 1}, stochastic={'episodes': 1})
            handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
            try:
                with patch.object(mf, 'evaluate_pair', interrupted):
                    self.assertEqual(mf.run(args), 75)
                ck = torch.load(root/'run/latest.pt', weights_only=False)
                self.assertEqual(ck['step'], 1)
                self.assertEqual(ck['progress']['last_eval_step'], 0)
                with patch.object(mf, 'evaluate_pair', return_value=row):
                    self.assertEqual(mf.run(args), 0)
                rows = [json.loads(x) for x in (root/'run/eval.jsonl').read_text().splitlines()]
                self.assertEqual([r['steps'] for r in rows], [1, 2])
                # Uninterrupted execution must give identical generator/Adam/RNG state.
                args.out = root/'continuous'; args.resume = False
                with patch.object(mf, 'evaluate_pair', return_value=row):
                    self.assertEqual(mf.run(args), 0)
                resumed = torch.load(root/'run/latest.pt', weights_only=False)
                continuous = torch.load(root/'continuous/latest.pt', weights_only=False)
                for key in ('generator', 'optimizer', 'rng', 'data_rng'):
                    MAPPOContracts().assert_nested_equal(resumed[key], continuous[key])
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)

    def test_ac_signal_run_resume(self):
        from unittest.mock import patch
        import mappo
        class Stub:
            def __init__(self, task, n, obs_mode, vision=None):
                self.n = n
            def reset(self, seeds):
                self.t = 0
                return self.encode()
            def encode(self):
                return torch.zeros(self.n, 2, 44), torch.zeros(self.n, 2, 69)
            def step(self, actions):
                self.t += 1
                if self.t == 1 and send_signal[0]:
                    signal.raise_signal(signal.SIGUSR1)
                return None, torch.ones(self.n), None, None, dict(success=torch.ones(self.n, dtype=torch.bool))
            def close(self): pass
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
        row = dict(deterministic={'episodes': 1}, stochastic={'episodes': 1})
        try:
            for algo in ('teamflow', 'maogpo'):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    actor = self.make_actor(algo)
                    cp = dict(method='macflow' if algo == 'teamflow' else 'meanflow', obs_mode='dec',
                              contract={'task': TASKS[0]}, state_mean=torch.zeros(69), state_std=torch.ones(69))
                    if algo == 'teamflow':
                        cp.update(config=actor.base.config_dict(), model=actor.base.state_dict())
                    else:
                        cp.update(config=actor.config_dict(), generator=actor.path.generator.state_dict(),
                                  phase='pretrain_meanflow', preprocessing='raw_float32')
                    torch.save(cp, root/'source.pt')
                    args = mappo.parser().parse_args(['--checkpoint', str(root/'source.pt'), '--out', str(root/'run'),
                        '--algo', algo, '--device', 'cpu', '--envs', '1', '--steps', '100',
                        '--epochs', '1', '--eval-episodes', '1', '--seed', '1'])
                    send_signal = [True]
                    with patch.object(mappo, 'BimanualEnv', Stub), patch.object(mappo, 'evaluate_pair', return_value=row):
                        self.assertEqual(mappo.run(args), 75)
                        send_signal[0] = False
                        self.assertEqual(mappo.run(args), 0)
                    evaluations = [json.loads(x) for x in (root/'run/eval.jsonl').read_text().splitlines()]
                    self.assertEqual([x['steps'] for x in evaluations], [0, 100])
                    if algo == 'teamflow':
                        self.assertEqual(evaluations[0]['pretrained_start'], 'teacher_10step')
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)

    def test_history_reset_evaluation(self):
        from unittest.mock import patch
        import evaluate as ev
        class Stub:
            device = torch.device('cpu')
            def __init__(self, *a, **kw): pass
            def reset(self, seeds): return self.encode()
            def encode(self): return torch.zeros(1, 2, 44), None
            def step(self, actions):
                return None, torch.ones(1), None, None, dict(success=torch.ones(1, dtype=torch.bool))
            def close(self): pass
        actor = self.make_actor('teamflow')
        for reset in (False, True):
            with patch.object(ev, 'BimanualEnv', Stub), patch.object(ev, 'HORIZON', 3), \
                    patch.object(actor, 'encode_step', wraps=actor.encode_step) as encode:
                ev.evaluate_pair(actor, TASKS[0], 'dec', 1, history_reset=reset)
                self.assertEqual(len(encode.call_args_list), 6)
                for i, call in enumerate(encode.call_args_list):
                    self.assertEqual(call.args[1] is None, reset or i % 3 == 0)


if __name__ == '__main__':
    unittest.main()
