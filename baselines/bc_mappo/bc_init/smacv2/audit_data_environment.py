"""Audit all converted Replay arrays and the real Terran 5 vs 5 environment."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from common import atomic_json, load_environment_factory, observe, seed_environment, sha256
from sc2_port_lease import install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    cfg = json.loads(Path(parser.parse_args().config).read_text())
    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)
    folder = Path(cfg['data_root']) / cfg['task'] / 'Replay'
    arrays = {k: np.load(folder/(k+'.npy'), mmap_mode='r') for k in
              ['obs', 'actions', 'rewards', 'legals', 'path_lengths', 'states']}
    lengths = arrays['path_lengths']; total = len(arrays['obs']); n = arrays['obs'].shape[1]
    checks = dict(length_sum=int(lengths.sum()) == total, positive_lengths=bool((lengths > 0).all()),
                  all_rows_aligned=all(len(v) == total for k, v in arrays.items() if k != 'path_lengths'))
    illegal = empty = bad_ids = 0; finite = shared = True
    for start in range(0, total, 8192):
        a = {k: v[start:start+8192] for k, v in arrays.items() if k != 'path_lengths'}
        finite &= all(bool(np.isfinite(v).all()) for v in a.values())
        shared &= bool(np.all(a['rewards'] == a['rewards'][:, :1]))
        bad_ids += int(np.any(a['obs'][..., -n:] != np.eye(n), axis=(1, 2)).sum())
        assert a['actions'].dtype.kind in 'iu' and a['actions'].min() >= 0 and a['actions'].max() < arrays['legals'].shape[-1]
        illegal += int((np.take_along_axis(a['legals'], a['actions'][..., None], -1)[..., 0] == 0).sum())
        empty += int((a['legals'].sum(-1) == 0).sum())
    checks.update(all_finite=finite, recorded_actions_legal=illegal == 0,
                  nonempty_legal_masks=empty == 0, shared_team_rewards=shared,
                  dataset_suffix_agent_ids_exact=bad_ids == 0)
    install()
    env = load_environment_factory(cfg['project_root'])(cfg['task'], seed=90000)
    try:
        checks['seeded_capability_distributions'] = seed_environment(env, 90000) > 0
        env.reset(); obs, state, legal = observe(env); info = env.get_env_info()
        checks.update(agent_count=info['n_agents'] == n,
            observation_dimensions=obs.shape == (n, arrays['obs'].shape[-1]-n),
            augmented_observation_dimensions=np.concatenate([obs,np.eye(n)],axis=-1).shape == arrays['obs'].shape[1:],
            action_dimensions=legal.shape == arrays['legals'].shape[1:],
            state_dimensions=state.shape == arrays['states'].shape[1:])
        for _ in range(30):
            actions = [int(np.flatnonzero(row)[-1]) for row in legal]
            _, done, _ = env.step(actions)
            if done: env.reset()
            obs, state, legal = observe(env)
    finally:
        env.close()
    sys.path.insert(0, str(Path(cfg['project_root'])/'mac_flow_pytorch_smacv2'))
    from macflow_torch.data import ConvertedSMACv2Sequences, SMACv2DatasetSpec
    data = ConvertedSMACv2Sequences(SMACv2DatasetSpec(task=cfg['task'], data_root=cfg['data_root']), 20)
    sample = data.sample_numpy(32)
    checks['sequence_sampler_shapes'] = sample['obs'].shape == (32, 20, n, arrays['obs'].shape[-1])
    checks['short_episodes_excluded'] = bool((lengths[data.eligible_episodes] >= 20).all())
    starts = np.r_[0, np.cumsum(lengths[:-1])]
    returns = np.add.reduceat(arrays['rewards'][:, 0], starts)
    metadata = json.loads((folder/'conversion_metadata.json').read_text())
    checks['conversion_identity_and_count'] = metadata['scenario'] == cfg['task'] and metadata['retained_steps'] == total
    result = dict(status='verified' if all(checks.values()) else 'failed', checks=checks,
        task=cfg['task'], split='Replay', data_path=str(folder), env_info=info, rows=total,
        episodes=len(lengths), episode_length_range=[int(lengths.min()), int(lengths.max())],
        excluded_short_episodes=int((lengths < 20).sum()), eligible_episodes=len(data.eligible_episodes),
        mean_dataset_team_return=float(returns.mean()), recorded_success_rate=None,
        observation_transform='Offline arrays contain trailing agent IDs; PPO appends identical IDs to raw observations.',
        conversion_metadata=metadata, conversion_metadata_sha256=sha256(folder/'conversion_metadata.json'),
        files={k: dict(path=str(folder/(k+'.npy')), shape=list(v.shape), dtype=str(v.dtype),
                       sha256=sha256(folder/(k+'.npy'))) for k, v in arrays.items()})
    atomic_json(out/'data_environment_audit.json', result)
    print(json.dumps(result), flush=True)
    assert result['status'] == 'verified', checks


if __name__ == '__main__': main()
