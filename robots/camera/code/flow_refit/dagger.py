"""DAgger from a state R_B teacher to a recurrent RGB2 one-step student."""
import argparse
import copy
import json
import math
from pathlib import Path
import signal
import time
import sys

# Modules shared with the online camera code live one directory up (code/); this directory
# (flow_refit/) comes first so its own policy.py / ac_policy.py are used.
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import torch

from data import sha256
from env_adapter import BimanualEnv, HORIZON
from evaluate import EvaluationInterrupted, evaluate_pair
from r_online import OnlinePolicy
from obs_builder import TASKS
from policy import Policy
from train import save_atomic
from vision import add_dino_arguments, checkpoint_camera_res, verify_weights
from training_seeds import add_seed_arguments, run_per_seed

TEACHER_RATES = {TASKS[0]: (.97, .97), TASKS[1]: (.99, .98)}


def decision(rows, task, max_rounds=6):
    """Rates are fractions; round zero is the baseline for improvement."""
    last = rows[-1]
    det = last['deterministic']['native_final_success_rate']
    stoch = last['stochastic']['native_final_success_rate']
    td, ts = TEACHER_RATES[task]
    if det >= .7*td and stoch >= .7*ts:
        return 'passed'
    if len(rows) >= 3:
        rates = [r['deterministic']['native_final_success_rate'] for r in rows[-3:]]
        if all(b-a < .03-1e-12 for a, b in zip(rates, rates[1:])):
            return 'plateau'
    if last['round'] >= max_rounds:
        return 'failed' if det < .30 and stoch < .30 else 'max_rounds'
    return 'continue'


class CollectionInterrupted(Exception):
    pass


@torch.no_grad()
def collect_round(env, student, teacher, path, round_number, steps=30000,
                  should_stop=lambda: False, contract=None, driver='student'):
    """Stream one vector episode at a time; commit only a complete round file.

    Keep exactly ceil(steps/100) whole episodes. The last vector batch may have
    unused lanes, whose physical steps are reported separately.
    """
    path = Path(path)
    partial = path.with_suffix('.partial.h5')
    episodes = math.ceil(steps/HORIZON)
    student.eval(); teacher.eval()
    stored = executed = 0
    seed0 = 20_000_000 + round_number*100000
    generator = torch.Generator(device=env.device).manual_seed(seed0)
    try:
        with h5py.File(partial, 'w') as f:
            f.attrs['contract'] = json.dumps(contract or {})
            f.attrs['round'] = round_number
            f.attrs['driver'] = driver
            data = f.create_group('data')
            while stored < episodes:
                if should_stop():
                    raise CollectionInterrupted()
                seeds = list(range(seed0+stored, seed0+stored+env.n))
                obs, state = env.reset(seeds)
                sc = tc = None
                keep = min(env.n, episodes-stored)
                # Only one vector episode is held in host memory (~335 MB).
                buffers = dict(obs_rgb2=np.empty((keep, HORIZON, 2, obs.shape[-1]), np.float16),
                               state=np.empty((keep, HORIZON, 2, state.shape[-1]), np.float32),
                               z=np.empty((keep, HORIZON, 2, 8), np.float32),
                               actions=np.empty((keep, HORIZON, 2, 8), np.float32),
                               success=np.empty((keep, HORIZON), bool))
                for t in range(HORIZON):
                    if should_stop():
                        raise CollectionInterrupted()
                    sf, sc = student.encode_step(obs, sc)
                    tf, tc = teacher.encode_step(env.teacher_obs(), tc)
                    z = torch.randn((env.n, 2, 8), device=env.device, generator=generator)
                    labels = teacher.distribution(tf, z).mean
                    # Round 0 (driver='teacher', beta=1): teacher executes its own labels; later rounds beta=0.
                    actions = labels if driver == 'teacher' else student.distribution(sf, z).mean
                    for key, value in [('obs_rgb2', obs), ('state', state), ('z', z), ('actions', labels)]:
                        buffers[key][:, t] = value[:keep].cpu().numpy()
                    # ManiSkill GPU envs flag terminated on success but never auto-reset; like the R_online
                    # rollouts and evaluate.py, every episode runs the full HORIZON and flags are ignored.
                    _, _, _, _, info = env.step(actions)
                    buffers['success'][:, t] = info['success'][:keep].reshape(-1).cpu().numpy()
                    if t < HORIZON-1:
                        obs, state = env.encode()
                for i in range(keep):
                    g = data.create_group(f'episode_{stored+i:06d}')
                    g.attrs['seed'] = seeds[i]
                    for key, value in buffers.items():
                        g.create_dataset(key, data=value[i])
                stored += keep
                executed += env.n*HORIZON
            f.attrs['transitions'] = stored*HORIZON
            f.attrs['executed_steps'] = executed
            f.attrs['complete'] = True
        if should_stop():
            raise CollectionInterrupted()
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return dict(episodes=stored, transitions=stored*HORIZON, executed_steps=executed)


class AggregatedData:
    """GPU cache filled episode by episode from each file; no aggregate CPU copy."""
    def __init__(self, paths, device='cuda', contract=None):
        self.device = torch.device(device)
        metadata = []
        dims = None
        for path in paths:
            with h5py.File(path, 'r') as f:
                if not f.attrs.get('complete', False):
                    raise ValueError(f'Incomplete round: {path}')
                if contract is not None and json.loads(f.attrs['contract']) != contract:
                    raise ValueError(f'Round provenance differs: {path}')
                for name in sorted(f['data']):
                    g = f['data'][name]
                    t, n, d = g['obs_rgb2'].shape
                    sd = g['state'].shape[-1]
                    if not 1 <= t <= HORIZON or n != 2 or g['obs_rgb2'].dtype != np.dtype('float16'):
                        raise ValueError(f'Invalid RGB2 episode: {path}/{name}')
                    if dims is not None and dims != (d, sd):
                        raise ValueError('Round dimensions differ')
                    dims = (d, sd)
                    for key, shape in [('state', (t, 2, sd)), ('z', (t, 2, 8)),
                                       ('actions', (t, 2, 8)), ('success', (t,))]:
                        if g[key].shape != shape:
                            raise ValueError(f'Invalid {key}: {path}/{name}')
                    metadata.append((path, name, t))
        if not metadata:
            raise ValueError('Empty aggregate')
        count = len(metadata)
        self.cpu_sizes = np.array([x[2] for x in metadata], np.int64)
        self.sizes = torch.as_tensor(self.cpu_sizes, device=device)
        self.tensors = {}
        for key, shape, dtype in [('obs', (count, HORIZON, 2, dims[0]), torch.float16),
                                  ('state', (count, HORIZON, 2, dims[1]), torch.float32),
                                  ('z', (count, HORIZON, 2, 8), torch.float32),
                                  ('actions', (count, HORIZON, 2, 8), torch.float32),
                                  ('success', (count, HORIZON), torch.bool)]:
            self.tensors[key] = torch.zeros(shape, device=device, dtype=dtype)
        index = 0
        for path in paths:
            with h5py.File(path, 'r') as f:
                for name in sorted(f['data']):
                    t = self.cpu_sizes[index]
                    for key, target in self.tensors.items():
                        arr = f['data'][name]['obs_rgb2' if key == 'obs' else key][:]
                        if not np.isfinite(arr).all():
                            raise ValueError(f'Nonfinite {path}/{name}/{key}')
                        target[index, :t].copy_(torch.as_tensor(arr, device=device))
                    index += 1

    def batch(self, rng, count=64, length=20):
        if count < 1 or not 1 <= length <= HORIZON:
            raise ValueError('Invalid sampler counts')
        candidates = np.flatnonzero(self.cpu_sizes >= length)
        if not len(candidates):
            raise ValueError('No episode supports sequence length')
        chosen = rng.choice(candidates, count)
        starts = torch.as_tensor(rng.integers(0, self.cpu_sizes[chosen]-length+1), device=self.device)
        ep = torch.as_tensor(chosen, device=self.device)
        ix = starts[:, None] + torch.arange(length, device=self.device)
        px = torch.arange(HORIZON, device=self.device)
        prefix = self.tensors['obs'][ep[:, None], px[None]].clone()
        prefix *= (px[None] < starts[:, None])[..., None, None]
        t = self.sizes[ep, None]
        coverage = torch.minimum(ix, t-length)-torch.maximum(torch.zeros_like(ix), ix-length+1)+1
        return dict(prefix=prefix.float(), lengths=starts,
                    obs_sequence=self.tensors['obs'][ep[:, None], ix].float(),
                    z=self.tensors['z'][ep[:, None], ix], actions=self.tensors['actions'][ep[:, None], ix],
                    time_weights=((t-length+1)*length/(t*coverage)).float())


def configure_student(model):
    model.requires_grad_(False)
    model.encoder.requires_grad_(True)
    model.actor_onestep_flow.requires_grad_(True)
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)


def dagger_loss(model, batch):
    carry = model.warmup(batch['prefix'], batch['lengths'])
    features, _ = model.encode_sequence(batch['obs_sequence'], carry)
    prediction = model.distribution(features, batch['z']).mean
    # Euclidean norm over the eight action coordinates, mean over time/arms/windows.
    return ((prediction-batch['actions']).square().sum(-1)*batch['time_weights'][..., None]).mean()


def train_step(model, optimizer, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = dagger_loss(model, batch)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.)
    if not bool(torch.isfinite(loss) & torch.isfinite(norm)):
        raise FloatingPointError('Nonfinite DAgger update')
    optimizer.step()
    return float(loss.detach())


def rng_state(rng):
    return dict(sampler=copy.deepcopy(rng.bit_generator.state), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state, rng):
    rng.bit_generator.state = state['sampler']
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def student_checkpoint(source, model, optimizer, progress, rng):
    # Preserve all train.py metadata, including normalization and DINO provenance.
    cp = {k: v for k, v in source.items() if k not in ('model', 'optimizer', 'algo', 'version', 'params')}
    random_state = rng_state(rng)
    cp.update(model=model.state_dict(), optimizer=optimizer.state_dict(), config=model.config_dict(),
              method='R', obs_mode='rgb2', camera_res=224,
              step=source['step']+progress['total_updates'], batch=64, sequence=20,
              sampler_rng=random_state['sampler'], torch_rng=random_state['torch'],
              cuda_rng=random_state['cuda'], elapsed_seconds=progress.get('elapsed_seconds', 0.),
              dagger=copy.deepcopy(progress), dagger_rng=random_state)
    return cp


def write_rows(path, rows):
    temp = path.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    temp.replace(path)


def run(a):
    if min(a.envs, a.steps, a.updates, a.batch, a.sequence, a.eval_episodes, a.save_every) < 1:
        raise ValueError('Counts must be positive')
    if a.sequence > HORIZON or a.rounds != 6 or a.steps > 100000:
        raise ValueError('Require sequence <=100, six-round limit and <=100000 steps per round')
    a.out.mkdir(parents=True, exist_ok=True)
    stop = {'value': False}
    def request_stop(sig, frame):
        stop['value'] = True
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    should_stop = lambda: stop['value']
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    source = torch.load(a.student, map_location='cpu', weights_only=False)
    teacher_cp = torch.load(a.teacher, map_location='cpu', weights_only=False)
    if (source['method'] != 'R' or source['obs_mode'] != 'rgb2'
            or source['contract']['task'] != a.task or checkpoint_camera_res(source, 224) != 224
            or not source['config'].get('q_state_input') or source['config']['obs_dim'] != 13080
            or source['config'].get('memoryless', False)):
        raise ValueError('Expected recurrent RGB2 224 privileged-Q R student')
    if (teacher_cp.get('algo') != 'var_b' or teacher_cp['obs_mode'] != 'dec'
            or teacher_cp['contract']['task'] != a.task):
        raise ValueError('Expected matching state R_B teacher snapshot')
    for cp in (source, teacher_cp):
        cfg = cp['config']
        if cfg['student_latent_std'] != 1. or not cfg['critic_detach_encoder'] or cfg['num_agents'] != 2 or cfg['action_dim'] != 8:
            raise ValueError('Invalid teacher/student execution config')
    for key in ('state_mean', 'state_std', 'dino_weights_sha256'):
        if key not in source:
            raise ValueError(f'Missing student provenance: {key}')
    teacher_cp.pop('optimizer', None)
    verify_weights(a.dino_weights, source['dino_weights_sha256'])
    params = {k: getattr(a, k) for k in ('task', 'envs', 'steps', 'teacher_steps', 'updates', 'batch', 'sequence', 'eval_episodes', 'rounds', 'seed')}
    params.update(student_sha256=sha256(a.student), teacher_sha256=sha256(a.teacher),
                  dino_weights_sha256=source['dino_weights_sha256'])
    model = Policy(source['config']).cuda()
    model.load_state_dict(source['model'])
    optimizer = configure_student(model)
    # Do not retain the large pretraining optimizer in host memory.
    source.pop('optimizer', None); source.pop('model', None)
    progress = dict(round=0, stage='evaluate', update=0, total_updates=0, rows=[],
                    loss_sum=0., collection={}, elapsed_seconds=0., params=params)
    latest = a.out/'latest.pt'
    if latest.exists():
        cp = torch.load(latest, map_location='cpu', weights_only=False)
        if cp['dagger']['params'] != params or cp['config'] != model.config_dict():
            raise ValueError('Resume parameters/source/config differ')
        model.load_state_dict(cp['model']); optimizer.load_state_dict(cp['optimizer'])
        progress = cp['dagger']; restore_rng(cp['dagger_rng'], rng)
        del cp
    write_rows(a.out/'rounds.jsonl', progress['rows'])
    clock = time.monotonic()
    def save():
        nonlocal clock
        now = time.monotonic()
        progress['elapsed_seconds'] += now-clock
        clock = now
        cp = student_checkpoint(source, model, optimizer, progress, rng)
        cp.update(batch=a.batch, sequence=a.sequence)
        save_atomic(cp, latest)
        return cp
    save()
    try:
        while progress['stage'] != 'done':
            if should_stop():
                save(); return 75
            r = progress['round']
            if progress['stage'] == 'collect':
                # An uncommitted complete file is regenerated from this stage's RNG.
                with torch.random.fork_rng(devices=[0]):
                    teacher = OnlinePolicy(Policy(teacher_cp['config']), 'var_b').cuda()
                    teacher.load_state_dict(teacher_cp['model']); teacher.requires_grad_(False)
                    env = BimanualEnv(a.task, a.envs, 'rgb2', camera_res=224,
                                      dino_source=a.dino_source, dino_weights=a.dino_weights,
                                      dino_weights_sha256=source['dino_weights_sha256'])
                    try:
                        if r == 1 and not (a.out/'round_0.h5').exists():
                            # Standard DAgger: aggregate the teacher's own trajectories (round 0) with every student round.
                            progress['teacher_collection'] = collect_round(env, model, teacher, a.out/'round_0.h5', 0,
                                                                           a.teacher_steps, should_stop, params, driver='teacher')
                            save()
                        progress['collection'] = collect_round(env, model, teacher, a.out/f'round_{r}.h5', r,
                                                               a.steps, should_stop, params)
                    finally:
                        env.close(); del env, teacher
                        torch.cuda.empty_cache()
                progress.update(stage='train', update=0, loss_sum=0.)
                save()
            elif progress['stage'] == 'train':
                data = AggregatedData([a.out/f'round_{i}.h5' for i in range(0, r+1)], contract=params)
                while progress['update'] < a.updates and not should_stop():
                    loss = train_step(model, optimizer, data.batch(rng, a.batch, a.sequence))
                    progress['update'] += 1; progress['total_updates'] += 1
                    progress['loss_sum'] += loss
                    if progress['update'] % a.save_every == 0:
                        save()
                        print(json.dumps(dict(round=r, update=progress['update'], loss=loss)), flush=True)
                del data
                torch.cuda.empty_cache()
                if progress['update'] == a.updates:
                    progress['stage'] = 'evaluate'
                save()
            elif progress['stage'] == 'evaluate':
                cp = save()
                save_atomic(cp, a.out/f'student_r{r}.pt')
                # Wrapper construction freezes encoder; restore trainability afterwards.
                with torch.random.fork_rng(devices=[0]):
                    online = OnlinePolicy(model, 'var_b')
                    try:
                        result = evaluate_pair(online, a.task, 'rgb2', a.eval_episodes,
                                               should_stop=should_stop, dino_source=a.dino_source,
                                               dino_weights=a.dino_weights,
                                               dino_weights_sha256=source['dino_weights_sha256'],
                                               det_seed=5_000_000, stoch_seed=5_100_000, camera_res=224)
                    finally:
                        model.requires_grad_(False)
                        model.encoder.requires_grad_(True); model.actor_onestep_flow.requires_grad_(True)
                        del online
                row = dict(round=r, **result, collection=progress['collection'],
                           aggregate_transitions=progress.get('teacher_collection', {}).get('transitions', 0)
                                                 + sum(x.get('collection', {}).get('transitions', 0) for x in progress['rows'])
                                                 + progress['collection'].get('transitions', 0),
                           training_loss=progress['loss_sum']/progress['update'] if progress['update'] else None,
                           elapsed_seconds=progress['elapsed_seconds']+time.monotonic()-clock)
                previous_time = progress['rows'][-1]['elapsed_seconds'] if progress['rows'] else 0.
                row['round_seconds'] = row['elapsed_seconds']-previous_time
                progress['rows'].append(row)
                row['decision'] = decision(progress['rows'], a.task, a.rounds) if r else 'baseline'
                if r and row['decision'] != 'continue':
                    progress['stage'] = 'done'
                else:
                    progress.update(round=r+1, stage='collect', update=0, loss_sum=0., collection={})
                save()
                write_rows(a.out/'rounds.jsonl', progress['rows'])
                print(json.dumps(row), flush=True)
            else:
                raise ValueError(f'Unknown stage: {progress["stage"]}')
    except (CollectionInterrupted, EvaluationInterrupted):
        save(); return 75
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--student', type=Path, required=True, help='camera student checkpoint to start DAgger from')
    p.add_argument('--teacher', type=Path, required=True, help='state teacher checkpoint that labels every visited step')
    p.add_argument('--out', type=Path, required=True)
    for name, default in [('envs', 64), ('steps', 80000), ('teacher-steps', 80000), ('updates', 8000), ('batch', 64),
                          ('sequence', 20), ('eval-episodes', 100), ('rounds', 6), ('save-every', 500)]:
        p.add_argument('--'+name, type=int, default=default)
    # Training seeds: --seeds 0 1 2 (default) or --seed k; seed k writes to <out>/seed<k>/.
    add_seed_arguments(p)
    add_dino_arguments(p)
    return p


if __name__ == '__main__':
    raise SystemExit(run_per_seed(parser().parse_args(), run))
