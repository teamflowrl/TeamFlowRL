"""State MAC-Flow and BC, resumable training and in-process evaluations."""
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from data import Demonstrations
from learner import MACFlowLearner
from policy import Config
from evaluate import EvaluationInterrupted
from vision import add_dino_arguments, verify_weights, checkpoint_camera_res


def save_atomic(obj, path):
    temp = path.with_suffix('.tmp'); torch.save(obj, temp); temp.replace(path)


def append_json(path, row):
    with path.open('a') as f: f.write(json.dumps(row)+'\n')


def configure_bc(learner):
    model = learner.model
    model.requires_grad_(False)
    model.encoder.requires_grad_(True); model.actor_onestep_flow.requires_grad_(True)
    learner.optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=model.cfg.lr)


def action_mse(model, b):
    carry = model.warmup(b['prefix'], b['lengths'])
    features, _ = model.encode_sequence(b['obs_sequence'], carry)
    prediction = model.distribution(features[:, :-1], torch.zeros_like(b['actions'])).mean
    return ((prediction-b['actions']).square()*b['time_weights'][..., None, None]).mean()


def bc_step(learner, b):
    learner.model.train()
    loss = action_mse(learner.model, b)
    learner.optim.zero_grad(set_to_none=True); loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in learner.model.parameters() if p.requires_grad], 10.)
    if not bool(torch.isfinite(norm) & torch.isfinite(loss)): raise FloatingPointError('BC nonfinite')
    learner.optim.step()
    return dict(loss=float(loss.detach()), grad_norm=float(norm))


def drop_indices(data, names):
    if not names:
        return ()
    fields = data.contract.get('fields', {}).get('obs_dec')
    if not fields:
        raise ValueError('Data contract has no obs_dec field schema; cannot drop fields')
    indices, start, found = [], 0, set()
    for field in fields:
        stop = start + field['size']
        if field['name'] in names:
            indices.extend(range(start, stop)); found.add(field['name'])
        start = stop
    if start != data.obs_dim or found != set(names):
        raise ValueError(f'Field schema mismatch or unknown fields: {sorted(set(names) - found)}')
    return tuple(indices)


def run(a):
    if min(a.updates, a.eval_every, a.eval_episodes, a.batch, a.sequence, a.seconds) <= 0:
        raise ValueError('Training counts and limits must be positive')
    a.out.mkdir(parents=True, exist_ok=True)
    a.resume = a.resume or (a.out/'latest.pt').exists()
    if any((a.out/x).exists() for x in ('latest.pt', 'train.jsonl', 'eval.jsonl')) and not a.resume:
        raise FileExistsError('Existing run; use --resume or a new output directory')
    stop = {'value': False, 'signal': None}
    def request_stop(sig, frame):
        stop.update(value=True, signal=sig)
    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT): signal.signal(sig, request_stop)
    torch.cuda.set_device(0)
    torch.set_num_threads(2); torch.set_float32_matmul_precision('high')
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    valid_rng = np.random.default_rng(a.seed+100000)
    data = Demonstrations(a.data, a.obs_mode, 'cuda'); stats = data.statistics()
    dino_hash = verify_weights(a.dino_weights, data.contract.get('dino_weights_sha256')) if a.obs_mode in ('rgb', 'rgb2') else None
    camera_res = checkpoint_camera_res(dict(obs_mode=a.obs_mode, contract=data.contract), getattr(a, 'camera_res', 224))
    # Fail before training if a selected sequence cannot be validated.
    for split, indices in data.split_indices.items():
        if not np.any(data.cpu_sizes[indices] >= a.sequence):
            raise ValueError(f'No {split} episode supports sequence={a.sequence}')
    cfg = Config(obs_dim=data.obs_dim, num_agents=data.num_agents, action_dim=data.action_dim, state_dim=data.state_dim,
                 initial_std=.02, q_gradient_ratio=.25, balance_window_edges=True,
                 q_weight=a.q_weight, critic_detach_encoder=a.critic_detach_encoder,
                 student_latent_std=a.student_latent_std, q_state_input=getattr(a, 'q_state_input', False),
                 memoryless=getattr(a, 'memoryless', False), obs_drop=drop_indices(data, a.drop_obs_fields))
    learner = MACFlowLearner(cfg, 'cuda'); model = learner.model
    model.obs_mean.copy_(torch.as_tensor(stats['obs_mean'], device='cuda'))
    model.obs_std.copy_(torch.as_tensor(stats['obs_std'], device='cuda'))
    if cfg.q_state_input:
        model.state_mean.copy_(torch.as_tensor(stats['state_mean'], device='cuda'))
        model.state_std.copy_(torch.as_tensor(stats['state_std'], device='cuda'))
    if a.method == 'bc': configure_bc(learner)
    step = 0; last_eval = 0; prior_seconds = 0.
    if a.resume:
        cp = torch.load(a.out/'latest.pt', map_location='cuda', weights_only=False)
        if (cp['data_sha256'] != data.fingerprint or cp['method'] != a.method
                or cp.get('dino_weights_sha256') != dino_hash or cp['obs_mode'] != a.obs_mode or cp['config'] != model.config_dict()
                or cp['sequence'] != a.sequence or cp['batch'] != a.batch or cp['seed'] != a.seed):
            raise ValueError('Resume dataset/method/observation/config/sampler differs')
        if a.obs_mode == 'rgb2':
            checkpoint_camera_res(cp, camera_res)
        model.load_state_dict(cp['model']); learner.optim.load_state_dict(cp['optimizer'])
        step, last_eval = cp['step'], cp['last_eval_step']
        prior_seconds = cp['elapsed_seconds']
        rng.bit_generator.state = cp['sampler_rng']; valid_rng.bit_generator.state = cp['valid_rng']
        torch.set_rng_state(cp['torch_rng'].cpu())
        if a.obs_mode in ('rgb', 'rgb2'):
            torch.cuda.set_rng_state(cp['cuda_rng'][0].cpu(), device=0)
        else:
            torch.cuda.set_rng_state_all([s.cpu() for s in cp['cuda_rng']])
    started = time.monotonic(); log_time = started; log_step = step
    failure = None
    def checkpoint():
        return dict(model=model.state_dict(), optimizer=learner.optim.state_dict(), config=model.config_dict(),
                    method=a.method, obs_mode=a.obs_mode, step=step, last_eval_step=last_eval,
                    seed=a.seed, sequence=a.sequence, batch=a.batch, contract=data.contract,
                    data_sha256=data.fingerprint, dino_weights_sha256=dino_hash, state_mean=stats['state_mean'], state_std=stats['state_std'],
                    sampler_rng=rng.bit_generator.state, valid_rng=valid_rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(), cuda_rng=[torch.cuda.get_rng_state(0)] if a.obs_mode in ('rgb', 'rgb2') else torch.cuda.get_rng_state_all(),
                    elapsed_seconds=prior_seconds+time.monotonic()-started,
                    **({'camera_res': camera_res} if a.obs_mode == 'rgb2' else {}))
    def save(): save_atomic(checkpoint(), a.out/'latest.pt')
    def closed_loop():
        nonlocal last_eval
        from evaluate import evaluate_pair
        path = a.out/f'ckpt_{step}.pt'
        save_atomic(checkpoint(), path)
        # Do not let simulation initialization or evaluation alter training RNG.
        with torch.random.fork_rng(devices=[0] if a.obs_mode in ('rgb', 'rgb2') else list(range(torch.cuda.device_count()))):
            result = evaluate_pair(model, data.contract['task'], a.obs_mode, a.eval_episodes,
                                   should_stop=lambda: stop['value'] or time.monotonic()-started >= a.seconds,
                                   dino_source=a.dino_source, dino_weights=a.dino_weights,
                                   dino_weights_sha256=dino_hash, camera_res=camera_res, contract=data.contract)
        append_json(a.out/'eval.jsonl', dict(step=step, checkpoint=path.name, task=data.contract['task'],
                                           method=a.method, obs_mode=a.obs_mode, **result))
        last_eval = step
        save()
    try:
        # Retry an interrupted scheduled evaluation before more updates.
        if step and step > last_eval and (step % a.eval_every == 0 or step >= a.updates) and not stop['value']:
            closed_loop()
        while step < a.updates and time.monotonic()-started < a.seconds and not stop['value']:
            b = data.batch(rng, a.batch, a.sequence)
            metrics = learner.train_step(b) if a.method == 'macflow' else bc_step(learner, b)
            step += 1
            validation = {}
            if step % 2000 == 0 and not stop['value']:
                model.eval()
                with torch.no_grad():
                    validation['validation_action_mse'] = float(action_mse(model, data.batch(valid_rng, a.batch, a.sequence, 'valid')))
                model.train()
            if step % 500 == 0 or step == a.updates:
                now = time.monotonic()
                row = dict(step=step, method=a.method, obs_mode=a.obs_mode,
                           elapsed_seconds=prior_seconds+now-started,
                           updates_per_second=(step-log_step)/(now-log_time), **metrics, **validation)
                append_json(a.out/'train.jsonl', row); print(json.dumps(row), flush=True)
                log_time, log_step = now, step
                save()
            if (step % a.eval_every == 0 or step == a.updates) and not stop['value']:
                closed_loop()
    except EvaluationInterrupted:
        pass
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        save()
        complete = step >= a.updates and last_eval >= step
        status = dict(status='failed' if failure else ('complete' if complete else 'checkpointed'),
                      step=step, last_eval_step=last_eval, target=a.updates, method=a.method, obs_mode=a.obs_mode,
                      signal=stop['signal'], error=failure, elapsed_seconds=prior_seconds+time.monotonic()-started,
                      excluded_short_episodes={k: int((data.cpu_sizes[v] < a.sequence).sum()) for k, v in data.split_indices.items()})
        (a.out/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    return 0 if complete else 75


if __name__ == '__main__':
    from training_seeds import add_seed_arguments, run_per_seed
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--method', choices=['macflow', 'bc'], default='macflow')
    p.add_argument('--obs-mode', choices=['dec', 'central', 'rgb', 'rgb2'], default='dec')
    p.add_argument('--updates', type=int, default=50000); p.add_argument('--eval-every', type=int, default=10000)
    p.add_argument('--eval-episodes', type=int, default=20); p.add_argument('--batch', type=int, default=128)
    p.add_argument('--sequence', type=int, default=20); p.add_argument('--seconds', type=float, default=float('inf'))
    p.add_argument('--resume', action='store_true')
    # Training seeds: --seeds 0 1 2 (default) or --seed k; seed k writes to <out>/seed<k>/.
    add_seed_arguments(p)
    p.add_argument('--q-weight', type=float, default=1.0)
    p.add_argument('--critic-detach-encoder', action='store_true')
    p.add_argument('--q-state-input', action='store_true')
    p.add_argument('--memoryless', action='store_true')
    p.add_argument('--drop-obs-fields', nargs='*', default=[], help='obs_dec field names hidden from the policy, e.g. own_qvel')
    p.add_argument('--camera-res', type=int, choices=[128, 224], default=224)
    # The dfpo branch needs a one-step student that consumes a random latent (1.0); 0 keeps the latent-free student.
    p.add_argument('--student-latent-std', type=float, default=0.)
    add_dino_arguments(p)
    raise SystemExit(run_per_seed(p.parse_args(), run))
