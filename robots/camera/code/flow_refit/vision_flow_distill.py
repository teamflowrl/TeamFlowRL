"""Visual R_A (A) initialization: fit the 10-step flow teacher on the DAgger aggregate with a frozen encoder.

Why: A optimizes the K=10 flow path itself. The DAgger student's one-step head is not that object, and its
`actor_bc_flow` was trained on pre-DAgger encoder features (the encoder changed during DAgger), so neither can be
used as A's start. We therefore freeze the DAgger student's encoder (the visual representation everyone shares)
and refit only `actor_bc_flow` by the same conditional flow-matching loss as R pretraining
(learner.py: x_t=(1-t)x0+t a, regress a-x0), on the teacher-labelled DAgger data the student was trained on
(rounds 0..R). Labels a are teacher actions at sampled latents, so the teacher's action distribution is kept.
Output keeps the train.py checkpoint format so ac_policy.from_pretrained(..., 'var_a') imports it.
"""
import argparse
import json
from pathlib import Path
import signal
import time
import sys

# Modules shared with the online camera code live one directory up (code/); this directory
# (flow_refit/) comes first so its own policy.py is used.
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from dagger import AggregatedData
from policy import Policy
from train import save_atomic, append_json
from training_seeds import add_seed_arguments, run_per_seed


def flow_loss(model, batch):
    with torch.no_grad():
        carry = model.warmup(batch['prefix'], batch['lengths'])
        features, _ = model.encode_sequence(batch['obs_sequence'], carry)
    actions = batch['actions']
    x0 = torch.randn_like(actions)
    t = torch.rand_like(actions[..., :1])
    xt = (1-t)*x0 + t*actions
    velocity = model.actor_bc_flow(features, xt, model.time_sin_embed(t))
    error = (velocity-(actions-x0)).square()
    weights = batch['time_weights']
    return (error*weights.reshape(*weights.shape, 1, 1)).mean()


def main(a):
    stop = {'flag': False}
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(flag=True))
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    source = torch.load(a.student, map_location='cuda:0', weights_only=False)
    model = Policy(source['config']).cuda()
    model.load_state_dict(source['model'])
    if source['config']['flow_steps'] != 10:
        raise ValueError('R_A needs the 10-step flow head')
    model.requires_grad_(False); model.actor_bc_flow.requires_grad_(True)
    optimizer = torch.optim.Adam(model.actor_bc_flow.parameters(), lr=a.lr)
    a.out.mkdir(parents=True, exist_ok=True)
    latest = a.out/'latest.pt'
    step = 0
    if latest.exists():
        cp = torch.load(latest, map_location='cuda:0', weights_only=False)
        model.load_state_dict(cp['model']); optimizer.load_state_dict(cp['flow_optimizer'])
        step = cp['flow_distill_step']; rng.bit_generator.state = cp['flow_rng']
    rounds = [a.dagger/f'round_{r}.h5' for r in range(0, a.rounds+1)]
    data = AggregatedData(rounds)
    started = time.monotonic()
    def save(path):
        cp = {k: v for k, v in source.items() if k not in ('model', 'optimizer', 'dagger', 'dagger_rng')}
        cp.update(model=model.state_dict(), flow_optimizer=optimizer.state_dict(), flow_distill_step=step,
                  flow_rng=rng.bit_generator.state, method='R', obs_mode='rgb2',
                  flow_distill=dict(student=str(a.student), rounds=[str(r) for r in rounds], updates=a.updates,
                                    lr=a.lr, batch=a.batch, sequence=a.sequence,
                                    note='encoder frozen; actor_bc_flow refit by conditional flow matching on teacher labels'))
        save_atomic(cp, path)
    while step < a.updates and not stop['flag']:
        model.train()
        loss = flow_loss(model, data.batch(rng, a.batch, a.sequence))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor_bc_flow.parameters(), 10.)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite flow loss')
        optimizer.step(); step += 1
        if step in (4000, 8000, 12000, 16000, 20000):
            save(a.out/f'ckpt_{step}.pt')
        if step % 500 == 0:
            append_json(a.out/'train.jsonl', dict(step=step, flow_loss=float(loss.detach()),
                                                  seconds=time.monotonic()-started))
            save(latest)
    save(latest)
    if step >= a.updates:
        save(a.out/'flow_teacher.pt')
        return 0
    return 75


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--student', type=Path, required=True); p.add_argument('--dagger', type=Path, required=True)
    p.add_argument('--rounds', type=int, required=True, help='use DAgger rounds 0..R (those the student was trained on)')
    p.add_argument('--out', type=Path, required=True); p.add_argument('--updates', type=int, default=20000)
    p.add_argument('--batch', type=int, default=64); p.add_argument('--sequence', type=int, default=20)
    p.add_argument('--lr', type=float, default=3e-4)
    # Training seeds: --seeds 0 1 2 (default) or --seed k; seed k writes to <out>/seed<k>/.
    add_seed_arguments(p)
    raise SystemExit(run_per_seed(p.parse_args(), main))
