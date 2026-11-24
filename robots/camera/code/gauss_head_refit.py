"""BC+MAPPO counterpart of flow_refit/vision_flow_distill.py for the symmetric camera ablation.

TeamFlowRL's camera start freezes the DAgger student's encoder and refits the 10-step flow head
(actor_bc_flow) for 20k updates on the teacher-labelled DAgger aggregate (rounds 0..R). Here the same
frozen encoder, data, updates, batch, sequence length, optimizer and gradient clip refit the one-step
Gaussian head (actor_onestep_flow) that BC+MAPPO executes, with the DAgger student loss
(flow_refit/dagger.py:dagger_loss: squared error of the head mean at the sampled latent to the teacher
label). Checkpoints every 4,000 updates keep the train.py format, so policycore --algo base and the
pretraining evaluation script load them unchanged.
"""
import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Camera code directory (this directory); flow_refit/ holds the DAgger modules.
R_CODE = Path(__file__).resolve().parent
# flow_refit/ has its own policy.py and dagger.py (the ones vision_flow_distill.py uses); it must come first.
sys.path.insert(0, str(R_CODE))
sys.path.insert(0, str(R_CODE / 'flow_refit'))
from dagger import AggregatedData, dagger_loss  # noqa: E402
from policy import Policy  # noqa: E402
from train import save_atomic, append_json  # noqa: E402
from training_seeds import add_seed_arguments, run_per_seed  # noqa: E402

SAVE_AT = (4000, 8000, 12000, 16000, 20000)


def main(a):
    stop = {'flag': False}
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(flag=True))
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    source = torch.load(a.student, map_location='cuda:0', weights_only=False)
    # Same name translation as flow_refit/run_flow_refit.py: flow_refit/policy.py only accepts action_head='R_onestep'.
    # The original name is restored in every saved checkpoint so policycore loads them unchanged.
    original_head = source['config'].get('action_head')
    model = Policy(dict(source['config'], action_head='R_onestep')).cuda()
    model.load_state_dict(source['model'])
    model.requires_grad_(False); model.actor_onestep_flow.requires_grad_(True)
    optimizer = torch.optim.Adam(model.actor_onestep_flow.parameters(), lr=a.lr)
    a.out.mkdir(parents=True, exist_ok=True)
    latest = a.out / 'latest.pt'
    step = 0
    if latest.exists():
        cp = torch.load(latest, map_location='cuda:0', weights_only=False)
        model.load_state_dict(cp['model']); optimizer.load_state_dict(cp['head_optimizer'])
        step = cp['head_refit_step']; rng.bit_generator.state = cp['head_rng']
    rounds = [a.dagger / f'round_{r}.h5' for r in range(0, a.rounds + 1)]
    data = AggregatedData(rounds)
    started = time.monotonic()

    def save(path):
        cp = {k: v for k, v in source.items() if k not in ('model', 'optimizer', 'dagger', 'dagger_rng')}
        cp['config'] = dict(source['config'], action_head=original_head)
        cp.update(model=model.state_dict(), head_optimizer=optimizer.state_dict(), head_refit_step=step,
                  head_rng=rng.bit_generator.state, method='R', obs_mode='rgb2',
                  head_refit=dict(student=str(a.student), rounds=[str(r) for r in rounds], updates=a.updates,
                                  lr=a.lr, batch=a.batch, sequence=a.sequence, head='actor_onestep_flow',
                                  note='encoder frozen; one-step Gaussian head refit with the DAgger student loss'))
        save_atomic(cp, path)

    while step < a.updates and not stop['flag']:
        model.train()
        loss = dagger_loss(model, data.batch(rng, a.batch, a.sequence))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor_onestep_flow.parameters(), 10.)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite head loss')
        optimizer.step(); step += 1
        if step in SAVE_AT:
            save(a.out / f'ckpt_{step}.pt')
        if step % 500 == 0:
            append_json(a.out / 'train.jsonl', dict(step=step, head_loss=float(loss.detach()),
                                                    seconds=time.monotonic() - started))
            save(latest)
    save(latest)
    if step >= a.updates:
        save(a.out / 'head_refit.pt')
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
