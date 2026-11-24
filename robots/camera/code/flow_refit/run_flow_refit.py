"""Flow-head refit on the frozen DAgger encoder (TeamFlowRL camera start).

Loads checkpoints with the action-head key renamed to this directory's name
('R_onestep'), exactly as the original launcher did, and then runs
vision_flow_distill.py with the given arguments. Training seeds 0, 1, 2 by default
(--seed k for one seed); seed k writes to <out>/seed<k>/.

    python flow_refit/run_flow_refit.py --student <dagger>/seed{seed}/student.pt \\
        --dagger <dagger>/seed{seed}/data --rounds 5 --out <flow_refit>/pick \\
        --updates 20000 --batch 64 --sequence 20 --lr 3e-4
"""
import runpy
import sys
from pathlib import Path

import torch

_load = torch.load


def load(*args, **kwargs):
    cp = _load(*args, **kwargs)
    if isinstance(cp, dict) and isinstance(cp.get('config'), dict) and 'action_head' in cp['config']:
        cp['config']['action_head'] = 'R_onestep'
    return cp


if __name__ == '__main__':
    torch.load = load
    torch.set_num_threads(2)
    script = Path(__file__).resolve().parent / 'vision_flow_distill.py'
    sys.argv = [str(script)] + sys.argv[1:]
    runpy.run_path(str(script), run_name='__main__')
