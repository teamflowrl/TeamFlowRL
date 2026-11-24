"""Training seeds for the training entry points in this directory.

Every training command runs the three training seeds 0, 1 and 2 by default. Each
seed is a complete, independent run that writes to its own sub-directory
``<output>/seed<k>/``, so runs with different seeds never overwrite each other.

    --seeds 0 1 2   (default) run the listed training seeds one after another
    --seed k        run a single training seed (e.g. to run seeds in parallel)

For every seed the training code receives the
arguments (or configuration) with only the seed and the output location replaced.
Evaluation and environment seeds are set separately.
"""
import copy
import json
import os
from pathlib import Path

DEFAULT_TRAINING_SEEDS = (0, 1, 2)


def add_seed_arguments(parser):
    """Add ``--seeds`` (default 0 1 2) and ``--seed`` (single-seed override)."""
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_TRAINING_SEEDS),
                        help='training seeds (default: 0 1 2); seed k writes to <output>/seed<k>/')
    parser.add_argument('--seed', type=int, default=None,
                        help='run a single training seed (overrides --seeds)')
    return parser


def selected_seeds(args):
    """Training seeds selected on the command line, in order, without duplicates."""
    if getattr(args, 'seed', None) is not None:
        return [int(args.seed)]
    return [int(s) for s in dict.fromkeys(args.seeds)]


def seed_dir(path, seed):
    """``<path>/seed<k>``; keeps the type (str or Path) of ``path``."""
    out = Path(path) / f'seed{int(seed)}'
    return out if isinstance(path, Path) else str(out)


def run_per_seed(args, train, path_attrs=('out',)):
    """Call ``train(args_k)`` once per selected training seed.

    ``args_k`` is a copy of ``args`` in which ``seed`` is the training seed and
    every attribute named in ``path_attrs`` (output locations) is replaced by its
    ``seed<k>`` sub-directory. A literal ``{seed}`` in any other string or path
    argument is replaced by the seed, so a stage can read the output of the same
    seed of an earlier stage (e.g. ``--checkpoint runs/pretrain/seed{seed}/ckpt.pt``).
    Stops at the first non-zero return code (for example the resumable-interruption
    code 75) and returns it.
    """
    for seed in selected_seeds(args):
        a = copy.copy(args)
        if hasattr(a, 'seeds'):
            delattr(a, 'seeds')
        for key, value in vars(a).items():
            if isinstance(value, (str, Path)) and '{seed}' in str(value):
                setattr(a, key, type(value)(str(value).replace('{seed}', str(seed))))
        a.seed = seed
        for attr in path_attrs:
            if getattr(a, attr, None) is not None:
                setattr(a, attr, seed_dir(getattr(a, attr), seed))
        code = train(a)
        if code not in (None, 0):
            return code
    return 0


def _expand(obj):
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return os.path.expandvars(obj) if isinstance(obj, str) else obj


def seed_config(cfg, seed, output_keys=('output', 'output_dir'), path_keys=('checkpoint',)):
    """Copy of a JSON training configuration for one training seed.

    * ``${VAR}`` placeholders are expanded from the environment;
    * every ``"seed"`` entry (top level and nested) is set to the training seed;
    * output locations get a ``seed<k>`` level: ``"output": R`` becomes
      ``R/seed<k>`` and ``"output_dir": R/phase`` becomes ``R/seed<k>/phase``;
      the keys in ``path_keys`` (e.g. the pretrained checkpoint that a PPO
      configuration reads from the pretraining output) are moved accordingly.

    Returns ``(config, seed_root)`` where ``seed_root`` is the per-seed directory.
    """
    cfg = _expand(copy.deepcopy(cfg))
    roots = []

    def collect(d):
        for key, value in d.items():
            if isinstance(value, dict):
                collect(value)
            elif key == 'output' and isinstance(value, str):
                roots.append(value.rstrip('/'))
            elif key == 'output_dir' and isinstance(value, str):
                roots.append(os.path.dirname(value.rstrip('/')))
    collect(cfg)
    if not roots:
        raise ValueError('configuration has no "output" or "output_dir" entry')

    def move(value):
        for root in sorted(set(roots), key=len, reverse=True):
            if value == root or value.startswith(root + '/'):
                return root + '/' + f'seed{int(seed)}' + value[len(root):]
        return value

    def rewrite(d):
        for key, value in d.items():
            if isinstance(value, dict):
                rewrite(value)
            elif key == 'seed':
                d[key] = int(seed)
            elif key in output_keys + path_keys and isinstance(value, str):
                d[key] = move(value)
    rewrite(cfg)
    return cfg, Path(roots[0]) / f'seed{int(seed)}'


def write_seed_config(cfg, seed, name, **kwargs):
    """Write the per-seed configuration to ``<seed_root>/<name>`` and return its path."""
    cfg, root = seed_config(cfg, seed, **kwargs)
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + '\n')
    return str(path)
