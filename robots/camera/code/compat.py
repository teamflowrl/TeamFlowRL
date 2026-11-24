"""Translate metadata tags found in upstream checkpoints to this deployment's names."""
import torch


ALGOS = ('base', 'var_a', 'var_b', 'var_c')
METHODS = ('flowcore', 'meanflow', 'bc')


def infer_algo(cp):
    cfg = cp.get('config') or {}
    model = cp.get('model') or {}
    if isinstance(cfg, dict) and isinstance(cfg.get('base_config'), dict):
        return 'var_a'
    if any(k.startswith('noise.') for k in model):
        return 'var_b'
    if any(k.startswith('base.') for k in model):
        return 'base'
    spec = (cfg.get('generator_spec') or {}).get('algorithm', '') if isinstance(cfg, dict) else ''
    if isinstance(spec, str) and spec.startswith('C_'):
        return 'var_c'
    if cp.get('phase') == 'pretrain_meanflow':
        return 'var_c'
    return None


def _spec(value):
    if isinstance(value, str) and len(value) > 2 and value[1] == '_':
        if value.startswith('A_'):
            return 'A_TF_POLICYCORE'
        if value.startswith('B_'):
            return 'B_SG_POLICYCORE'
        if value.startswith('C_'):
            return 'C_MF_POLICYCORE'
    return value


def normalize_checkpoint(cp):
    """In-place, non-destructive view of a checkpoint dictionary.

    The checkpoint files are not rewritten; only in-memory tags are renamed so the
    deployment code never needs to spell the upstream method identifiers.
    """
    if cp.get('algo') not in ALGOS:
        inferred = infer_algo(cp)
        if inferred is not None:
            cp['algo'] = inferred
    if cp.get('method') not in METHODS:
        cp['method'] = 'flowcore'
    cfg = cp.get('config')
    if isinstance(cfg, dict):
        if cfg.get('action_head') not in (None, 'flowcore_onestep'):
            cfg['action_head'] = 'flowcore_onestep'
        spec = cfg.get('generator_spec')
        if isinstance(spec, dict):
            spec['algorithm'] = _spec(spec.get('algorithm'))
        base = cfg.get('base_config')
        if isinstance(base, dict):
            if base.get('action_head') not in (None, 'flowcore_onestep'):
                base['action_head'] = 'flowcore_onestep'
            if isinstance(base.get('generator_spec'), dict):
                base['generator_spec']['algorithm'] = _spec(base['generator_spec'].get('algorithm'))
    params = cp.get('params')
    if isinstance(params, dict):
        if params.get('algo') not in ALGOS and cp.get('algo') in ALGOS:
            params['algo'] = cp['algo']
        if params.get('method') not in METHODS:
            params['method'] = 'flowcore'
    return cp


def load_checkpoint(path, map_location='cpu'):
    return normalize_checkpoint(torch.load(path, map_location=map_location, weights_only=False))
