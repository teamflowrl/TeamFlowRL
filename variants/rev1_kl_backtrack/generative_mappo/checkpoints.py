"""Versioned, algorithm-tagged checkpoints and strict legacy offline import."""
import hashlib
import os
from pathlib import Path
import torch
from .models import Generator, PathPolicy, FrozenSMACEncoder

FORMAT = 'generative_mappo_v1'


def load(path):
    # Only load trusted local research checkpoints (some contain NumPy RNG state).
    try:
        return torch.load(path,map_location='cpu',weights_only=False)
    except TypeError:  # PyTorch 1.12
        return torch.load(path,map_location='cpu')


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temp);os.replace(temp,path)


def infer_hidden(state):
    weights=sorted([(int(k.split('.')[0]),v) for k,v in state.items() if k.endswith('.weight')],key=lambda p:p[0])
    if not weights:
        raise ValueError('No MLP weights found')
    return [w.shape[0] for _,w in weights[:-1]]


def import_offline(path, algorithm, policy_options=None):
    ck=load(path)
    if ck.get('format') == FORMAT:
        if ck.get('phase')!='pretrain_meanflow' or algorithm!='C_MF_MAPPO' or ck['algorithm']!=algorithm:
            raise ValueError('Fresh online training requires the matching OFFLINE checkpoint; use --resume for online')
        generator=Generator(ck['generator_spec']);generator.load_state_dict(ck['generator'],strict=True)
        encoder=None
    else:
        if algorithm not in ('A_TF_MAPPO','B_SG_MAPPO'):
            raise ValueError('C requires its own pretrained MeanFlow, not MAC-Flow weights')
        if ck.get('phase') not in (None,'pretrain') or 'actor' in ck or 'model' not in ck:
            raise ValueError('Expected an offline MAC-Flow checkpoint, not a PPO checkpoint')
        state=ck['model'];teacher=algorithm=='A_TF_MAPPO';encoder=None
        continuous=any(k.startswith('teacher.') for k in state)
        prefix=('teacher.' if teacher else 'student.') if continuous else (
            'actor_bc_flow.net.' if teacher else 'actor_onestep_flow.net.')
        selected={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        hidden=infer_hidden(selected)
        keys=sorted((k for k in selected if k.endswith('.weight')),key=lambda k:int(k.split('.')[0]))
        action_dim=selected[keys[-1]].shape[0]
        if continuous:
            obs_dim=selected[keys[0]].shape[1]-action_dim-int(teacher)
            spec=dict(algorithm=algorithm,obs_dim=obs_dim,action_dim=action_dim,hidden=hidden,approximate='tanh')
        else:
            cfg=ck['config'];encoder=FrozenSMACEncoder(cfg)
            encoder.load_state_dict({k[8:]:v for k,v in state.items() if k.startswith('encoder.')},strict=True)
            spec=dict(algorithm=algorithm,obs_dim=cfg.get('lstm_hidden_dim',64),action_dim=action_dim,
                      hidden=hidden,approximate='none',time_frequencies=cfg.get('t_embed_frequencies',8))
        generator=Generator(spec);generator.net.load_state_dict(selected,strict=True)
    actor=PathPolicy(generator,**(policy_options or {}))
    provenance=dict(source_checkpoint=str(Path(path).resolve()),source_sha256=sha256(path),
                    source_step=ck.get('step',ck.get('steps')),source_phase=ck.get('phase','legacy_offline'))
    return actor,encoder,provenance


def actor_payload(actor, encoder=None):
    return dict(format=FORMAT,algorithm=actor.algorithm,generator_spec=actor.generator.spec,
                policy_options=actor.options,actor=actor.state_dict(),
                encoder_config=encoder.cfg if encoder else None,
                encoder=encoder.state_dict() if encoder else None)


def restore_actor(payload,expected_algorithm=None):
    if payload.get('format')!=FORMAT:
        raise ValueError('Not a generative MAPPO checkpoint')
    if expected_algorithm and payload['algorithm']!=expected_algorithm:
        raise ValueError('Algorithm mismatch: refusing to mix A, B and C')
    actor=PathPolicy(Generator(payload['generator_spec']),**payload['policy_options'])
    if actor.algorithm!=payload['algorithm']:
        raise ValueError('Generator/checkpoint algorithm mismatch')
    actor.load_state_dict(payload['actor'],strict=True)
    encoder=None
    if payload.get('encoder_config'):
        encoder=FrozenSMACEncoder(payload['encoder_config']);encoder.load_state_dict(payload['encoder'],strict=True)
    return actor,encoder

