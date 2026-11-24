"""Offline MeanFlow data: existing runtime datasets or explicit NPY arrays."""
from pathlib import Path
import numpy as np
import torch
from .environments import legacy_module,with_ids


class OfflineData:
    def __init__(self,c,seed):
        self.c=c;self.rng=np.random.RandomState(seed);self.runtime=None
        if c['format']=='runtime':
            self.runtime=legacy_module(c['runtime_dir'],'data').Dataset(c['root'],c['task'],c['split'],seed)
        elif c['format']=='npy':
            p=Path(c['path'])
            self.obs=np.load(p/c.get('obs_file','obs.npy'),mmap_mode='r')
            self.actions=np.load(p/c.get('action_file','actions.npy'),mmap_mode='r')
            if len(self.obs)!=len(self.actions) or len(self.obs)==0:
                raise ValueError('Mismatched/empty offline arrays')
        else:
            raise ValueError('Data format must be runtime or npy')

    def sample(self,batch_size,device,action_dim,action_type):
        if self.runtime:
            batch=self.runtime.sample(device,batch_size,1)
            obs,act=batch['obs'][:,0],batch['actions'][:,0]
        else:
            idx=self.rng.randint(0,len(self.obs),batch_size)
            obs=np.array(self.obs[idx],dtype=np.float32);act=np.array(self.actions[idx])
            if self.c.get('add_agent_ids',False):
                obs=with_ids(obs)
            obs=torch.as_tensor(obs,device=device);act=torch.as_tensor(act,device=device)
        if action_type=='discrete':
            if act.ndim==obs.ndim and act.shape[-1]==1:
                act=act.squeeze(-1)
            if act.ndim==obs.ndim-1:
                if not bool((act==act.long()).all()) or bool(((act<0)|(act>=action_dim)).any()):
                    raise ValueError('Invalid discrete action indices')
                act=torch.nn.functional.one_hot(act.long(),action_dim)
        act=act.float()
        if obs.ndim!=3 or act.shape!=(*obs.shape[:-1],action_dim):
            raise ValueError('Offline data must yield [batch,agents,obs/action_dim]')
        if not bool(torch.isfinite(obs).all() and torch.isfinite(act).all()):
            raise ValueError('Nonfinite offline data')
        # No silent clipping/normalization of demonstrations.
        if action_type=='continuous' and bool((act.abs()>1.0001).any()):
            raise ValueError('Normalize continuous action data to [-1,1] explicitly')
        return obs,act

    def rng_state(self):
        return self.runtime.rng.get_state() if self.runtime else self.rng.get_state()

    def restore_rng(self,state):
        (self.runtime.rng if self.runtime else self.rng).set_state(state)


def validate_smac_observation_layout(config):
    """Fail before training if a C checkpoint's offline ID layout differs online.

    A/B use legacy encoders and their original data adapters. C consumes NPY
    observations directly, so matching dimensions alone is insufficient.
    """
    if config['algorithm']!='C_MF_MAPPO' or config.get('environment',{}).get('family') not in ('smac','smacv2'):
        return
    data=config['data']
    if data['format']!='npy':
        raise ValueError('SMAC MeanFlow observation layout requires explicit NPY contract')
    obs=np.load(Path(data['path'])/data.get('obs_file','obs.npy'),mmap_mode='r')
    if obs.ndim!=3 or len(obs)==0:raise ValueError('Invalid SMAC observation array')
    n=obs.shape[1];sample=np.asarray(obs[np.linspace(0,len(obs)-1,min(256,len(obs)),dtype=int)])
    if data.get('add_agent_ids',False):sample=with_ids(sample)
    position=config['environment'].get('agent_id_position','trailing')
    if position not in ('leading','trailing'):raise ValueError('Unknown agent ID position')
    ids=sample[...,:n] if position=='leading' else sample[...,-n:]
    if not np.all(ids==np.eye(n,dtype=sample.dtype)):
        raise ValueError('Offline/online agent-ID layout mismatch: expected '+position)
    if sample.shape[-1]!=config['generator']['obs_dim']:
        raise ValueError('Offline/online observation dimension mismatch')
