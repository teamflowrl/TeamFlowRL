"""Adapters to the existing, unchanged environment runtimes.

Each training process imports one legacy runtime only. This keeps legacy module
names (data/common) isolated. SMAC workers are inherited from that runtime.
"""
import importlib
import sys
from pathlib import Path
import numpy as np


def legacy_module(directory,name):
    directory=Path(directory).resolve()
    previous=sys.modules.get(name)
    if previous is not None and Path(previous.__file__).resolve().parent!=directory:
        raise RuntimeError(f'Conflicting legacy module {name}; use separate processes per environment family')
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))
    return importlib.import_module(name)


def with_ids(obs,position="trailing"):
    n=obs.shape[-2]
    ids=np.broadcast_to(np.eye(n,dtype=np.float32),(*obs.shape[:-2],n,n))
    if position not in ("leading","trailing"):
        raise ValueError("Unknown agent ID position")
    return np.concatenate([ids,obs] if position=="leading" else [obs,ids],-1).astype(np.float32)


class RuntimeEnv:
    def __init__(self,c,seed=None):
        self.c=c;self.family=c['family'];self.n_envs=c.get('n_envs',1)
        seed=c.get('seed',0) if seed is None else seed
        if self.family in ('smac','smacv2'):
            self.pool=legacy_module(c['runtime_dir'],'vector_env').ParallelEnv(
                c['project_root'],c['task'],self.n_envs,seed,log_dir=c.get('worker_log_dir'))
            self.action_dim=self.pool.info['n_actions'];self.action_type='discrete'
            self.current=with_ids(self.pool.obs,c.get("agent_id_position","trailing"));self.states=self.pool.state.copy();self.legal=self.pool.legal.copy()
        else:
            module='mpe_env' if self.family=='mpe' else 'mamujoco_env'
            self.pool=legacy_module(c['runtime_dir'],module).Pool(c['task'],c['data_root'],self.n_envs,c.get('workers',1),seed)
            self.action_dim=c['action_dim'];self.action_type='continuous'
            self.current=self.pool.current.copy()
            self.states=self.current.reshape(self.n_envs,-1).copy() if self.family=='mpe' else self.pool.states.copy()
            self.legal=None
        self.n_agents=self.current.shape[1]
        self.resets=np.ones(self.n_envs,bool)

    def step(self,actions):
        n=len(actions)
        if self.family in ('smac','smacv2'):
            padded=np.zeros((self.n_envs,self.n_agents),np.int64);padded[:n]=actions
            result=self.pool.step(padded,active=np.arange(self.n_envs)<n)
            self.current=with_ids(result['obs'],self.c.get('agent_id_position','trailing'));self.states=result['state'];self.legal=result['legal']
            final=result['bootstrap_state'][:n];reward=result['reward'][:n];done=result['done'][:n];term=result['terminated'][:n]
            wins=result['win'][:n]
        elif self.family=='mpe':
            nxt,final_obs,reward,done=self.pool.step(actions)
            self.current[:n]=nxt;self.states[:n]=nxt.reshape(n,-1)
            final=final_obs.reshape(n,-1);reward=reward.mean(-1);term=np.zeros(n,bool);wins=None
        else:
            nxt,state,final,reward,done,term=self.pool.step(actions)
            self.current[:n]=nxt;self.states[:n]=state;reward=reward.mean(-1);wins=None
        self.resets[:n]=done
        return dict(reward=np.asarray(reward,np.float32),done=np.asarray(done,bool),
                    terminated=np.asarray(term,bool),bootstrap_state=np.asarray(final,np.float32),win=wins)

    def close(self):
        self.pool.close()


class ToyEnv:
    """Explicit synthetic integration fixture, never a reported benchmark."""
    def __init__(self,c,seed=None):
        self.n_envs=c.get('n_envs',4);self.n_agents=c.get('n_agents',2)
        self.action_dim=c.get('action_dim',2);self.action_type=c.get('action_type','continuous')
        self.rng=np.random.RandomState(c.get('seed',0) if seed is None else seed)
        self.current=np.zeros((self.n_envs,self.n_agents,c.get('obs_dim',4)),np.float32)
        self.time=np.zeros(self.n_envs,int);self.resets=np.ones(self.n_envs,bool)
        self.current[:]=self.rng.normal(size=self.current.shape)
        self.states=self.current.reshape(self.n_envs,-1).copy()
        self.legal=np.ones((self.n_envs,self.n_agents,self.action_dim),bool) if self.action_type=='discrete' else None

    def step(self,actions):
        n=len(actions);self.time[:n]+=1
        if self.action_type=='continuous':
            reward=-(actions-.5*np.tanh(self.current[:n,:,:self.action_dim])).square().mean((1,2)) if hasattr(actions,'square') else -np.mean((actions-.5*np.tanh(self.current[:n,:,:self.action_dim]))**2,(1,2))
        else:
            reward=(actions==self.current[:n,:,0].clip(0,self.action_dim-1).astype(int)).mean(-1)
        self.current[:n]=self.rng.normal(size=self.current[:n].shape)
        final=self.current[:n].reshape(n,-1).copy();done=self.time[:n]>=7
        term=done & (np.arange(n)%2==0)
        self.current[:n][done]=self.rng.normal(size=self.current[:n][done].shape)
        self.time[:n][done]=0;self.states=self.current.reshape(self.n_envs,-1).copy();self.resets[:n]=done
        return dict(reward=np.asarray(reward,np.float32),done=done,terminated=term,bootstrap_state=final,win=None)

    def close(self):
        pass


def make_env(config,seed=None):
    if config['family']=='toy':
        return ToyEnv(config,seed)
    if config['family'] not in ('mpe','mamujoco','smac','smacv2'):
        raise ValueError('Unknown environment family')
    return RuntimeEnv(config,seed)

