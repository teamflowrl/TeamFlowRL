"""Separate CLI for A/B/C pretraining, fresh online MAPPO, resume and evaluation."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import numpy as np
import torch
from . import ALGORITHMS
from .checkpoints import FORMAT,load,save,sha256,import_offline,actor_payload,restore_actor
from .data import OfflineData
from .environments import make_env
from .meanflow import meanflow_loss
from .models import Generator,PathPolicy
from .ppo import MAPPO,gae,gradient_norm

STOP=False


def stop(signum,frame):
    global STOP
    STOP=True


def json_write(path,obj):
    path=Path(path);temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def rng_state():
    return dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    torch.set_rng_state(state['torch']);np.random.set_state(state['numpy']);random.setstate(state['python'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


class Log:
    def __init__(self,c,phase,resume=False):
        self.path=Path(c['output'])/phase
        if self.path.exists() and any(self.path.iterdir()) and not resume:
            raise FileExistsError('Refusing to overwrite an existing run: '+str(self.path))
        self.path.mkdir(parents=True,exist_ok=True)
        self.algorithm=c['algorithm'];self.start=time.monotonic();self.step=0
        self.handle=(self.path/'metrics.jsonl').open('a',buffering=1)
        config_name=f'resume_request_{time.time_ns()}.json' if resume else 'config.json'
        json_write(self.path/config_name,c)
        self.status('initializing')

    def emit(self,kind,**kw):
        obj=dict(kind=kind,algorithm=self.algorithm,algorithm_name=ALGORITHMS[self.algorithm],
                 step=self.step,time=time.time(),elapsed_s=time.monotonic()-self.start,**kw)
        line=json.dumps(obj,allow_nan=False);self.handle.write(line+'\n');print(line,flush=True)

    def status(self,state,**kw):
        json_write(self.path/'status.json',dict(status=state,algorithm=self.algorithm,step=self.step,
                                              pid=os.getpid(),updated=time.time(),**kw))


def features(actor,encoder,env,device,carry):
    # CPU tensors must not alias mutable environment arrays across env.step().
    obs=torch.tensor(env.current,device=device,dtype=torch.float32)
    if encoder:
        reset=np.broadcast_to(env.resets[:,None],env.current.shape[:2]).copy()
        if env.legal is not None:
            reset |= env.legal[:,:,0]
        obs,carry=encoder.step(obs,carry,torch.as_tensor(reset,device=device))
    if obs.shape[-1]!=actor.generator.obs_dim:
        raise ValueError('Environment features do not match generator input dimension')
    return obs.to(dtype=next(actor.parameters()).dtype),carry


def evaluate(actor,encoder,env_config,episodes=20,seed=10000,max_env_steps=1000000):
    if episodes<1:
        raise ValueError('Evaluation episode count must be positive')
    saved=rng_state();seed_all(seed);env=None
    device=next(actor.parameters()).device
    try:
        c=dict(env_config,n_envs=min(episodes,env_config.get('n_envs',1)),workers=1)
        env=make_env(c,seed);carry=None;returns=np.zeros(env.n_envs);results=[];wins=[];steps=0
        while len(results)<episodes:
            obs,carry=features(actor,encoder,env,device,carry)
            legal=torch.as_tensor(env.legal,device=device) if env.legal is not None else None
            # Base z remains random. Only transition exploration is disabled.
            sample=actor.sample(obs,legal,exploration=False)
            result=env.step(sample.actions.cpu().numpy());returns+=result['reward'];steps+=env.n_envs
            for i in np.flatnonzero(result['done']):
                if len(results)<episodes:
                    results.append(float(returns[i]))
                    if result['win'] is not None:wins.append(float(result['win'][i]))
                returns[i]=0
            if steps>max_env_steps:
                raise TimeoutError('Evaluation did not finish; check episode boundaries')
        return dict(episodes=episodes,mean_return=float(np.mean(results)),return_std=float(np.std(results)),
                    win_rate=float(np.mean(wins)) if wins else None,
                    evaluation_mode='random_base_latent_no_transition_noise',generator_calls=actor.steps)
    finally:
        if env:env.close()
        restore_rng(saved)


def pretrain(c,log,resume=None):
    if c['algorithm']!='C_MF_MAPPO':
        raise ValueError('A/B import existing MAC-Flow checkpoints; only C has new pretraining here')
    device=c.get('device','cpu');seed_all(c.get('seed',0));p=c['pretrain']
    if p['steps']<1 or p.get('batch_size',256)<1:
        raise ValueError('Pretraining steps and batch size must be positive')
    spec=dict(c['generator'],algorithm=c['algorithm'])
    generator=Generator(spec).to(device)
    data=OfflineData(c['data'],c.get('seed',0))
    optim=torch.optim.Adam(generator.parameters(),lr=p.get('lr',3e-4));start=0
    if resume:
        ck=load(resume)
        if ck.get('format')!=FORMAT or ck.get('phase')!='pretrain_meanflow' or ck['generator_spec']!=spec:
            raise ValueError('Incompatible MeanFlow pretraining resume')
        generator.load_state_dict(ck['generator']);optim.load_state_dict(ck['optimizer']);start=ck['step']
        data.restore_rng(ck['data_rng']);restore_rng(ck['rng'])
    def payload():
        return dict(format=FORMAT,algorithm=c['algorithm'],phase='pretrain_meanflow',step=log.step,
                    generator_spec=spec,generator=generator.state_dict(),optimizer=optim.state_dict(),
                    config=c,rng=rng_state(),data_rng=data.rng_state())
    log.step=start
    if not resume:save(log.path/'checkpoint_0.pt',payload())
    log.status('running')
    for step in range(start+1,p['steps']+1):
        obs,actions=data.sample(p.get('batch_size',256),device,spec['action_dim'],c.get('policy',{}).get('action_type','continuous'))
        if obs.shape[-1]!=spec['obs_dim']:
            raise ValueError('Offline observation width does not match MeanFlow')
        loss,metrics=meanflow_loss(generator,obs,actions,p.get('flow_ratio',.5),p.get('dispersive_weight',0.))
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite MeanFlow loss')
        optim.zero_grad(set_to_none=True);loss.backward();gn=gradient_norm(generator)
        torch.nn.utils.clip_grad_norm_(generator.parameters(),p.get('max_grad_norm',10.),error_if_nonfinite=True);optim.step()
        log.step=step
        if step==start+1 or step%p.get('log_every',1000)==0:
            log.emit('pretrain',loss=float(loss.detach()),generator_grad_norm=gn,**metrics);log.status('running')
        if step%p.get('save_every',50000)==0 or step==p['steps'] or STOP:
            save(log.path/f'checkpoint_{step}.pt',payload())
        if STOP:
            log.status('paused');return
    save(log.path/'checkpoint_final.pt',payload());log.status('complete')
    log.emit('complete',checkpoint=str(log.path/'checkpoint_final.pt'))


def online(c,log,resume=None):
    device=c.get('device','cpu');seed_all(c.get('seed',0));env=None
    o=dict(steps=50000000,rollout_steps=128,eval_every=250000,eval_episodes=20,gamma=.99,gae_lambda=.95)
    o.update(c.get('online',{}))
    if o['steps']<1 or o['rollout_steps']<1 or o['eval_every']<1:
        raise ValueError('Invalid online step budget/interval')
    if resume:
        ck=load(resume)
        if ck.get('phase')!='online':raise ValueError('Resume requires an online checkpoint')
        actor,encoder=restore_actor(ck,c['algorithm']);provenance=ck['provenance']
        for key in ('family','task','data_root','project_root'):
            if ck['config']['environment'].get(key)!=c['environment'].get(key):
                raise ValueError('Cannot resume into a different environment')
        if ck['ppo_options']!={**MAPPO_DEFAULTS,**c.get('ppo',{})}:
            raise ValueError('Resume PPO settings changed; create an explicit new study instead')
    else:
        actor,encoder,provenance=import_offline(c['source_checkpoint'],c['algorithm'],c.get('policy'))
    actor=actor.to(device).eval()
    if encoder:encoder=encoder.to(device).eval()
    try:
        env=make_env(c['environment'],c.get('seed',0)+1000)
        if env.action_type!=actor.action_type or env.action_dim!=actor.generator.action_dim:
            raise ValueError('Policy action space differs from environment')
        trainer=MAPPO(actor,env.states.shape[-1],c.get('ppo'))
        if resume:
            trainer.restore(ck);log.step=ck['step'];restore_rng(ck['rng'])
            log.emit('resume',checkpoint=resume,environment_reset=True,optimizers_restored=True)
        initial={k:v.detach().clone() for k,v in actor.generator.state_dict().items()}
        def payload():
            return dict(**actor_payload(actor,encoder),**trainer.state(),phase='online',step=log.step,
                        state_dim=env.states.shape[-1],config=c,provenance=provenance,rng=rng_state())
        if not resume:save(log.path/'checkpoint_0.pt',payload())
        if o['eval_episodes']:
            log.emit('eval',**evaluate(actor,encoder,c['environment'],o['eval_episodes']))
        log.status('running');carry=None;episode_returns=np.zeros(env.n_envs)
        while log.step<o['steps'] and not STOP:
            remaining=o['steps']-log.step
            active=min(env.n_envs,remaining)
            horizon=min(o['rollout_steps'],remaining//active)
            records=[];completed=[]
            for _ in range(horizon):
                with torch.no_grad():
                    obs,carry=features(actor,encoder,env,device,carry);obs=obs[:active]
                    states=torch.tensor(env.states[:active],device=device)
                    legal=torch.as_tensor(env.legal[:active],device=device) if env.legal is not None else None
                    sample=actor.sample(obs,legal)
                    mask=(legal.sum(-1)>1).float() if legal is not None else torch.ones(obs.shape[:-1],device=device)
                    values=trainer.critic(states)
                    result=env.step(sample.actions.cpu().numpy())
                    next_values=trainer.critic(torch.as_tensor(result['bootstrap_state'],device=device))
                    records.append(dict(obs=obs.clone(),state=states.clone(),chain=sample.chain,
                        log_prob=sample.log_prob,mask=mask,value=values,next_value=next_values,
                        reward=torch.as_tensor(result['reward'],device=device),
                        terminated=torch.as_tensor(result['terminated'],device=device,dtype=torch.float32),
                        done=torch.as_tensor(result['done'],device=device,dtype=torch.float32)))
                    episode_returns[:active]+=result['reward']
                    for i in np.flatnonzero(result['done']):
                        completed.append(float(episode_returns[i]));episode_returns[i]=0
                log.step+=active
            batch={k:torch.stack([r[k] for r in records]) for k in records[0]}
            adv,returns=gae(batch['reward'],batch['value'],batch['next_value'],batch['terminated'],batch['done'],o['gamma'],o['gae_lambda'])
            batch.update(advantage=adv,returns=returns)
            batch={k:v.flatten(0,1) for k,v in batch.items()}
            try:
                metrics=trainer.update(batch)
            except (FloatingPointError,AssertionError):
                # Diagnostic only: a partially updated actor is NOT a valid
                # resume checkpoint. Preserve the exact inputs for regression.
                save(log.path/'failure_debug.pt',dict(diagnostic_only=True,
                     algorithm=c['algorithm'],step=log.step,batch=batch,
                     actor=actor_payload(actor,encoder),trainer=trainer.state()))
                raise
            delta=sum((v-initial[k]).double().square().sum().item() for k,v in actor.generator.state_dict().items())**.5
            log.emit('train',generator_parameter_delta=delta,rollout_env_steps=horizon*active,
                     episode_return=float(np.mean(completed)) if completed else None,**metrics)
            log.status('running')
            boundary=log.step//o['eval_every']!=(log.step-horizon*active)//o['eval_every']
            if boundary or log.step==o['steps'] or STOP:
                save(log.path/f'checkpoint_{log.step}.pt',payload())
                if o['eval_episodes'] and not STOP:
                    log.emit('eval',**evaluate(actor,encoder,c['environment'],o['eval_episodes']))
        if sha256(provenance['source_checkpoint'])!=provenance['source_sha256']:
            raise AssertionError('Source pretrained checkpoint changed during training')
        save(log.path/('checkpoint_paused.pt' if STOP else 'checkpoint_final.pt'),payload())
        log.status('paused' if STOP else 'complete')
        log.emit('paused' if STOP else 'complete',actor_updates=trainer.actor_updates,source_unchanged=True)
    finally:
        if env:env.close()


# Explicit defaults are also checked during resume; keep in one public constant.
MAPPO_DEFAULTS=dict(actor_lr=2e-5,critic_lr=3e-4,epochs=4,batch_size=256,clip=.2,
                    target_kl=.02,max_grad_norm=.5,reference_coef=0.,bc_coef=0.,critic_warmup_iters=1)


DEFAULT_TRAINING_SEEDS=(0,1,2)


def expand_env(obj):
    """Expand ${VAR} placeholders (e.g. ${DATA_ROOT}) in configuration strings."""
    if isinstance(obj,dict):return {k:expand_env(v) for k,v in obj.items()}
    if isinstance(obj,list):return [expand_env(v) for v in obj]
    return os.path.expandvars(obj) if isinstance(obj,str) else obj


def seed_config(c,seed):
    """Copy of the configuration for one training seed.

    config["seed"] is overridden and every output goes to <output>/seed<k>/, so
    runs with different seeds never overwrite each other. A literal "{seed}" in
    source_checkpoint is replaced by the seed (e.g. a per-seed pretraining run).
    """
    c=copy.deepcopy(c);base=str(c['output']).rstrip('/');out=str(Path(base)/f'seed{seed}')
    c['seed']=int(seed);c['output']=out
    env=c.get('environment',{})
    for key,value in list(env.items()):
        if isinstance(value,str) and (value==base or value.startswith(base+'/')):env[key]=out+value[len(base):]
    if isinstance(c.get('source_checkpoint'),str):c['source_checkpoint']=c['source_checkpoint'].replace('{seed}',str(seed))
    return c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--phase',choices=['pretrain','online','evaluate'],required=True)
    parser.add_argument('--resume');parser.add_argument('--checkpoint')
    parser.add_argument('--seeds',type=int,nargs='+',default=list(DEFAULT_TRAINING_SEEDS),
                        help='training seeds (default: 0 1 2); each run writes to <output>/seed<k>/')
    parser.add_argument('--seed',type=int,default=None,help='run a single training seed (overrides --seeds)')
    args=parser.parse_args();c=expand_env(json.loads(Path(args.config).read_text()))
    seeds=[args.seed] if args.seed is not None else list(dict.fromkeys(args.seeds))
    if args.resume and len(seeds)>1 and '{seed}' not in args.resume:
        raise ValueError('--resume names one run: pass --seed k, or put {seed} in the checkpoint path')
    if c['algorithm'] not in ALGORITHMS:raise ValueError('Use A_TF_MAPPO, B_SG_MAPPO or C_MF_MAPPO')
    # Linux's loader reads LD_LIBRARY_PATH at process startup. Re-exec only this
    # CLI when an explicit MuJoCo runtime configuration needs extra libraries.
    libraries=c.get('environment',{}).get('library_paths',[])
    current=[p for p in os.environ.get('LD_LIBRARY_PATH','').split(':') if p]
    missing=[p for p in libraries if p not in current]
    if missing:
        child_env=dict(os.environ,LD_LIBRARY_PATH=':'.join(missing+current))
        os.execve(sys.executable,[sys.executable,'-m','generative_mappo.train',*sys.argv[1:]],child_env)
    if c.get('environment',{}).get('sc2path'):os.environ['SC2PATH']=c['environment']['sc2path']
    torch.set_num_threads(c.get('torch_threads',2))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.phase=='evaluate':
        if not args.checkpoint:raise ValueError('--checkpoint is required for evaluation')
        # A "{seed}" in --checkpoint evaluates one checkpoint per training seed.
        for seed in (seeds if '{seed}' in args.checkpoint else [None]):
            path=args.checkpoint.replace('{seed}',str(seed))
            ck=load(path);actor,encoder=restore_actor(ck,c['algorithm'])
            actor.to(c.get('device','cpu')).eval()
            if encoder:encoder.to(c.get('device','cpu')).eval()
            result=evaluate(actor,encoder,c['environment'],c.get('online',{}).get('eval_episodes',20))
            print(json.dumps(dict(algorithm=c['algorithm'],training_seed=seed,checkpoint_sha256=sha256(path),**result)))
        return
    for seed in seeds:
        if STOP:break
        cs=seed_config(c,seed);resume=args.resume.replace('{seed}',str(seed)) if args.resume else None
        log=Log(cs,args.phase,bool(resume))
        try:
            {'online':online,'pretrain':pretrain}[args.phase](cs,log,resume)
        except BaseException as exc:
            log.status('failed',error=repr(exc));raise
        finally:log.handle.close()


if __name__=='__main__':
    main()
