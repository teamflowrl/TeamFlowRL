"""Hands-away release evaluation over the 150-control-step horizon with full episode telemetry."""
import os,sys,json,time,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
# Robot code: CORE_CODE (default robots/core). RoboFactory release code: RF_CODE (default robots/core;
# it must also provide the RoboFactory release task, see robots/hands_away/README.md).
CORE_CODE=Path(os.environ.get('CORE_CODE',str(Path(__file__).resolve().parents[2]/'core')))
RF_CODE=Path(os.environ.get('RF_CODE',str(CORE_CODE)))
DOMAIN=os.environ.get('AWAY_DOMAIN','ms')
THRESHOLDS=None

def install(domain,threshold_file):
    global DOMAIN,THRESHOLDS
    DOMAIN=domain
    THRESHOLDS=json.loads(Path(threshold_file).read_text())['thresholds']
    if domain=='rf':
        from env_robofactory import register_robofactory
        register_robofactory()
    import away_metrics
    away_metrics.install(domain,THRESHOLDS)

def make_env(domain,n):
    if domain=='ms':
        from ms_release.env_adapter import BimanualEnv
        return BimanualEnv('TwoRobotStackCube-v1',n,obs_mode='dec')
    from env_robofactory import RoboFactoryEnv
    return RoboFactoryEnv('TwoRobotsStackCubeReleasePPO',n,obs_mode='dec')

def evaluate(policy,domain,episodes,seed_start,stochastic,should_stop=None):
    import torch
    from away_metrics import PhysicsTelemetry,EpisodeRecorder
    started=time.monotonic()
    env=make_env(domain,episodes)
    training=policy.training
    tele=None
    try:
        seeds=list(range(seed_start,seed_start+episodes))
        obs,_=env.reset(seeds)
        tele=PhysicsTelemetry(env.base)
        record=EpisodeRecorder(episodes,seeds,THRESHOLDS)
        generator=torch.Generator(device=env.device).manual_seed(seed_start)
        carry=None; returns=torch.zeros(episodes,device=env.device)
        policy.eval()
        with torch.no_grad():
            for step in range(1,151):
                if should_stop and should_stop():
                    from evaluate import EvaluationInterrupted
                    raise EvaluationInterrupted('Hands-away evaluation interrupted')
                features,carry=policy.encode_step(obs,carry)
                action,_,_=policy.act(features,stochastic=stochastic,generator=generator)
                tele.begin_step()
                _,reward,_,_,info=env.step(action.clamp(-1,1))
                record.append(step,tele.read(info)); returns+=reward.reshape(-1)
                obs,_=env.encode()
        result=record.finish()
        result.update(stochastic=stochastic,seeds=seeds,returns=returns.cpu().tolist(),elapsed_seconds=time.monotonic()-started)
        # Retain archived trainer summary keys while keeping detailed telemetry.
        for key in ('original','release','away'):
            vals=[e[key+'_final'] for e in result['episodes']]
            if any(v is None for v in vals): continue
            result[key+'_final_success_rate']=sum(vals)/episodes
            result[key+'_final_success']=vals
            result[key+'_success_once_rate']=sum(e[key] for e in result['episodes'])/episodes
        result['native_final_success_rate']=result.get('release_final_success_rate',sum(e['original_final'] for e in result['episodes'])/episodes)
        result['native_success_once_rate']=result.get('release_success_once_rate',result['rates']['original'])
        return result
    finally:
        if tele: tele.close()
        env.close(); policy.train(training)

def ms_pair(policy,task,episodes=100,det_seed=5000000,stoch_seed=5100000,should_stop=None):
    return dict(deterministic=evaluate(policy,'ms',episodes,det_seed,False,should_stop),stochastic=evaluate(policy,'ms',episodes,stoch_seed,True,should_stop))

def evaluate_pair(policy,task,obs_mode='dec',episodes=100,video=None,should_stop=None,det_seed=5000000,stoch_seed=5100000,**kwargs):
    return dict(deterministic=evaluate(policy,'rf',episodes,det_seed,False,should_stop),stochastic=evaluate(policy,'rf',episodes,stoch_seed,True,should_stop))

def main():
    p=argparse.ArgumentParser();p.add_argument('--domain',choices=['ms','rf'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--episodes',type=int,default=100);p.add_argument('--thresholds');p.add_argument('--seed',type=int,default=4000000)
    a=p.parse_args()
    sys.path.insert(0,str(CORE_CODE if a.domain=='ms' else RF_CODE))
    import torch
    torch.set_num_threads(2); torch.manual_seed(1)
    if a.thresholds: install(a.domain,a.thresholds)
    from eval_post import load
    policy,cp=load(a.checkpoint,'cuda:0')
    expected='TwoRobotStackCube-v1' if a.domain=='ms' else 'TwoRobotsStackCubePPO'
    if cp['contract']['task']!=expected: raise ValueError(f"Expected original task {expected}: {cp['contract']}")
    result=evaluate(policy,a.domain,a.episodes,a.seed,False)
    result.update(run_name=a.out.parent.name,checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),algo=cp['algo'],source_step=cp.get('step'),phase='baseline' if a.episodes==100 else 'smoke',task=a.domain)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.out.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(a.out)
    print(json.dumps(dict(out=str(a.out),rates=result['rates'],elapsed_seconds=result['elapsed_seconds'])),flush=True)
if __name__=='__main__': main()
