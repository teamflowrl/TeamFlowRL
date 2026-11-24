"""Read-only checkpoint evaluation in the unchanged 150-step release environment."""
import argparse, hashlib, json, time
from pathlib import Path
import torch
from eval_post import load
from ms_release.env_adapter import BimanualEnv
from ms_metrics import Telemetry, summarize

class AwayCounter:
    def __init__(self, n, device):
        self.count=torch.zeros(n,dtype=torch.long,device=device)
    def update(self, info):
        qualifying=info['success_original'].bool() & ~info['robot_contact'].bool() & info['retracted'].bool()
        self.count=torch.where(qualifying,self.count+1,0)
        return qualifying & (self.count>=10)

@torch.no_grad()
def evaluate(policy, stochastic):
    seed=6100000 if stochastic else 6000000
    env=BimanualEnv('TwoRobotStackCube-v1',100)
    try:
        obs,_=env.reset(list(range(seed,seed+100)))
        counter=AwayCounter(100,env.device)
        telemetry=Telemetry(env); metrics=[]
        gen=torch.Generator(device=env.device).manual_seed(seed)
        carry=None
        ever={k:torch.zeros(100,dtype=torch.bool,device=env.device) for k in ['away','release','original']}
        rows=[]
        for step in range(1,151):
            features,carry=policy.encode_step(obs,carry)
            actions,_,_=policy.act(features,stochastic=stochastic,generator=gen)
            telemetry.begin()
            _,_,_,_,info=env.step(actions.clamp(-1,1))
            metrics.append({k:v.cpu().tolist() for k,v in telemetry.read(info).items()})
            final={'away':counter.update(info),'release':info['success_release'].bool(),'original':info['success_original'].bool()}
            assert not (final['release'] & ~final['away']).any()
            for k in ever: ever[k] |= final[k]
            rows.append({k:v.cpu().tolist() for k,v in final.items()})
            obs,_=env.encode()
        result={'episodes':100,'seeds':list(range(seed,seed+100)),'stochastic':stochastic,'horizon':150,'trajectory_success':rows}
        for k in ever:
            result[k+'_final_success']=final[k].cpu().tolist()
            result[k+'_final_successes']=int(final[k].sum())
            result[k+'_success_once']=ever[k].cpu().tolist()
        result['supplement']=summarize(metrics,env.base.control_freq)
        result['telemetry']=metrics
        result['final_diagnostics']={k:info[k].cpu().tolist() for k in ['robot_contact','retracted','opened','min_tcp_distance','success_original']}
        return result
    finally: env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_device(0)
    started=time.monotonic();policy,cp=load(a.checkpoint,'cuda:0')
    result={'checkpoint':str(a.checkpoint),'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'algo':cp['algo'],'steps':cp.get('step',cp.get('progress',{}).get('steps'))}
    for mode in ['deterministic','stochastic']:
        result[mode]=evaluate(policy,mode=='stochastic')
        print(mode,{k:result[mode][k+'_final_successes'] for k in ['away','release','original']},flush=True)
    result['elapsed_seconds']=time.monotonic()-started
    a.out.write_text(json.dumps(result,indent=2)+'\n')
