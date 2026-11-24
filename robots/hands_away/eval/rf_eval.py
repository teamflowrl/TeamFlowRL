"""Hands-away and inter-arm safety evaluation; unchanged release environment and policies."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from eval_post import load
from tasks import make_env

class ArmContacts:
    def __init__(self,base):
        self.base=base
        pairs=[]
        for l in base.left_agent.robot.get_links():
            for r in base.right_agent.robot.get_links(): pairs.extend(zip(l._bodies,r._bodies))
        self.query=base.scene.px.gpu_create_contact_pair_impulse_query(pairs)
        self.hit=torch.zeros(base.num_envs,dtype=torch.bool,device=base.device)
        self.original=base._after_simulation_step
        base._after_simulation_step=self.sample
    def sample(self):
        self.original()
        b=self.base
        b.scene.px.gpu_query_contact_pair_impulses(self.query)
        forces=self.query.cuda_impulses.torch().reshape(-1,b.num_envs,3)/b.scene.px.timestep
        self.hit|=(torch.linalg.norm(forces,dim=-1)>1e-3).any(0)

@torch.no_grad()
def evaluate(policy,cp,stochastic):
    seed=6100000 if stochastic else 6000000
    n=100
    env=make_env('TwoRobotsStackCubeReleasePPO',n,cp['obs_mode'],seed,contract=cp['contract'])
    try:
        obs,_=env.reset(list(range(seed,seed+n)));b=env.base
        collision=ArmContacts(b)
        count=torch.zeros(n,dtype=torch.long,device=b.device)
        ever={k:torch.zeros(n,dtype=torch.bool,device=b.device) for k in ['away','release','original']}
        gen=torch.Generator(device=b.device).manual_seed(seed);carry=None;rows=[]
        for step in range(1,151):
            collision.hit.zero_()
            features,carry=policy.encode_step(obs,carry)
            actions,_,_=policy.act(features,stochastic=stochastic,generator=gen)
            _,_,_,_,info=env.step(actions)
            qualifying=info['success_original'].bool() & ~info['robot_contact'].bool() & info['retracted'].bool()
            count=torch.where(qualifying,count+1,0)
            final={'away':qualifying & (count>=10),'release':info['success_release'].bool(),'original':info['success_original'].bool()}
            for k in ever:ever[k]|=final[k]
            a,c=b.cubeA.pose.p,b.cubeB.pose.p
            placed=(torch.linalg.norm(a[:,:2]-b.goal_region.pose.p[:,:2],dim=-1)<b.goal_radius)&((a[:,2]-b.cube_half_size[2]).abs()<0.01)
            d=torch.minimum(torch.linalg.norm(b.left_agent.tcp.pose.p-a,dim=-1),torch.linalg.norm(b.left_agent.tcp.pose.p-c,dim=-1))
            # Operational onset: upper cube enters the lower cube's vertical placement corridor.
            stacking=(torch.linalg.norm(c[:,:2]-a[:,:2],dim=-1)<2*b.cube_half_size[0]) & (c[:,2]>a[:,2]+b.cube_half_size[2])
            row={k:v.cpu().tolist() for k,v in final.items()}
            row.update(arm_collision=collision.hit.cpu().tolist(),first_placed=placed.cpu().tolist(),first_tcp_distance=d.cpu().tolist(),second_stacking=stacking.cpu().tolist())
            rows.append(row);obs,_=env.encode()
        episodes=[]
        for i in range(n):
            placement=next((s+1 for s,r in enumerate(rows) if r['first_placed'][i]),None)
            away=next((s+1 for s,r in enumerate(rows) if placement is not None and s+1>=placement and r['first_tcp_distance'][i]>=.15),None)
            second=next((s+1 for s,r in enumerate(rows) if r['second_stacking'][i]),None)
            hit=[s+1 for s,r in enumerate(rows) if r['arm_collision'][i]]
            episodes.append(dict(seed=seed+i,arm_collision=bool(hit),arm_collision_steps=hit,first_placement_step=placement,first_away_step=away,withdrawal_delay_steps=None if away is None else away-placement,second_stacking_start_step=second,away_before_second=None if second is None or placement is None else away is not None and away<second and second>1 and rows[second-2]['first_tcp_distance'][i]>=.15))
        delays=[e['withdrawal_delay_steps'] for e in episodes if e['withdrawal_delay_steps'] is not None]
        eligible=[e for e in episodes if e['away_before_second'] is not None]
        result=dict(episodes=n,seeds=list(range(seed,seed+n)),stochastic=stochastic,horizon=150,control_freq=float(b.control_freq),sim_freq=float(b.sim_freq),trajectory_success=rows,episode_metrics=episodes,arm_collision_rate=sum(e['arm_collision'] for e in episodes)/n,withdrawal_delay_median_steps=float(np.median(delays)) if delays else None,withdrawal_observed_count=len(delays),placement_count=sum(e['first_placement_step'] is not None for e in episodes),away_before_second_count=sum(e['away_before_second'] for e in eligible),away_before_second_eligible_count=len(eligible))
        for k in ever:
            result[k+'_final_success']=final[k].cpu().tolist();result[k+'_final_successes']=int(final[k].sum());result[k+'_success_once']=ever[k].cpu().tolist()
        return result
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    torch.set_num_threads(2);torch.cuda.set_device(0)
    started=time.monotonic();policy,cp=load(a.checkpoint,'cuda:0')
    result=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),algo=cp['algo'],steps=cp.get('step',cp.get('progress',{}).get('steps')),metric_definition={'away':'original success, no robot-object contact >0.001 N, both TCP >=0.15 m from both cubes, 10 consecutive control steps; no opening or velocity criterion','collision':'all left/right robot link pairs >0.001 N at any physics substep, aggregated into control step','first_placement':'first cube A within original goal radius and tabletop z tolerance 0.01 m; may occur while held','first_withdrawal':'first subsequent left TCP distance >=0.15 m from BOTH cubes','second_stacking':'first upper cube B xy distance to A <2*cube half size and B.z>A.z+cube half size','delay_censoring':'median only of observed withdrawals; placement and observed counts supplied; horizon 150'})
    for mode in ['deterministic','stochastic']:
        result[mode]=evaluate(policy,cp,mode=='stochastic');print(mode,{k:result[mode][k] for k in ['away_final_successes','release_final_successes','original_final_successes','arm_collision_rate','withdrawal_delay_median_steps']},flush=True)
    result['elapsed_seconds']=time.monotonic()-started
    a.out.write_text(json.dumps(result,indent=2)+'\n')
