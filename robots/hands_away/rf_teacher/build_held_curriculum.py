"""Collect early teacher stack states while preserving closed gripper commands (curriculum of
start states for the RoboFactory release teacher). Set RF_STACK_TEACHER and PPO_SCRIPT."""
import argparse,json
from pathlib import Path
import torch,numpy as np,gymnasium as gym
import os
import release_env
# Definitions previously imported from a check script: ManiSkill's PPO baseline script, the
# RoboFactory stacking PPO teacher, and the state-dict flattening helper.
PPO=os.path.expandvars(os.environ.get('PPO_SCRIPT','${CODE_ROOT}/vendor/maniskill/examples/baselines/ppo/ppo.py'))
TEACHER=Path(os.path.expandvars(os.environ.get('RF_STACK_TEACHER','${ROBOT_ROOT}/runs/ppo_expert/TwoRobotsStackCube/final_ckpt.pt')))
def flat(tree, prefix=''):
    out={}
    for k,v in tree.items():
        if isinstance(v,dict):out.update(flat(v,prefix+k+'/'))
        else:out[prefix+k]=v
    return out
from record_ppo import load_ppo_module
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(21);np.random.seed(21)
env=FlattenActionSpaceWrapper(gym.make('TwoRobotsStackCubeRelease-dense',num_envs=64,obs_mode='state',control_mode='pd_joint_delta_pos',sim_backend='physx_cuda'))
base=env.unwrapped
class Spaces:
 single_observation_space=base.single_observation_space
 single_action_space=env.single_action_space
agent=load_ppo_module(PPO).Agent(Spaces()).to('cuda');agent.load_state_dict(torch.load(TEACHER,map_location='cuda',weights_only=False));agent.eval()
states=[];metas=[];candidates=[];candidate_meta=[];summary=[]
with torch.no_grad():
 for seed,mode in [(21,'unmodified'),(22,'closed'),(23,'close_after_stack')]:
  obs,info=env.reset(seed=seed);hit=torch.zeros(64,device='cuda',dtype=torch.bool)
  counts={'both_grasp':0,'stacked':0,'closed_stacked':0}
  for t in range(150):
   action=agent.get_action(obs,deterministic=True).clamp(-1,1)
   if mode=='closed':action[:,[7,15]]=-1
   if mode=='close_after_stack':
    hit|=info['stacked']
    action[hit]=0;action[hit,7]=-1;action[hit,15]=-1
   obs,_,_,_,info=env.step(action)
   both=info['is_cubeA_grasped'] & info['is_cubeB_grasped']
   closed=torch.stack([x.robot.get_qpos()[:,-2:].sum(-1)<0.055 for x in base.agent.agents],-1).all(-1)
   mask=info['stacked'] & both
   counts['both_grasp']+=int(mask.sum());counts['stacked']+=int(info['stacked'].sum());counts['closed_stacked']+=int((info['stacked']&closed).sum())
   if t%3==0 and (mask.any() or (info['stacked']&closed).any()):
    snap=flat(base.get_state_dict())
    for idx in (info['stacked']&(both|closed)).nonzero().flatten().tolist():
     row={k:v[idx].detach().cpu().clone() for k,v in snap.items()}
     meta=dict(seed=seed,mode=mode,step=t+1,env=idx,both_grasp=bool(both[idx]),both_closed=bool(closed[idx]))
     candidates.append(row);candidate_meta.append(meta)
     if both[idx]:states.append(row);metas.append(meta)
   if t%50==49:print(mode,t+1,counts,flush=True)
  summary.append(dict(mode=mode,seed=seed,**counts))
env.close()
if not states:states,metas=candidates,candidate_meta
if states:torch.save(dict(states={k:torch.stack([s[k] for s in states]) for k in states[0]},meta=metas),a.out/'curriculum_held.pt')
result=dict(runs=summary,saved=len(states),saved_both_grasp=sum(x['both_grasp'] for x in metas),saved_both_closed=sum(x['both_closed'] for x in metas))
(a.out/'curriculum_held.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
