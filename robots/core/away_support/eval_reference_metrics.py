"""Unmodified metric classes extracted from HANDS_AWAY evaluator sources."""
import torch

class AwayCounter:
    def __init__(self, n, device):
        self.count=torch.zeros(n,dtype=torch.long,device=device)
    def update(self, info):
        qualifying=info['success_original'].bool() & ~info['robot_contact'].bool() & info['retracted'].bool()
        self.count=torch.where(qualifying,self.count+1,0)
        return qualifying & (self.count>=10)

class Telemetry:
    def __init__(self, env):
        self.env=env; self.base=b=env.base
        pairs=[]
        for left in b.left_agent.robot.get_links():
            for right in b.right_agent.robot.get_links():
                pairs.extend(zip(left._bodies,right._bodies))
        self.query=b.scene.px.gpu_create_contact_pair_impulse_query(pairs)
        self.collision=torch.zeros(env.n,dtype=torch.bool,device=env.device)
        self.peak=torch.zeros(env.n,device=env.device)
        self.old_hook=b._after_simulation_step
        def hook():
            self.old_hook()
            b.scene.px.gpu_query_contact_pair_impulses(self.query)
            forces=self.query.cuda_impulses.torch()/b.scene.px.timestep
            peak=torch.linalg.vector_norm(forces.reshape(-1,env.n,3),dim=-1).amax(0)
            self.collision |= peak>1e-3
            self.peak=torch.maximum(self.peak,peak)
        b._after_simulation_step=hook
    def begin(self):
        self.collision.zero_(); self.peak.zero_()
    def read(self,info):
        b=self.base
        distances=torch.stack([torch.stack([torch.linalg.vector_norm(a.tcp.pose.p-c.pose.p,dim=-1) for c in (b.cubeA,b.cubeB)],-1).amin(-1) for a in (b.left_agent,b.right_agent)],-1)
        delta=b.cubeA.pose.p-b.cubeB.pose.p
        approach=(torch.linalg.vector_norm(delta[:,:2],dim=-1)<=b.goal_radius)&(delta[:,2]>.02)
        return dict(collision=self.collision.clone(),collision_peak_force=self.peak.clone(),right_away=distances[:,1]>=.12,first_placed=info['cubeB_placed'].bool(),second_approach=approach,second_stacked=info['is_cubeA_on_cubeB'].bool(),tcp_distances=distances)

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

SOURCE_PROVENANCE = {'AwayCounter': {'source': 'away_eval/ms_eval.py', 'source_sha256': 'f080c9d2fe564717cac4e43203b25f7350fa11eb6574b3b314808010a2c940af'}, 'Telemetry': {'source': 'away_eval/ms_metrics.py', 'source_sha256': 'b3c279e82e21690ebfcd3e24ab5cfcf5304219d27ca5979b5e46a64aee67db38'}, 'ArmContacts': {'source': 'away_eval/rf_eval.py', 'source_sha256': 'feb0587b059eda60bd145aa23f3eab0eec14727742cf89d14b8305e9688fcade'}}
