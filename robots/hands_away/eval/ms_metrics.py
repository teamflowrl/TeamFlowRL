"""Supplemental telemetry; observes physics without changing actions/rewards."""
import torch

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

def summarize(rows,frequency):
    import numpy as np
    n=len(rows[0]['collision']); episodes=[]
    for i in range(n):
        first=lambda key,start=0:next((t+1 for t,r in enumerate(rows) if t>=start and r[key][i]),None)
        placed=first('first_placed'); away=first('right_away',placed-1) if placed else None
        approach=first('second_approach'); stacked=first('second_stacked')
        collisions=[t+1 for t,r in enumerate(rows) if r['collision'][i]]
        before=None if approach is None or placed is None else (away is not None and away<approach and rows[approach-2]['right_away'][i] if approach>1 else False)
        episodes.append(dict(collision_steps=collisions,collision_count=len(collisions),first_placement_step=placed,first_away_step=away,retreat_delay_steps=away-placed if away else None,second_approach_step=approach,second_stack_step=stacked,away_before_second=before))
    delays=[e['retreat_delay_steps'] for e in episodes if e['retreat_delay_steps'] is not None]; eligible=[e for e in episodes if e['away_before_second'] is not None]
    return dict(control_frequency=frequency,episodes=episodes,collision_rate=100*sum(bool(e['collision_steps']) for e in episodes)/n,median_collision_steps=float(np.median([e['collision_count'] for e in episodes])),placement_count=sum(e['first_placement_step'] is not None for e in episodes),retreated_count=len(delays),median_retreat_delay_steps=float(np.median(delays)) if delays else None,away_before_second_count=sum(e['away_before_second'] for e in eligible),second_approach_eligible_count=len(eligible))
