"""Hands-away TCP-to-nearest-cube reward, force telemetry and calibrated distance thresholds.

No environment registration or policy assumptions: attach to a ManiSkill base env.
Episode collectors MUST disable autoreset and run the complete 150-step horizon.
"""
import json
import math
from pathlib import Path
import torch


class PhysicsTelemetry:
    """All-arm/all-cube forces and inter-arm collision at every physics substep."""
    def __init__(self, base):
        self.base = base
        self.scene = base.scene
        self.n = base.num_envs
        self.agents = (base.left_agent, base.right_agent)
        self.cubes = (base.cubeA, base.cubeB)
        self.groups = [[(link, cube) for link in agent.robot.get_links() for cube in self.cubes]
                       for agent in self.agents]
        self.groups.append([(left, right) for left in self.agents[0].robot.get_links()
                            for right in self.agents[1].robot.get_links()])
        self.query = None
        if base.gpu_sim_enabled:
            pairs, self.sizes = [], []
            for group in self.groups:
                start = len(pairs)
                for left, right in group:
                    pairs.extend(zip(left._bodies, right._bodies))
                self.sizes.append(len(pairs) - start)
            self.query = base.scene.px.gpu_create_contact_pair_impulse_query(pairs)
        self.peaks = torch.zeros((self.n, 3), device=base.device)
        self.placed = torch.zeros(self.n, dtype=torch.bool, device=base.device)
        self.origin = torch.zeros((self.n, 2, 3), device=base.device)
        self.active = True
        self.previous_hook = base._after_simulation_step
        def hook():
            self.previous_hook()
            # A scene can be replaced before episode initialization runs.
            if self.active and self.base.scene is self.scene:
                self.peaks = torch.maximum(self.peaks, self.forces())
        self.hook = hook
        base._after_simulation_step = hook

    def close(self):
        self.active = False
        if self.base._after_simulation_step is self.hook:
            self.base._after_simulation_step = self.previous_hook

    def reset(self, env_idx=None):
        idx = slice(None) if env_idx is None else env_idx
        self.peaks[idx] = 0
        self.placed[idx] = False
        self.origin[idx] = 0

    def begin_step(self):
        self.peaks.zero_()

    def forces(self):
        b = self.base
        if b.scene is not self.scene:
            raise RuntimeError('Telemetry belongs to a replaced scene; recreate after reset')
        if self.query is not None:
            b.scene.px.gpu_query_contact_pair_impulses(self.query)
            values = self.query.cuda_impulses.torch() / b.scene.px.timestep
            return torch.stack([torch.linalg.vector_norm(v.reshape(-1, self.n, 3), dim=-1).amax(0)
                                for v in values.split(self.sizes, dim=0)], -1)
        return torch.stack([torch.stack([torch.linalg.vector_norm(
            b.scene.get_pairwise_contact_forces(left, right), dim=-1)
            for left, right in group], -1).amax(-1) for group in self.groups], -1)

    def read(self, info):
        b = self.base
        tcp = torch.stack([a.tcp.pose.p for a in self.agents], 1)
        original = info.get('success_original', info['success']).bool()
        # Placement is geometric onset, not release-conditioned original success.
        if 'stacked' in info:
            stacked = info['stacked'].bool()
        elif 'is_cubeA_on_cubeB' in info and 'cubeB_placed' in info:
            stacked = info['is_cubeA_on_cubeB'].bool() & info['cubeB_placed'].bool()
        else:
            raise KeyError('Supply geometric stacked in info for this task')
        fresh = stacked & ~self.placed
        self.origin[fresh] = tcp[fresh]
        self.placed |= stacked
        displacement = torch.linalg.vector_norm(tcp - self.origin, dim=-1)
        displacement = torch.where(self.placed[:, None], displacement, torch.zeros_like(displacement))
        distance = torch.stack([torch.stack([torch.linalg.vector_norm(a.tcp.pose.p-c.pose.p, dim=-1)
                                            for c in self.cubes], -1).amin(-1) for a in self.agents], -1)
        self.peaks = torch.maximum(self.peaks, self.forces())
        return dict(original=original, stacked=stacked, placed=self.placed.clone(),
                    tcp_position_m=tcp.clone(), placement_tcp_position_m=self.origin.clone(),
                    tcp_displacement_m=displacement, tcp_cube_distance_m=distance,
                    robot_cube_peak_force_n=self.peaks[:, :2].clone(),
                    arm_collision_peak_force_n=self.peaks[:, 2].clone(),
                    opening_m=torch.stack([a.robot.get_qpos()[:, -2:].sum(-1) for a in self.agents], -1),
                    cube_linear_speed_m_s=torch.stack([torch.linalg.vector_norm(c.linear_velocity, dim=-1) for c in self.cubes], -1),
                    cube_angular_speed_rad_s=torch.stack([torch.linalg.vector_norm(c.angular_velocity, dim=-1) for c in self.cubes], -1))


def qualify(metrics, thresholds):
    """Both arms must clear the calibrated nearest-cube distance; release adds opening."""
    m, t = metrics, thresholds
    no_contact = (m['robot_cube_peak_force_n'] <= t['contact_force_n']).all(-1)
    away = (m['original'] & m['placed'] & no_contact
            & (m['tcp_cube_distance_m'] >= t['tcp_cube_distance_m']).all(-1))
    released = away & (m['opening_m'] >= t['opening_m']).all(-1)
    if t.get('linear_speed_m_s') is not None:
        released &= (m['cube_linear_speed_m_s'] < t['linear_speed_m_s']).all(-1)
    if t.get('angular_speed_rad_s') is not None:
        released &= (m['cube_angular_speed_rad_s'] < t['angular_speed_rad_s']).all(-1)
    return away, released


class EpisodeRecorder:
    def __init__(self, num_envs, seeds, thresholds=None):
        assert num_envs == len(seeds)
        self.n, self.seeds, self.thresholds = num_envs, list(seeds), thresholds
        self.rows = []
        self.counts = None

    def append(self, step, metrics):
        row = dict(step=int(step), **{k: v.detach().cpu().tolist() if torch.is_tensor(v) else v
                                    for k, v in metrics.items()})
        if self.thresholds:
            predicates = torch.stack(qualify(metrics, self.thresholds), -1)
            if self.counts is None:
                self.counts = torch.zeros_like(predicates, dtype=torch.long)
            self.counts = torch.where(predicates, self.counts+1, 0)
            row['away'] = (self.counts[:, 0] >= self.thresholds['hold_steps']).cpu().tolist()
            row['release'] = (self.counts[:, 1] >= self.thresholds['hold_steps']).cpu().tolist()
        self.rows.append(row)

    def finish(self):
        if not self.rows:
            raise ValueError('No trajectory collected')
        episodes = []
        for i, seed in enumerate(self.seeds):
            post = [r for r in self.rows if r['placed'][i]]
            peak = lambda key: max((max(r[key][i]) for r in post), default=None)
            episode = dict(seed=seed, original=any(r['original'][i] for r in self.rows),
                original_final=self.rows[-1]['original'][i],
                placement_step=next((r['step'] for r in self.rows if r['stacked'][i]), None),
                tcp_cube_distance_m=max((min(r['tcp_cube_distance_m'][i]) for r in post), default=None),
                tcp_displacement_m=max((min(r['tcp_displacement_m'][i]) for r in post), default=None),
                robot_cube_peak_force_n=peak('robot_cube_peak_force_n'),
                opening_m=max((min(r['opening_m'][i]) for r in post), default=None),
                cube_linear_speed_m_s=peak('cube_linear_speed_m_s'),
                cube_angular_speed_rad_s=peak('cube_angular_speed_rad_s'),
                collision=any(r['arm_collision_peak_force_n'][i] > 1e-3 for r in self.rows),
                collision_peak_force_n=max(r['arm_collision_peak_force_n'][i] for r in self.rows))
            for key in ('away', 'release'):
                episode[key] = any(r[key][i] for r in self.rows) if self.thresholds else None
                episode[key+'_final'] = self.rows[-1][key][i] if self.thresholds else None
            episodes.append(episode)
        rates = {k: sum(e[k] for e in episodes)/self.n for k in ('original', 'collision')}
        if self.thresholds:
            rates.update({k: sum(e[k] for e in episodes)/self.n for k in ('away', 'release', 'original_final', 'away_final', 'release_final')})
        return dict(episodes=episodes, trajectory=self.rows, rates=rates, thresholds=self.thresholds,
                    episode_count=self.n, horizon=len(self.rows))


def percentiles(values):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(v))
    if not values:
        return dict(n=0, p10=None, p50=None, p90=None, maximum=None)
    def at(p):
        x = (len(values)-1)*p
        low, high = math.floor(x), math.ceil(x)
        return values[low] + (values[high]-values[low])*(x-low)
    return dict(n=len(values), p10=at(.1), p50=at(.5), p90=at(.9), maximum=values[-1])


def calibrate(baselines, task):
    """Legacy displacement calibration is forbidden for v3."""
    raise RuntimeError('v3 requires independent TCP-distance calibration; use calibrate_v3.py')


def write_calibration(result, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/'baseline_distribution.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Calibrated release thresholds', '', result['derivation'], '',
             '| criterion | value |', '|---|---|']
    lines += [f'| {k} | {v} |' for k, v in result['thresholds'].items()]
    lines += ['', 'Successful baseline episodes and p10/p50/p90:', '', '| run | quantity | n | p10 | p50 | p90 |', '|---|---|---:|---:|---:|---:|']
    for name, dist in result['distributions'].items():
        for key, value in dist.items():
            if isinstance(value, dict):
                lines.append(f"| {name} | {key} | {value['n']} | {value['p10']} | {value['p50']} | {value['p90']} |")
    (out/'THRESHOLDS.md').write_text('\n'.join(lines)+'\n')


def install(domain, thresholds):
    """Patch imported release_env class BEFORE make_env; preserve reward formulas.

    domain: ms/rf. threshold dict from calibrate()['thresholds'].
    This is process-local and never modifies source files or original checkpoints.
    """
    if domain not in ('ms', 'rf'):
        raise ValueError('domain must be ms or rf')
    if not math.isfinite(thresholds['tcp_cube_distance_m']) or thresholds['tcp_cube_distance_m'] <= 0:
        raise ValueError('A positive calibrated TCP distance threshold is required')
    import importlib
    release_env = importlib.import_module('ms_release.release_env' if domain == 'ms' else 'release_env')
    cls = (release_env.TwoRobotStackCubeRelease if domain == 'ms'
           else release_env.TwoRobotsStackCubeRelease)
    if getattr(cls, '_away_installed', False):
        raise RuntimeError('Hands-away patch already installed in this process')
    original_evaluate = cls.evaluate
    original_initialize = cls._initialize_episode
    original_reconfigure = getattr(cls, '_reconfigure', None)
    t = dict(thresholds)

    def detach(self):
        if hasattr(self, '_away_telemetry'):
            self._away_telemetry.close()
            del self._away_telemetry
        for key in ('_away_cached_metrics', '_away_count', '_away_last_step'):
            if hasattr(self, key):
                delattr(self, key)

    def reconfigure(self, *args, **kwargs):
        # Detach before scene teardown or reset-settling simulation can run.
        detach(self)
        return original_reconfigure(self, *args, **kwargs)

    def initialize(self, env_idx, options):
        if hasattr(self, '_away_telemetry') and self._away_telemetry.scene is not self.scene:
            detach(self)
        original_initialize(self, env_idx, options)
        if hasattr(self, '_away_telemetry'):
            self._away_telemetry.reset(env_idx)
            self._away_count[env_idx] = 0
            self._away_last_step[env_idx] = -1
            # Explicitly invalidate reset rows; untouched environments retain state.
            if hasattr(self, '_away_cached_metrics'):
                for value in self._away_cached_metrics.values():
                    value[env_idx] = 0

    def evaluate(self):
        if hasattr(self, '_away_telemetry') and self._away_telemetry.scene is not self.scene:
            detach(self)
        info = original_evaluate(self)
        if not hasattr(self, '_away_telemetry'):
            self._away_telemetry = PhysicsTelemetry(self)
            self._away_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self._away_last_step = torch.full_like(self._away_count, -1)
        tele = self._away_telemetry
        metrics = tele.read(info)
        changed = self.elapsed_steps != self._away_last_step
        if hasattr(self, '_away_cached_metrics'):
            metrics = {k: torch.where(changed.reshape((-1,)+(1,)*(v.ndim-1)), v, self._away_cached_metrics[k])
                       for k, v in metrics.items()}
        self._away_cached_metrics = {k: v.clone() for k, v in metrics.items()}
        away, released = qualify(metrics, t)
        fresh = (self.elapsed_steps != self._away_last_step) & (self.elapsed_steps > 0)
        self._away_count = torch.where(fresh, torch.where(released, self._away_count+1, 0), self._away_count)
        self._away_last_step = self.elapsed_steps.clone()
        # The original distance shaping scale is separate from the calibrated success threshold.
        self._stable_count = self._away_count.clone()
        retreat_progress = (metrics['tcp_cube_distance_m']/(.12 if domain == 'ms' else .15)).clamp(0, 1)
        retreat_progress = retreat_progress.mean(-1) if domain == 'ms' else retreat_progress.sum(-1)
        retracted = (metrics['tcp_cube_distance_m'] >= t['tcp_cube_distance_m']).all(-1)
        success = released & (self._away_count >= t['hold_steps'])
        info.update(success=success, success_release=success, stable_count=self._away_count.clone(),
                    retracted=retracted, retreat_progress=retreat_progress,
                    away_tcp_displacement=metrics['tcp_displacement_m'],
                    away_tcp_cube_distance=metrics['tcp_cube_distance_m'],
                    away_robot_cube_peak_force=metrics['robot_cube_peak_force_n'])
        # retain old opened/released reward stages, but enforce substep contact signal
        info['released'] = ((metrics['opening_m'] >= t['opening_m']).all(-1)
                            & (metrics['robot_cube_peak_force_n'] <= t['contact_force_n']).all(-1))
        info['robot_contact'] = (metrics['robot_cube_peak_force_n'] > t['contact_force_n']).any(-1)
        if changed.any():
            tele.begin_step()
        return info

    cls.evaluate = evaluate
    cls._initialize_episode = initialize
    if original_reconfigure is not None:
        cls._reconfigure = reconfigure
    cls._away_installed = True
    return cls
