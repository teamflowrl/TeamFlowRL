"""One observation contract for restored demonstrations and online environments."""
import torch

TASKS = ('TwoRobotPickCube-v1', 'TwoRobotStackCube-v1')
HORIZON = 100
AGENT_IDS = ('panda_wristcam-0', 'panda_wristcam-1')


def leaves(tree, prefix=''):
    if isinstance(tree, dict):
        for key in sorted(tree):
            yield from leaves(tree[key], f'{prefix}.{key}' if prefix else key)
    else:
        yield prefix, tree


def build_observations(raw, task, elapsed, agent_ids=None, return_schema=False):
    if task not in TASKS:
        from tasks import get_task
        cfg = get_task(task)
        if cfg.env_type == 'aloha':
            from env_aloha import build_aloha_observations
            return build_aloha_observations(raw, task, elapsed, agent_ids, return_schema)
        from env_robofactory import build_observations as build_rf_observations
        return build_rf_observations(raw, task, elapsed, return_schema)
    ids = tuple(sorted(raw['agent'])) if agent_ids is None else tuple(agent_ids)
    if len(ids) != 2:
        raise ValueError('Expected two agents in left/right order')
    ref = raw['agent'][ids[0]]['qpos']
    ref = torch.as_tensor(ref)
    device, n = ref.device, ref.shape[0]
    def tensor(x):
        return torch.as_tensor(x, device=device, dtype=torch.float32).reshape(n, -1)
    extra = raw['extra']
    clock = 1 - torch.as_tensor(elapsed, device=device, dtype=torch.float32).expand(n).reshape(n, 1) / HORIZON
    eye = torch.eye(2, device=device).expand(n, -1, -1)
    dec, fields = [], None
    for i, arm in enumerate(('left', 'right')):
        tcp = tensor(extra[f'{arm}_arm_tcp'])
        qpos, qvel = (tensor(raw['agent'][ids[i]][k]) for k in ('qpos', 'qvel'))
        if qpos.shape[-1] != 9 or qvel.shape[-1] != 9 or tcp.shape[-1] != 7:
            raise ValueError('Expected Panda qpos=9, qvel=9, TCP pose=7')
        parts = [('own_qpos', qpos), ('own_qvel', qvel), ('own_tcp_pose', tcp)]
        if task == TASKS[0]:
            cube = tensor(extra['cube_pose'])
            delta = tensor(extra['cube_to_goal_pos'])
            parts += [('cube_pose', cube), ('goal_pos', cube[:, :3] + delta),
                      ('own_tcp_to_cube', cube[:, :3] - tcp[:, :3]), ('cube_to_goal', delta)]
        else:
            ca, cb = tensor(extra['cubeA_pose']), tensor(extra['cubeB_pose'])
            parts += [('cubeA_pose', ca), ('cubeB_pose', cb),
                      ('goal_region_pos', tensor(extra['goal_region_pos'])),
                      ('own_tcp_to_cubeA', ca[:, :3] - tcp[:, :3]),
                      ('own_tcp_to_cubeB', cb[:, :3] - tcp[:, :3]),
                      ('cubeA_to_cubeB', cb[:, :3] - ca[:, :3])]
        parts += [('remaining_time', clock), ('agent_one_hot', eye[:, i])]
        dec.append(torch.cat([v for _, v in parts], -1))
        fields = [{'name': k, 'size': v.shape[-1]} for k, v in parts]
    central_parts = [(k, tensor(v)) for k, v in leaves(raw)]
    joint = torch.cat([v for _, v in central_parts] + [clock], -1)
    central = torch.cat([joint[:, None].expand(-1, 2, -1), eye], -1)
    result = dict(obs_dec=torch.stack(dec, 1), obs_central=central, state=central)
    if any(not torch.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite observation')
    if return_schema:
        schema = dict(agent_ids=list(ids), obs_dec=fields,
                      obs_central=[{'name': k, 'size': v.shape[-1]} for k, v in central_parts]
                      + [{'name': 'remaining_time', 'size': 1}, {'name': 'agent_one_hot', 'size': 2}])
        schema['state'] = schema['obs_central']
        return result, schema
    return result


@torch.no_grad()
def build_rgb_observations(raw, base, task, elapsed, vision):
    """RGB actor input only; privileged state is retained separately in the H5."""
    from vision import actor_observations
    if task not in TASKS:
        raise ValueError(task)
    ids = tuple(base.agent.agents_dict)
    if len(ids) != 2:
        raise ValueError('Expected two agents in action order')
    rgb = raw['sensor_data']['base_camera']['rgb']
    if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (128, 128, 3) or rgb.dtype != torch.uint8:
        raise ValueError(f'Expected default base_camera uint8 RGB (N,128,128,3), got {rgb.shape}/{rgb.dtype}')
    proprio = torch.stack([torch.cat([raw['agent'][k]['qpos'], raw['agent'][k]['qvel']], -1)
                           for k in ids], 1).float()
    if tuple(proprio.shape) != (len(rgb), 2, 18):
        raise ValueError('Expected Panda qpos=9 and qvel=9')
    target = base.goal_site if task == TASKS[0] else base.goal_region
    goal = target.pose.p.to(proprio.device).float()
    clock = 1 - torch.as_tensor(elapsed, device=proprio.device).expand(len(rgb)).float() / HORIZON
    visual = vision(rgb)
    if tuple(visual.shape) != (len(rgb), 1920):
        raise ValueError('Expected 1920 DINO features')
    result = actor_observations(visual, proprio, goal, clock)
    if result.shape[-1] != 1944 or not torch.isfinite(result).all():
        raise ValueError('Invalid RGB observation')
    return result


def rgb2_cameras(raw, agent_ids, camera_res=224):
    """Resolve wrist sensors by full agent identity, never by dict order/index guess."""
    import re
    sensors = raw['sensor_data']
    ids = tuple(agent_ids)
    if len(ids) != 2 or len(set(ids)) != 2 or 'base_camera' not in sensors:
        raise ValueError(f'Expected base camera and two distinct agents: {list(sensors)}')
    keys = ['base_camera']
    for agent in ids:
        matches = [key for key in sensors if 'hand_camera' in key and
                   re.search(r'(?<![A-Za-z0-9])' + re.escape(agent) + r'(?![A-Za-z0-9])', key)]
        if len(matches) != 1:
            raise ValueError(f'Cannot uniquely match hand camera for {agent}: {list(sensors)}')
        keys.append(matches[0])
    if len(set(keys)) != 3:
        raise ValueError('Wrist camera ownership is ambiguous')
    images = [sensors[key]['rgb'] for key in keys]
    for key, rgb in zip(keys, images):
        if (rgb.dtype != torch.uint8 or rgb.ndim != 4 or
                tuple(rgb.shape[1:]) != (camera_res, camera_res, 3) or len(rgb) != len(images[0])):
            raise ValueError(f'{key}: expected uint8 (N,{camera_res},{camera_res},3), got {rgb.shape}/{rgb.dtype}; check sensor_configs overrides')
    return keys, images


@torch.no_grad()
def build_rgb2_observations(raw, base, task, elapsed, vision, camera_res=224):
    if task not in TASKS:
        raise ValueError(task)
    ids = tuple(base.agent.agents_dict)
    _, images = rgb2_cameras(raw, ids, camera_res)
    n = len(images[0])
    # Exactly three encoded images per environment frame, shared base encoded once.
    features = vision(torch.cat(images, 0))
    if tuple(features.shape) != (3*n, 6528):
        raise ValueError('Expected grid=4 DINO features (6528 per camera)')
    base_feature, hand0, hand1 = features.split(n)
    proprio = torch.stack([torch.cat([raw['agent'][key]['qpos'], raw['agent'][key]['qvel']], -1)
                           for key in ids], 1).float()
    if tuple(proprio.shape) != (n, 2, 18):
        raise ValueError('Expected Panda qpos=9 and qvel=9')
    target = base.goal_site if task == TASKS[0] else base.goal_region
    goal = target.pose.p.to(proprio.device).float()
    if tuple(goal.shape) != (n, 3):
        raise ValueError('Expected public goal position (N,3)')
    clock = 1 - torch.as_tensor(elapsed, device=proprio.device).expand(n).float()/HORIZON
    result = torch.cat([base_feature[:, None].expand(-1, 2, -1), torch.stack([hand0, hand1], 1),
                        proprio, goal[:, None].expand(-1, 2, -1),
                        clock[:, None, None].expand(-1, 2, -1),
                        torch.eye(2, device=proprio.device).expand(n, -1, -1)], -1)
    if result.shape != (n, 2, 13080) or not torch.isfinite(result).all():
        raise ValueError('Invalid RGB2 actor observation')
    return result
