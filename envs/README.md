# Environment adapters

`generative_mappo.environments` imports one benchmark-specific adapter module
by path and calls it through a small common interface. The adapters are kept
per runtime version, so a configuration
must reference the runtime directory listed in
`configs/benchmarks/index.json`.

## Layout

```
envs/mpe/runtimes/<runtime>/          mpe_env.py, data.py, vendor/
envs/mamujoco/runtimes/<runtime>/     mamujoco_env.py, data.py
envs/omiga/runtimes/<runtime>/        mamujoco_env.py, data.py
envs/smac/runtimes/<runtime>/         vector_env.py, common.py, sc2_port_lease.py,
                                      project/mac_flow_pytorch_smacv2/macflow_torch/smacv1_env.py
envs/smacv2/runtimes/<runtime>/       vector_env.py, common.py, sc2_port_lease.py,
                                      project/mac_flow_pytorch_smacv2/macflow_torch/smacv2_env.py
```

`generative_mappo.environments.RuntimeEnv` imports:

| family | module | expected directory |
| --- | --- | --- |
| `mpe` | `mpe_env` | `envs/mpe/runtimes/<runtime>` |
| `mamujoco` (OG-MARL) | `mamujoco_env` | `envs/mamujoco/runtimes/<runtime>` |
| `mamujoco` with OMIGA tasks | `mamujoco_env` | `envs/omiga/runtimes/<runtime>` |
| `smac` | `vector_env` | `envs/smac/runtimes/<runtime>` |
| `smacv2` | `vector_env` | `envs/smacv2/runtimes/<runtime>` |

## Runtime versions

Several runtime directories exist per benchmark (`<family>_rt1`, `<family>_rt2`,
...). They are the adapter versions used by different cells and differ in
scenario support (for example `simple_world` and the prey policy in MPE, or the
task a MA-MuJoCo adapter accepts). Use the `runtime_dir` given in the cell's
configuration (also listed in `configs/benchmarks/index.json`).

## Runtime-specific notes

### MPE

`mpe_env.py` inserts `<runtime>/vendor` on `sys.path` and imports the vendored
`multiagent` and `ddpg_agent` packages, so each MPE runtime directory contains
its own `vendor/` copy. `simple_world` support and the prey policy differ
between runtime versions; use the version listed for the cell.

Offline MPE arrays, when used, follow:

```
${DATA_ROOT}/mpe/<task>/<split>/seed_<i>_data/{obs,next_obs,acs,rews,dones}_<j>.npy
```

with five data seeds and three controlled agents, as expected by
`mpe_env`'s companion `data.py`.

### MA-MuJoCo (OG-MARL)

Requires `multiagent-mujoco` (`multiagent_mujoco.mujoco_multi.MujocoMulti`)
and MuJoCo 2.1.0. Each runtime is specialised for a scenario group; the
`4ant` adapter, for example, asserts its task. Offline arrays follow:

```
${DATA_ROOT}/mamujoco/<task>/<split>/{obs,actions,rewards,discounts,path_lengths}.npy
```

### OMIGA

Requires `gym` and MuJoCo 2.1.0. The adapter uses the published fully
observed MuJoCo tasks (`Hopper-v2`, `HalfCheetah-v2`, `Ant-v2`) and a flat
action concatenation, not OG-MARL joint control. Offline arrays follow:

```
${DATA_ROOT}/omiga/<task>/<split>/{obs,actions,rewards,terminals,truncations,path_lengths}.npy
```

and the split directory contains `audit.json`.

### SMAC and SMACv2

Require the corresponding Python packages and StarCraft II installations.
`environment.sc2path` (default `${SC2PATH}`) is exported as `SC2PATH` before training. SMAC v1
uses the `3m`, `2s3z`, `5m_vs_6m`, and `8m` maps; SMACv2 uses the
`terran_5_vs_5`, `zerg_5_vs_5`, and `terran_10_vs_10` distributions.
`vector_env.py` spawns one process per environment and isolates each worker's
process group so that StarCraft II child processes are cleaned up together.

`common.py` loads the environment factory from
`<project_root>/mac_flow_pytorch_smacv2/macflow_torch/smacv1_env.py`
(SMAC) or `smacv2_env.py` (SMACv2). The configuration template points
`project_root` at the matching `runtimes/<runtime>/project` directory.
