# Configurations

`benchmarks/<benchmark>__<task>__<dataset>.json` is the configuration of one
benchmark cell; `benchmarks/index.json` lists every cell with its benchmark,
task, dataset, code variant (`root` = `generative_mappo/`, otherwise a directory
in `variants/`), environment runtime (`envs/`) and online budget.

A configuration holds the policy settings (`policy`), the environment adapter
(`environment`), the online budget and evaluation interval (`online`), the MAPPO
settings (`ppo`), the offline source checkpoint (`source_checkpoint`) and the
output directory (`output`). Common defaults: actor / critic learning rates
`2e-5` / `3e-4`, discount `0.99`, GAE `0.95`, clipping `0.2`, trajectory-KL
threshold `0.02`, 4 epochs, minibatch 256, gradient-norm clip `0.5`, `K = 10`
denoising steps, learnable transition noise initialized at `0.1`, rollout length
128. Task-specific changes are stored in the corresponding file.

The `"seed"` field is overridden by the training seed: the trainer runs seeds
0, 1, 2 by default and writes seed `k` to `<output>/seed<k>/`. `${VAR}`
placeholders are expanded from the environment.
