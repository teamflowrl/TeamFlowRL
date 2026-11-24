# Trajectory-ratio ablation

Both ablations change only how the probability ratio of an agent's denoising
trajectory is formed and clipped; every other setting follows the default
configuration of the same cell (50 million environment steps). Default
TeamFlowRL forms, for each agent, the ratio of its complete denoising trajectory
(the product of the `K = 10` transition ratios) and clips it separately with one
PPO clip, using the shared team advantage.

| Directory | Change | Configuration field |
| --- | --- | --- |
| `per_step_clip/` | clip the ratio of each of the `K` denoising steps separately and sum the `K` clipped surrogate losses (as DPPO does for single-agent denoising transitions); the trajectory-KL guard is unchanged | `ppo.ratio_mode = "per_step"` |
| `avg_logprob_clamp/` | divide the trajectory log probability by `K * action_dim`, clamp it element-wise to `[-5, 2]`, multiply back, then form the ratio (clamped elements receive no gradient) | `ppo.logprob_clamp = [-5.0, 2.0]` |

Each directory contains its own `generative_mappo/` package and three
configurations: MPE Spread Medium, MPE Spread Medium-Replay and MA-MuJoCo 2-Ant
Poor. Run from the repository root; every command trains the three training
seeds 0, 1, 2 (`<output>/seed<k>/`), or one seed with `--seed k`:

```bash
python tools/run_with_variant.py --variant ablations/ratio/per_step_clip -- \
  --config ablations/ratio/per_step_clip/configs/mpe__simple_spread__medium.json --phase online

python tools/run_with_variant.py --variant ablations/ratio/avg_logprob_clamp -- \
  --config ablations/ratio/avg_logprob_clamp/configs/mamujoco__2ant__poor.json --phase online
```

Evaluation of all three seeds:

```bash
python -m eval.evaluate_main_table --variant ablations/ratio/per_step_clip \
  --config ablations/ratio/per_step_clip/configs/mpe__simple_spread__medium.json \
  --checkpoint '<output>/seed{seed}/online/checkpoint_final.pt' --seeds 0 1 2
```

The default rows of the ablation table are the corresponding benchmark cells in
`configs/benchmarks/`. `per_step_clip/stepwise.patch` is the diff of the
per-step change against the core package, and `per_step_clip/test_stepwise.py`
tests it:

```bash
cd ablations/ratio/per_step_clip && python -m unittest test_stepwise -v
```
