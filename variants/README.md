# Code variants

Each directory is a complete, self-contained copy of the `generative_mappo`
package. A cell that uses a variant is run through the launcher, which puts the
variant first on `sys.path`:

```bash
python tools/run_with_variant.py --variant variants/<variant> -- \
  --config configs/benchmarks/<cell>.json --phase online          # seeds 0 1 2
```

Evaluation takes the same `--variant` argument (`eval/evaluate_main_table.py`).
All variants accept `--seeds` / `--seed` exactly like the root package (training
seeds 0, 1, 2 by default, seed k in `<output>/seed<k>/`). All of them keep the
ten-step denoising trajectory, the shared team advantage and the separate
clipping of each agent's complete trajectory ratio; they differ only in the
numerical or optimization details listed below.

| Variant | Difference from `generative_mappo/` | Cells |
| --- | --- | --- |
| `bounded_mean/` | `models.py`: optional `policy.transition_mean_bound`; each transition mean becomes `bound * tanh(mean / bound)` (configurations use `bound = 10`, with `kl_backtrack_steps = 12`) | OMIGA 6-HalfCheetah Expert / Medium / Medium-Replay, 3-Hopper Expert / Medium |
| `rev2/` | earlier numerical revision: PPO branch selected in log space, FP64 replay tolerance and `network_dtype`, but without the SMAC observation-layout validator (agent ids appended at the end only) | SMAC 5m_vs_6m Good |
| `rev1_kl_backtrack/` | first numerical revision (FP32 policy, no `network_dtype`) plus opt-in KL backtracking (`ppo.kl_backtrack_steps`) | SMAC 5m_vs_6m Medium |
| `minibatch_kl/` | trajectory-KL guard on the current minibatch (`ppo.kl_check_scope = "minibatch"`) with cached old transition statistics, critic value normalization (`critic_valuenorm`), KL normalized by the number of steps, scheduled actor learning rate and entropy coefficient | MPE Spread Random |
| `chunked_rollout_kl/` | rollout trajectory KL is evaluated in chunks of 20,480 samples and accumulated in float64 (evaluation only; the PPO minibatch is unchanged) | MA-MuJoCo 2-HalfCheetah Good |
