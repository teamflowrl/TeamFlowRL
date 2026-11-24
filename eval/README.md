# Benchmark evaluation

`evaluate_main_table.py` evaluates online checkpoints with the benchmark
protocol: three evaluation groups of 20 episodes with evaluation seeds 10000,
80000 and 90000 (these are evaluation seeds, not training seeds). If
`--checkpoint` contains `{seed}`, the checkpoint of every training seed in
`--seeds` (default 0 1 2) is evaluated, and the report also gives the mean and
sample standard deviation across training seeds.

```bash
python -m eval.evaluate_main_table --config configs/benchmarks/<cell>.json \
  --checkpoint 'outputs/<cell>/seed{seed}/online/checkpoint_final.pt' \
  --seeds 0 1 2 --out outputs/<cell>/eval_all_seeds.json [--variant variants/<variant>]
```
