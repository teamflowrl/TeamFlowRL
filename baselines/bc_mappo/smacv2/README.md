# SMACv2 BC+MAPPO runs

BC+MAPPO on SMACv2 Terran 5v5 Good, Zerg 5v5 Good, Zerg 5v5 Replay and
Terran 10v10 Replay. The training and evaluation code is `../bc_init/smacv2/`;
only task and data fields differ between configurations. BC uses 1,000,000
updates and MAPPO uses 50,000,000 environment steps.

## Layout

* `configs/<task>_<dataset>/`: pretraining, MAPPO and final-test configurations
  (`terran_5_vs_5_Good`, `zerg_5_vs_5_Replay`; the other runs change only
  `task`, `split` and `data_root`). Their `"seed"` is overridden by the training
  seed (0, 1, 2 by default).
* `scripts/convert_good.py`: converts the Good episode shards to the flat array
  layout of the Replay datasets.
* `scripts/eval_record.py`: held-out evaluation that also records per-episode
  returns (`--training-seed k` selects the run of seed `k`).

## Run (training seeds 0, 1, 2)

```bash
export SC2PATH=/path/to/StarCraftII_smacv2
export DATA_ROOT=/path/to/datasets RUN_ROOT=/path/to/runs CODE_ROOT=$PWD/baselines/bc_mappo
C=$CODE_ROOT/smacv2/configs/terran_5_vs_5_Good
cd $CODE_ROOT/bc_init/smacv2
python pretrain_smac.py --config $C/pretrain_config.json          # -> $RUN_ROOT/.../seed<k>/pretrain
python train_mechanism_ablation.py --config $C/ppo_config.json    # -> $RUN_ROOT/.../seed<k>/ppo
R=$RUN_ROOT/smacv2/terran_5_vs_5_Good
for k in 0 1 2; do
  python evaluate_mechanism_checkpoint.py --config $C/test_final_50m_config.json --training-seed $k \
    --env-info $R/seed$k/ppo/run_config.json --checkpoint $R/seed$k/ppo/checkpoint_final.pt
  for s in 10000 80000 90000; do
    python $CODE_ROOT/smacv2/scripts/eval_record.py --config $C/ppo_config.json --training-seed $k \
      --env-info $R/seed$k/ppo/run_config.json --checkpoint $R/seed$k/ppo/checkpoint_final.pt \
      --episodes 20 --seed $s --policies deployment --output $R/seed$k/paper_eval_seed_$s
  done
done
```

`pretrain_smac.py` reads the dataset/environment specification named by
`data_audit` in the pretraining configuration; create it once with
`../bc_init/smacv2/audit_data_environment.py`. Evaluation uses evaluation seeds
10000, 80000 and 90000 with 20 deterministic episodes each.
