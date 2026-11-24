# BC-initialized MAPPO

Code for MPE, MA-MuJoCo, SMAC and SMACv2.

1. Initialize a random actor and run 1,000,000 pure BC updates on the task dataset
   (SMAC/SMACv2: masked cross entropy on dataset actions, recurrent encoder,
   four-layer 256-wide action head, zero latent input; MPE/MA-MuJoCo: MSE
   regression from observations to dataset actions, four-layer 512-wide policy,
   zero latent input).
2. Start online MAPPO from the BC actor with fresh optimizers and a newly
   initialized centralized value critic for 50,000,000 environment steps.
3. Evaluate the final checkpoint with the TeamFlowRL protocol (evaluation seeds
   10000, 80000, 90000; 20 episodes each).

No teacher, flow matching, distillation or Q guidance is used.

## Layout

* `mpe/`, `mamujoco/`: `train.py` (`--phase pretrain` / `--phase ppo`),
  `evaluate.py`, vendored environment code under `vendor/`.
* `smac/`, `smacv2/`: `audit_data_environment.py` (dataset/environment
  specification used as `--env-info` by the BC evaluations), `pretrain_smac.py`
  (BC), `train_mechanism_ablation.py` (MAPPO), `evaluate_mechanism_checkpoint.py`,
  and the policy/environment package under `project/mac_flow_pytorch_smacv2/`.
* `training_seeds.py` in every family: training seeds 0, 1, 2 by default.

## Run (training seeds 0, 1, 2)

MPE / MA-MuJoCo (from the family directory):

```bash
cd mpe   # or mamujoco
python train.py --config <config.json> --phase pretrain        # -> <output>/seed<k>/pretrain
python train.py --config <config.json> --phase ppo             # -> <output>/seed<k>/ppo
for k in 0 1 2; do for s in 10000 80000 90000; do
  python evaluate.py --config <output>/seed$k/config_ppo.json \
    --checkpoint <output>/seed$k/ppo/checkpoint_50000000.pt \
    --output <output>/seed$k/eval_seed_$s.json --episodes 20 --seed $s
done; done
```

SMAC / SMACv2 (set `SC2PATH` first):

```bash
cd smacv2   # or smac
python audit_data_environment.py --config <audit_config.json>     # writes the data_audit file of the pretrain config
python pretrain_smac.py --config <pretrain_config.json>           # -> <run>/seed<k>/pretrain
python train_mechanism_ablation.py --config <ppo_config.json>     # -> <run>/seed<k>/ppo
for k in 0 1 2; do
  python evaluate_mechanism_checkpoint.py --config <eval_config.json> --training-seed $k \
    --env-info <run>/seed$k/ppo/run_config.json --checkpoint <run>/seed$k/ppo/checkpoint_final.pt
done
```

Example SMACv2 configurations are in `../smacv2/configs/`.
