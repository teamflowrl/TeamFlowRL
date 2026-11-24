# Camera observations

TeamFlowRL and BC+MAPPO on ManiSkill3 Pick (`TwoRobotPickCube-v1`) and Stack
(`TwoRobotStackCube-v1`) with camera observations. Each arm receives a global
view shared by both arms and its own wrist view, encoded by a frozen DINOv2
model, together with its proprioceptive state, the goal position, the remaining
time and an agent indicator. Cube poses are available only to the state teacher
and to the centralized critic.

The camera code uses deployment names for the actor ids and the core package
(`compat.py` applies the same renaming to checkpoint tags):

| Camera code | Meaning |
| --- | --- |
| `var_a` | TeamFlowRL (ten-step flow head) |
| `base` | BC+MAPPO (one-step Gaussian head) |
| `var_b`, `var_c` | the `dfpo` / `maogpo` actor variants of `robots/core` |
| `flowcore`, `FlowCore*`, `R`, `RFlow*` | MAC-Flow pretraining method / modules |
| `policycore.py`, `PolicyCore`, `r_online` | online trainer (`mappo.py`) / MAPPO update |
| `algo_ref.generative_policycore`, `algo_ref.generative_R` | the TeamFlowRL core package; `code/algo_ref/` builds both from `generative_mappo/` |

## Layout

```
camera/code/
  policycore.py          online fine-tuning with camera actors and a state critic
  R_L2/policycore.py     the same trainer with an --initial-std option (Pick, BC+MAPPO)
  run_train.sh           online launcher, seeds 0 1 2
  train.py               offline pretraining of the camera policy (--obs-mode rgb2)
  flow_refit/dagger.py   DAgger from the state teacher to the recurrent camera policy
  flow_refit/run_flow_refit.py, vision_flow_distill.py
                         TeamFlowRL start: freeze the DAgger encoder, refit the ten-step flow head
  gauss_head_refit.py    BC+MAPPO start: same frozen encoder, refit the one-step Gaussian head
  eval_checkpoint.py     evaluation of a refitted Gaussian-head checkpoint (100 + 100 episodes)
  ac_policy.py, compat.py, evaluate.py, env_adapter.py, obs_builder.py, policy.py, vision.py, ...
  references/            MAC-Flow policy modules
  algo_ref/              import alias of the TeamFlowRL core package
```

## Pipeline (training seeds 0, 1, 2 at every stage)

Every stage runs seeds 0, 1, 2 by default (`--seed k` for one seed) and writes
seed `k` to `<out>/seed<k>/`; `{seed}` in an input path selects the same seed of
the previous stage. Set `DINO_SOURCE` and `DINO_WEIGHTS` (DINOv2 ViT-S/14).

```bash
C=robots/camera/code
# 1. RGB observations of the demonstrations and camera pretraining (DAgger start)
python robots/core/extract_rgb.py --task TwoRobotPickCube-v1 --mode rgb2 --camera-res 224 \
  --state-h5 <state>.h5 --h5 <demos>.h5 --out <rgb2>.h5
python $C/train.py --data <rgb2>.h5 --out <camera_pretrain>/pick --method flowcore --obs-mode rgb2 --updates 50000
# 2. DAgger from the state teacher (an online-trained state policy, robots/core)
python $C/flow_refit/dagger.py --task TwoRobotPickCube-v1 \
  --student '<camera_pretrain>/pick/seed{seed}/ckpt_50000.pt' \
  --teacher '<state_teacher>/seed{seed}/<checkpoint>.pt' --out <dagger>/pick
# 3a. TeamFlowRL start: flow-head refit on the frozen DAgger encoder (Pick: rounds 0..5, Stack: 0..2)
python $C/flow_refit/run_flow_refit.py --student '<dagger>/pick/seed{seed}/student_r5.pt' \
  --dagger '<dagger>/pick/seed{seed}' --rounds 5 --out <flow_refit>/pick \
  --updates 20000 --batch 64 --sequence 20 --lr 3e-4
# 3b. BC+MAPPO start: Gaussian-head refit on the same data
python $C/gauss_head_refit.py --student '<dagger>/pick/seed{seed}/student_r5.pt' \
  --dagger '<dagger>/pick/seed{seed}' --rounds 5 --out <head_refit>/pick \
  --updates 20000 --batch 64 --sequence 20 --lr 3e-4
# 4. online fine-tuning, 1M steps (1,004,800 = 157 rollouts of 6,400 transitions)
COMMON="--camera-res 224 --envs 64 --steps 1004800 --eval-every-steps 500000 --eval-episodes 100"
bash $C/run_train.sh TwoRobotPickCube-v1 var_a '<flow_refit>/pick/seed{seed}/flow_teacher.pt' <online>/tf_pick $COMMON
POLICY_SCRIPT=R_L2/policycore.py bash $C/run_train.sh TwoRobotPickCube-v1 base \
  '<head_refit>/pick/seed{seed}/head_refit.pt' <online>/bc_pick $COMMON \
  --actor-lr 2e-5 --critic-lr 3e-4 --minibatch 128 --epochs 4 --initial-std 0.02
bash $C/run_train.sh TwoRobotStackCube-v1 var_a '<flow_refit>/stack/seed{seed}/flow_teacher.pt' <online>/tf_stack $COMMON
bash $C/run_train.sh TwoRobotStackCube-v1 base '<head_refit>/stack/seed{seed}/head_refit.pt' <online>/bc_stack $COMMON
```

The points before online step zero in the camera learning curves come from the
refit stage: `eval_checkpoint.py` evaluates the Gaussian-head checkpoints saved
every 4,000 updates (`ckpt_4000.pt` ... `ckpt_20000.pt`) of each seed:

```bash
for k in 0 1 2; do for s in 4000 8000 12000 16000 20000; do
  python $C/eval_checkpoint.py --checkpoint <head_refit>/pick/seed$k/ckpt_$s.pt \
    --out <head_refit>/pick/seed$k/eval_$s.json
done; done
```

Online runs evaluate every 500,000 steps and at the end with 100 deterministic
and 100 stochastic episodes (`<online>/<run>/seed<k>/eval.jsonl`).
