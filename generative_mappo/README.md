# Core package (`generative_mappo`)

TeamFlowRL core used by all benchmark experiments (and, through an import alias,
by the robot code).

| Module | Contents |
| --- | --- |
| `models.py` | flow generator; `PathPolicy`: the `K`-step denoising trajectory as Gaussian transitions with learnable noise, exact trajectory log probability, conditional trajectory KL; centralized critic |
| `ppo.py` | `MAPPO`: shared team advantage (GAE), separate clipping of each agent's complete trajectory ratio, trajectory-KL stop, critic warm-up |
| `meanflow.py` | MeanFlow loss (only for the `C_MF_MAPPO` variant) |
| `checkpoints.py` | versioned checkpoints and import of offline flow-policy checkpoints |
| `data.py`, `environments.py` | offline data and benchmark adapters (`envs/`) |
| `train.py` | CLI: `--phase pretrain|online|evaluate`; training seeds 0 1 2 by default (`--seeds`, `--seed`) |
| `tests/` | CPU unit tests: `python -m unittest generative_mappo.tests.test_algorithms -v` |

Algorithm ids: `A_TF_MAPPO` = TeamFlowRL (ten-step flow policy initialized from
the offline teacher); `B_SG_MAPPO` and `C_MF_MAPPO` are one-step generator
variants kept for completeness.
