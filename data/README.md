# Data

Datasets are not included. The configurations read them from `DATA_ROOT`:

```
${DATA_ROOT}/mpe/        MPE offline arrays
${DATA_ROOT}/mamujoco/   MA-MuJoCo offline arrays
${DATA_ROOT}/omiga/      OMIGA offline arrays
```

SMAC and SMACv2 need a StarCraft II installation given by `SC2PATH`. The array
layout of each benchmark is described in `envs/README.md`.
