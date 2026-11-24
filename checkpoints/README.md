# Offline checkpoints

The online phase starts from an offline flow-policy checkpoint
(`source_checkpoint` in each configuration). The configurations expect

```
${CKPT_ROOT}/<task_dataset>/<checkpoint file>
```

where `<task_dataset>` is `experiment.task_dataset` in the configuration (for
example `simple_spread_expert` or `6halfcheetah_Medium`). Set `CKPT_ROOT`, or
edit `source_checkpoint` directly. `generative_mappo.checkpoints.import_offline`
validates the checkpoint format and rejects a checkpoint whose algorithm tag does
not match the configuration.
