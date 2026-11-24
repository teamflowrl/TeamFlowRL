"""Evaluate online checkpoints with the benchmark protocol.

Each checkpoint is evaluated in three evaluation groups of 20 episodes. Each
group has its own evaluation seed (10000, 80000, 90000); the per-checkpoint
return is the mean of the three group means, with the sample standard
deviation across group means.

If ``--checkpoint`` contains ``{seed}``, one checkpoint per training seed
(``--seeds``, default 0 1 2) is evaluated and the report also contains the
mean and sample standard deviation across training seeds.
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

EVALUATION_SEEDS = (10000, 80000, 90000)


def summarize(values):
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="task JSON (see configs/benchmarks)")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="online checkpoint; may contain {seed}, e.g. outputs/<cell>/seed{seed}/online/checkpoint_final.pt",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="training seeds for {seed}")
    parser.add_argument("--episodes", type=int, default=20, help="episodes per evaluation group")
    parser.add_argument("--device", default=None, help="torch device; defaults to config.device")
    parser.add_argument("--out", default=None, help="optional output JSON path")
    parser.add_argument(
        "--variant",
        default=None,
        help="optional variant directory, e.g. variants/bounded_mean",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.variant:
        sys.path.insert(0, str(root / args.variant))
    sys.path.insert(1, str(root))

    from generative_mappo.checkpoints import load, restore_actor
    from generative_mappo.train import evaluate, expand_env

    config = expand_env(json.loads(Path(args.config).read_text()))
    if config.get("environment", {}).get("sc2path"):
        os.environ["SC2PATH"] = config["environment"]["sc2path"]
    device = args.device or config.get("device", "cpu")

    runs = []
    training_seeds = args.seeds if "{seed}" in args.checkpoint else [None]
    for training_seed in training_seeds:
        checkpoint = args.checkpoint.replace("{seed}", str(training_seed))
        payload = load(checkpoint)
        actor, encoder = restore_actor(payload, config["algorithm"])
        actor = actor.to(device).eval()
        if encoder is not None and hasattr(encoder, "to"):
            encoder = encoder.to(device).eval()

        groups = []
        for seed in EVALUATION_SEEDS:
            result = evaluate(actor, encoder, config["environment"], args.episodes, seed)
            groups.append(dict(seed=seed, **result))

        summary = {"mean_return": summarize([g["mean_return"] for g in groups])}
        win_rates = [g["win_rate"] for g in groups]
        if all(value is not None for value in win_rates):
            summary["win_rate"] = summarize(win_rates)
        runs.append(dict(training_seed=training_seed, checkpoint=checkpoint, groups=groups, summary=summary))

    report = {
        "protocol": {
            "episodes_per_group": args.episodes,
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "per_checkpoint_uncertainty": "sample SD across evaluation-group means",
        },
        "config": str(args.config),
        "runs": runs,
    }
    if len(runs) > 1:
        report["across_training_seeds"] = {
            key: summarize([run["summary"][key]["mean"] for run in runs])
            for key in runs[0]["summary"]
            if all(key in run["summary"] for run in runs)
        }
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
