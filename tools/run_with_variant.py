"""Run the training CLI against a selected code variant.

Usage (from the repository root):

    python tools/run_with_variant.py --variant variants/bounded_mean \
        -- --config configs/benchmarks/<cell>.json --phase online

The variant directory is placed first on ``sys.path`` so its
``generative_mappo`` package shadows the repository-root package. Relative
paths in the configuration still resolve from the repository root.
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="variant directory")
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    variant = str((root / args.variant).resolve())
    sys.path.insert(0, variant)
    sys.path.insert(1, str(root))
    # The trainer may re-execute itself as `python -m generative_mappo.train` (to apply
    # MuJoCo library paths); PYTHONPATH keeps the variant package first in that process too.
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [variant, str(root)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
    sys.argv = ["generative_mappo.train"] + [
        item for item in args.train_args if item != "--"
    ]

    from generative_mappo.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
