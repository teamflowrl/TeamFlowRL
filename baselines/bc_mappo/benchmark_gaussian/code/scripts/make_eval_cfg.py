"""Create a fresh evaluation config from a PPO config."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--ppo', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--output-dir', required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--episodes', type=int, required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
c = json.loads(Path(a.ppo).read_text())
c.update(output_dir=a.output_dir, eval_seed=a.seed, eval_episodes=a.episodes)
Path(a.out).write_text(json.dumps(c, indent=2, sort_keys=True) + '\n')
print(a.out)
