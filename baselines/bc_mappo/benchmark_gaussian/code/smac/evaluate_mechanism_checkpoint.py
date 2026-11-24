"""Independent held-out evaluation using the validated parallel evaluator."""
import argparse
import json
from pathlib import Path
from sc2_port_lease import install
install()
from common import atomic_json, sha256
from train import DEFAULTS
from mechanism_evaluation import evaluate_parallel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--env-info',required=True);parser.add_argument('--checkpoint')
    parser.add_argument('--training-seed',type=int,help='evaluate the run of this training seed: output_dir and the '
                        'source checkpoint of the configuration are moved to <run>/seed<k>/ (see training_seeds.py)')
    args=parser.parse_args();cfg=dict(DEFAULTS,**json.loads(Path(args.config).read_text()))
    if args.training_seed is not None:
        from training_seeds import seed_config
        cfg=seed_config(cfg,args.training_seed)[0]
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=False)
    info=json.loads(Path(args.env_info).read_text())['env_info']
    try:
        if cfg['variant']=='jax':
            from jax_backend import JaxBackend
            backend=JaxBackend(cfg,info)
        else:
            from mechanism_backend import MechanismBackend as TorchBackend
            backend=TorchBackend(cfg,info)
        cfg['source_checkpoint_sha256']=sha256(cfg['checkpoint'])
        if args.checkpoint:backend.restore(args.checkpoint)
        atomic_json(out/'run_config.json',dict(cfg,evaluated_checkpoint=args.checkpoint,
                    environment_rng='smac_v1_game_seed_and_worker_numpy_v1'))
        with open(out/'evaluation.jsonl','w',buffering=1) as log:
            for deterministic in [True,False]:
                result=evaluate_parallel(backend,cfg,cfg['eval_episodes'],deterministic)
                record=dict(variant=cfg['variant'],task=cfg['task'],checkpoint=args.checkpoint,**result)
                line=json.dumps(record,allow_nan=False);print(line,flush=True);log.write(line+'\n')
        atomic_json(out/'status.json',dict(status='complete'))
    except BaseException as exc:
        atomic_json(out/'status.json',dict(status='failed',error=repr(exc)));raise


if __name__=='__main__':main()
