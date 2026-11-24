"""Evaluate one camera checkpoint of the one-step Gaussian head (BC+MAPPO start; head refit) with
100 deterministic and 100 stochastic episodes, calling the unchanged online policy and evaluation code.

    python eval_checkpoint.py --checkpoint <head_refit>/seed<k>/ckpt_20000.pt --out <eval>.json
"""
import argparse, importlib.util, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from compat import normalize_checkpoint
from policy import Policy
from evaluate import evaluate_pair
from data import sha256
from vision import checkpoint_camera_res, DEFAULT_DINO_SOURCE, DEFAULT_DINO_WEIGHTS
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
assert not a.out.exists(), 'Do not overwrite results'
torch.cuda.set_device(0);torch.set_num_threads(2)
torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
random.seed(1);np.random.seed(1);torch.manual_seed(1)
cp=torch.load(a.checkpoint,map_location='cuda',weights_only=False)
original_method=cp.get('method');normalize_checkpoint(cp)
task=cp['contract']['task'];assert cp['obs_mode']=='rgb2'
source=HERE/('R_L2/policycore.py' if 'Pick' in task else 'policycore.py')
spec=importlib.util.spec_from_file_location('online_original',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
base=Policy(cp['config']).cuda();base.load_state_dict(cp['model'],strict=True)
policy=mod.OnlinePolicy(base,'base',initial_std=.02).cuda() if 'Pick' in task else mod.OnlinePolicy(base,'base').cuda()
policy.eval().requires_grad_(False)
root=HERE
meta=dict(checkpoint_path=str(a.checkpoint),sha256=sha256(a.checkpoint),task=task,checkpoint_name=a.checkpoint.name,update=cp.get('step'),dagger_round=cp.get('dagger',{}).get('round'),dagger_total_updates=cp.get('dagger',{}).get('total_updates'),original_method=original_method,task_id=os.environ.get('TASK_ID',''),run_name=a.out.stem,script_path=str(Path(__file__).resolve()),policy_script=str(source),initial_exploration_std=.02 if 'Pick' in task else .2,camera_res=checkpoint_camera_res(cp,224),history_reset=False,matmul_precision='highest',matmul_tf32=False,cudnn_tf32=False,code_sha256={str(f.relative_to(HERE)):sha256(f) for f in HERE.rglob('*.py')})
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.with_suffix('.started.json').write_text(json.dumps(meta,indent=2))
result=evaluate_pair(policy,task,'rgb2',100,det_seed=5000000,stoch_seed=5100000,history_reset=False,camera_res=224,dino_source=DEFAULT_DINO_SOURCE,dino_weights=Path(DEFAULT_DINO_WEIGHTS),dino_weights_sha256=cp['dino_weights_sha256'])
meta.update(result);meta['checkpoint_sha256_after']=sha256(a.checkpoint);assert meta['sha256']==meta['checkpoint_sha256_after']
import importlib.metadata as md
meta['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'sapien':md.version('sapien'),'mani_skill':md.version('mani_skill'),'python':sys.version}
meta['peak_torch_reserved_bytes']=torch.cuda.max_memory_reserved()
a.out.write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({k:result[k]['native_final_successes'] for k in result}),flush=True)
