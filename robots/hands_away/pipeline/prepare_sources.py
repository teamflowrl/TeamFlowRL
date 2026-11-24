import os,sys,json,hashlib,random
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent
sys.path.insert(0,os.environ.get('RF_CODE',str(R.parents[1]/'core')))
from policy import Policy
from mappo import OnlinePolicy
from ac_policy import from_pretrained
hashes={'macflow_30000.pt':'1fa5b3b70a52e0b2ce55ca64cf1b2737bca4c835cdb5c4d9be358208c3157fc3','ckpt_400000.pt':'3b7a8e5eb78b7957ad39bd15e32c1364257801882f82dbfb8004c3d1f4412ad5'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for n,s in hashes.items():assert sha(R/'rf_pretrained'/n)==s,n
rows=[]
for algo in ['teamflow','dfpo','maogpo','fppo']:
 random.seed(1);np.random.seed(1);torch.manual_seed(1)
 src=R/'rf_pretrained'/('ckpt_400000.pt' if algo=='maogpo' else 'macflow_30000.pt')
 cp=torch.load(src,map_location='cpu',weights_only=False)
 if algo in ('teamflow','maogpo'):
  policy=from_pretrained(cp,algo)
  if algo=='maogpo':
   bounds=json.loads((R/'release_bounds.json').read_text())
   assert bounds['dataset_sha256']=='3c8f08f26e18d84cdb724c913244dc74f4de0c104ac888609da1dc46418a3eff'
   policy.set_obs_clip('minmax',bounds['lo'],bounds['hi'])
 else:
  base=Policy(cp['config']);base.load_state_dict(cp['model']);policy=OnlinePolicy(base,algo)
 path=R/'rf_original'/f'{algo}.pt'
 torch.save(dict(version=2,method=cp['method'],algo=algo,model=policy.state_dict(),config=policy.config_dict(),contract=cp['contract'],obs_mode=cp['obs_mode'],step=0,source_sha256=sha(src)),path)
 stats=R/'rf_checkpoints'/('meanflow/ckpt_400000.stats.json' if algo=='maogpo' else 'macflow/ckpt_30000.stats.json')
 old=json.loads(stats.read_text());old=old.get('critic',old)
 for k in ('state_mean','state_std'):assert torch.equal(torch.as_tensor(old[k]),torch.as_tensor(cp[k])),(algo,k)
 rows.append(dict(domain='rf',algo=algo,path=str(path),sha256=sha(path),pretrained=str(src),pretrained_sha256=sha(src),stats=str(stats),stats_sha256=sha(stats),release_bounds_dataset_sha256=bounds['dataset_sha256'] if algo=='maogpo' else None))
expected_ms=dict(teamflow='7822c00e9dcdd678780c821a34069cd37fdbef25b903a2091d65821d69807d1c',dfpo='8ed71f0ad7b7435f272cd9bfc5c266187a64ab7139d2b2a74dba372478bd7278',maogpo='ad52d66419ac730960e84d22808bc0354b1f7ca8eb1e270018e57b9c5aaf123f',fppo='61ac7ac41718fe1f36e0d65e0105e39ff3c877ffd1c4458c0fde31102dee4076')
for algo in ['teamflow','dfpo','maogpo','fppo']:
 p=R/'checkpoints'/f'{algo}-dec/ckpt_0.pt';assert sha(p)==expected_ms[algo];rows.append(dict(domain='ms',algo=algo,path=str(p),sha256=sha(p)))
(R/'checkpoint_manifest.json').write_text(json.dumps(rows,indent=2))
(R/'MILESTONE_V3_SOURCES.json').write_text(json.dumps(dict(status='complete',sources=rows),indent=2))
print('Verified pretrained SHA, matching critic statistics, release minmax; constructed all 8 sources.')
