"""ManiSkill hands-away training: runs ms_release/train_release.py (robots/core) with the
release evaluation of away_eval.py (thresholds from $AWAY_THRESHOLDS). Training seeds 0, 1, 2 by
default (--seed k for one seed); seed k writes to <out>/seed<k>/."""
import os,sys,runpy,importlib.util,json
from pathlib import Path
root=Path(__file__).resolve().parent
# Robot code: CORE_CODE (default robots/core). RoboFactory release code: RF_CODE (default robots/core;
# it must also provide the RoboFactory release task, see robots/hands_away/README.md).
CORE_CODE=Path(os.environ.get('CORE_CODE',str(Path(__file__).resolve().parents[2]/'core')))
RF_CODE=Path(os.environ.get('RF_CODE',str(CORE_CODE)))
sys.path.insert(0,str(root))
sys.path.insert(0,str(CORE_CODE))
import away_eval
away_eval.install('ms',os.environ['AWAY_THRESHOLDS'])
import ms_release.eval_release as ev
ev.evaluate_pair=away_eval.ms_pair
if '--checkpoint' in sys.argv and 'maogpo' in sys.argv[sys.argv.index('--checkpoint')+1]:
    name='teamflowrl_ref.generative_mappo.away_stability_ppo'
    spec=importlib.util.spec_from_file_location(name,RF_CODE/'away_support/stability_ppo.py')
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m)
    old=m.MAPPO.update
    def update(self,batch):
        self.options.update(target_kl=.02,max_grad_norm=.5,bc_coef=0.,kl_backtrack_steps=12)
        return old(self,batch)
    m.MAPPO.update=update
    import ac_policy
    ac_policy.MAPPO=m.MAPPO
runpy.run_module('ms_release.train_release',run_name='__main__')
