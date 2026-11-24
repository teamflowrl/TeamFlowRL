"""Import aliases of the TeamFlowRL core package used by the camera code.

The camera code was run with deployment names for the core package
``generative_mappo`` (see ``compat.py``, which applies the same renaming to the
tags stored in checkpoints):

* ``algo_ref.generative_policycore``: algorithm tags ``A_TF_POLICYCORE`` /
  ``B_SG_POLICYCORE`` / ``C_MF_POLICYCORE`` and trainer class ``PolicyCore``
  (used by ``ac_policy.py``);
* ``algo_ref.generative_R``: algorithm tags ``A_TF_R_online`` / ... and trainer
  class ``R_online`` (used by ``flow_refit/ac_policy.py``).

Both are built from the repository's ``generative_mappo/`` modules; the only
change is the spelling of the three algorithm tags, and the trainer class is
the core ``MAPPO`` class under another name. Set ``TEAMFLOWRL_ROOT`` to the
repository root if the camera code is copied elsewhere.
"""
import os
import sys
import types
from pathlib import Path

_MODULES = ('meanflow', 'models', 'ppo', 'checkpoints')   # dependency order
_ALIASES = {'generative_policycore': ('POLICYCORE', 'PolicyCore'),
            'generative_R': ('R_online', 'R_online')}


def _core_dir():
    candidates = [Path(os.environ['TEAMFLOWRL_ROOT'])] if os.environ.get('TEAMFLOWRL_ROOT') else []
    candidates += list(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / 'generative_mappo' / '__init__.py').is_file():
            return candidate / 'generative_mappo'
    raise ImportError('generative_mappo/ not found; set TEAMFLOWRL_ROOT to the repository root')


def _source(path, suffix):
    text = path.read_text()
    for tag in ('A_TF', 'B_SG', 'C_MF'):
        text = text.replace(f"'{tag}_MAPPO'", f"'{tag}_{suffix}'")
    return text


def _build(name, suffix, trainer):
    core = _core_dir()
    package_name = f'{__name__}.{name}'
    package = types.ModuleType(package_name)
    package.__path__, package.__package__, package.__file__ = [], package_name, str(core / '__init__.py')
    sys.modules[package_name] = package
    exec(compile(_source(core / '__init__.py', suffix), package.__file__, 'exec'), package.__dict__)
    for module_name in _MODULES:
        path = core / f'{module_name}.py'
        module = types.ModuleType(f'{package_name}.{module_name}')
        module.__package__, module.__file__ = package_name, str(path)
        sys.modules[module.__name__] = module
        exec(compile(_source(path, suffix), str(path), 'exec'), module.__dict__)
        setattr(package, module_name, module)
    setattr(package.ppo, trainer, package.ppo.MAPPO)
    return package


for _name, (_suffix, _trainer) in _ALIASES.items():
    globals()[_name] = _build(_name, _suffix, _trainer)
