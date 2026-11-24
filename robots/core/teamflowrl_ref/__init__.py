"""Import alias for the TeamFlowRL core package.

The robot code imports the core package (flow policy, denoising-trajectory
likelihood, MAPPO update with a shared team advantage) as
``teamflowrl_ref.generative_mappo``. This package makes that name resolve to the
repository's ``generative_mappo/`` package, so no copy of the core is needed.

The repository root is found by walking up from this file; set
``TEAMFLOWRL_ROOT`` to the repository root if the robot code is copied elsewhere.
"""
import os
from pathlib import Path


def _repository_root():
    candidates = [Path(os.environ['TEAMFLOWRL_ROOT'])] if os.environ.get('TEAMFLOWRL_ROOT') else []
    candidates += list(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / 'generative_mappo' / '__init__.py').is_file():
            return candidate
    raise ImportError('generative_mappo/ not found; set TEAMFLOWRL_ROOT to the repository root')


# Submodules of teamflowrl_ref are looked up in the repository root, so
# teamflowrl_ref.generative_mappo.<module> loads <root>/generative_mappo/<module>.py.
__path__ = [str(_repository_root())]
