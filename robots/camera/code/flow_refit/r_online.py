"""Alias used by dagger.py: the online camera trainer is ``policycore.py`` in the parent
directory (code/); ``OnlinePolicy`` and the rollout/update functions are defined there."""
from policycore import *  # noqa: F401,F403
from policycore import OnlinePolicy  # noqa: F401
