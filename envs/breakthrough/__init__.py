"""Breakthrough environments.

Imports are resolved lazily (PEP 562) so that importing the package does
not eagerly import the submodules. This mirrors the parent ``envs`` package
and, importantly, avoids the ``runpy`` double-execution warning when a
submodule is run as a script via ``python -m envs.breakthrough.<module>``
(eager re-exports would place the module in ``sys.modules`` before runpy
executes it as ``__main__``).
"""

import importlib

# Public name -> module providing it (relative to this package).
_EXPORTS = {
    "BreakthroughEnv": ".breakthrough_env",
    "BreakthroughException": ".breakthrough_env",
    "VectorBreakthroughEnv": ".vector_breakthrough_env",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)
