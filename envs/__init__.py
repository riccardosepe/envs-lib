"""Aggregator for the environment classes.

Imports are resolved lazily (PEP 562) so that importing a single env
subpackage (e.g. ``envs.sailingdomain``) does not force every other env
to be importable. This lets consumers materialize only a subset of envs
on disk (e.g. via git sparse-checkout) without breaking ``import envs``.
"""

import importlib

# Public name -> module providing it (relative to this package).
_EXPORTS = {
    "BarterGameEnv": ".bartergame.bartergame_env",
    "BlockStackEnv": ".blockstack.blockstack_env",
    "BreakthroughEnv": ".breakthrough.breakthrough_env",
    "CliffWorldEnv": ".cliffworld.cliffworld_env",
    "Connect4Env": ".connect4.connect4_env",
    "FrozenLakeEnv": ".frozenlake.frozenlake_env",
    "FruitCollectionEnv": ".fruitcollection.fruitcollection_env",
    "SailingDomainEnv": ".sailingdomain.sailingdomain_env",
    "TicTacToeEnv": ".tictactoe.tictactoe_env",
    "TowersOfHanoiEnv": ".hanoitowers.hanoitowers_env",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)
