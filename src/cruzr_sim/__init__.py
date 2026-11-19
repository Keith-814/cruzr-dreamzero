"""Cruzr simulation foundation; pure policy utilities do not initialize MuJoCo."""
__all__ = ["CruzrEnvironment", "load_config"]


def __getattr__(name):
    if name == "CruzrEnvironment":
        from .environment import CruzrEnvironment
        return CruzrEnvironment
    if name == "load_config":
        from .config import load_config
        return load_config
    raise AttributeError(name)
