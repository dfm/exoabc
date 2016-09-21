# -*- coding: utf-8 -*-

__version__ = "0.0.1.dev0"

try:
    __EXOABC_SETUP__
except NameError:
    __EXOABC_SETUP__ = False

if not __EXOABC_SETUP__:
    pass
    # __all__ = ["Simulator"]
    # from .model import Simulator
