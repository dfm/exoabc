# -*- coding: utf-8 -*-

__version__ = "0.0.1.dev0"

try:
    __EXOPOP_SETUP__
except NameError:
    __EXOPOP_SETUP__ = False

if not __EXOPOP_SETUP__:
    __all__ = ["Simulator"]
    from .model import Simulator
