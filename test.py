#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import requests
import numpy as np
import pandas as pd
from io import BytesIO

from exopop import Simulator, data

NMAX = 10
PERIOD_RNG = (5., 300.)
RADIUS_RNG = (0.3, 12.)

# Get the datasets.
stlr = data.get_burke_gk()
kois = data.get_candidates(stlr, period_range=PERIOD_RNG,
                           radius_range=RADIUS_RNG)
multiplicity = data.compute_multiplicity(len(stlr), kois, NMAX)

print("Selected {0} targets after cuts".format(len(stlr)))
print("Selected {0} KOIs after cuts".format(len(kois)))
print("Observed multiplicity distribution: {0}".format(multiplicity))

print("Setting up simulator...")
strt = time.time()
sim = Simulator(stlr.iloc[:500], NMAX, RADIUS_RNG[0], RADIUS_RNG[1],
                PERIOD_RNG[0], PERIOD_RNG[1])
print("    took {0} seconds".format(time.time() - strt))

# Choose some reasonable parameters.
multi = np.ones(NMAX)  # np.arange(1, NMAX+1) ** -1.0
multi /= np.sum(multi)
pars = np.append([
    -0.5, -2.0, 3.0,    # radius
    -0.5, 0.1, 100.0,  # period
    0.01                # inclination
], np.log(multi[:-1]))

print("Timing observations...")
K = 50
strt = time.time()
for i in range(K):
    counts = sim.observe(pars)
print("    time per observation: {0} seconds".format((time.time() - strt) / K))

print("Observed counts: {0}".format(counts))
