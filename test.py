#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import requests
import numpy as np
import pandas as pd
from io import BytesIO

from exopop import Simulator

NMAX = 10
PERIOD_RNG = (5., 300.)
RADIUS_RNG = (0.3, 12.)


def get_catalog(name, basepath="data"):
    fn = os.path.join(basepath, "{0}.h5".format(name))
    if os.path.exists(fn):
        return pd.read_hdf(fn, name)
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    print("Downloading {0}...".format(name))
    url = ("http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/"
           "nph-nstedAPI?table={0}&select=*").format(name)
    r = requests.get(url)
    if r.status_code != requests.codes.ok:
        r.raise_for_status()
    fh = BytesIO(r.content)
    df = pd.read_csv(fh)
    df.to_hdf(fn, name, format="t")
    return df

kois = get_catalog("q1_q16_koi")
stlr = get_catalog("q1_q16_stellar")

# Select G & K dwarfs.
m = (4200 <= stlr.teff) & (stlr.teff <= 6100)
m &= stlr.radius <= 1.15

# Only include stars with sufficient data coverage.
m &= stlr.dataspan > 365.25*2.
m &= stlr.dutycycle > 0.6
m &= stlr.rrmscdpp07p5 <= 1000.

# Only select stars with mass & radius estimates.
m &= np.isfinite(stlr.mass) & np.isfinite(stlr.radius)
stlr = pd.DataFrame(stlr[m])
print("Selected {0} targets after cuts".format(len(stlr)))

# Join on the stellar list.
kois = pd.merge(kois, stlr[["kepid"]], on="kepid", how="inner")

# Only select the KOIs in the relevant part of parameter space.
m = kois.koi_pdisposition == "CANDIDATE"
m &= np.isfinite(kois.koi_prad)
m &= (PERIOD_RNG[0] < kois.koi_period) & (kois.koi_period < PERIOD_RNG[1])
m &= (RADIUS_RNG[0] < kois.koi_prad) & (kois.koi_prad < RADIUS_RNG[1])
kois = pd.DataFrame(kois[m])
print("Selected {0} KOIs after cuts".format(len(kois)))

# Compute the counts.
unique_kois = kois.groupby("kepid")[["koi_pdisposition"]].count()
data = np.zeros(NMAX, dtype=int)
data[0] = len(stlr) - len(unique_kois)
counts = np.array(unique_kois.groupby("koi_pdisposition")
                  .koi_pdisposition.count())
data[1:1+len(counts)] = counts
print(data)

sim = Simulator(stlr.iloc[:100], NMAX, RADIUS_RNG[0], RADIUS_RNG[1],
                PERIOD_RNG[0], PERIOD_RNG[1])

multi = np.arange(1, NMAX+1) ** -1.0
multi /= np.sum(multi)
pars = np.append([-1.0, -2.5, 1.0, -0.5, 0.01], np.log(multi[:-1]))

K = 50
strt = time.time()
for i in range(K):
    counts = sim.observe(pars)
print((time.time() - strt) / K)
print(counts)
