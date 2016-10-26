#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import time
import argparse
from collections import Counter

import h5py
import numpy as np
from scipy.stats import ks_2samp

import tqdm

from exoabc import Simulator, data

parser = argparse.ArgumentParser()
parser.add_argument("prefix", choices=["q1_q16", "q1_q17_dr24"])
parser.add_argument("runid")
parser.add_argument("--num-per", type=int, default=1000)
parser.add_argument("--maxn", type=int, default=8)
parser.add_argument("--poisson", action="store_true")
parser.add_argument("--broken", action="store_true")
args = parser.parse_args()

if args.prefix == "q1_q17_dr24":
    period_range = (10, 300)
    prad_range = (0.75, 2.5)
    depth_range = (0, 1000)
    maxn = args.maxn
    prefix = "q1_q17_dr24"
    stlr = data.get_burke_gk(prefix=prefix)
    kois = data.get_candidates(stlr=stlr, prefix=prefix, mesthresh=15.0)
    params = data.calibrate_completeness(stlr, period_range=period_range)
elif args.prefix == "q1_q16":
    params = None
    period_range = (50, 300)
    prad_range = (0.75, 2.5)
    depth_range = (0, 1000)
    maxn = args.maxn
    prefix = "q1_q16"
    stlr = data.get_burke_gk(prefix=prefix)
    kois = data.get_candidates(stlr=stlr, prefix=prefix)
else:
    assert False, "Invalid prefix"

if args.poisson:
    multi_params = 0.0
    min_log_multi = -10.0
    max_log_multi = 1.0
else:
    multi_params = np.ones(maxn + 1) / (maxn + 1.0)
    min_log_multi = -5.0
    max_log_multi = 0.0

sim = Simulator(
    stlr,
    period_range[0], period_range[1], 0.0,
    prad_range[0], prad_range[1], -2.0,
    -3.0, multi_params,
    min_period_slope=-5.0, max_period_slope=3.0,
    min_radius_slope=-5.0, max_radius_slope=3.0,
    min_log_sigma=-5.0, max_log_sigma=np.log(np.radians(90)),
    min_log_multi=min_log_multi, max_log_multi=max_log_multi,
    release=prefix, completeness_params=params,
    poisson=args.poisson, broken_radius=args.broken,
    seed=int(os.getpid() + 1000*time.time()) % 20000,
)

def compute_stats(catalog):
    m = (period_range[0] <= catalog.koi_period)
    m &= (catalog.koi_period <= period_range[1])
    m &= (prad_range[0] <= catalog.koi_prad)
    m &= (catalog.koi_prad <= prad_range[1])
    m &= (depth_range[0] <= catalog.koi_depth)
    m &= (catalog.koi_depth <= depth_range[1])
    c = catalog[m]

    # Multiplicity
    h = Counter(Counter(c.kepid).values())
    hist = np.zeros(maxn+1, dtype=int)
    for i in range(1, maxn+1):
        hist[i] = h.get(i, 0)
    hist[0] = len(stlr) - np.sum(hist[1:])

    return (
        hist, np.array(c.koi_period), np.array(c.koi_depth),
        np.array(c.koi_duration)
    )
obs_stats = compute_stats(kois)

def compute_distances(ds1, ds2):
    n1 = np.sum(ds1[0][1:])
    n2 = np.sum(ds2[0][1:])
    multi_dist_1 = (np.log(n1)-np.log(n2))**2
    multi_dist_2 = np.mean((np.log(ds1[0]+1) - np.log(ds2[0]+1))**2.0)
    period_dist = ks_2samp(ds1[1], ds2[1]).statistic
    depth_dist = ks_2samp(ds1[2], ds2[2]).statistic
    dur_dist = ks_2samp(ds1[3], ds2[3]).statistic
    return (multi_dist_1, multi_dist_2, period_dist, depth_dist, dur_dist)

def sample(initial):
    # Sample the hyperparamters
    lp = sim.sample_parameters()
    if not np.isfinite(lp):
        return None

    # Simulate the population
    df = sim.sample_population()
    if len(df) <= 1:
        return None

    # Build the output
    pars, state = sim.get_parameters(), sim.state
    mu = sim.mean_multiplicity()
    zero = sim.evaluate_multiplicity(np.zeros(1))[0]
    dist = compute_distances(obs_stats, compute_stats(df))
    return dist, pars, state, mu, zero

def parse_samples(samples):
    samples = [s for s in samples if s is not None and
               np.all(np.isfinite(s[0]))]
    return map(np.array, zip(*samples))

fn = "{0}.h5".format(args.runid)
with h5py.File(fn, "w") as f:
    f.attrs["maxn"] = maxn
    for i in range(len(obs_stats)):
        f.attrs["obs_stats_{0}".format(i)] = obs_stats[i]


for it in range(500):
    N = args.num_per
    rhos, thetas, states, mus, zeros = parse_samples(map(
        sample, tqdm.tqdm((None for _ in range(N)), total=N)))
    weights = np.ones(len(rhos)) / len(rhos)

    with h5py.File(fn, "a") as f:
        f.attrs["iteration"] = it
        if it == 0:
            f.create_dataset("rho", data=rhos,
                             maxshape=tuple([None]+list(rhos.shape[1:])))
            f.create_dataset("theta", data=thetas,
                             maxshape=tuple([None]+list(thetas.shape[1:])))
            f.create_dataset("state", data=states,
                             maxshape=tuple([None]+list(states.shape[1:])))
            f.create_dataset("expected_number", data=mus,
                             maxshape=tuple([None]+list(mus.shape[1:])))
            f.create_dataset("log_rate_zero", data=zeros,
                             maxshape=tuple([None]+list(zeros.shape[1:])))
        else:
            for k, v in zip(["rho", "theta", "state", "expected_number",
                             "log_rate_zero"],
                            [rhos, thetas, states, mus, zeros]):
                n = f[k].shape[0]
                f[k].resize(n + v.shape[0], axis=0)
                f[k][n:] = v
