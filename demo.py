#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import time
from functools import partial
from collections import Counter

import h5py
import numpy as np
from scipy.stats import ks_2samp

import tqdm
from schwimmbad import MPIPool

from exoabc import Simulator, data

__all__ = []


period_range = (50, 300)
prad_range = (0.75, 2.5)
depth_range = (0, 1000)
maxn = 2

prefix = "q1_q16"
stlr = data.get_burke_gk(prefix=prefix)
kois = data.get_candidates(stlr=stlr, prefix=prefix)

sim = Simulator(
    stlr,
    period_range[0], period_range[1], 0.0,
    prad_range[0], prad_range[1], -2.0,
    -3.0, np.zeros(maxn),
    release=prefix,
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
    hist = np.zeros(maxn, dtype=int)
    for i in range(maxn):
        hist[i] = h.get(i + 1, 0)

    return (
        hist, np.array(c.koi_period), np.array(c.koi_depth),
        np.array(c.koi_duration)
    )
obs_stats = compute_stats(kois)

def compute_distance(ds1, ds2):
    norm = max(np.max(ds1[0]), np.max(ds2[0]))
    multi_dist = np.sum((ds1[0] - ds2[0])**2.0) / norm**2
    period_dist = ks_2samp(ds1[1], ds2[1]).statistic
    depth_dist = ks_2samp(ds1[2], ds2[2]).statistic
    return multi_dist + period_dist + depth_dist

def sample(initial):
    if initial is None:
        lp = sim.sample_parameters()
        if not np.isfinite(lp):
            return np.inf, sim.get_parameters(), sim.state
    else:
        lp = sim.set_parameters(initial)
        if not np.isfinite(lp):
            return np.inf, sim.get_parameters(), sim.state

    pars, state = sim.get_parameters(), sim.state
    df = sim.sample_population()
    if len(df) <= 1:
        return np.inf, pars, state
    return compute_distance(obs_stats, compute_stats(df)), pars, state

def pmc_sample_one(eps, tau, theta0, weights, initial=None):
    # Sample until a suitable sample is found.
    rho = np.inf
    while rho > eps or not np.isfinite(rho):
        theta_star = theta0[np.random.choice(np.arange(len(weights)),
                                             p=weights)]
        theta_i = theta_star + tau * np.random.randn(len(theta_star))
        p, _, state_i = sample(theta_i)
        rho = np.sum(p)

    # Re-weight the samples.
    log_prior = sim.log_pdf()
    norm = 0.5*((theta0 - theta_i)/tau[None, :])**2 + np.log(tau[None, :])
    norm = np.log(weights) - np.sum(norm, axis=1)
    log_weight = log_prior - np.logaddexp.reduce(norm)
    return rho, theta_i, state_i, log_weight

def parse_samples(samples):
    rho = np.array([s[0] for s in samples])
    m = np.isfinite(rho)
    rho = rho[m]
    params = np.array([s[1] for s in samples])[m]
    states = np.array([s[2] for s in samples])[m]
    if len(samples[0]) == 3:
        return rho, params, states
    log_w = np.array([s[3] for s in samples])[m]
    return rho, params, states, np.exp(log_w - np.logaddexp.reduce(log_w))

def update_target_density(rho, params, weights, percentile=25.0):
    norm = np.sum(weights)
    mu = np.sum(params * weights[:, None], axis=0) / norm
    tau = np.sqrt(2 * np.sum((params-mu)**2*weights[:, None], axis=0) / norm)
    eps = np.percentile(rho, percentile)
    return eps, tau

with MPIPool() as pool:
    pool.wait()

    # Run step 1 of PMC method.
    N = 1000
    rhos, thetas, states = parse_samples(list(pool.map(
        sample, tqdm.tqdm((None for N in range(N)), total=N))))
    weights = np.ones(len(rhos)) / len(rhos)

    os.makedirs("results", exist_ok=True)
    for it in range(10):
        eps, tau = update_target_density(rhos, thetas, weights)
        func = partial(pmc_sample_one, eps, tau, thetas, weights)
        rhos, thetas, states, weights = parse_samples(list(pool.map(
            func, tqdm.tqdm((None for N in range(N)), total=N))))

        with h5py.File("results/{0:03d}.h5".format(it), "w") as f:
            f.attrs["iteration"] = it
            f.attrs["eps"] = eps
            f.attrs["tau"] = tau
            f.create_dataset("rho", data=rhos)
            f.create_dataset("theta", data=thetas)
            f.create_dataset("weight", data=weights)
