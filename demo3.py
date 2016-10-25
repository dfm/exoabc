#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import time
import argparse
from collections import Counter

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

import tqdm
import corner
from schwimmbad import MPIPool

from exoabc import Simulator, data

__all__ = []

parser = argparse.ArgumentParser(
    description="collect some search and injection/recovery results"
)
parser.add_argument("prefix", choices=["q1_q16", "q1_q17_dr24"])
parser.add_argument("--poisson", action="store_true")
args = parser.parse_args()

if args.prefix == "q1_q17_dr24":
    period_range = (10, 300)
    prad_range = (0.75, 2.5)
    depth_range = (0, 1000)
    maxn = 8
    prefix = "q1_q17_dr24"
    stlr = data.get_burke_gk(prefix=prefix)
    kois = data.get_candidates(stlr=stlr, prefix=prefix, mesthresh=15.0)
    params = data.calibrate_completeness(stlr, period_range=period_range)
elif args.prefix == "q1_q16":
    params = None
    period_range = (50, 300)
    prad_range = (0.75, 2.5)
    depth_range = (0, 1000)
    maxn = 5
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
    multi_params = np.zeros(maxn)
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
    poisson=args.poisson,
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

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    os.makedirs("results", exist_ok=True)
    stlr.to_hdf(os.path.join("results", "stlr.h5"), "stlr", format="t")
    kois.to_hdf(os.path.join("results", "kois.h5"), "kois", format="t")
    for it in range(500):
        N = 500
        rhos, thetas, states, mus, zeros = parse_samples(list(pool.map(
            sample, tqdm.tqdm((None for _ in range(N)), total=N))))
        weights = np.ones(len(rhos)) / len(rhos)

        with h5py.File(os.path.join("results", "{0:03d}.h5".format(it)),
                       "w") as f:
            f.attrs["maxn"] = maxn
            f.attrs["iteration"] = it
            for i in range(len(obs_stats)):
                f.attrs["obs_stats_{0}".format(i)] = obs_stats[i]
            f.create_dataset("rho", data=rhos)
            f.create_dataset("theta", data=thetas)
            f.create_dataset("state", data=states)
            f.create_dataset("expected_number", data=mus)
            f.create_dataset("log_rate_zero", data=zeros)

        rhos = np.sum(rhos, axis=1)
        eps = np.percentile(rhos, 25)
        m = rhos < eps
        thetas = thetas[m]
        states = states[m]
        rhos = rhos[m]

        fig = corner.corner(thetas)
        fig.savefig(os.path.join("results", "corner-{0:03d}.png".format(it)))
        plt.close(fig)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Observed distributions
        dur_range = (obs_stats[3].min(), obs_stats[3].max())
        for i in np.random.choice(len(thetas), size=100):
            p = thetas[i]
            sim.set_parameters(p)
            sim.state = states[i]
            pop = sim.sample_population()
            sim_stats = compute_stats(pop)

            axes[0, 0].hist(sim_stats[1], range=period_range, histtype="step",
                            color="k", alpha=0.2)
            axes[0, 1].hist(sim_stats[2], range=depth_range, histtype="step",
                            color="k", alpha=0.2)
            axes[0, 2].hist(sim_stats[3], range=dur_range, histtype="step",
                            color="k", alpha=0.2)
            axes[0, 3].plot(sim_stats[0], color="k", alpha=0.2)

        axes[0, 0].hist(obs_stats[1], range=period_range, histtype="step",
                        color="g", lw=2)
        axes[0, 1].hist(obs_stats[2], range=depth_range, histtype="step",
                        color="g", lw=2)
        axes[0, 2].hist(obs_stats[3], range=dur_range, histtype="step",
                        color="g", lw=2)
        axes[0, 3].plot(obs_stats[0], color="g", lw=2)
        axes[0, 0].set_yscale("log")
        axes[0, 1].set_yscale("log")
        axes[0, 2].set_yscale("log")
        axes[0, 3].set_yscale("log")
        axes[0, 0].set_ylim(0, axes[0, 0].get_ylim()[1])
        axes[0, 1].set_ylim(0, axes[0, 1].get_ylim()[1])
        axes[0, 0].set_xlabel("period")
        axes[0, 1].set_xlabel("depth")
        axes[0, 2].set_xlabel("duration")
        axes[0, 3].set_xlabel("multiplicity")
        # axes[0, 0].set_yticklabels([])
        # axes[0, 1].set_yticklabels([])
        axes[0, 0].set_ylabel("observed distributions")

        # True distributions
        for n, rng, ax in zip(thetas[:, :2].T, (period_range, prad_range),
                              axes[1, :2]):
            x = np.linspace(rng[0], rng[1], 5000)
            norm = (n + 1) / (rng[1]**(n+1) - rng[0]**(n+1))
            d = x[:, None]**n[None, :] * norm[None, :]
            q = np.percentile(d, [16, 50, 84], axis=1)
            ax.fill_between(x, q[0], q[2], color="k", alpha=0.1)
            ax.plot(x, q[1], color="k", lw=2)
            ax.set_xlim(*rng)

        # Plot the multiplicity distribution.
        multi = np.empty((len(thetas), maxn+1))
        inds = np.arange(maxn+1).astype(np.float64)
        for i in range(len(thetas)):
            p = thetas[i]
            sim.set_parameters(p)
            multi[i] = sim.evaluate_multiplicity(inds)
        ax = axes[1, 3]
        q = np.percentile(np.exp(multi), [16, 50, 84], axis=0)
        ax.fill_between(inds, q[0], q[2], color="k", alpha=0.1)
        ax.plot(inds, q[1], color="k", lw=2)
        ax.set_xlim(0, maxn)

        axes[1, 0].set_yscale("log")
        axes[1, 1].set_yscale("log")
        axes[1, 3].set_yscale("log")
        axes[1, 0].set_xlabel("period")
        axes[1, 1].set_xlabel("radius")
        axes[1, 3].set_xlabel("multiplicity")
        axes[1, 0].set_ylabel("underlying distributions")

        fig.tight_layout()
        fig.savefig(os.path.join("results", "params-{0:03d}.png".format(it)),
                    bbox_inches="tight")
        plt.close(fig)
