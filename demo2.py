#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import time
import argparse
from math import factorial
from functools import partial
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
args = parser.parse_args()

if args.prefix == "q1_q17_dr24":
    period_range = (10, 300)
    prad_range = (0.75, 2.5)
    depth_range = (0, 1000)
    maxn = 8
    prefix = "q1_q17_dr24"
    stlr = data.get_burke_gk(prefix=prefix)
    kois = data.get_candidates(stlr=stlr, prefix=prefix, mesthresh=15.0)
    params, fig = data.calibrate_completeness(stlr, period_range=period_range,
                                              plot=True)
    fig.savefig("completeness.png")
    plt.close(fig)
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

sim = Simulator(
    stlr,
    period_range[0], period_range[1], 0.0,
    prad_range[0], prad_range[1], -2.0,
    -3.0, np.zeros(maxn),
    min_period_slope=-5.0, max_period_slope=3.0,
    min_radius_slope=-5.0, max_radius_slope=3.0,
    min_log_sigma=-5.0, max_log_sigma=np.log(np.radians(90)),
    min_log_multi=-10.0, max_log_multi=100.0,
    release=prefix, completeness_params=params,
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

def compute_distance(ds1, ds2):
    multi_dist = np.mean((np.log(ds1[0]+1) - np.log(ds2[0]+1))**2.0)
    period_dist = ks_2samp(ds1[1], ds2[1]).statistic
    depth_dist = ks_2samp(ds1[2], ds2[2]).statistic
    # dur_dist = ks_2samp(ds1[3], ds2[3]).statistic
    return multi_dist + period_dist + depth_dist  # + dur_dist

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

def update_target_density(rho, params, weights, percentile=30.0):
    norm = np.sum(weights)
    mu = np.sum(params * weights[:, None], axis=0) / norm
    tau = np.sqrt(2 * np.sum((params-mu)**2*weights[:, None], axis=0) / norm)
    eps = np.percentile(rho, percentile)
    return eps, tau

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    # from cycler import cycler
    # from matplotlib import rcParams
    # rcParams["font.size"] = 16
    # rcParams["font.family"] = "sans-serif"
    # rcParams["font.sans-serif"] = ["Computer Modern Sans"]
    # rcParams["text.usetex"] = True
    # rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
    # rcParams["axes.prop_cycle"] = cycler("color", (
    #     "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    #     "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    # ))  # d3.js color cycle

    # Run step 1 of PMC method.
    N = 1500
    rhos, thetas, states = parse_samples(list(pool.map(
        sample, tqdm.tqdm((None for N in range(N)), total=N))))
    weights = np.ones(len(rhos)) / len(rhos)

    os.makedirs("results", exist_ok=True)
    stlr.to_hdf(os.path.join("results", "stlr.h5"), "stlr", format="t")
    kois.to_hdf(os.path.join("results", "kois.h5"), "kois", format="t")
    for it in range(100):
        eps, tau = update_target_density(rhos, thetas, weights)
        func = partial(pmc_sample_one, eps, tau, thetas, weights)
        rhos, thetas, states, weights = parse_samples(list(pool.map(
            func, tqdm.tqdm((None for N in range(N)), total=N))))

        with h5py.File(os.path.join("results", "{0:03d}.h5".format(it)),
                       "w") as f:
            f.attrs["maxn"] = maxn
            f.attrs["iteration"] = it
            f.attrs["eps"] = eps
            f.attrs["tau"] = tau
            for i in range(len(obs_stats)):
                f.attrs["obs_stats_{0}".format(i)] = obs_stats[i]
            f.create_dataset("rho", data=rhos)
            f.create_dataset("theta", data=thetas)
            f.create_dataset("weight", data=weights)
            f.create_dataset("state", data=states)

        fig = corner.corner(thetas, weights=weights)
        fig.savefig(os.path.join("results", "corner-{0:03d}.png".format(it)))
        plt.close(fig)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Observed distributions
        dur_range = (obs_stats[3].min(), obs_stats[3].max())
        for i in np.random.choice(len(weights), p=weights, size=100):
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
        axes[0, 3].set_yscale("log")
        axes[0, 0].set_xlabel("period")
        axes[0, 1].set_xlabel("depth")
        axes[0, 2].set_xlabel("duration")
        axes[0, 3].set_xlabel("multiplicity")
        axes[0, 0].set_yticklabels([])
        axes[0, 1].set_yticklabels([])
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

        n = np.concatenate((np.ones((len(thetas), 1)),
                            np.exp(thetas[:, -maxn:])),
                           axis=1)
        n /= np.sum(n, axis=1)[:, None]
        q = np.percentile(n, [16, 50, 84], axis=0)
        ax = axes[1, 3]
        x = np.arange(maxn+1)
        ax.fill_between(x, q[0], q[2], color="k", alpha=0.1)
        ax.plot(x, q[1], color="k", lw=2)
        lam = np.exp(-0.25089448)
        ax.plot(x, lam**x * np.exp(-lam) / np.array(list(map(factorial, x))),
                color="g", lw=2)
        ax.set_xlim(0, maxn)

        axes[1, 3].set_yscale("log")
        axes[1, 0].set_xlabel("period")
        axes[1, 1].set_xlabel("radius")
        axes[1, 3].set_xlabel("multiplicity")
        axes[1, 0].set_ylabel("underlying distributions")

        fig.tight_layout()
        fig.savefig(os.path.join("results", "params-{0:03d}.png".format(it)),
                    bbox_inches="tight")
        plt.close(fig)
