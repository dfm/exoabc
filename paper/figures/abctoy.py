#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

from mcmc import mh
from plotting import setup, SQUARE_FIGSIZE, savefig

setup()
np.random.seed(42)

# The simulation model:
def simulator(ln_mu, size=None):
    return np.random.poisson(np.exp(ln_mu), size=size)

true_rate = 6.78
prior_range = np.array([-5.0, 3.0])
# true_rate = 50.345
# prior_range = np.array([0, 4.5])

# Simulated data:
N_obs = simulator(np.log(true_rate))

# Correct probabilistic model:

def lnprior(ln_mu):
    m = prior_range[0] <= ln_mu
    m &= ln_mu <= prior_range[1]
    return np.log(1.0 * m)

def lnprob(ln_mu):
    return lnprior(ln_mu) + N_obs * ln_mu - np.exp(ln_mu)

mu_grid = np.linspace(np.exp(prior_range[0]), np.exp(prior_range[1]), 10000)
true_posterior = np.exp(lnprob(np.log(mu_grid)) - np.log(mu_grid))
true_posterior /= simps(true_posterior, mu_grid)

# Heuristic method:
def heuristic_log_probability_function(N_obs, ln_mu):
    N_sim = simulator(float(ln_mu))
    return lnprior(ln_mu) + N_obs * np.log(N_sim) - N_sim

heuristic_log_probability = partial(heuristic_log_probability_function, N_obs)
heuristic_chain, _ = mh(heuristic_log_probability, np.log([N_obs]), 200000)

# ABC method:
def pseudo_log_probability_function(N_obs, S, eps, ln_mu):
    N_sim = simulator(float(ln_mu), size=S)
    dist = N_sim - N_obs
    return lnprior(ln_mu) + np.logaddexp.reduce(-0.5 * (dist / eps)**2)

S = 50
eps_pow = -3
pseudo_log_probability = partial(pseudo_log_probability_function, N_obs, S,
                                 10**eps_pow)
pseudo_chain, _ = mh(pseudo_log_probability, np.log([N_obs]), 500000)

fig, ax = plt.subplots(1, 1, figsize=SQUARE_FIGSIZE)

bins = np.linspace(0, np.exp(prior_range[1]), 50)
# x = 0.5*(bins[1:] + bins[:-1])
y, _ = np.histogram(np.exp(heuristic_chain), bins, density=True)
ax.step(bins, np.append(0, y), lw=1.5, label="heuristic")
y, _ = np.histogram(np.exp(pseudo_chain), bins, density=True)
ax.step(bins, np.append(0, y), lw=1.5, label="abc")
ax.plot(mu_grid, true_posterior, color="k", lw=1.5, label="exact", alpha=0.8)
ax.axvline(true_rate, lw=3, color="k", alpha=0.3)

ax.set_yticklabels([])
ax.set_xlabel("$\mu$")
ax.set_ylabel("$p(\mu\,|\,N_\mathrm{obs})$")
ax.set_xlim(0, np.exp(prior_range[1]))
ax.set_ylim(0, 1.1 * true_posterior.max())
ax.legend(fontsize=12)

with open("abctoy.tex", "w") as f:
    f.write("\\newcommand{{\\abctoytruth}}{{{{\ensuremath{{{0:.2f}}}}}}}\n"
            .format(true_rate))
    f.write("\\newcommand{{\\abctoynobs}}{{{{\ensuremath{{{0:.0f}}}}}}}\n"
            .format(N_obs))
    f.write("\\newcommand{{\\abctoyeps}}{{{{\ensuremath{{10^{{{0:d}}}}}}}}}\n"
            .format(eps_pow))
    f.write("\\newcommand{{\\abctoyS}}{{{{\ensuremath{{{0:d}}}}}}}\n"
            .format(S))

    q = np.percentile(np.exp(heuristic_chain), [16, 50, 84])
    f.write("\\newcommand{\\abctoyheuristic}{{\ensuremath{")
    f.write("{0:.2f}^{{+{1:.2f}}}_{{{2:.2f}}}"
            .format(q[1], q[2]-q[1], q[1]-q[0]))
    f.write("}}}\n")

    q = np.percentile(np.exp(pseudo_chain), [16, 50, 84])
    f.write("\\newcommand{\\abctoyabc}{{\ensuremath{")
    f.write("{0:.2f}^{{+{1:.2f}}}_{{{2:.2f}}}"
            .format(q[1], q[2]-q[1], q[1]-q[0]))
    f.write("}}}\n")

savefig(fig, "abctoy.pdf")
