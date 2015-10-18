# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "RealDataset", "SimulatedDataset", "ExopopABCModel",
    "PowerLawMultiDist", "PoissonMultiDist", "PoissonMixtureMultiDist",
]

import numpy as np
import matplotlib.pyplot as pl
from scipy.special import factorial

from .data import compute_multiplicity


class RealDataset(object):

    def __init__(self, stlr, kois, nplanets):
        self.stlr = stlr
        self.kois = kois
        self.multiplicity = compute_multiplicity(len(stlr), kois, nplanets)
        self.radii = np.array(self.kois.koi_prad)
        self.periods = np.array(self.kois.koi_period)


class SimulatedDataset(object):

    def __init__(self, multiplicity, periods, radii):
        self.multiplicity = multiplicity
        self.periods = periods
        self.radii = radii


class ExopopABCModel(object):

    def __init__(self, simulator, dataset, initial_theta, multi_dist=None,
                 thawed_parameters=None):
        self.simulator = simulator
        self.theta = initial_theta
        self.cached_theta = np.array(self.theta)
        self.dataset = dataset

        if multi_dist is None:
            multi_dist = PowerLawMultiDist()
        self.multi_dist = multi_dist

        if thawed_parameters is None:
            thawed_parameters = np.arange(len(self.theta))
        self.thawed_parameters = np.array(thawed_parameters, dtype=int)

        if self.simulator.nplanets != len(self.dataset.multiplicity):
            raise ValueError("dimension mismatch: the dataset and simulator "
                             "must expect the same number of planets")

    def get_simulated_dataset(self):
        r = self.simulator.observe(self.transform_parameters(self.theta))
        return SimulatedDataset(r[0], r[2][:, 0], r[2][:, 1])

    def transform_parameters(self, theta):
        n = len(self.multi_dist)
        multi = self.multi_dist.transform(theta[:n], self.simulator.nplanets)
        return np.concatenate((
            theta[n:-1],
            [np.radians(np.exp(theta[-1]))],
            np.log(multi[:-1])
        ))

    def log_prior(self):
        n = len(self.multi_dist)
        lp = self.multi_dist.log_prior(self.theta[:n])
        if not np.isfinite(lp):
            return -np.inf

        # Radius slopes
        if np.any((self.theta[n:n+2] < -5) | (self.theta[n:n+2] > 5)):
            return -np.inf

        # Radius break
        rng = self.simulator.radius_range
        if not (rng[0] < self.theta[n+2] < rng[1]):
            return -np.inf

        # Period slopes
        if np.any((self.theta[n+3:n+5] < -5) | (self.theta[n+3:n+5] > 5)):
            return -np.inf

        # Period break
        rng = self.simulator.period_range
        if not (rng[0] < self.theta[n+5] < rng[1]):
            return -np.inf

        # Mutual inclination width
        if not (-3 < self.theta[-1] < np.log(90.0)):
            return -np.inf

        return lp

    def negative_log_distance(self):
        # Run the simulation and observe the synthetic catalog.
        try:
            r = self.simulator.observe(self.transform_parameters(self.theta))
        except RuntimeError:
            return -np.inf
        counts = r[0]

        # Assert that the model predicts at least one count in every
        # multiplicity bin with a non-zero observation.
        multi = self.dataset.multiplicity
        if np.any((multi > 0) & (counts == 0)):
            return -np.inf

        # Compute the Poisson likelihood with the simulation providing the
        # expected counts.
        m = counts > 0
        nld = multi[m] * np.log(counts[m]) - counts[m]
        nld[multi > 1] -= logfactorial(multi[m & (multi > 1)])
        return np.sum(nld)

    def perturb(self, nld_thresh=-np.inf, initial_nld=None):
        self.cached_theta = np.array(self.theta)

        # Sometime update the simulation variables.
        if np.random.rand() < 0.5:
            self.simulator.perturb()
            nld = self.negative_log_distance()
            if nld > nld_thresh:
                return nld
            self.simulator.revert()
            return (
                self.negative_log_distance()
                if initial_nld is None else initial_nld
            )

        # Other times, update the hyperparameters.
        lny = self.log_prior() + np.log(np.random.rand())
        i = np.random.choice(self.thawed_parameters)
        I = self.step_out(i, lny, nld_thresh, 3.0)
        return self.slice_sample(i, lny, nld_thresh, I)

    def revert(self):
        self.theta = np.array(self.cached_theta)
        self.simulator.revert()

    def step_out(self, ind, lny, nld_thresh, w, m=100):
        u, v = np.random.rand(2)
        x0 = np.array(self.theta[ind])
        L = self.theta[ind] - w*u
        R = L + w
        J = int(np.floor(v*m))
        K = int(m - 1 - J)

        for j in range(J, 0, -1):
            L -= w
            self.theta[ind] = L
            nld = self.negative_log_distance()
            if lny > self.log_prior() or nld_thresh > nld:
                break

        for k in range(K, 0, -1):
            R += w
            self.theta[ind] = R
            nld = self.negative_log_distance()
            if lny > self.log_prior() or nld_thresh > nld:
                break

        self.theta[ind] = x0

        return L, R

    def slice_sample(self, ind, lny, nld_thresh, I):
        L, R = I
        x0 = self.theta[ind]
        while True:
            u = np.random.rand()
            self.theta[ind] = L + u * (R - L)
            nld = self.negative_log_distance()
            if lny < self.log_prior() and nld_thresh < nld:
                return nld
            if self.theta[ind] < x0:
                L = self.theta[ind]
            else:
                R = self.theta[ind]
            if np.allclose(L, R):
                raise RuntimeError("slice sampling didn't converge")

    def sample(self, theta, niter, nld_thresh=-np.inf):
        self.theta = np.array(theta)
        chain = np.empty((niter, len(self.theta)))
        nlds = np.empty(niter)

        # Save the initial location.
        chain[0, :] = theta
        nlds[0] = self.negative_log_distance()

        for i in range(1, niter):
            nlds[i] = self.perturb(initial_nld=nlds[i-1],
                                   nld_thresh=nld_thresh)
            chain[i, :] = self.theta

        return chain, nlds

    def plot(self, period_range=None, radius_range=None, nsamps=1):
        theta = self.theta

        if period_range is None:
            period_range = self.simulator.period_range
        if radius_range is None:
            radius_range = self.simulator.radius_range

        # Run the simulations.
        pars = self.transform_parameters(theta)
        counts = []
        catalogs = []
        for _ in range(nsamps):
            if nsamps > 1:
                self.simulator.resample()
            r = self.simulator.observe(pars)
            counts.append(r[0])
            catalogs.append(r[2])

        # Set up the figure.
        fig, axes = pl.subplots(4, 2, figsize=(6, 7))

        # Plot the periods and radii.
        ax = axes[0, 0]
        ax.plot(np.log10(catalogs[0][:, 0]), np.log10(catalogs[0][:, 1]),
                "og", ms=2, mec="none")
        ax.plot(np.log10(self.dataset.periods), np.log10(self.dataset.radii),
                "ok", ms=2, mec="none")
        ax.set_xlim(np.log10(period_range))
        ax.set_ylim(np.log10(radius_range))
        ax.set_ylabel(r"$\log_{10} R / R_\oplus$")
        ax.set_xlabel(r"$\log_{10} P / \mathrm{day}$")

        # Plot the multiplicity.
        ax = axes[1, 0]
        for c in counts:
            ax.plot(c, "o-g", alpha=0.5, ms=2, mec="none")
        ax.plot(self.dataset.multiplicity, "o:k", ms=2, mec="none")
        ax.set_xlim(0, 9.5)
        ax.set_yscale("log")
        ax.set_xlabel("number of transiting planets")
        ax.set_ylabel("number of systems")
        ax.set_title("observed multiplicity")

        # Plot the period histogram.
        ax = axes[2, 0]
        bins = np.exp(np.linspace(np.log(period_range[0]),
                                  np.log(period_range[1]), 10))
        for c in catalogs:
            ax.hist(c[:, 0], bins, histtype="step", color="g", alpha=0.5)
        ax.hist(self.dataset.periods, bins, histtype="step", color="k",
                linestyle="dashed")
        ax.set_xlim(period_range)
        ax.set_xscale("log")
        ax.set_yticklabels([])
        ax.set_xlabel(r"$P / \mathrm{day}$")
        ax.set_title("observed period dist.")

        # Plot the radius histogram.
        ax = axes[3, 0]
        bins = np.exp(np.linspace(np.log(radius_range[0]),
                                  np.log(radius_range[1]), 10))
        for c in catalogs:
            ax.hist(c[:, 1], bins, histtype="step", color="g", alpha=0.5)
        ax.hist(self.dataset.radii, bins, histtype="step", color="k",
                linestyle="dashed")
        ax.set_xlim(radius_range)
        ax.set_yticklabels([])
        ax.set_xscale("log")
        ax.set_xlabel(r"$R / R_\oplus$")
        ax.set_title("observed radius dist.")

        # Plot the inclination distribution.
        ax = axes[0, 1]
        sig = np.exp(self.theta[-1])
        x = np.linspace(0, 4*sig, 500)
        y = x * np.exp(-0.5 * x**2 / sig**2) / sig**2
        ax.plot(x, y, "g")
        ax.set_xlim(x[0], x[-1])
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(r"$p(\delta i)$")
        ax.set_xlabel(r"mutual inclination $\delta i$")
        ax.set_title("true mutual incl. dist.")

        # Plot the true multiplicity.
        ax = axes[1, 1]
        p = np.append(np.exp(pars[-self.simulator.nplanets+1:]), 0.0)
        p[-1] = 1.0 - np.sum(p[:-1])
        ax.plot(p, ".-g")
        ax.set_yscale("log")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("$p(N)$")
        ax.set_xlabel("number of planets $N$")
        ax.set_title("true multiplicity")

        # Plot the period distribution.
        ax = axes[2, 1]
        x = np.linspace(np.log10(period_range[0]), np.log10(period_range[1]),
                        500)
        y = broken_power_law(pars[3:6], period_range, 10**x)
        ax.plot(10**x, y, "g")
        ax.set_xlim(period_range)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("$p(P)$")
        ax.set_xlabel(r"$\log_{10} P / \mathrm{day}$")
        ax.set_title("true period dist.")

        # Plot the radius distribution.
        ax = axes[3, 1]
        x = np.linspace(np.log10(radius_range[0]), np.log10(radius_range[1]),
                        500)
        y = broken_power_law(pars[:3], radius_range, 10**x)
        ax.plot(10**x, y, "g")
        ax.set_xlim(radius_range)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("$p(R)$")
        ax.set_xlabel(r"$R / R_\oplus$")
        ax.set_title("true radius dist.")

        fig.subplots_adjust(hspace=0.8)

        return fig


class PowerLawMultiDist(object):

    def __len__(self):
        return 2

    def transform(self, theta, ntot):
        multi = (np.arange(ntot) + np.exp(theta[1])) ** theta[0]
        multi /= np.sum(multi)
        return multi

    def log_prior(self, theta):
        if not (-5 < theta[0] < 1):
            return -np.inf
        if not (-3 < theta[1] < 3):
            return -np.inf
        return 0.0


class PoissonMultiDist(object):

    def __len__(self):
        return 1

    def transform(self, theta, ntot):
        rate = np.exp(theta[0])
        n = np.arange(ntot)
        multi = rate ** n * np.exp(-rate) / factorial(n)
        multi /= np.sum(multi)
        return multi


class PoissonMixtureMultiDist(PoissonMultiDist):

    def __len__(self):
        return 2

    def transform(self, theta, ntot):
        multi = super(PoissonMixtureMultiDist, self).transform(theta[:1], ntot)
        multi *= np.exp(theta[1])
        multi[1] = 1.0 - multi[0] - np.sum(multi[2:])
        return multi


def logfactorial(n):
    # Ramanujan's formula
    return (n * np.log(n) - n + np.log(n * (1.0+4*n*(1+2*n))) / 6.0
            + 0.5*np.log(np.pi)) * (n > 1.0)


def broken_power_law(params, rng, x):
    xmin, xmax = rng
    a1, a2, x0 = params
    a11 = a1 + 1.0
    a21 = a2 + 1.0
    x0da = x0 ** (a2-a1)

    if a11 == 0.0:
        fmin1 = np.log(xmin)
        N1 = x0da*(np.log(x0)-fmin1)
    else:
        fmin1 = xmin ** a11
        N1 = x0da*(x0**a11-fmin1)/a11
    if a21 == 0.0:
        fmin2 = np.log(x0)
        N2 = np.log(xmax)-fmin2
    else:
        fmin2 = x0 ** a21
        N2 = (xmax**a21-fmin2)/a21
    N = N1 + N2

    return (x0da * x**a1 * (x < x0) + x**a2 * (x >= x0)) / N
