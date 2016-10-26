# -*- coding: utf-8 -*-

from __future__ import division, print_function

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

from .sim import DR24CompletenessModel
from .data import get_q1_q17_dr24_injections
from .plot_setup import setup, SQUARE_FIGSIZE, D3_COLORS

__all__ = ["calibrate_dr24_completness"]

def calibrate_dr24_completness(stlr, plot=None, **kwargs):
    # Get the injections table.
    inj = get_q1_q17_dr24_injections()

    # Join on the stellar sample.
    inj = pd.merge(inj, stlr[["kepid"]], on="kepid")

    # Extract the relevant quantities.
    x = np.array(inj.period)
    y = np.array(inj.expect_mes)
    z = 1.0 * np.array((inj.recovered == 1) & (inj.Disp == "PC"))

    # Build the period grid.
    period_bins = np.linspace(
        kwargs.get("period", (0, None))[0],
        kwargs.get("period", (None, 800))[1],
        kwargs.get("num_period_bins", 1) + 1,
    )
    params = np.array([(0.7, 7.1, 0.0) for _ in range(len(period_bins)-1)])
    params = params.flatten()

    # Optimize the parameters.
    model = DR24CompletenessModel()
    resid = lambda p: z - model.get_pdet(p, period_bins, x, y)
    params, _, info, msg, flag = leastsq(resid, params, full_output=True)
    if flag not in [1, 2, 3, 4]:
        logging.warn("completeness calibration failed with message: \n{0}"
                     .format(msg))

    # Plot the results.
    if plot is not None:
        setup()
        fig, ax = plt.subplots(1, 1, figsize=SQUARE_FIGSIZE)
        bins = np.linspace(2, 20, 25)
        y2 = np.linspace(2, 20, 1000)

        for i, c in enumerate(D3_COLORS[:len(period_bins)-1]):
            a, b = period_bins[i:i+2]
            x0 = 0.5 * (a + b)
            m = (a <= x) & (x < b)

            n_tot, _ = np.histogram(y[m], bins)
            n_rec, _ = np.histogram(y[m][z[m] > 0], bins)
            n = n_rec / n_tot
            ax.step(bins[1:], n, color=c)
            ax.plot(y2, model.get_pdet(params, period_bins,
                                       x0+np.zeros_like(y2), y2),
                    color=c,
                    label="${0:.0f} \le P < {1:.0f}$".format(a, b))
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.legend(loc=4, fontsize=12)
        ax.set_xlabel("MES")
        ax.set_ylabel("recovery rate")
        fig.savefig(plot, bbox_inches="tight")

    return period_bins, params
