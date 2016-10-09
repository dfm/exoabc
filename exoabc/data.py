# -*- coding: utf-8 -*-

import os
import requests
import numpy as np
import pandas as pd
from io import BytesIO  # Python 3 only!
import matplotlib.pyplot as pl
from scipy.optimize import leastsq

from .sim import DR24CompletenessModel

__all__ = ["get_catalog", "get_burke_gk", "get_candidates",
           "compute_multiplicity", "calibrate_completeness"]


def get_catalog(name, prefix="q1_q16", basepath=None):
    """
    Download a catalog from the Exoplanet Archive by name and save it as a
    Pandas HDF5 file.

    :param name:     the table name
    :param basepath: the directory where the downloaded files should be saved
                     (default: ``data`` in the current working directory)

    """
    if basepath is None:
        basepath = os.environ.get("EXOABC_DATA", "data")

    basepath = os.path.join(basepath, prefix)
    fn = os.path.join(basepath, "{0}.h5".format(name))
    if os.path.exists(fn):
        return pd.read_hdf(fn, name)
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    fullname = prefix+"_"+name
    print("Downloading {0}...".format(fullname))
    url = ("http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/"
           "nph-nstedAPI?table={0}&select=*").format(fullname)
    r = requests.get(url)
    if r.status_code != requests.codes.ok:
        r.raise_for_status()
    fh = BytesIO(r.content)
    df = pd.read_csv(fh)
    df.to_hdf(fn, name, format="t")
    return df


def get_burke_gk(**kwargs):
    """
    Get the stellar catalog of G/K dwarfs as selected by Burke et al. (2015).
    The output is a Pandas data frame with all the columns from the ``stellar``
    tables on the Exoplanet Archive.

    """
    stlr = get_catalog("stellar", **kwargs)

    # Select G & K dwarfs.
    m = (4200 <= stlr.teff) & (stlr.teff <= 6100)
    m &= stlr.radius <= 1.15

    # Only include stars with sufficient data coverage.
    m &= stlr.rrmscdpp07p5 <= 1000.
    m &= stlr.dataspan > 365.25*2.
    m &= stlr.dutycycle > 0.3

    # Only select stars with mass & radius estimates.
    m &= np.isfinite(stlr.mass) & np.isfinite(stlr.radius)
    return pd.DataFrame(stlr[m])


def get_ballard_m(**kwargs):
    """
    Get the stellar catalog of M dwarfs as selected by Ballard & Johnson
    (2014). The output is a Pandas data frame with all the columns from the
    ``stellar`` tables on the Exoplanet Archive.

    """
    stlr = get_catalog("stellar", **kwargs)

    # Select M dwarfs.
    m = (3950 <= stlr.teff) & (stlr.teff <= 4200)
    m &= stlr.radius <= 1.15

    # Only include stars with sufficient data coverage.
    m &= stlr.dataspan > 365.25*2.
    m &= stlr.dutycycle > 0.3

    # Only select stars with mass & radius estimates.
    m &= np.isfinite(stlr.mass) & np.isfinite(stlr.radius)
    return pd.DataFrame(stlr[m])


def get_candidates(stlr=None, mesthresh=None, period_range=None,
                   radius_range=None, **kwargs):
    """
    Get the Q1-Q17 candidates from the KOI table on the Exoplanet Archive.

    :param stlr:         optionally join on a specific stellar sample
    :param mesthresh:    the target MES threshold
    :param period_range: restrict to a range of periods
    :param radius_range: restrict to a range of radii

    """
    kois = get_catalog("koi", **kwargs)

    # Join on the stellar list.
    if stlr is not None:
        kois = pd.merge(kois, stlr[["kepid"]], on="kepid", how="inner")

    # Only select the KOIs in the relevant part of parameter space.
    m = kois.koi_pdisposition == "CANDIDATE"
    m &= np.isfinite(kois.koi_prad)
    if period_range is not None:
        m &= period_range[0] < kois.koi_period
        m &= kois.koi_period < period_range[1]
    if radius_range is not None:
        m &= radius_range[0] < kois.koi_prad
        m &= kois.koi_prad < radius_range[1]

    # Apply the MES thresholding.
    if mesthresh is not None:
        m &= np.isfinite(kois.koi_max_mult_ev)
        m &= (kois.koi_max_mult_ev > mesthresh)

    return pd.DataFrame(kois[m])


def compute_multiplicity(nstars, kois, nmax=None):
    """
    Compute the observed multiplicity distribution of a candidate list. This
    returns an array of integer counts. The first element is the number of
    systems with zero observed transiting planets, the second gives the number
    of singles, and so on.

    :param nstars: the total number of stars in the stellar sample
    :param kois:   the Pandas data frame of candidates
    :param nmax:   the maximum number of planets to allow per system; the
                   distribution will be padded with zeros if this is larger
                   than the maximum observed multiplicity

    """
    # Compute the counts.
    unique_kois = kois.groupby("kepid")[["koi_pdisposition"]].count()
    counts = np.array(unique_kois.groupby("koi_pdisposition")
                      .koi_pdisposition.count())

    # Build the bins.
    if nmax is None:
        data = np.zeros(len(counts) + 1, dtype=int)
    else:
        data = np.zeros(nmax, dtype=int)

    # Don't forget the "zero" bin!
    data[0] = nstars - len(unique_kois)
    n = min(len(data), len(counts) - 1)
    data[1:1+n] = counts[:n]
    return data


def calibrate_completeness(stlr, mesthresh=15.0, mesmax=50.0,
                           period_range=None, plot=False):
    """
    Calibrate the completeness model using injections and given a stellar
    catalog.

    :param stlr:      optionally join on a specific stellar sample
    :param mesthresh: the target MES threshold

    """
    # Load the injection catalog.
    names = [
        "kepid", "sky", "period", "epoch", "t_depth", "t_dur", "t_b", "t_ror",
        "t_aor", "offset_from_source", "offset_distance", "expect_mes",
        "recovered", "meas_mes", "r_period", "r_epoch", "r_depth", "r_dur",
        "r_b", "r_ror", "r_aor"
    ]
    inj = pd.read_csv(os.path.join("data", "q1_q17_dr24", "injections.txt"),
                      delim_whitespace=True, skiprows=4, header=None,
                      names=names, na_values="null")
    m = inj.offset_from_source == 0
    if period_range is not None:
        m &= period_range[0] < inj.period
        m &= inj.period < period_range[1]
    inj = pd.DataFrame(inj[m])

    # Restrict to the stellar sample.
    inj = pd.merge(inj, stlr[["kepid"]], on="kepid", how="inner")

    y = np.array(inj.expect_mes, dtype=float)
    m = y < mesmax
    y = y[m]
    x = np.array(inj.period, dtype=float)[m]
    z = np.array(inj.recovered, dtype=float)[m]
    z[np.array(inj.meas_mes[m] < mesthresh)] = 0.0

    # Compute the weights (prior) model.
    N, X, Y = np.histogram2d(x, y, (22, 23))
    inds_x = np.clip(np.digitize(x, X) - 1, 0, len(X) - 2)
    inds_y = np.clip(np.digitize(y, Y) - 1, 0, len(Y) - 2)
    w = np.sqrt(N[inds_x, inds_y])

    # Fit the completeness model.
    p0 = np.array([0.0, 0.7, 0.0, mesthresh, 0.0, 0.0])
    completeness_model = DR24CompletenessModel()
    resid = lambda p: (z - completeness_model.get_pdet(p, x, y)) / w
    params, _, info, msg, flag = leastsq(resid, p0, full_output=True)
    print(msg)
    if flag not in [1, 2, 3, 4]:
        print("Warning: completeness calibration did not converge. Message:")
        print(msg)

    if not plot:
        return params

    fig = pl.figure()
    ax = fig.add_subplot(111)
    q = np.percentile(x, [25, 50, 75])
    b = np.linspace(0, 3*mesthresh, 15)
    y2 = np.linspace(0, 3*mesthresh, 1000)
    for mn, mx, c in zip(np.append(x.min(), q), np.append(q, x.max()), "rgbk"):
        m = (mn <= x) & (x < mx)
        n_tot, _ = np.histogram(y[m], b)
        n_rec, _ = np.histogram(y[m][z[m] > 0], b)
        n = n_rec / n_tot
        ax.errorbar(0.5*(b[:-1] + b[1:]), n, yerr=n / np.sqrt(n_rec), fmt=".",
                    color=c)
        z2 = completeness_model.get_pdet(params, 0.5*(mn+mx)+np.zeros_like(y2),
                                         y2)
        ax.plot(y2, z2, color=c)
    ax.set_xlim(0, 3*mesthresh)
    ax.set_xlabel("expected MES")
    ax.set_ylabel("completeness")
    return params, fig
