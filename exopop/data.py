# -*- coding: utf-8 -*-

__all__ = ["get_catalog", "get_burke_gk", "get_candidates",
           "compute_multiplicity"]

import os
import requests
import numpy as np
import pandas as pd
from io import BytesIO  # Python 3 only!


def get_catalog(name, basepath="data"):
    """
    Download a catalog from the Exoplanet Archive by name and save it as a
    Pandas HDF5 file.

    :param name:     the table name
    :param basepath: the directory where the downloaded files should be saved
                     (default: ``data`` in the current working directory)

    """
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


def get_burke_gk(**kwargs):
    """
    Get the stellar catalog of G/K dwarfs as selected by Burke et al. (2015).
    The output is a Pandas data frame with all the columns from the ``stellar``
    tables on the Exoplanet Archive.

    """
    stlr = get_catalog("q1_q16_stellar", **kwargs)

    # Select G & K dwarfs.
    m = (4200 <= stlr.teff) & (stlr.teff <= 6100)
    m &= stlr.radius <= 1.15

    # Only include stars with sufficient data coverage.
    m &= stlr.dataspan > 365.25*2.
    m &= stlr.dutycycle > 0.6
    m &= stlr.rrmscdpp07p5 <= 1000.

    # Only select stars with mass & radius estimates.
    m &= np.isfinite(stlr.mass) & np.isfinite(stlr.radius)
    return pd.DataFrame(stlr[m])


def get_candidates(stlr=None, period_range=None, radius_range=None, **kwargs):
    """
    Get the Q1-Q16 candidates from the KOI table on the Exoplanet Archive.

    :param stlr:         optionally join on a specific stellar sample
    :param period_range: restrict to a range of periods
    :param radius_range: restrict to a range of radii

    """
    kois = get_catalog("q1_q16_koi", **kwargs)

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
