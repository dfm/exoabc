# -*- coding: utf-8 -*-

import os
import requests
import numpy as np
import pandas as pd
from io import BytesIO  # Python 3 only!

__all__ = ["get_catalog", "get_sample", "get_q1_q17_dr24_injections"]


def get_q1_q17_dr24_injections(basepath=None, clobber=False):
    if basepath is None:
        basepath = os.environ.get("EXOABC_DATA", "data")
    fn = os.path.join(basepath, "q1_q17_dr24_injection_robovetter_join.csv")
    if os.path.exists(fn) and not clobber:
        return pd.read_csv(fn)
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    url = ("https://zenodo.org/record/163405/files/"
           "q1_q17_dr24_injection_robovetter_join.csv")
    r = requests.get(url)
    if r.status_code != requests.codes.ok:
        r.raise_for_status()
    fh = BytesIO(r.content)
    df = pd.read_csv(fh)
    df.to_csv(fn, index=False)
    return df


def get_catalog(name, prefix="q1_q16", basepath=None, clobber=False):
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
    if os.path.exists(fn) and not clobber:
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


def get_sample(name, prefix="q1_q16", basepath=None, clobber=False, join=None,
               join_on="kepid", **kwargs):
    df = get_catalog(name, prefix=prefix, basepath=basepath, clobber=clobber)

    if join is not None:
        df = pd.merge(df, join, on=join_on, how="inner")

    m = np.ones(len(df), dtype=bool)
    for column, value in kwargs.items():
        # Special values and strings:
        if value == "finite":
            m &= np.isfinite(df[column])
            continue
        if isinstance(value, str):
            m &= df[column] == value
            continue

        # Single values:
        try:
            len(value)
        except TypeError:
            m &= df[column] == value
            continue

        # Ranges:
        if len(value) != 2:
            raise ValueError("unrecognized argument: {0} = {1}".format(
                column, value
            ))

        m &= np.isfinite(df[column])
        if value[0] is not None:
            m &= value[0] <= df[column]
        if value[1] is not None:
            m &= df[column] < value[1]

    return pd.DataFrame(df[m])
