# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

__all__ = ["mh"]

def mh(log_p_func, theta0, niter, sigma=0.1):
    ndim = len(theta0)
    theta = np.array(theta0)
    chain = np.empty((niter, ndim))
    lp = log_p_func(theta0)
    acc = 0
    for i in range(niter):
        q = np.array(theta)
        ind = np.random.randint(ndim)
        q[ind] += sigma * np.random.randn()
        lq = log_p_func(q)

        u = np.log(np.random.rand())
        if u < lq - lp:
            theta = q
            lp = lq
            acc += 1

        chain[i] = theta

    return chain, acc / niter
