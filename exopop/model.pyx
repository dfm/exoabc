# distutils: language = c++
from __future__ import division

cimport cython
from libcpp.vector cimport vector

import time
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DTYPE_u = np.uint32
ctypedef np.uint32_t DTYPE_u_t

cdef extern from "simulation.h" namespace "abcsim":

    cdef cppclass CatalogRow:
        unsigned starid
        double period
        double radius

    cdef cppclass Star:
        Star ()
        Star (
            double mass, double radius,
            double dataspan, double dutycycle,
            unsigned n_cdpp, const double* cdpp_x, const double* cdpp_y,
            unsigned n_thresh, const double* thresh_x, const double* thresh_y
        )

    cdef cppclass Simulation:
        Simulation (
            unsigned nstars, unsigned nplanets, double min_radius,
            double max_radius, double min_period, double max_period,
            unsigned seed
        )
        void add_star (Star* star)
        void resample_radii ()
        void resample_periods ()
        void resample_eccens ()
        void resample_incls ()
        void resample_delta_incls ()
        void resample_omegas ()
        vector[CatalogRow] observe (double* params, unsigned* counts, int* flag)


cdef class Simulator:
    """
    This class provides functionality for simulating a population of exoplanets
    around a set of Kepler targets. This uses the Burke et al. (2015)
    semi-analytic completeness model.

    """

    cdef unsigned nplanets
    cdef Simulation* simulator

    def __cinit__(self, stars, unsigned nplanets,
                  double min_radius, double max_radius,
                  double min_period, double max_period,
                  seed=None):
        # Seed with the time if no seed is provided.
        if seed is None:
            seed = time.time()

        # Build the simulation.
        self.nplanets = nplanets
        cdef unsigned nstars = len(stars)
        self.simulator = new Simulation(nstars, nplanets,
                                        min_radius, max_radius,
                                        min_period, max_period,
                                        seed)

        # Add the stars from the import catalog.
        cdef Star* starobj
        cdef np.ndarray[DTYPE_t, ndim=1] cdpp_x
        cdef np.ndarray[DTYPE_t, ndim=1] cdpp_y
        cdef np.ndarray[DTYPE_t, ndim=1] thr_x
        cdef np.ndarray[DTYPE_t, ndim=1] thr_y
        for _, star in stars.iterrows():
            # Pull out the CDPP values.
            cdpp_cols = [k for k in star.keys() if k.startswith("rrmscdpp")]
            cdpp_x = np.array([k[-4:].replace("p", ".") for k in cdpp_cols],
                              dtype=float)
            inds = np.argsort(cdpp_x)
            cdpp_x = np.ascontiguousarray(cdpp_x[inds], dtype=np.float64)
            cdpp_y = np.ascontiguousarray(star[cdpp_cols][inds],
                                          dtype=np.float64)

            # And the MES thresholds.
            thr_cols = [k for k in star.keys() if k.startswith("mesthres")]
            thr_x = np.array([k[-4:].replace("p", ".") for k in thr_cols],
                             dtype=float)
            inds = np.argsort(thr_x)
            thr_x = np.ascontiguousarray(thr_x[inds], dtype=np.float64)
            thr_y = np.ascontiguousarray(star[thr_cols][inds],
                                         dtype=np.float64)

            starobj = new Star(
                star.mass, star.radius, star.dataspan, star.dutycycle,
                cdpp_x.shape[0], <double*>cdpp_x.data, <double*>cdpp_y.data,
                thr_x.shape[0], <double*>thr_x.data, <double*>thr_y.data,
            )
            self.simulator.add_star(starobj)

    def __dealloc__(self):
        del self.simulator

    def observe(self, np.ndarray[DTYPE_t, ndim=1] params):
        """
        Observe the current simulation for a given set of hyperparameters.
        The parameters are as follows:

        .. code-block:: python
            [radius_power1, radius_power2, radius_break,
             period_power1, period_power2, period_break,
             std_of_incl_distribution, ln_multiplicity...(nmax parameters)]

        """
        if params.shape[0] != 7 + self.nplanets - 1:
            raise ValueError("dimension mismatch")

        cdef np.ndarray[DTYPE_u_t, ndim=1] counts = np.empty(self.nplanets,
                                                             dtype=DTYPE_u)

        cdef int flag
        cdef vector[CatalogRow] catalog = \
            self.simulator.observe(<double*>params.data,
                                   <unsigned*>counts.data,
                                   &flag)
        if flag:
            raise RuntimeError("simulator failed with status {0}".format(flag))

        # Copy the catalog.
        cdef int i
        cdef np.ndarray[DTYPE_t, ndim=2] cat_out = np.empty((catalog.size(), 2),
                                                            dtype=DTYPE)
        for i in range(cat_out.shape[0]):
            cat_out[i, 0] = catalog[i].period
            cat_out[i, 1] = catalog[i].radius

        return counts, cat_out
