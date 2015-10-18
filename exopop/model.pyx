# distutils: language = c++
from __future__ import division

cimport cython
from libc.math cimport exp
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

    cdef cppclass Simulation:
        Simulation (
            unsigned nstars, unsigned nplanets, double min_radius,
            double max_radius, double min_period, double max_period,
            unsigned seed
        )
        void clean_up ()
        Simulation* copy()
        void add_star (Star* star)
        vector[CatalogRow] observe (double* params, unsigned* counts, int* flag)

        # Re-sampling methods can perturb individual sets of parameters or all
        # of them.
        void resample ()
        void resample_multis()
        void resample_q1()
        void resample_q2()
        void resample_incls()
        void resample_radii()
        void resample_periods()
        void resample_eccens()
        void resample_omegas()
        void resample_mutual_incls()
        void resample_delta_incls()
        void resample_obs_randoms()

cdef extern from "completeness.h" namespace "abcsim":

    cdef cppclass CompletenessModel:
        CompletenessModel ()

    cdef cppclass Q1_Q16_CompletenessModel(CompletenessModel):
        Q1_Q16_CompletenessModel (double a, double b)

    cdef cppclass Q1_Q17_CompletenessModel(CompletenessModel):
        Q1_Q17_CompletenessModel (double m, double b, double mesthresh, double width)

    cdef cppclass Star:
        Star ()
        Star (
            CompletenessModel* completeness_model,
            double mass, double radius,
            double dataspan, double dutycycle,
            unsigned n_cdpp, const double* cdpp_x, const double* cdpp_y,
            unsigned n_thresh, const double* thresh_x, const double* thresh_y
        )


cdef class Simulator:
    """
    This class provides functionality for simulating a population of exoplanets
    around a set of Kepler targets. This uses the Burke et al. (2015)
    semi-analytic completeness model.

    """

    cdef unsigned nplanets
    cdef unsigned cached
    cdef object period_range
    cdef object radius_range
    cdef Simulation* simulator
    cdef Simulation* cached_simulator
    cdef CompletenessModel* completeness_model

    def __cinit__(self, stars, unsigned nplanets,
                  double min_radius, double max_radius,
                  double min_period, double max_period,
                  np.ndarray[DTYPE_t, ndim=1] completeness_params,
                  seed=None, int release=17):
        # Parse the completeness model.
        if release == 16:
            self.completeness_model = new Q1_Q16_CompletenessModel(
                completeness_params[0], completeness_params[1],
            )
        elif release == 17:
            self.completeness_model = new Q1_Q17_CompletenessModel(
                completeness_params[0], completeness_params[1],
                completeness_params[2], exp(completeness_params[3]),
            )
        else:
            raise ValueError("invalid 'release'")

        # Seed with the time if no seed is provided.
        if seed is None:
            seed = time.time()

        # Build the simulation.
        self.cached = 0
        self.nplanets = nplanets
        cdef unsigned nstars = len(stars)
        self.radius_range = (min_radius, max_radius)
        self.period_range = (min_period, max_period)
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
                self.completeness_model,
                star.mass, star.radius, star.dataspan, star.dutycycle,
                cdpp_x.shape[0], <double*>cdpp_x.data, <double*>cdpp_y.data,
                thr_x.shape[0], <double*>thr_x.data, <double*>thr_y.data,
            )
            self.simulator.add_star(starobj)

    def __dealloc__(self):
        if self.cached:
            del self.cached_simulator
        self.simulator.clean_up()
        del self.simulator
        del self.completeness_model

    property nplanets:
        def __get__(self):
            return self.nplanets

    property period_range:
        def __get__(self):
            return self.period_range

    property radius_range:
        def __get__(self):
            return self.radius_range

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
        cdef np.ndarray[DTYPE_u_t, ndim=1] starids = np.empty(catalog.size(),
                                                              dtype=DTYPE_u)
        cdef np.ndarray[DTYPE_t, ndim=2] cat_out = np.empty((catalog.size(), 2),
                                                            dtype=DTYPE)
        for i in range(cat_out.shape[0]):
            starids[i] = catalog[i].starid
            cat_out[i, 0] = catalog[i].period
            cat_out[i, 1] = catalog[i].radius

        return counts, starids, cat_out

    def resample(self):
        """
        Re-sample all the per-planet and per-star parameters from their priors.

        """
        self.simulator.resample()

    def revert(self):
        if self.cached:
            del self.simulator
            self.simulator = self.cached_simulator
            self.cached = 0

    def perturb(self):
        """
        Randomly re-sample one set of per-planet or per-star parameters (i.e.
        all the periods or radii or whatever) from the prior.

        """
        # First, cache the current simulation state for reverting.
        if self.cached:
            del self.cached_simulator
        self.cached_simulator = self.simulator.copy()
        self.cached = 1

        # Then select a parameter to update.
        cdef int ind = np.random.randint(11)
        if ind == 0:
            self.simulator.resample_multis()
        elif ind == 1:
            self.simulator.resample_q1()
        elif ind == 2:
            self.simulator.resample_q2()
        elif ind == 3:
            self.simulator.resample_incls()
        elif ind == 4:
            self.simulator.resample_radii()
        elif ind == 5:
            self.simulator.resample_periods()
        elif ind == 6:
            self.simulator.resample_eccens()
        elif ind == 7:
            self.simulator.resample_omegas()
        elif ind == 8:
            self.simulator.resample_mutual_incls()
        elif ind == 9:
            self.simulator.resample_delta_incls()
        elif ind == 10:
            self.simulator.resample_obs_randoms()
