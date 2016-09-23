# distutils: language = c++
from __future__ import division

cimport cython
from libc.math cimport exp
from libcpp.vector cimport vector
from libcpp.string cimport string

import time
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DTYPE_u = np.uint32
ctypedef np.uint32_t DTYPE_u_t


cdef extern from "boost/random.hpp" namespace "boost::random":

    cdef cppclass mt19937:
        pass


cdef extern from "exoabc/exoabc.h" namespace "exoabc":

    ctypedef mt19937 random_state_t

    # Distributions
    cdef cppclass Distribution:
        pass

    # Observation model
    cdef cppclass BaseStar:
        pass

    cdef cppclass Star[T]:
        Star (
            const T completeness_model,
            double mass, double radius,
            double dataspan, double dutycycle,
            unsigned n_cdpp, const double* cdpp_x, const double* cdpp_y,
            unsigned n_thresh, const double* thresh_x, const double* thresh_y
        )

    # Simulation
    cdef cppclass CompletenessModel:
        pass

    cdef cppclass Q1_Q16_CompletenessModel:
        pass

    cdef cppclass Q1_Q17_CompletenessModel:
        Q1_Q17_CompletenessModel (
            double qmax_m, double qmax_b,
            double mes0_m, double mes0_b,
            double lnw_m, double lnw_b
        )

    cdef cppclass CatalogRow:
        unsigned starid
        double period
        double radius
        double duration
        double depth

    cdef cppclass Simulation:
        Simulation (
            Distribution* period_distribution,
            Distribution* radius_distribution,
            Distribution* eccen_distribution,
            Distribution* width_distribution,
            Distribution* multi_distribution
        )

    #     void clean_up ()
    #     Simulation* copy()
    #     string get_state ()
    #     string get_state (unsigned seed)
    #     void set_state (string state)

    #     void add_star (Star* star)
    #     vector[CatalogRow] sample_population () nogil

# cdef extern from "completeness.h" namespace "exopop":

    # cdef cppclass CompletenessModel:
    #     CompletenessModel ()

    # cdef cppclass Q1_Q17_CompletenessModel(CompletenessModel):
    #     Q1_Q17_CompletenessModel (double, double, double, double, double, double)

    # cdef cppclass Star:
    #     Star ()
    #     Star (
    #         CompletenessModel* completeness_model,
    #         double mass, double radius,
    #         double dataspan, double dutycycle,
    #         unsigned n_cdpp, const double* cdpp_x, const double* cdpp_y,
    #         unsigned n_thresh, const double* thresh_x, const double* thresh_y
    #     )


cdef class Simulator:
    """
    This class provides functionality for simulating a population of exoplanets
    around a set of Kepler targets. This uses the Burke et al. (2015)
    semi-analytic completeness model.

    """

    # cdef unsigned nplanets
    # cdef unsigned cached
    # cdef object period_range
    # cdef object radius_range
    # cdef Simulation* simulator
    # cdef Simulation* cached_simulator
    cdef CompletenessModel* completeness_model

    def __cinit__(self, stars, release=None, completeness_params=None):
        # Figure out which completeness model to use
        if release is None:
            release = "q1_q16"
        if release == "q1_q16":
            self.completeness_model = new Q1_Q16_CompletenessModel()
        elif release == "q1_q17":
            completeness_params = np.atleast_1d(completeness_params)
            if not completeness_params.shape == (6, ):
                raise ValueError("completeness parameters dimension mismatch")
            self.completeness_model = new Q1_Q17_CompletenessModel(
                *(completeness_params)
            )
        else:
            raise ValueError("unrecognized release: '{0}'".format(release))

        # Set up the simulation distributions

    def __dealloc__(self):
        del self.completeness_model

    # def __cinit__(self, stars, unsigned nplanets,
    #               double min_radius, double max_radius,
    #               double min_period, double max_period,
    #               np.ndarray[DTYPE_t, ndim=1] completeness_params,
    #               seed=None):
    #     # Parse the completeness model.
    #     if completeness_params.shape[0] != 6:
    #         raise ValueError("dimension mismatch")
    #     self.completeness_model = new Q1_Q17_CompletenessModel(
    #         completeness_params[0], completeness_params[1],
    #         completeness_params[2], completeness_params[3],
    #         completeness_params[4], completeness_params[5],
    #     )

    #     # Seed with the time if no seed is provided.
    #     if seed is None:
    #         seed = time.time()

    #     # Build the simulation.
    #     self.cached = 0
    #     self.nplanets = nplanets
    #     cdef unsigned nstars = len(stars)
    #     self.radius_range = (min_radius, max_radius)
    #     self.period_range = (min_period, max_period)
    #     self.simulator = new Simulation(nstars, nplanets,
    #                                     min_radius, max_radius,
    #                                     min_period, max_period,
    #                                     seed)

    #     # Add the stars from the import catalog.
    #     cdef Star* starobj
    #     cdef np.ndarray[DTYPE_t, ndim=1] cdpp_x
    #     cdef np.ndarray[DTYPE_t, ndim=1] cdpp_y
    #     cdef np.ndarray[DTYPE_t, ndim=1] thr_x
    #     cdef np.ndarray[DTYPE_t, ndim=1] thr_y
    #     for _, star in stars.iterrows():
    #         # Pull out the CDPP values.
    #         cdpp_cols = [k for k in star.keys() if k.startswith("rrmscdpp")]
    #         cdpp_x = np.array([k[-4:].replace("p", ".") for k in cdpp_cols],
    #                           dtype=float)
    #         inds = np.argsort(cdpp_x)
    #         cdpp_x = np.ascontiguousarray(cdpp_x[inds], dtype=np.float64)
    #         cdpp_y = np.ascontiguousarray(star[cdpp_cols][inds],
    #                                       dtype=np.float64)

    #         # And the MES thresholds.
    #         thr_cols = [k for k in star.keys() if k.startswith("mesthres")]
    #         thr_x = np.array([k[-4:].replace("p", ".") for k in thr_cols],
    #                          dtype=float)
    #         inds = np.argsort(thr_x)
    #         thr_x = np.ascontiguousarray(thr_x[inds], dtype=np.float64)
    #         thr_y = np.ascontiguousarray(star[thr_cols][inds],
    #                                      dtype=np.float64)

    #         starobj = new Star(
    #             self.completeness_model,
    #             star.mass, star.radius, star.dataspan, star.dutycycle,
    #             cdpp_x.shape[0], <double*>cdpp_x.data, <double*>cdpp_y.data,
    #             thr_x.shape[0], <double*>thr_x.data, <double*>thr_y.data,
    #         )
    #         self.simulator.add_star(starobj)

    # def __dealloc__(self):
    #     if self.cached:
    #         del self.cached_simulator
    #     self.simulator.clean_up()
    #     del self.simulator
    #     del self.completeness_model

    # property nplanets:
    #     def __get__(self):
    #         return self.nplanets

    # property period_range:
    #     def __get__(self):
    #         return self.period_range

    # property radius_range:
    #     def __get__(self):
    #         return self.radius_range

    # def observe(self, np.ndarray[DTYPE_t, ndim=1] params, state=None):
    #     """
    #     Observe the current simulation for a given set of hyperparameters.
    #     The parameters are as follows:

    #     .. code-block:: python
    #         [radius_power1, radius_power2, radius_break,
    #          period_power1, period_power2, period_break,
    #          std_of_incl_distribution, ln_multiplicity...(nmax parameters)]

    #     :param state: (optional)
    #         The random state can be provided to ensure a specific catalog.

    #     """
    #     if params.shape[0] != 7 + self.nplanets - 1:
    #         raise ValueError("dimension mismatch")

    #     cdef np.ndarray[DTYPE_u_t, ndim=1] counts = np.empty(self.nplanets,
    #                                                          dtype=DTYPE_u)

    #     if state is not None:
    #         self.simulator.set_state(state)
    #     else:
    #         state = self.simulator.get_state()
    #     self.simulator.resample()

    #     cdef int flag
    #     cdef vector[CatalogRow] catalog
    #     cdef Simulation* sim = self.simulator
    #     cdef double* p = <double*>params.data
    #     cdef unsigned* c = <unsigned*>counts.data
    #     with nogil:
    #         catalog = sim.observe(p, c, &flag)

    #     if flag:
    #         raise RuntimeError("simulator failed with status {0}".format(flag))

    #     # Copy the catalog.
    #     cdef int i
    #     cdef np.ndarray[DTYPE_u_t, ndim=1] starids = np.empty(catalog.size(),
    #                                                           dtype=DTYPE_u)
    #     cdef np.ndarray[DTYPE_t, ndim=2] cat_out = np.empty((catalog.size(), 2),
    #                                                         dtype=DTYPE)
    #     for i in range(cat_out.shape[0]):
    #         starids[i] = catalog[i].starid
    #         cat_out[i, 0] = catalog[i].period
    #         cat_out[i, 1] = catalog[i].radius

    #     return counts, starids, cat_out, state

    # def resample(self):
    #     """
    #     Re-sample all the per-planet and per-star parameters from their priors.

    #     """
    #     self.simulator.resample()

    # def revert(self):
    #     if self.cached:
    #         del self.simulator
    #         self.simulator = self.cached_simulator
    #         self.cached = 0

    # def perturb(self):
    #     """
    #     Randomly re-sample one set of per-planet or per-star parameters (i.e.
    #     all the periods or radii or whatever) from the prior.

    #     """
    #     # First, cache the current simulation state for reverting.
    #     if self.cached:
    #         del self.cached_simulator
    #     self.cached_simulator = self.simulator.copy()
    #     self.cached = 1

    #     # Then select a parameter to update.
    #     cdef int ind = np.random.randint(11)
    #     if ind == 0:
    #         self.simulator.resample_multis()
    #     elif ind == 1:
    #         self.simulator.resample_q1()
    #     elif ind == 2:
    #         self.simulator.resample_q2()
    #     elif ind == 3:
    #         self.simulator.resample_incls()
    #     elif ind == 4:
    #         self.simulator.resample_radii()
    #     elif ind == 5:
    #         self.simulator.resample_periods()
    #     elif ind == 6:
    #         self.simulator.resample_eccens()
    #     elif ind == 7:
    #         self.simulator.resample_omegas()
    #     elif ind == 8:
    #         self.simulator.resample_mutual_incls()
    #     elif ind == 9:
    #         self.simulator.resample_delta_incls()
    #     elif ind == 10:
    #         self.simulator.resample_obs_randoms()

    # def get_state(self, seed=None):
    #     if seed is None:
    #         return self.simulator.get_state()
    #     return self.simulator.get_state(int(seed))

    # def set_state(self, bytes state):
    #     self.simulator.set_state(state)
