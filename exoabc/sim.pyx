# distutils: language = c++
from __future__ import division

cimport cython
from libc.math cimport exp, log
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
        mt19937()
        mt19937(unsigned seed)


cdef extern from "exoabc/exoabc.h" namespace "exoabc":

    ctypedef mt19937 random_state_t
    cdef string serialize_state (const random_state_t s)
    cdef random_state_t deserialize_state (const string s)

    # Distributions
    cdef cppclass BaseParameter:
        pass
    cdef cppclass Distribution:
        pass
    cdef cppclass Uniform(Distribution):
        Uniform(double, double)
    cdef cppclass Delta:
        Delta(double)
    cdef cppclass Parameter(BaseParameter):
        Parameter(double)
        Parameter(Distribution*)
        Parameter(Distribution*, random_state_t)
        Parameter(Distribution*, double)
    cdef cppclass PowerLaw(Distribution):
        PowerLaw(double, double, BaseParameter*)
    cdef cppclass Normal(Distribution):
        Normal(BaseParameter*, BaseParameter*)
    cdef cppclass Beta(Distribution):
        Beta(BaseParameter*, BaseParameter*)
    cdef cppclass Rayleigh(Distribution):
        Rayleigh(BaseParameter*)
    cdef cppclass Multinomial(Distribution):
        Multinomial(BaseParameter*)
        void add_bin(Parameter*)

    # Observation model
    cdef cppclass CompletenessModel:
        pass
    cdef cppclass Q1_Q16_CompletenessModel(CompletenessModel):
        pass
    cdef cppclass Q1_Q17_CompletenessModel(CompletenessModel):
        Q1_Q17_CompletenessModel (
            double qmax_m, double qmax_b,
            double mes0_m, double mes0_b,
            double lnw_m, double lnw_b
        )
    cdef cppclass Star:
        Star (
            const CompletenessModel* completeness_model,
            double mass, double radius, double dataspan, double dutycycle,
            unsigned n_cdpp, const double* cdpp_x, const double* cdpp_y,
            unsigned n_thresh, const double* thresh_x, const double* thresh_y
        )

    # Simulation
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
        size_t size ()
        void add_star (Star* star)
        void sample_parameters (random_state_t) nogil
        vector[CatalogRow] sample_population (random_state_t) nogil
        void get_parameter_values (double* params)
        double set_parameter_values (const double* params)
        double log_pdf ()


cdef class Simulator:
    """
    This class provides functionality for simulating a population of exoplanets
    around a set of Kepler targets. This uses the Burke et al. (2015)
    semi-analytic completeness model.

    """

    cdef random_state_t     state
    cdef Simulation*        simulation
    cdef CompletenessModel* completeness_model

    def __cinit__(self, stars,
                  double min_period, double max_period, double period_slope,
                  double min_radius, double max_radius, double radius_slope,
                  double log_sigma, log_multi_params,
                  double min_period_slope=-4.0, double max_period_slope=3.0,
                  double min_radius_slope=-4.0, double max_radius_slope=3.0,
                  double min_log_sigma=-10.0, double max_log_sigma=1.0,
                  double min_log_multi=-5.0, double max_log_multi=2.0,
                  eccen_params=(0.867, 3.03),
                  seed=None, release=None, completeness_params=None):
        # Set up the random state
        cdef unsigned iseed
        if seed is None:
            iseed = int(time.time())
        else:
            iseed = int(seed)
        self.state = random_state_t(iseed)

        # Figure out which completeness model to use
        if release is None:
            release = "q1_q16"
        if release == "q1_q16":
            self.completeness_model = new Q1_Q16_CompletenessModel()
        elif release == "q1_q17_dr24":
            completeness_params = np.atleast_1d(completeness_params)
            if not completeness_params.shape == (6, ):
                raise ValueError("completeness parameters dimension mismatch")
            self.completeness_model = new Q1_Q17_CompletenessModel(
                completeness_params[0], completeness_params[1],
                completeness_params[2], completeness_params[3],
                completeness_params[4], completeness_params[5],
            )
        else:
            raise ValueError("unrecognized release: '{0}'".format(release))

        # Set up the simulation distributions
        cdef PowerLaw* period = new PowerLaw(
            min_period, max_period,
            new Parameter(new Uniform(min_period_slope, max_period_slope),
                          period_slope)
        )
        cdef PowerLaw* radius = new PowerLaw(
            min_radius, max_radius,
            new Parameter(new Uniform(min_radius_slope, max_radius_slope),
                          radius_slope)
        )
        cdef Beta* eccen = new Beta(new Parameter(log(eccen_params[0])),
                                    new Parameter(log(eccen_params[1])))
        cdef Rayleigh* width = new Rayleigh(
            new Parameter(new Uniform(min_log_sigma, max_log_sigma), log_sigma)
        )
        cdef np.ndarray[DTYPE_t, ndim=1] lmp = np.atleast_1d(log_multi_params)
        cdef Multinomial* multi = new Multinomial(new Parameter(0.0))
        cdef Parameter* par
        cdef double v
        for v in log_multi_params:
            par = new Parameter(new Uniform(min_log_multi, max_log_multi), v)
            multi.add_bin(par)

        # Build the simulator
        self.simulation = new Simulation(period, radius, eccen, width, multi)

        # Add in the stars from the catalog
        cdef Star* starobj
        cdef np.ndarray[DTYPE_t, ndim=1] cdpp_x
        cdef np.ndarray[DTYPE_t, ndim=1] cdpp_y
        cdef np.ndarray[DTYPE_t, ndim=1] thr_x
        cdef np.ndarray[DTYPE_t, ndim=1] thr_y

        # Which columns have the cdpp
        cdpp_cols = [k for k in stars.keys() if k.startswith("rrmscdpp")]
        cdpp_x = np.array([k[-4:].replace("p", ".") for k in cdpp_cols], dtype=float)
        inds = np.argsort(cdpp_x)

        # And the thresholds
        thr_cols = [k for k in stars.keys() if k.startswith("mesthres")]
        thr_x = np.array([k[-4:].replace("p", ".") for k in thr_cols], dtype=float)
        inds = np.argsort(thr_x)

        for _, star in stars.iterrows():
            # Pull out the CDPP values.
            cdpp_x = np.ascontiguousarray(cdpp_x[inds], dtype=np.float64)
            cdpp_y = np.ascontiguousarray(star[cdpp_cols][inds],
                                          dtype=np.float64)

            # And the MES thresholds.
            thr_x = np.ascontiguousarray(thr_x[inds], dtype=np.float64)
            thr_y = np.ascontiguousarray(star[thr_cols][inds],
                                         dtype=np.float64)

            # Put the star together
            starobj = new Star(
                self.completeness_model,
                star.mass, star.radius, star.dataspan, star.dutycycle,
                cdpp_x.shape[0], <double*>cdpp_x.data, <double*>cdpp_y.data,
                thr_x.shape[0], <double*>thr_x.data, <double*>thr_y.data,
            )
            self.simulation.add_star(starobj)

    def __dealloc__(self):
        del self.simulation
        del self.completeness_model

    def sample_parameters(self):
        # Run the simulation
        cdef random_state_t state
        with nogil:
            self.simulation.sample_parameters(self.state)
        return self.simulation.log_pdf()

    def sample_population(self):
        # Run the simulation
        cdef vector[CatalogRow] catalog
        with nogil:
            catalog = self.simulation.sample_population(self.state)

        # Convert the simulation to a numpy array
        result = np.empty(catalog.size(), dtype=[
            ("kepid", int), ("koi_period", float), ("koi_prad", float),
            ("koi_duration", float), ("koi_depth", float)
        ])
        cdef int i
        for i in range(catalog.size()):
            result["kepid"][i] = catalog[i].starid
            result["koi_period"][i] = catalog[i].period
            result["koi_prad"][i] = catalog[i].radius
            result["koi_duration"][i] = catalog[i].duration
            result["koi_depth"][i] = catalog[i].depth
        return result

    def get_parameters(self):
        cdef np.ndarray[DTYPE_t, ndim=1] params = np.empty(self.simulation.size())
        self.simulation.get_parameter_values(<double*>params.data)
        return params

    def set_parameters(self, value):
        cdef np.ndarray[DTYPE_t, ndim=1] params = np.atleast_1d(value)
        if params.shape[0] != self.simulation.size():
            raise ValueError("dimension mismatch")
        return self.simulation.set_parameter_values(<double*>params.data)

    def log_pdf(self):
        return self.simulation.log_pdf()

    property state:
        def __get__(self):
            return serialize_state(self.state)

        def __set__(self, value):
            cdef string blob = value
            self.state = deserialize_state(blob)
