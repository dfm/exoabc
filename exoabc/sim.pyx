# distutils: language = c++
from __future__ import division

cimport cython
from libc.math cimport exp, log
from libcpp.vector cimport vector
from libcpp.string cimport string

import time
import numpy as np
cimport numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda f, *args, **kwargs: f

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
        double log_pdf(double x)
    cdef cppclass Uniform(Distribution):
        Uniform(double, double)
    cdef cppclass Delta:
        Delta(double)
    cdef cppclass Parameter(BaseParameter):
        Parameter(double)
        Parameter(string, Distribution*)
        Parameter(string, Distribution*, random_state_t)
        Parameter(string, Distribution*, double)
        Parameter(string, Distribution*, vector[double])
    cdef cppclass PowerLaw(Distribution):
        PowerLaw(double, double, BaseParameter*)
    cdef cppclass BrokenPowerLaw(Distribution):
        BrokenPowerLaw(double, double, BaseParameter*, BaseParameter*, BaseParameter*)
    cdef cppclass Normal(Distribution):
        Normal(BaseParameter*, BaseParameter*)
    cdef cppclass Beta(Distribution):
        Beta(BaseParameter*, BaseParameter*)
    cdef cppclass Rayleigh(Distribution):
        Rayleigh(BaseParameter*)
    cdef cppclass Multinomial(Distribution):
        Multinomial(BaseParameter*)
    cdef cppclass Poisson(Distribution):
        Poisson(BaseParameter*)
    cdef cppclass Dirichlet(Distribution):
        Dirichlet(size_t n)

    # Observation model
    cdef cppclass CompletenessModel:
        pass
    cdef cppclass Q1_Q16_CompletenessModel(CompletenessModel):
        pass
    cdef cppclass Q1_Q17_CompletenessModel(CompletenessModel):
        Q1_Q17_CompletenessModel (
            vector[double] period_bin_edges,
            vector[double] qmax,
            vector[double] mes0,
            vector[double] lnw
        )
        double get_pdet (double period, double mes, double mest)
    cdef cppclass Star:
        Star (
            const CompletenessModel* completeness_model,
            double ln_mass, double sig_ln_mass, double ln_radius, double sig_ln_radius,
            double dataspan, double dutycycle,
            unsigned n_cdpp, const double* cdpp_x, const double* cdpp_y,
            unsigned n_thresh, const double* thresh_x, const double* thresh_y
        )

    # Simulation
    cdef cppclass CatalogRow:
        long unsigned starid
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
        double evaluate_multiplicity (double n)
        double evaluate_period_distribution (double period)
        double evaluate_radius_distribution (double radius)
        double mean_multiplicity ()


cdef class DR24CompletenessModel:

    def get_pdet(self,
                 np.ndarray[DTYPE_t, ndim=1, mode='c'] params,
                 np.ndarray[DTYPE_t, ndim=1, mode='c'] period_bin_edges,
                 np.ndarray[DTYPE_t, ndim=1, mode='c'] period,
                 np.ndarray[DTYPE_t, ndim=1, mode='c'] mes):
        # Check the shapes.
        cdef int n = period.shape[0]
        if n != mes.shape[0]:
            raise ValueError("dimension mismatch (period/mes)")

        cdef int bin_edges = len(period_bin_edges)
        cdef int bins = bin_edges - 1

        # Build the completeness model.
        if params.shape[0] != bins * 3:
            raise ValueError("dimension mismatch (params)")

        cdef int i
        cdef vector[double] period_bin_edges_v = vector[double](bin_edges)
        cdef vector[double] qmax = vector[double](bins)
        cdef vector[double] mes0 = vector[double](bins)
        cdef vector[double] lnw = vector[double](bins)
        for i in range(bins):
            period_bin_edges_v[i] = period_bin_edges[i]
            qmax[i] = params[3*i]
            mes0[i] = params[3*i+1]
            lnw[i] = params[3*i+2]
        period_bin_edges_v[bins] = period_bin_edges[bins]

        # Build the completeness model
        cdef Q1_Q17_CompletenessModel* model = new Q1_Q17_CompletenessModel(
            period_bin_edges_v, qmax, mes0, lnw
        )

        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] output = np.empty(n, dtype=DTYPE)
        for i in range(n):
            output[i] = model.get_pdet(period[i], mes[i], 0.0)
        del model
        return output


cdef class Simulator:
    """
    This class provides functionality for simulating a population of exoplanets
    around a set of Kepler targets. This uses the Burke et al. (2015)
    semi-analytic completeness model.

    """

    cdef random_state_t     state
    cdef Simulation*        simulation
    cdef CompletenessModel* completeness_model

    @classmethod
    def from_config_args(cls, stars, seed=None, release=None,
                         completeness_params=None, **kwargs):
        min_period, max_period = kwargs.pop("period")
        min_period_slope, max_period_slope = kwargs.pop("period_slope")
        period_slope = np.random.uniform(min_period_slope, max_period_slope)

        min_radius, max_radius = kwargs.pop("radius")
        min_radius_slope, max_radius_slope = kwargs.pop("radius_slope")
        radius_slope = np.random.uniform(min_period_slope, max_period_slope)

        min_log_sigma, max_log_sigma = kwargs.pop("log_sigma")
        log_sigma = np.random.uniform(min_log_sigma, max_log_sigma)

        if kwargs["poisson"]:
            kwargs.pop("maxn")
            min_log_multi, max_log_multi = kwargs.pop("log_mu")
            log_multi_params = np.random.uniform(min_log_multi, max_log_multi)
        else:
            kwargs.pop("log_mu")
            min_log_multi, max_log_multi = 0.0, 0.0
            maxn = kwargs.pop("maxn")
            log_multi_params = -np.log(maxn) + np.zeros(maxn+1)

        return cls(
            stars,
            min_period, max_period, period_slope,
            min_radius, max_radius, radius_slope,
            log_sigma, log_multi_params,
            min_period_slope, max_period_slope,
            min_radius_slope, max_radius_slope,
            min_log_sigma, max_log_sigma,
            min_log_multi, max_log_multi,
            seed=seed, release=release,
            completeness_params=completeness_params,
            **kwargs
        )

    def __cinit__(self, stars,
                  double min_period, double max_period, double period_slope,
                  double min_radius, double max_radius, double radius_slope,
                  double log_sigma, log_multi_params,
                  double min_period_slope=-4.0, double max_period_slope=3.0,
                  double min_radius_slope=-4.0, double max_radius_slope=3.0,
                  double min_log_sigma=-5.0, double max_log_sigma=1.0,
                  double min_log_multi=-8.0, double max_log_multi=8.0,
                  poisson=False, broken_radius=False,
                  eccen_params=(0.867, 3.03),
                  seed=None, release=None,
                  completeness_params=None,
                  period_bin_edges=None):
        # Set up the random state
        cdef unsigned iseed
        if seed is None:
            iseed = int(time.time())
        else:
            iseed = int(seed)
        self.state = random_state_t(iseed)

        cdef int i, bin_edges, bins
        cdef vector[double] period_bin_edges_v, qmax,  mes0, lnw

        # Figure out which completeness model to use
        if release is None:
            release = "q1_q16"
        if release == "q1_q16":
            self.completeness_model = new Q1_Q16_CompletenessModel()
        elif release == "q1_q17_dr24":
            bin_edges = len(period_bin_edges)
            bins = bin_edges - 1

            # Build the completeness model.
            if completeness_params.shape[0] != bins * 3:
                raise ValueError("completeness parameters dimension mismatch")

            period_bin_edges_v = vector[double](bin_edges)
            qmax = vector[double](bins)
            mes0 = vector[double](bins)
            lnw = vector[double](bins)
            for i in range(bins):
                period_bin_edges_v[i] = period_bin_edges[i]
                qmax[i] = completeness_params[3*i]
                mes0[i] = completeness_params[3*i+1]
                lnw[i] = completeness_params[3*i+2]
            period_bin_edges_v[bins] = period_bin_edges[bins]

            # Build the completeness model
            self.completeness_model = new Q1_Q17_CompletenessModel(
                period_bin_edges_v, qmax, mes0, lnw
            )
        else:
            raise ValueError("unrecognized release: '{0}'".format(release))

        # Set up the simulation distributions
        cdef string name = b"period_slope"
        cdef PowerLaw* period = new PowerLaw(
            min_period, max_period,
            new Parameter(name,
                          new Uniform(min_period_slope, max_period_slope),
                          period_slope)
        )
        cdef Distribution* radius
        if broken_radius:
            name = b"radius"
            radius = new BrokenPowerLaw(
                min_radius, max_radius,
                new Parameter(name,
                              new Uniform(min_radius, max_radius),
                              0.5 * (min_radius + max_radius)),
                new Parameter(name,
                              new Uniform(min_radius_slope, max_radius_slope),
                              radius_slope),
                new Parameter(name,
                              new Uniform(min_radius_slope, max_radius_slope),
                              radius_slope),
            )
        else:
            name = b"radius_slope"
            radius = new PowerLaw(
                min_radius, max_radius,
                new Parameter(name,
                              new Uniform(min_radius_slope, max_radius_slope),
                              radius_slope)
            )

        cdef Beta* eccen = new Beta(new Parameter(log(eccen_params[0])),
                                    new Parameter(log(eccen_params[1])))
        name = b"log_width"
        cdef Rayleigh* width = new Rayleigh(
            new Parameter(name,
                          new Uniform(min_log_sigma, max_log_sigma), log_sigma)
        )

        # Multiplicity
        cdef np.ndarray[DTYPE_t, ndim=1] lmp = np.atleast_1d(log_multi_params)
        cdef Distribution* multi
        cdef Parameter* par
        cdef double v
        cdef vector[double] vec

        if poisson:
            par = new Parameter(name, new Uniform(min_log_multi,
                                                  max_log_multi), lmp[0])
            multi = new Poisson(par)
        else:
            vec.resize(len(lmp))
            for i in range(len(lmp)):
                vec[i] = np.exp(lmp[i])
            par = new Parameter(name, new Dirichlet(len(lmp)), vec)
            multi = new Multinomial(par)

        # Build the simulator
        self.simulation = new Simulation(period, radius, eccen, width, multi)

        # Add in the stars from the catalog
        cdef Star* starobj
        cdef int dim
        cdef np.ndarray[DTYPE_t, ndim=1] cdpp_x, cdpp_y
        cdef np.ndarray[DTYPE_t, ndim=2] cdpp_y0
        cdef np.ndarray[DTYPE_t, ndim=1] thr_x, thr_y
        cdef np.ndarray[DTYPE_t, ndim=2] thr_y0

        # Which columns have the cdpp
        cdpp_cols = np.array([k for k in stars.keys()
                              if k.startswith("rrmscdpp")])
        cdpp_x = np.array([k[-4:].replace("p", ".") for k in cdpp_cols], dtype=float)
        cdpp_inds = np.argsort(cdpp_x)
        cdpp_x = np.ascontiguousarray(cdpp_x[cdpp_inds], dtype=np.float64)
        cdpp_cols = cdpp_cols[cdpp_inds]
        cdpp_y0 = np.ascontiguousarray(np.array(stars[cdpp_cols]),
                                       dtype=np.float64)

        # And the thresholds
        thr_cols = np.array([k for k in stars.keys()
                             if k.startswith("mesthres")])
        thr_x = np.array([k[-4:].replace("p", ".") for k in thr_cols], dtype=float)
        thr_inds = np.argsort(thr_x)
        thr_x = np.ascontiguousarray(thr_x[thr_inds], dtype=np.float64)
        thr_cols = thr_cols[thr_inds]
        thr_y0 = np.ascontiguousarray(np.array(stars[thr_cols]),
                                      dtype=np.float64)

        cdef double mn, mx, sig_m, sig_r
        for i, (_, star) in tqdm(enumerate(stars.iterrows()),
                                 total=len(stars)):
            # Work out uncertainties.
            mx = log(star.mass + star.mass_err1)
            mn = log(star.mass + star.mass_err2)
            sig_m = 0.5 * (mx - mn)
            mx = log(star.radius + star.radius_err1)
            mn = log(star.radius + star.radius_err2)
            sig_r = 0.5 * (mx - mn)

            # Pull out the CDPP data.
            thr_y = thr_y0[i]
            cdpp_y = cdpp_y0[i]

            # Put the star together
            starobj = new Star(
                self.completeness_model,
                log(star.mass), sig_m, log(star.radius), sig_m,
                star.dataspan, star.dutycycle,
                cdpp_x.shape[0],
                <double*>cdpp_x.data,
                <double*>cdpp_y.data,
                thr_x.shape[0],
                <double*>thr_x.data,
                <double*>thr_y.data,
            )
            self.simulation.add_star(starobj)

    def __dealloc__(self):
        del self.simulation
        del self.completeness_model

    property state:
        def __get__(self):
            return serialize_state(self.state)

        def __set__(self, value):
            cdef string blob = value
            self.state = deserialize_state(blob)

    def evaluate_multiplicity(self, np.ndarray[DTYPE_t, ndim=1] n):
        cdef int i
        cdef np.ndarray[DTYPE_t, ndim=1] m = np.empty(n.shape[0])
        for i in range(n.shape[0]):
            m[i] = self.simulation.evaluate_multiplicity(n[i])
        return m

    def mean_multiplicity(self):
        return self.simulation.mean_multiplicity()

    def evaluate_period_distribution(self, np.ndarray[DTYPE_t, ndim=1] x):
        cdef int i
        cdef np.ndarray[DTYPE_t, ndim=1] d = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            d[i] = self.simulation.evaluate_period_distribution(x[i])
        return d

    def evaluate_radius_distribution(self, np.ndarray[DTYPE_t, ndim=1] x):
        cdef int i
        cdef np.ndarray[DTYPE_t, ndim=1] d = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            d[i] = self.simulation.evaluate_radius_distribution(x[i])
        return d

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
        return pd.DataFrame.from_records(result)

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
