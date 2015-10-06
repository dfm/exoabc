#ifndef _ABC_SIMULATION_H_
#define _ABC_SIMULATION_H_

#include <cmath>
#include <cfloat>
#include <vector>
#include <boost/random.hpp>
#include <boost/math/distributions.hpp>

using std::vector;
using boost::math::cdf;
using boost::math::gamma_distribution;

namespace abcsim {

// G / (4 * pi)
#define GRAV_OVER_4_PI 74.6094376947028

// Earth radius in Solar radii
#define RADIUS_EARTH   0.009171

// Limb darkening depth parameters
#define DELTA_C        1.0874
#define DELTA_S        1.0187

struct CatalogRow {
    unsigned starid;
    double period;
    double radius;
};

class Star {
public:
    Star () : gamma_(4.65, 0.98) {};
    Star (
        double mass, double radius,
        double dataspan, double dutycycle,
        unsigned n_cdpp, const double* cdpp_x, const double* cdpp_y,
        unsigned n_thresh, const double* thresh_x, const double* thresh_y
    ) : mass_(mass), radius_(radius),
        dataspan_(dataspan), dutycycle_(dutycycle),
        timefactor_(dataspan*dutycycle),
        cdpp_x_(n_cdpp), cdpp_y_(n_cdpp),
        thresh_x_(n_thresh), thresh_y_(n_thresh),
        gamma_(4.65, 0.98)
    {
        for (unsigned i = 0; i < n_cdpp; ++i) {
            cdpp_x_[i] = cdpp_x[i];
            cdpp_y_[i] = cdpp_y[i];
        }
        for (unsigned i = 0; i < n_thresh; ++i) {
            thresh_x_[i] = thresh_x[i];
            thresh_y_[i] = thresh_y[i];
        }
    };

    //
    // Estimate the approximate expected transit depth as a function
    // of radius ratio. There might be a typo here. In the paper it
    // uses c + s*k but in the public code, it is c - s*k:
    // https://github.com/christopherburke/KeplerPORTs
    //
    // :param k: the dimensionless radius ratio between the planet and
    //           the star
    //
    double get_delta (double k) const {
        return 0.84 * k * k * (DELTA_C + DELTA_S * k);
    };

    //
    // Estimate the multiple event statistic value for a transit.
    //
    // :param period: the period in days
    // :param rp:     the planet radius in Earth radii
    // :param tau:    the transit duration in hours
    //
    double get_mes (double period, double rp, double tau) const {
        double sigma = interp1d(tau, cdpp_x_, cdpp_y_),
               k = rp * RADIUS_EARTH / radius_,
               snr = get_delta(k) * 1.0e6 / sigma,
               ntrn = timefactor_ / period;
        return snr * sqrt(ntrn);
    };

    //
    // Compute the semi-major axis of an orbit in Solar radii.
    //
    // :param period: the period in days
    //
    double get_aor (double period) const {
        return pow(GRAV_OVER_4_PI * period * period * mass_, 1./3) / radius_;
    };

    //
    // Equation (1) from Burke et al. This estimates the transit
    // duration in the same units as the input period. There is a
    // typo in the paper (24/4 = 6 != 4).
    //
    // :param period: the period in any units of your choosing
    // :param aor:    the dimensionless semi-major axis (scaled
    //                by the stellar radius)
    // :param e:      the eccentricity of the orbit
    //
    double get_duration (double aor, double period, double e) const {
        return 0.25 * period * sqrt(1 - e * e) / aor;
    };

    //
    // Equation (5) from Burke et al. Estimate the detection efficiency
    // for a transit.
    //
    // :param star:   a pandas row giving the stellar properties
    // :param aor:    the dimensionless semi-major axis (scaled
    //                by the stellar radius)
    // :param period: the period in days
    // :param rp:     the planet radius in Earth radii
    // :param e:      the orbital eccentricity
    //
    double get_pdet (double aor, double period, double rp, double e) const {
        double tau = 24.0 * get_duration(aor, period, e),
               mes = get_mes(period, rp, tau),
               mest = interp1d(tau, thresh_x_, thresh_y_),
               x = mes - 4.1 - (mest - 7.1);
        if (x <= 0.0) return 0.0;
        return cdf(gamma_, x);
    };

    //
    // Equation (6) from Burke et al. Estimates the window function
    // using a binomial distribution.
    //
    // :param period: the period in days
    //
    double get_pwin (double period) const {
        double M = dataspan_ / period;
        if (M < 2.0) return 0.0;

        double f = dutycycle_,
               omf = 1.0 - f,
               pw = 1 - pow(omf, M) - M*f*pow(omf, M-1)
                    - 0.5*M*(M-1)*f*f*pow(omf, M-2);
        if (pw < 0.0) return 0.0;
        return pw;
    };

    //
    // A helper function to combine all the completeness effects.
    //
    // :param star:      a pandas row giving the stellar properties
    // :param period:    the period in days
    // :param rp:        the planet radius in Earth radii
    // :param e:         the orbital eccentricity
    //
    double get_completeness (double aor, double period, double rp, double e) const {
        return get_pwin(period) * get_pdet(aor, period, rp, e);
    };

private:
    double mass_, radius_, dataspan_, dutycycle_, timefactor_;
    vector<double> cdpp_x_, cdpp_y_, thresh_x_, thresh_y_;
    gamma_distribution<double> gamma_;

    double interp1d (double x0, const vector<double>& x, const vector<double>& y) const {
        unsigned n = x.size();
        if (x0 <= x[0]) return y[0];
        for (unsigned i = 0; i < n-1; ++i)
            if (x[i] < x0 && x0 <= x[i+1])
                return (y[i+1]-y[i])/(x[i+1]-x[i])*(x0-x[i]) + y[i];
        return y[n-1];
    };
};

class Simulation {
public:
    Simulation (unsigned nstars, unsigned nplanets,
                double min_radius, double max_radius,
                double min_period, double max_period,
                unsigned seed)
    : min_radius_(min_radius), max_radius_(max_radius),
      min_period_(min_period), max_period_(max_period),
      nstars_(nstars), nplanets_(nplanets), ntot_(nstars * nplanets),
      multi_randoms_(nstars),
      incl_randoms_(nstars),
      radius_randoms_(nstars*nplanets),
      period_randoms_(nstars*nplanets),
      eccen_randoms_(nstars*nplanets),
      omega_randoms_(nstars*nplanets),
      delta_incl_randoms_(nstars*nplanets),
      stars_(nstars), rng_(seed) {
        resample_multis();
        resample_incls();
        resample_radii();
        resample_periods();
        resample_eccens();
        resample_omegas();
        resample_delta_incls();
    };
    void add_star (Star star) {
        stars_.push_back(star);
    };

    // Re-sample the underlying parameters.
    void resample_multis () {
        for (unsigned i = 0; i < nstars_; ++i)
            multi_randoms_[i] = uniform_generator_(rng_);
    };
    void resample_incls () {
        for (unsigned i = 0; i < nstars_; ++i)
            incl_randoms_[i] = acos(uniform_generator_(rng_));
    };
    void resample_radii () {
        for (unsigned i = 0; i < ntot_; ++i)
            radius_randoms_[i] = uniform_generator_(rng_);
    };
    void resample_periods () {
        for (unsigned i = 0; i < ntot_; ++i)
            period_randoms_[i] = uniform_generator_(rng_);
    };
    void resample_eccens () {
        for (unsigned i = 0; i < ntot_; ++i)
            eccen_randoms_[i] = uniform_generator_(rng_);
    };
    void resample_omegas () {
        for (unsigned i = 0; i < ntot_; ++i)
            omega_randoms_[i] = M_PI * uniform_generator_(rng_);
    };
    void resample_delta_incls () {
        for (unsigned i = 0; i < ntot_; ++i)
            delta_incl_randoms_[i] = normal_generator_(rng_);
    };

    // Observe
    vector<CatalogRow> observe (double* params, unsigned* counts, int* flag) {
        *flag = 0;

        // Initialize to zero.
        unsigned i, j, n, count;
        for (i = 0; i < nplanets_; ++i) counts[i] = 0;

        // Initialize the catalog.
        vector<CatalogRow> catalog;

        // Unpack the parameters.
        vector<double> multi_params(nplanets_ - 1);
        double radius, period, incl, omega, eccen = 0.0, aor, Q, factor, r, b,
               prob, norm = 0.0,
               radius_power1 = params[0],
               radius_power2 = params[1],
               radius_break = params[2],
               period_power = params[3],
               incl_sigma = params[4];

        // Get the multiplicity parameters and fail if they are invalid.
        for (i = 0; i < nplanets_-1; ++i) {
            multi_params[i] = exp(params[5+i]);
            norm += multi_params[i];
        }
        if (norm > 1.0) { *flag = 1; return catalog; }

        for (i = 0; i < nstars_; ++i) {
            Star star = stars_[i];
            count = 0;
            prob = 0.0;
            for (j = 0; j < nplanets_; ++j) {
                // Impose the multiplicity constraint.
                prob += multi_params[j];
                if (prob >= multi_randoms_[i]) break;

                n = i*nplanets_ + j;

                // Samples
                radius = broken_power_law(radius_power1, radius_power2,
                                          radius_break, min_radius_,
                                          max_radius_, radius_randoms_[n]);
                period = power_law(period_power, min_period_, max_period_,
                                   period_randoms_[n]);
                incl = incl_randoms_[i] + incl_sigma * delta_incl_randoms_[n];
                omega = omega_randoms_[n];

                // Completeness model
                aor = star.get_aor(period);

                // Geometry
                factor = (1 - eccen * eccen) / (1 + eccen * sin(omega));
                b = std::abs(aor * cos(incl) * factor);
                if (b < 1.0) {
                    Q = star.get_completeness(aor, period, radius, eccen);
                    r = uniform_generator_(rng_);
                    if (r <= Q) {
                        CatalogRow row = {i, period, radius};
                        catalog.push_back(row);
                        count += 1;

                    }
                }
            }
            counts[count] += 1;
        }

        return catalog;
    };

private:
    double min_radius_, max_radius_, min_period_, max_period_;
    unsigned nstars_, nplanets_, ntot_;
    vector<double> multi_randoms_,
                   incl_randoms_,
                   radius_randoms_,
                   period_randoms_,
                   eccen_randoms_,
                   omega_randoms_,
                   delta_incl_randoms_;
    vector<Star> stars_;
    boost::random::mt19937 rng_;
    boost::random::uniform_01<> uniform_generator_;
    boost::random::normal_distribution<> normal_generator_;

    double power_law (double n, double mn, double mx, double u) const {
        if (fabs(n + 1.0) < DBL_EPSILON) {
            double lnmn = log(mn);
            return mn * exp(u * (log(mx) - lnmn));
        }
        double np1 = n+1.0,
               x0n = pow(mn, np1);
        return pow((pow(mx, np1) - x0n) * u + x0n, 1.0 / np1);
    };

    double broken_power_law (double a1, double a2, double x0,
                             double xmin, double xmax, double u) const {
        double a11 = a1 + 1.0,
               a21 = a2 + 1.0,
               x0da = pow(x0, a2-a1),
               fmin1, fmin2,
               N1, N2, N;

        if (fabs(a11) < DBL_EPSILON) {
            fmin1 = log(xmin);
            N1 = x0da*(log(x0)-fmin1);
        } else {
            fmin1 = pow(xmin, a11);
            N1 = x0da*(pow(x0, a11)-fmin1)/a11;
        }
        if (fabs(a21) < DBL_EPSILON) {
            fmin2 = log(x0);
            N2 = log(xmax)-fmin2;
        } else {
            fmin2 = pow(x0, a21);
            N2 = (pow(xmax, a21)-fmin2)/a21;
        }
        N = N1 + N2;

        // Low x
        if (u <= N1 / N) {
            if (fabs(a11) < DBL_EPSILON) return xmin*exp(u*N/x0da);
            return pow(a11*u*N/x0da + fmin1, 1.0/a11);
        }

        // High x
        u -= N1 / N;
        if (fabs(a21) < DBL_EPSILON) return xmin*exp(u*N);
        return pow(a21*u*N + fmin2, 1.0/a21);
    };
};

};

#endif  // _ABC_SIMULATION_H_
