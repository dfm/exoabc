#ifndef _ABC_SIMULATION_H_
#define _ABC_SIMULATION_H_

#include <cmath>
#include <cfloat>
#include <vector>
#include <string>
#include <sstream>
#include <boost/random.hpp>
#include <boost/math/distributions.hpp>
#include <boost/random/beta_distribution.hpp>

#include "completeness.h"

using std::vector;
using std::string;
using boost::math::cdf;

using exopop::Star;

namespace abcsim {

struct CatalogRow {
    unsigned starid;
    double period;
    double radius;
};

class Simulation {
public:
    Simulation (unsigned nstars, unsigned nplanets,
                double min_radius, double max_radius,
                double min_period, double max_period,
                unsigned seed, double alpha=0.867, double beta=3.03)
    : min_radius_(min_radius), max_radius_(max_radius),
      min_period_(min_period), max_period_(max_period),
      nstars_(nstars), nplanets_(nplanets), ntot_(nstars * nplanets),
      multi_randoms_(nstars),
      q1_randoms_(nstars),
      q2_randoms_(nstars),
      incl_randoms_(nstars),
      radius_randoms_(nstars*nplanets),
      period_randoms_(nstars*nplanets),
      eccen_randoms_(nstars*nplanets),
      omega_randoms_(nstars*nplanets),
      mutual_incl_randoms_(nstars),
      delta_incl_randoms_(nstars*nplanets),
      obs_randoms_(nstars*nplanets),
      stars_(), rng_(seed),
      beta_generator_(alpha, beta)
    {
        resample();
    };
    ~Simulation () {};
    void clean_up () {
        for (unsigned i = 0; i < stars_.size(); ++i) delete stars_[i];
    };
    void add_star (Star* star) {
        stars_.push_back(star);
    };
    Simulation* copy () {
        return new Simulation(*this);
    };

    string get_state () const {
        std::stringstream ss;
        ss << rng_;
        return ss.str();
    };

    string get_state (unsigned seed) const {
        std::stringstream ss;
        boost::random::mt19937 gen(seed);
        ss << gen;
        return ss.str();
    };

    void set_state (string state) {
        std::stringstream ss(state);
        ss >> rng_;
        uniform_generator_.reset();
        normal_generator_.reset();
        beta_generator_.reset();
    };

    // Re-sample the underlying parameters.
    void resample () {
        resample_multis();
        resample_q1();
        resample_q2();
        resample_incls();
        resample_radii();
        resample_periods();
        resample_eccens();
        resample_omegas();
        resample_mutual_incls();
        resample_delta_incls();
        resample_obs_randoms();
    };
    void resample_multis () {
        for (unsigned i = 0; i < nstars_; ++i)
            multi_randoms_[i] = uniform_generator_(rng_);
    };
    void resample_q1 () {
        for (unsigned i = 0; i < nstars_; ++i)
            q1_randoms_[i] = uniform_generator_(rng_);
    };
    void resample_q2 () {
        for (unsigned i = 0; i < nstars_; ++i)
            q2_randoms_[i] = uniform_generator_(rng_);
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
            eccen_randoms_[i] = beta_generator_(rng_);
    };
    void resample_omegas () {
        for (unsigned i = 0; i < ntot_; ++i)
            omega_randoms_[i] = M_PI * uniform_generator_(rng_);
    };
    void resample_mutual_incls () {
        for (unsigned i = 0; i < nstars_; ++i)
            mutual_incl_randoms_[i] = uniform_generator_(rng_);
    };
    void resample_delta_incls () {
        for (unsigned i = 0; i < ntot_; ++i)
            delta_incl_randoms_[i] = normal_generator_(rng_);
    };
    void resample_obs_randoms () {
        for (unsigned i = 0; i < ntot_; ++i)
            obs_randoms_[i] = uniform_generator_(rng_);
    };

    vector<CatalogRow> observe (double* params, unsigned* counts, int* flag) {
        *flag = 0;

        // Initialize to zero.
        unsigned i, j, n, count;
        for (i = 0; i < nplanets_; ++i) counts[i] = 0;

        // Initialize the catalog.
        vector<CatalogRow> catalog;

        // Unpack the parameters.
        vector<double> multi_params(nplanets_ - 1);
        double radius, period, incl, omega, eccen, Q, r,
               prob, norm = 0.0, mutual_incl, q1, q2,
               radius_power1 = params[0],
               radius_power2 = params[1],
               radius_break = params[2],
               period_power1 = params[3],
               period_power2 = params[4],
               period_break = params[5],
               incl_sigma = params[6];

        // Check that the break locations are in the correct ranges.
        if (radius_break < min_radius_ || max_radius_ < radius_break ||
                period_break < min_period_ || max_period_ < period_break) {
            *flag = 2;
            return catalog;
        }

        // Get the multiplicity parameters and fail if they are invalid.
        for (i = 0; i < nplanets_-1; ++i) {
            multi_params[i] = exp(params[7+i]);
            norm += multi_params[i];
        }
        if (norm > 1.0) { *flag = 1; return catalog; }

        for (i = 0; i < nstars_; ++i) {
            Star* star = stars_[i];
            count = 0;
            prob = 0.0;

            // The mutual inclination is drawn from a Rayleigh distribution.
            mutual_incl = incl_sigma * sqrt(-2*log(mutual_incl_randoms_[i]));

            // The limb darkening coefficients are drawn from a uniform distribution.
            q1 = q1_randoms_[i];
            q2 = q2_randoms_[i];

            for (j = 0; j < nplanets_; ++j) {
                // Impose the multiplicity constraint.
                prob += multi_params[j];
                if (prob >= multi_randoms_[i]) break;

                n = i*nplanets_ + j;

                // Samples
                radius = broken_power_law(radius_power1, radius_power2,
                                          radius_break, min_radius_,
                                          max_radius_, radius_randoms_[n]);
                period = broken_power_law(period_power1, period_power2,
                                          period_break, min_period_,
                                          max_period_, period_randoms_[n]);
                incl = incl_randoms_[i] + mutual_incl * delta_incl_randoms_[n];
                eccen = eccen_randoms_[n];
                omega = omega_randoms_[n];

                // Compute the completeness.
                Q = star->get_completeness(q1, q2, period, radius, incl, eccen, omega);
                r = obs_randoms_[n];
                if (r <= Q) {
                    CatalogRow row = {i, period, radius};
                    catalog.push_back(row);
                    count += 1;
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
                   q1_randoms_,
                   q2_randoms_,
                   incl_randoms_,
                   radius_randoms_,
                   period_randoms_,
                   eccen_randoms_,
                   omega_randoms_,
                   mutual_incl_randoms_,
                   delta_incl_randoms_,
                   obs_randoms_;
    vector<Star*> stars_;
    boost::random::mt19937 rng_;
    boost::random::uniform_01<> uniform_generator_;
    boost::random::normal_distribution<> normal_generator_;
    boost::random::beta_distribution<> beta_generator_;

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
