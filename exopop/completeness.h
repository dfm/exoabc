#ifndef _ABC_COMPLETENESS_H_
#define _ABC_COMPLETENESS_H_

#include <cmath>
#include <cfloat>
#include <vector>
#include <boost/math/distributions.hpp>

#include "quad.h"

using std::vector;
using boost::math::cdf;
using boost::math::gamma_distribution;

using transit::QuadraticLimbDarkening;

namespace abcsim {

// G / (4 * pi)
#define GRAV_OVER_4_PI 74.6094376947028

// Earth radius in Solar radii
#define RADIUS_EARTH   0.009171

class CompletenessModel {
public:
    CompletenessModel () {};
    virtual ~CompletenessModel () {};

    virtual double get_mes (double period, double depth, double sigma, double total_time) const {
        double snr = depth / sigma,
               ntrn = total_time / period;
        return snr * sqrt(ntrn);
    };

    virtual double get_pdet (double mes, double mesthresh) const {
        if (mes >= mesthresh) return 1.0;
        return 0.0;
    };

    virtual double get_pwin (double period, double dataspan, double dutycycle) const {
        double M = dataspan / period;
        if (M <= 2.0) return 0.0;
        double f = dutycycle,
               omf = 1.0 - f,
               pw = 1 - pow(omf, M) - M*f*pow(omf, M-1) - 0.5*M*(M-1)*f*f*pow(omf, M-2);
        if (pw < 0.0) return 0.0;
        return pw;
    };
};

class Q1_Q16_CompletenessModel : public CompletenessModel {
public:
    Q1_Q16_CompletenessModel (double a=4.65, double b=0.98) : gamma_(a, b) {};

    double get_mes (double period, double depth, double sigma, double total_time) const {
        double mean_depth = 0.84 * depth,
               snr = mean_depth / sigma,
               ntrn = total_time / period;
        return snr * sqrt(ntrn);
    };

    double get_pdet (double mes, double mesthresh) const {
        double x = mes - 4.1 - (mesthresh - 7.1);
        if (x <= 0.0) return 0.0;
        return cdf(gamma_, x);
    };

private:
    gamma_distribution<double> gamma_;
};

class Q1_Q17_CompletenessModel : public CompletenessModel {
public:
    Q1_Q17_CompletenessModel (double m, double b, double mesthresh=15.0, double width=0.5)
        : m_(m), b_(b), mesthresh_(mesthresh), width_(width) {};

    double get_mes (double period, double depth, double sigma, double total_time) const {
        double snr = depth / sigma,
               ntrn = total_time / period;
        return snr * sqrt(ntrn);
    };

    double get_pdet (double mes, double nothing) const {
        double y = (m_ * mes + b_) / (1.0 + exp(-(mes-mesthresh_)/width_));
        if (y > 1.0) return 1.0;
        else if (y <= DBL_EPSILON) return 0.0;
        return y;
    };

private:
    double m_, b_, mesthresh_, width_;
};


class Star {
public:
    Star (
        CompletenessModel* completeness_model,
        double mass, double radius,
        double dataspan, double dutycycle,
        unsigned n_cdpp, const double* cdpp_x, const double* cdpp_y,
        unsigned n_thresh, const double* thresh_x, const double* thresh_y
    ) : mass_(mass), radius_(radius),
        dataspan_(dataspan), dutycycle_(dutycycle),
        timefactor_(dataspan*dutycycle),
        cdpp_x_(n_cdpp), cdpp_y_(n_cdpp),
        thresh_x_(n_thresh), thresh_y_(n_thresh),
        completeness_model_(completeness_model)
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

    double get_mass () const { return mass_; };

    double get_aor (double period) const {
        return pow(GRAV_OVER_4_PI * period * period * mass_, 1./3) / radius_;
    };

    double get_impact (double aor, double incl, double e, double omega) const {
        return std::abs(aor * cos(incl) * (1.0 - e * e) / (1.0 + e * sin(omega)));
    };

    double get_duration (double aor, double period, double ror, double b, double incl, double e, double omega) const {
        double duration, opr, arg, b2 = b*b;
        opr = 1.0 + ror;
        opr *= opr;
        if (b2 >= opr) return 0.0;
        arg = sqrt(opr - b2) / aor / sin(incl);
        duration = period / M_PI * asin(arg);
        if (e > DBL_EPSILON)
            duration *= sqrt(1.0 - e*e) / (1.0 + e * sin(omega));
        return duration;
    }

    double get_maximum_depth (double q1, double q2, double ror, double b) const {
        double params[2] = {q1, q2};
        QuadraticLimbDarkening ld;
        return 1.0e6 * (1.0 - ld(params, ror, b));
    };

    double get_completeness (double q1, double q2, double period, double rp,
                             double incl, double e, double omega) const {
        double b, aor, ror, tau, depth, sigma, mest, mes, pdet, pwin;

        // Compute the duration; it will be zero if there is no transit.
        aor = get_aor (period);
        b = get_impact (aor, incl, e, omega);
        ror = rp * RADIUS_EARTH / radius_;
        tau = 24.0 * get_duration(aor, period, ror, b, incl, e, omega);
        if (tau <= DBL_EPSILON) return 0.0;

        // Get the depth.
        depth = get_maximum_depth(q1, q2, ror, b);
        if (depth <= DBL_EPSILON) return 0.0;

        // Interpolate to get the CDPP and MES threshold at the correct duration.
        sigma = interp1d(tau, cdpp_x_, cdpp_y_);
        mest = interp1d(tau, thresh_x_, thresh_y_);

        // Compute the MES detection efficiency.
        mes = completeness_model_->get_mes(period, depth, sigma, timefactor_);
        pdet = completeness_model_->get_pdet(mes, mest);

        // Get the window function.
        pwin = completeness_model_->get_pwin(period, dataspan_, dutycycle_);

        return pdet * pwin;
    };


private:
    double mass_, radius_, dataspan_, dutycycle_, timefactor_;
    vector<double> cdpp_x_, cdpp_y_, thresh_x_, thresh_y_;
    CompletenessModel* completeness_model_;

    double interp1d (double x0, const vector<double>& x, const vector<double>& y) const {
        unsigned n = x.size();
        if (x0 <= x[0]) return y[0];
        for (unsigned i = 0; i < n-1; ++i)
            if (x[i] < x0 && x0 <= x[i+1])
                return (y[i+1]-y[i])/(x[i+1]-x[i])*(x0-x[i]) + y[i];
        return y[n-1];
    };
};

};

#endif  // _ABC_COMPLETENESS_H_
