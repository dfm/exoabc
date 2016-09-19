#ifndef _EXOABC_DISTRIBUTIONS_
#define _EXOABC_DISTRIBUTIONS_

#include <cmath>
#include <cfloat>
#include <boost/math/distributions.hpp>
#include "exoabc/parameters.h"

namespace exoabc {

class Distribution {
public:
  virtual double scale_random (double u) const {
    return 0.0;
  };
  virtual double sample (random_state_t& state) {
    return this->scale_random(rng_(state));
  };
  size_t size () const { return 0; };

protected:
  boost::random::uniform_01<> rng_;
};

class Uniform : public Distribution {
public:
  Uniform (double mn, double mx) : mn_(mn), mx_(mx) {};
  double sample (random_state_t& state) {
    double u = rng_(state);
    return mn_ + (mx_ - mn_) * u;
  };

private:
  double mn_, mx_;
  boost::random::uniform_01<> rng_;
};


class Delta : public Distribution {
public:
  Delta (double value) : value_(value) {};
  size_t size () const { return 0; };
  double scale_random (double u) {
    return value_;
  };
  double sample (random_state_t& state) {
    return value_;
  };

private:
  double value_;
};




template <typename T1>
class PowerLaw : public Distribution {
public:
  PowerLaw (double mn, double mx, Parameter<T1>& n) : mn_(mn), mx_(mx), n_(n) {};
  size_t size () const { return 1; };
  double scale_random (double u) const {
    double n = n_.value();
    if (fabs(n + 1.0) < DBL_EPSILON) {
      double lnmn = log(mn_);
      return mn_ * exp(u * (log(mx_) - lnmn));
    }
    double np1 = n+1.0,
           x0n = pow(mn_, np1);
    return pow((pow(mx_, np1) - x0n) * u + x0n, 1.0 / np1);
  };

private:
  double mn_, mx_;
  Parameter<T1> n_;
};


template <typename T1, typename T2>
class Normal : public Distribution {
public:
  Normal (Parameter<T1>& mu, Parameter<T2>& log_sig) : mu_(mu), log_sig_(log_sig) {};
  size_t size () const { return 2; };
  double sample (random_state_t& state) {
    return mu_.value() + exp(log_sig_.value()) * rng_(state);
  };

private:
  Parameter<T1> mu_;
  Parameter<T2> log_sig_;
};


template <typename T1, typename T2>
class Beta : public Distribution {
public:
  Beta (Parameter<T1>& log_a, Parameter<T2>& log_b) : log_a_(log_a), log_b_(log_b) {};
  size_t size () const { return 2; };
  double sample (random_state_t& state) {
    rng_ = boost::random::beta_distribution<>(exp(log_a_.value()), exp(log_b_.value()));
    return rng_(state);
  };

private:
  boost::random::beta_distribution<> rng_;
  Parameter<T1> log_a_;
  Parameter<T2> log_b_;
};

template <typename T1>
class Rayleigh : public Distribution {
public:
  Rayleigh (Parameter<T1>& log_sig) : log_sig_(log_sig) {};
  size_t size () const { return 1; };
  double scale_random (double u) const {
    return sqrt(-2.0 * exp(2.0 * log_sig_.value()) * log(1.0 - u));
  };

private:
  Parameter<T1> log_sig_;
};

}; // namespace exoabc

#endif  // _EXOABC_DISTRIBUTIONS_
