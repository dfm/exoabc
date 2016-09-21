#ifndef _EXOABC_DISTRIBUTIONS_
#define _EXOABC_DISTRIBUTIONS_

#include <cmath>
#include <cfloat>
#include <vector>
#include <boost/math/distributions.hpp>

#include "exoabc/parameter.h"

namespace exoabc {

class Distribution {
public:
  virtual double scale_random (double u) const {
    return 0.0;
  };
  virtual double sample (random_state_t& state) {
    return this->scale_random(rng_(state));
  };
  size_t size () const { return parameters_.size(); };
  std::vector<BaseParameter*> parameters () { return parameters_; };

protected:
  boost::random::uniform_01<> rng_;
  std::vector<BaseParameter*> parameters_;
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
  PowerLaw (double mn, double mx, Parameter<T1>& n) : mn_(mn), mx_(mx) {
    this->parameters_.push_back(&n);
  };
  double scale_random (double u) const {
    double n = this->parameters_[0]->value();
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
};

template <typename T1, typename T2>
class Normal : public Distribution {
public:
  Normal (Parameter<T1>& mu, Parameter<T2>& log_sig) {
    this->parameters_.push_back(&mu);
    this->parameters_.push_back(&log_sig);
  };
  double sample (random_state_t& state) {
    boost::random::normal_distribution<> normal_rng;
    return this->parameters_[0]->value() + exp(this->parameters_[1]->value()) * normal_rng(state);
  };
};

template <typename T1, typename T2>
class Beta : public Distribution {
public:
  Beta (Parameter<T1>& log_a, Parameter<T2>& log_b) {
    this->parameters_.push_back(&log_a);
    this->parameters_.push_back(&log_b);
  };
  double sample (random_state_t& state) {
    boost::random::beta_distribution<> beta_rng(
      exp(this->parameters_[0]->value()), exp(this->parameters_[1]->value())
    );
    return beta_rng(state);
  };
};

template <typename T1>
class Rayleigh : public Distribution {
public:
  Rayleigh (Parameter<T1>& log_sig) {
    this->parameters_.push_back(&log_sig);
  };
  double scale_random (double u) const {
    return sqrt(-2.0 * exp(2.0 * this->parameters_[0]->value()) * log(1.0 - u));
  };
};

class Multinomial : public Distribution {
public:

  void add_bin (BaseParameter* parameter) {
    this->parameters_.push_back(parameter);
  };

  double scale_random (double u) const {
    size_t n = this->parameters_.size();
    double norm = 0.0, value = 0.0;
    for (size_t i = 0; i < n; ++i) norm += this->parameters_[i]->value();
    for (size_t i = 0; i < n-1; ++i) {
      value += this->parameters_[i]->value();
      if (value / norm > u) return 1.0 * i;
    }
    return n-1.0;
  };
};

}; // namespace exoabc

#endif  // _EXOABC_DISTRIBUTIONS_
