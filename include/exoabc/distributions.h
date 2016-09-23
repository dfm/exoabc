#ifndef _EXOABC_DISTRIBUTIONS_
#define _EXOABC_DISTRIBUTIONS_

#include <cmath>
#include <cfloat>
#include <vector>
#include <string>
#include <sstream>
#include <boost/random.hpp>
#include <boost/math/distributions.hpp>

namespace exoabc {

typedef boost::random::mt19937 random_state_t;

// Random state serialization
std::string serialize_state (const random_state_t& state) {
  std::stringstream ss;
  ss << state;
  return ss.str();
}
random_state_t deserialize_state (const std::string& blob) {
  std::stringstream ss(blob);
  random_state_t state;
  ss >> state;
  return state;
}

class BaseParameter {
public:
  virtual ~BaseParameter () {};
  virtual double value () const = 0;
  virtual double value (double value) = 0;
  virtual double sample (random_state_t& state) = 0;
  virtual bool is_frozen () const = 0;
  virtual double log_pdf () const = 0;
};

class Distribution {
public:
  virtual double scale_random (double u) const {
    return 0.0;
  };
  virtual double sample (random_state_t& state) {
    return this->scale_random(rng_(state));
  };
  virtual double log_pdf (double x) const { return 0.0; };
  virtual ~Distribution () {
    for (size_t i = 0; i < size(); ++i) delete parameters_[i];
  };
  size_t size () const { return parameters_.size(); };
  std::vector<BaseParameter*> parameters () { return parameters_; };

protected:
  boost::random::uniform_01<> rng_;
  std::vector<BaseParameter*> parameters_;
};

class Uniform : public Distribution {
public:
  Uniform (double mn, double mx) : mn_(mn), dx_(mx - mn) {};
  double sample (random_state_t& state) {
    double u = rng_(state);
    return mn_ + dx_ * u;
  };
  double log_pdf (double x) const {
    if (x < mn_ || x > mn_ + dx_) return -INFINITY;
    return -log(dx_);
  };

private:
  double mn_, dx_;
  boost::random::uniform_01<> rng_;
};

class Delta : public Distribution {
public:
  Delta (double value) : value_(value) {};
  double scale_random (double u) const {
    return value_;
  };
  double sample (random_state_t& state) {
    return value_;
  };
  double log_pdf (double x) const {
    if (std::abs(x - value_) < DBL_EPSILON) return 0.0;
    return -INFINITY;
  };

private:
  double value_;
};

class Parameter : public BaseParameter {
public:
  Parameter (double value)
    : prior_(new Delta(value)), frozen_(true), value_(value) {};
  Parameter (Distribution* prior)
    : prior_(prior), frozen_(false) {};
  Parameter (Distribution* prior, random_state_t& state)
    : prior_(prior), frozen_(false), value_(prior->sample(state)) {};
  Parameter (Distribution* prior, double value, bool frozen = false)
    : prior_(prior), frozen_(frozen), value_(value) {};

  ~Parameter () {
    delete prior_;
  };

  void freeze () { frozen_ = true; };
  void thaw () { frozen_ = false; };
  bool is_frozen () const { return frozen_; };

  double value () const { return value_; };
  double value (double value) {
    value_ = value;
    return prior_->log_pdf(value);
  };
  double sample (random_state_t& state) {
    value_ = prior_->sample(state);
    return value_;
  };
  double log_pdf () const { return prior_->log_pdf(value_); };

private:
  Distribution* prior_;
  bool frozen_;
  double value_;
};

class PowerLaw : public Distribution {
public:
  PowerLaw (double mn, double mx, BaseParameter* n) : mn_(mn), mx_(mx) {
    this->parameters_.push_back(n);
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
  double log_pdf (double x) const {
    if (x < mn_ || x > mx_) return -INFINITY;
    double n = this->parameters_[0]->value();
    if (fabs(n + 1.0) < DBL_EPSILON)
      return -log(log(mx_) - log(mn_)) + n * log(x);
    double np1 = n+1.0;
    return log(np1) - log(pow(mx_, np1) - pow(mn_, np1)) + n * log(x);
  };

private:
  double mn_, mx_;
};

class Normal : public Distribution {
public:
  Normal (BaseParameter* mu, BaseParameter* log_sig) {
    this->parameters_.push_back(mu);
    this->parameters_.push_back(log_sig);
  };
  double sample (random_state_t& state) {
    boost::random::normal_distribution<> normal_rng;
    return this->parameters_[0]->value() + exp(this->parameters_[1]->value()) * normal_rng(state);
  };
  double log_pdf (double x) const {
    double mu = this->parameters_[0]->value(),
           sig = exp(this->parameters_[1]->value()),
           chi = (x - mu) / sig;
    return -0.5 * (chi*chi + log(2.0*M_PI*sig*sig));
  };
};

class Beta : public Distribution {
public:
  Beta (BaseParameter* log_a, BaseParameter* log_b) {
    this->parameters_.push_back(log_a);
    this->parameters_.push_back(log_b);
  };
  double sample (random_state_t& state) {
    boost::random::beta_distribution<> beta_rng(
      exp(this->parameters_[0]->value()), exp(this->parameters_[1]->value())
    );
    return beta_rng(state);
  };
  double log_pdf (double x) const {
    boost::math::beta_distribution<> beta(
      exp(this->parameters_[0]->value()), exp(this->parameters_[1]->value())
    );
    return log(boost::math::pdf(beta, x));
  };
};

class Rayleigh : public Distribution {
public:
  Rayleigh (BaseParameter* log_sig) {
    this->parameters_.push_back(log_sig);
  };
  double scale_random (double u) const {
    return sqrt(-2.0 * exp(2.0 * this->parameters_[0]->value()) * log(1.0 - u));
  };
  double log_pdf (double x) const {
    if (x < 0.0) return -INFINITY;
    double sig = this->parameters_[0]->value(), chi = x / sig;
    return -0.5 * chi*chi + log(x) - log(sig*sig);
  };
};

class Multinomial : public Distribution {
public:
  Multinomial (BaseParameter* parameter) {
    this->parameters_.push_back(parameter);
  };
  void add_bin (BaseParameter* parameter) {
    this->parameters_.push_back(parameter);
  };
  double scale_random (double u) const {
    size_t n = this->parameters_.size();
    double norm = 0.0, value = 0.0;
    for (size_t i = 0; i < n; ++i) norm += exp(this->parameters_[i]->value());
    for (size_t i = 0; i < n-1; ++i) {
      value += exp(this->parameters_[i]->value());
      if (value / norm > u) return 1.0 * i;
    }
    return n-1.0;
  };
  double log_pdf (double x) const {
    size_t n = this->parameters_.size(), ind = size_t(x);
    if (x < 0 || x >= this->parameters_.size()) return -INFINITY;
    double norm = 0.0;
    for (size_t i = 0; i < n; ++i) norm += exp(this->parameters_[i]->value());
    return this->parameters_[ind]->value() - log(norm);
  };
};

}; // namespace exoabc

#endif  // _EXOABC_DISTRIBUTIONS_
