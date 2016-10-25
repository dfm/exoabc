#ifndef _EXOABC_DISTRIBUTIONS_H_
#define _EXOABC_DISTRIBUTIONS_H_

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
  virtual size_t dimension () const = 0;
  virtual std::string name() const = 0;
  virtual bool is_frozen () const = 0;
  virtual std::vector<double> value () const = 0;
  virtual double value_1d () const = 0;
  virtual std::vector<double> sample (random_state_t& state) = 0;
  virtual double value (std::vector<double> value) = 0;
  virtual double log_pdf () const = 0;
};

class Distribution {
public:
  virtual ~Distribution () {
    for (size_t i = 0; i < size(); ++i) delete parameters_[i];
  };

  size_t size () const { return parameters_.size(); };
  virtual size_t dimension () const { return 1; };

  std::vector<BaseParameter*> parameters () { return parameters_; };

  virtual double mean_1d () const { return 0.0; };
  virtual std::vector<double> mean () const {
    std::vector<double> result(this->dimension());
    for (size_t n = 0; n < this->dimension(); ++n) result[n] = this->mean_1d();
    return result;
  };
  virtual std::vector<double> sample (random_state_t& state) {
    std::vector<double> result(this->dimension());
    for (size_t n = 0; n < this->dimension(); ++n) result[n] = this->sample_1d(state);
    return result;
  };
  virtual double scale_random (double u) const {
    return 0.0;
  };
  virtual double sample_1d (random_state_t& state) {
    return this->scale_random(rng_(state));
  };
  virtual double log_prior () const { return 0.0; };
  virtual double log_pdf_1d (double x) const { return 0.0; };
  virtual double log_pdf (const std::vector<double>& x) const {
    return this->log_pdf_1d(x[0]);
  };

protected:
  boost::random::uniform_01<> rng_;
  std::vector<BaseParameter*> parameters_;
};

class Uniform : public Distribution {
public:
  Uniform (double mn, double mx) : mn_(mn), dx_(mx - mn) {};
  double scale_random (double u) const {
    return mn_ + dx_ * u;
  };
  double log_pdf_1d (double x) const {
    if (x < mn_ || x > mn_ + dx_) return -INFINITY;
    return -log(dx_);
  };

private:
  double mn_, dx_;
};

class Delta : public Distribution {
public:
  Delta (double value) : value_(value) {};
  double scale_random (double u) const {
    return value_;
  };
  double sample_1d (random_state_t& state) {
    return value_;
  };
  double log_pdf_1d (double x) const {
    if (std::abs(x - value_) < DBL_EPSILON) return 0.0;
    return -INFINITY;
  };

private:
  double value_;
};

class Parameter : public BaseParameter {
public:
  Parameter (double value)
    : prior_(new Delta(value)), frozen_(true), value_(1), name_("delta")
  {
    value_[0] = value;
  };
  Parameter (std::string name, double value)
    : prior_(new Delta(value)), frozen_(true), value_(1), name_(name)
  {
    value_[0] = value;
  };
  Parameter (std::string name, Distribution* prior)
    : prior_(prior), frozen_(false), value_(prior->dimension()), name_(name) {};
  Parameter (std::string name, Distribution* prior, random_state_t& state)
    : prior_(prior), frozen_(false), name_(name)
  {
    value_ = prior->sample(state);
  };
  Parameter (std::string name, Distribution* prior, std::vector<double> vector)
    : prior_(prior), frozen_(false), value_(vector), name_(name) {};
  Parameter (std::string name, Distribution* prior, double value, bool frozen = false)
    : prior_(prior), frozen_(frozen), value_(1), name_(name)
  {
    value_[0] = value;
  };

  ~Parameter () {
    delete prior_;
  };

  size_t dimension () const { return value_.size(); };
  void freeze () { frozen_ = true; };
  void thaw () { frozen_ = false; };
  bool is_frozen () const { return frozen_; };
  std::string name () const { return name_; };

  // Getters
  std::vector<double> value () const { return value_; };
  double value_1d () const { return value_[0]; };

  // Setters
  double value_1d (double value) {
    value_[0] = value;
    return prior_->log_pdf(value_);
  };
  double value (std::vector<double> value) {
    value_ = value;
    return prior_->log_pdf(value_);
  };

  std::vector<double> sample (random_state_t& state) {
    value_ = prior_->sample(state);
    return value_;
  };
  double log_pdf () const { return prior_->log_pdf(value_); };

private:
  Distribution* prior_;
  bool frozen_;
  std::vector<double> value_;
  std::string name_;
};

class PowerLaw : public Distribution {
public:
  PowerLaw (double mn, double mx, BaseParameter* n) : mn_(mn), mx_(mx) {
    this->parameters_.push_back(n);
  };
  double scale_random (double u) const {
    double n = this->parameters_[0]->value_1d();
    if (fabs(n + 1.0) < DBL_EPSILON) {
      double lnmn = log(mn_);
      return mn_ * exp(u * (log(mx_) - lnmn));
    }
    double np1 = n+1.0,
           x0n = pow(mn_, np1);
    return pow((pow(mx_, np1) - x0n) * u + x0n, 1.0 / np1);
  };
  double log_pdf_1d (double x) const {
    if (x < mn_ || x > mx_) return -INFINITY;
    double n = this->parameters_[0]->value_1d();
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
  double sample_1d (random_state_t& state) {
    boost::random::normal_distribution<> normal_rng;
    return this->parameters_[0]->value_1d() + exp(this->parameters_[1]->value_1d()) * normal_rng(state);
  };
  double log_pdf_1d (double x) const {
    double mu = this->parameters_[0]->value_1d(),
           sig = exp(this->parameters_[1]->value_1d()),
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
  double sample_1d (random_state_t& state) {
    boost::random::beta_distribution<> beta_rng(
      exp(this->parameters_[0]->value_1d()), exp(this->parameters_[1]->value_1d())
    );
    return beta_rng(state);
  };
  double log_pdf_1d (double x) const {
    boost::math::beta_distribution<> beta(
      exp(this->parameters_[0]->value_1d()), exp(this->parameters_[1]->value_1d())
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
    return sqrt(-2.0 * exp(2.0 * this->parameters_[0]->value_1d()) * log(1.0 - u));
  };
  double log_pdf_1d (double x) const {
    if (x < 0.0) return -INFINITY;
    double sig = this->parameters_[0]->value_1d(), chi = x / sig;
    return -0.5 * chi*chi + log(x) - log(sig*sig);
  };
};

double logsumexp (double a, double b) {
  if (a >= b) return a + log(1.0 + exp(b - a));
  return b + log(1.0 + exp(a - b));
}

class Multinomial : public Distribution {
public:
  Multinomial (BaseParameter* parameter) {
    this->parameters_.push_back(parameter);
  };
  double mean_1d () const {
    BaseParameter* par = this->parameters_[0];
    std::vector<double> value = par->value();
    size_t n = par->dimension();
    double mu = 0.0;
    for (size_t i = 1; i < n; ++i) mu += i * value[i];
    return mu;
  };
  double log_prior () const {
    return 0.0;
  };
  double scale_random (double u) const {
    BaseParameter* par = this->parameters_[0];
    std::vector<double> vec = par->value();
    double value = 0.0;
    size_t n = par->dimension();
    for (size_t i = 0; i < n; ++i) {
      value += vec[i];
      if (value > u) return i;
    }
    return n - 1.0;
  };
  double log_pdf_1d (double x) const {
    BaseParameter* par = this->parameters_[0];
    std::vector<double> vec = par->value();
    size_t n = par->dimension(), ind = size_t(x);
    if (x < 0 || x >= n) return -INFINITY;
    return vec[ind];
  };
};

class Poisson : public Distribution {
public:
  Poisson (BaseParameter* log_rate) {
    this->parameters_.push_back(log_rate);
  };
  double mean_1d () const {
    return exp(this->parameters_[0]->value_1d());
  };
  double sample_1d (random_state_t& state) {
    boost::random::poisson_distribution<> rng(exp(this->parameters_[0]->value_1d()));
    return double(rng(state));
  };
  double log_pdf_1d (double x) const {
    boost::math::poisson_distribution<> pois(exp(this->parameters_[0]->value_1d()));
    return log(boost::math::pdf(pois, x));
  };
};

class Dirichlet : public Distribution {
public:
  Dirichlet (size_t dim) : dim_(dim) {};
  size_t dimension () const { return dim_; };
  std::vector<double> sample (random_state_t& state) {
    std::vector<double> result(dim_);
    boost::random::gamma_distribution<> rng(1.0 / this->dim_);
    double norm = 0.0;
    for (size_t i = 0; i < dim_; ++i) {
      result[i] = rng(state);
      norm += result[i];
    }
    for (size_t i = 0; i < dim_; ++i) result[i] /= norm;
    return result;
  };
private:
  size_t dim_;
};

}; // namespace exoabc

#endif  // _EXOABC_DISTRIBUTIONS_
