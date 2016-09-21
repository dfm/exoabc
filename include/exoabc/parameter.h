#ifndef _EXOABC_PARAMETERS_
#define _EXOABC_PARAMETERS_

#include <boost/random.hpp>

namespace exoabc {

typedef boost::random::mt19937 random_state_t;

class BaseParameter {
public:
  virtual double value () const = 0;
  virtual double value (double value) = 0;
  virtual double sample (random_state_t& state) = 0;
  virtual bool is_frozen () const = 0;
};

template <typename Prior>
class Parameter : public BaseParameter {
public:
  Parameter (double value)
    : value_(value), frozen_(true), prior_(Prior(value)) {};
  Parameter (Prior& prior)
    : frozen_(false), prior_(prior) {};
  Parameter (Prior& prior, random_state_t& state)
    : value_(prior.sample(state)), frozen_(false), prior_(prior) {};
  Parameter (Prior& prior, double value, bool frozen = false)
    : value_(value), frozen_(frozen), prior_(prior) {};

  void freeze () { frozen_ = true; };
  void thaw () { frozen_ = false; };
  bool is_frozen () const { return frozen_; };

  double value () const { return value_; };
  double value (double value) {
    value_ = value;
    return prior_.log_pdf(value);
  };
  double sample (random_state_t& state) {
    value_ = prior_.sample(state);
    return value_;
  };

private:
  Prior prior_;
  bool frozen_;
  double value_;
};

}; // namespace exoabc

#endif  // _EXOABC_PARAMETERS_
