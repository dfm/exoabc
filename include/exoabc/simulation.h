#ifndef _EXOABC_SIMULATION_H_
#define _EXOABC_SIMULATION_H_

#include <vector>

#include "exoabc/distributions.h"
#include "exoabc/observation_model.h"

namespace exoabc {

struct CatalogRow {
    unsigned starid;
    double period;
    double radius;
    double duration;
    double depth;
};

class Simulation {
public:
  Simulation (
    Distribution* period_distribution,
    Distribution* radius_distribution,
    Distribution* eccen_distribution,
    Distribution* width_distribution,
    Distribution* multi_distribution
  )
  : period_distribution_(period_distribution)
  , radius_distribution_(radius_distribution)
  , eccen_distribution_ (eccen_distribution)
  , width_distribution_ (width_distribution)
  , multi_distribution_ (multi_distribution)
  {
    add_parameters(period_distribution_->parameters());
    add_parameters(radius_distribution_->parameters());
    add_parameters(eccen_distribution_->parameters());
    add_parameters(width_distribution_->parameters());
    add_parameters(multi_distribution_->parameters());
  };

  ~Simulation () {
    delete period_distribution_;
    delete radius_distribution_;
    delete eccen_distribution_;
    delete width_distribution_;
    delete multi_distribution_;
    for (size_t i = 0; i < stars_.size(); ++i) delete stars_[i];
  };

  void add_star (const BaseStar* star) {
    stars_.push_back(star);
  };

  template <typename ParamType>
  void add_parameter (ParamType* parameter) {
    parameters_.push_back(parameter);
  };

  void add_parameters (std::vector<BaseParameter*> parameters) {
    for (size_t i = 0; i < parameters.size(); ++i)
      parameters_.push_back(parameters[i]);
  };

  size_t size () {
    size_t count = 0;
    for (size_t i = 0; i < parameters_.size(); ++i)
      if (!(parameters_[i]->is_frozen())) count++;
    return count;
  };

  void sample_parameters (random_state_t& state) {
    for (size_t i = 0; i < parameters_.size(); ++i)
      parameters_[i]->sample(state);
  };

  std::vector<CatalogRow> sample_population (random_state_t& state) {
    std::vector<CatalogRow> catalog;
    for (size_t i = 0; i < stars_.size(); ++i) {
      // Get the star
      const BaseStar* star = stars_[i];

      // Sample a number of planets
      size_t N = size_t(multi_distribution_->sample(state));

      // Sample the mean inclination and the inclination width
      double mean_incl = M_PI * (2.0 * uniform_rng_(state) - 1.0),
             incl_width = width_distribution_->sample(state);

      // Loop over the planets
      for (size_t n = 0; n < N; ++n) {
        // Base parameters
        double radius = radius_distribution_->sample(state),
               period = period_distribution_->sample(state),
               eccen  = eccen_distribution_->sample(state),
               omega  = 2.0 * M_PI * uniform_rng_(state),
               q1     = uniform_rng_(state),
               q2     = uniform_rng_(state);

        // Inclination
        double incl = mean_incl + incl_width * normal_rng_(state);

        // Detection probability
        double duration, depth;
        double pdet = star->get_completeness(q1, q2, period, radius, incl, eccen, omega,
                                             &duration, &depth);

        // Is the planet detected?
        if (uniform_rng_(state) < pdet) {
          CatalogRow row = {i, period, radius, duration, depth};
          catalog.push_back(row);
        }
      }
    }
    return catalog;
  };

  std::vector<double> get_parameter_values () const {
    std::vector<double> params;
    for (size_t i = 0; i < parameters_.size(); ++i)
      if (!(parameters_[i]->is_frozen())) params.push_back(parameters_[i]->value());
    return params;
  };

  void get_parameter_values (double* params) const {
    size_t j = 0;
    for (size_t i = 0; i < parameters_.size(); ++i)
      if (!(parameters_[i]->is_frozen())) params[j++] = parameters_[i]->value();
  };

  double set_parameter_values (const std::vector<double>& vector) {
    return set_parameter_values((double*)&(vector[0]));
  };

  double set_parameter_values (const double* vector) {
    size_t j = 0;
    double log_prior = 0.0;
    for (size_t i = 0; i < parameters_.size(); ++i)
      if (!(parameters_[i]->is_frozen()))
        log_prior += parameters_[i]->value(vector[j++]);
    if (std::isinf(log_prior) || std::isnan(log_prior)) return -INFINITY;
    return log_prior;
  };

  double log_pdf () const {
    double log_prior = 0.0;
    for (size_t i = 0; i < parameters_.size(); ++i)
      if (!(parameters_[i]->is_frozen()))
        log_prior += parameters_[i]->log_pdf();
    if (std::isinf(log_prior) || std::isnan(log_prior)) return -INFINITY;
    return log_prior;
  };

private:
  std::vector<BaseParameter*> parameters_;
  std::vector<const BaseStar*> stars_;

  // Random number generators
  boost::random::uniform_01<> uniform_rng_;
  boost::random::normal_distribution<> normal_rng_;

  // Population distributions.
  Distribution* period_distribution_;
  Distribution* radius_distribution_;
  Distribution* eccen_distribution_;
  Distribution* width_distribution_;
  Distribution* multi_distribution_;

};

}; // namespace exoabc

#endif  // _EXOABC_SIMULATION_
