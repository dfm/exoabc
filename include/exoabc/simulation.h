#ifndef _EXOABC_SIMULATION_H_
#define _EXOABC_SIMULATION_H_

#include <vector>

#include "exoabc/parameter.h"
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

template <
  typename Period,
  typename Radius,
  typename Eccen,
  typename Width,
  typename Multi
>
class Simulation {
public:
  Simulation (
    Period& period_distribution,
    Radius& radius_distribution,
    Eccen&  eccen_distribution,
    Width&  width_distribution,
    Multi&  multi_distribution
  )
  : period_distribution_(period_distribution)
  , radius_distribution_(radius_distribution)
  , eccen_distribution_ (eccen_distribution)
  , width_distribution_ (width_distribution)
  , multi_distribution_ (multi_distribution)
  {
    add_parameters(period_distribution_.parameters());
    add_parameters(radius_distribution_.parameters());
    add_parameters(eccen_distribution_.parameters());
    add_parameters(width_distribution_.parameters());
    add_parameters(multi_distribution_.parameters());
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
    return parameters_.size();
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
      size_t N = size_t(multi_distribution_.sample(state));

      // Sample the mean inclination and the inclination width
      double mean_incl = M_PI * (2.0 * uniform_rng_(state) - 1.0),
             incl_width = width_distribution_.sample(state);

      // Loop over the planets
      for (size_t n = 0; n < N; ++n) {
        // Base parameters
        double radius = radius_distribution_.sample(state),
               period = period_distribution_.sample(state),
               eccen  = eccen_distribution_.sample(state),
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

private:
  std::vector<BaseParameter*> parameters_;
  std::vector<const BaseStar*> stars_;

  // Random number generators
  boost::random::uniform_01<> uniform_rng_;
  boost::random::normal_distribution<> normal_rng_;

  // Population distributions.
  Period period_distribution_;
  Radius radius_distribution_;
  Eccen  eccen_distribution_;
  Width  width_distribution_;
  Multi  multi_distribution_;

};

}; // namespace exoabc

#endif  // _EXOABC_SIMULATION_
