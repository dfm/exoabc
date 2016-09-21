#include <iostream>
#include <ctime>

#include "exoabc/distributions.h"
#include "exoabc/parameter.h"
#include "exoabc/simulation.h"
#include "exoabc/observation_model.h"

using namespace exoabc;

int main () {
  random_state_t state(time(NULL));

  // Period and radius
  double min_period = 5.0,
         max_period = 300.0,
         min_radius = 1.0,
         max_radius = 10.0;
  Uniform slope_prior(-4.0, 3.0);
  Parameter<Uniform> period_slope(slope_prior, -1.0124),
                     radius_slope(slope_prior, -2.0);
  PowerLaw<Uniform> period(min_period, max_period, period_slope),
                    radius(min_radius, max_radius, radius_slope);

  // Eccentricity
  Parameter<Delta> log_a(log(0.867)), log_b(log(3.03));
  Beta<Delta, Delta> eccen(log_a, log_b);

  // Mutual inclination
  Uniform log_scatter_prior(-0.1, 0.1);
  Parameter<Uniform> log_scatter(log_scatter_prior, 0.001);
  Rayleigh<Uniform> width(log_scatter);

  // Multiplicity
  Uniform rate_prior(0.0, 100.0);
  Multinomial multi;
  size_t nbins = 4;
  Delta base_rate(1.0);
  Parameter<Delta> rate0(base_rate, 1.0);
  Parameter<Uniform> rates[] = {
    Parameter<Uniform>(rate_prior, 10.0),
    Parameter<Uniform>(rate_prior, 100.0),
    Parameter<Uniform>(rate_prior, 300.0),
    Parameter<Uniform>(rate_prior, 10.0),
  };
  multi.add_bin(&rate0);
  for (size_t i = 0; i < nbins; ++i) multi.add_bin(&(rates[i]));

  // Set up the simulator
  Simulation<PowerLaw<Uniform>, PowerLaw<Uniform>, Beta<Delta, Delta>,
             Rayleigh<Uniform>, Multinomial>
    sim(period, radius, eccen, width, multi);

  std::cout << sim.size() << std::endl;

  // Set up the completeness model
  Q1_Q16_CompletenessModel comp;
  double cdpp_x[] = {1.5, 15.0},
         cdpp_y[] = {100.0, 200.0},
         mest_x[] = {1.5, 15.0},
         mest_y[] = {7.1, 7.1};
  Star<Q1_Q16_CompletenessModel> star(comp, 1.0, 1.0, 1000.0, 0.8,
      2, cdpp_x, cdpp_y, 2, mest_x, mest_y);

  for (size_t i = 0; i < 1000; ++i) sim.add_star(&star);

  std::vector<double> params = sim.get_parameter_values();
  std::cout << params[0] << std::endl;
  params[3] = -1.0;
  std::cout << sim.set_parameter_values(params) << std::endl;

  /* sim.sample_parameters(state); */
  std::vector<CatalogRow> sim_catalog = sim.sample_population(state);
  std::cout << sim_catalog.size() << std::endl;
  std::cout << sim_catalog[0].duration << " " << sim_catalog[0].period << std::endl;

  return 0;
}
