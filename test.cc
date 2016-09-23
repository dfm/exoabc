#include <iostream>
#include <ctime>

#include "exoabc/exoabc.h"

using namespace exoabc;

int main () {
  random_state_t state(time(NULL));

  // Period and radius
  double min_period = 5.0,
         max_period = 300.0,
         min_radius = 1.0,
         max_radius = 10.0;
  PowerLaw
    *period = new PowerLaw(min_period, max_period, new Parameter(new Uniform(-4.0, 3.0), -1.01)),
    *radius = new PowerLaw(min_radius, max_radius, new Parameter(new Uniform(-4.0, 3.0), -2.02));

  // Eccentricity
  Beta* eccen = new Beta(new Parameter(log(0.867)), new Parameter(log(3.03)));

  // Mutual inclination
  Rayleigh* width = new Rayleigh(new Parameter(new Uniform(-0.1, 0.1), 0.001));

  // Multiplicity
  Multinomial* multi = new Multinomial();
  multi->add_bin(new Parameter(1.0));
  multi->add_bin(new Parameter(new Uniform(-10.0, 10.0), 10.0));
  multi->add_bin(new Parameter(new Uniform(-10.0, 10.0), 100.0));
  multi->add_bin(new Parameter(new Uniform(-10.0, 10.0), 10.0));

  // Set up the simulator
  Simulation sim(period, radius, eccen, width, multi);

  std::cout << sim.size() << std::endl;

  // Set up the completeness model
  Q1_Q16_CompletenessModel comp;
  double cdpp_x[] = {1.5, 15.0},
         cdpp_y[] = {100.0, 200.0},
         mest_x[] = {1.5, 15.0},
         mest_y[] = {7.1, 7.1};
  for (size_t i = 0; i < 1000; ++i) sim.add_star(
    new Star<Q1_Q16_CompletenessModel>(&comp, 1.0, 1.0, 1000.0, 0.8,
      2, cdpp_x, cdpp_y, 2, mest_x, mest_y));

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
