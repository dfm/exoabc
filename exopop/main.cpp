#include <ctime>
#include <iostream>
#include "simulation.h"

using abcsim::Star;
using abcsim::Simulation;


int main () {
    Simulation sim (1, 10, 0.5, 10.0, 1.0, 500.0, std::time(0));

    /* double cdpp_x[] = {1.5, 2., 2.5, 3., 3.5, 4.5, 5., 6., 7.5, 9., 10.5, 12., */
    /*                    12.5, 15.}, */
    /*        cdpp_y[] = {170.559, 151.554, 138.079, 128.717, 121.001, 109.38, */
    /*                    104.957, 98.301, 90.919, 86.139, 83.249, 80.924, */
    /*                    80.341, 77.147}, */

    double cdpp_x[] = {100.},
           cdpp_y[] = {10.0},
           thresh_x[] = {100.},
           thresh_y[] = {7.1};
    sim.add_star(new Star (0.92, 0.839, 0.5 * M_PI, 1426.742, 0.8792,
                           1, cdpp_x, cdpp_y, 1, thresh_x, thresh_y));

    double params[] = {-2.5, -0.5, 0.01};
    unsigned counts[10];
    sim.observe(params, counts);

    for (unsigned i = 0; i < 10; ++i)
        std::cout << counts[i] << " ";
    std::cout << std::endl;

    // double periods[] = {100.0, 150.0, 200.0, 500.0};
    // for (unsigned i = 0; i < 4; ++i) {
    //     double aor = star.get_aor(periods[i]);
    //     std::cout << star.get_completeness(aor, periods[i], 2.0, 0.0) << std::endl;
    // }

    return 0;
}
