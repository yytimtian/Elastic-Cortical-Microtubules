This repository is under active development. Scripts may contain unused code that has not been deleted yet.

# Interacting MT Simulation Files

This is an event-driven simulation where the main objects are the event list, MTs, and bundles. Conceptual details of this simulation can be found in my Master's thesis (DOI:10.14288/1.0417339). A detailed guide with minimal working examples is under development.

## Scripts

Dependencies for these scripts: Python 3.13.1, NumPy 2.2.2, Scipy 1.15.1, Matplotlib 3.10.0 (only used for data analysis). The scripts below are used for the simulation itself. There are additional post-processing scripts that will be added in the future.

* `run_sim_array_rerun.py` is the main simulation file. It contains a single function that calls upon the simulation function from the below files. See script for info on how to run it.
* `parameters.py` contains all parameters and basic settings used in the simulation. This includes booleans that tell the simulation whether there is e.g. isotropic vs. LDD nucleation. All other files import these parameters.
* `sim_algs_fixed_region.py` contains the high-level functions responsible for stepping through the simulation. Among others, this includes `update_event_list()` which updates the event list (inserting and deleting events) after each step and `update()` updates the MT and bundle objects (adding vertices, changing MT state, etc.). `simulate()` brings together these functions and runs the simulations. In practice, I run this file interactively for small-scale runs and troubleshooting.
* `comparison_fns.py` contains the class definitions of MT and bundles, along with functions relating to the objects. This file also contains other misc. functions.
* `sim_anchoring_alg_test.py` contains functions simulating the anchoring mechanism. It uses Scipy for the Euler-Lagrange BVP, among other numerical methods.
* `zippering.py` contains the function responsible for deciding on collision resolution depending on the geometry of MTs. It was coded on a case-by-case basis, there is likely an easier way of doing this....
* `plotting.py` contains functions to plot and analyze the geometry of MTs (calculate order parameters).
