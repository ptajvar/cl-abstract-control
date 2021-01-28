# Closed loop abstraction based control synthesis
This repository provides python libraries for abstraction based control synthesis using local linearizations of nonlinear dynamics.
Main libraries:
* dynamics_library: a set of dynamical models along a with a continuous-to-discontinuous time translation function.
* zonotope_lib: defining a zonotope class with functialities to compute Minkowski sum and inner and outer box approximations.
* pwa_lib: Piecewise affine library that includes:
  * Hybridization function: continuous time nonlinear to discontinuous time piecewise affine
  * Pre-computation function: computing the backward reachable set of a target zonotope
The usage of different functionalities are shown in the "usage_* " scripts
