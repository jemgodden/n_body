# n_body

This program has the ability to perform a simulation of N number of point masses interacting via Newton's universal law of gravitation, in a 3D space.

Two different integration methods can be used during the simulation, Velocity Verlet and Runge-Kutta. Velocity Verlet is written using original code, whereas the Runge-Kutta method is written using the GSL library.

# Downloading

The easiest way to download the code is to press the download button on the home page of the GitHub repository.

# Requirements

To run the program a C compiler is required, along with the following libraries: stdio.h, stdlib.h, string.h, math.h, stdbool.h and various modules of the gsl library.

# Use

This program will run from the command line, after being compiled. 
The command line arguments take the form:

./a.out -integration_system -input_file -time_step -total_sim_time

The integration systems available are outlined at the top of this file and are expressed as v and r respectively.


The input file must take the following form:

Each line contains information on a single object. 

It will have information in this order: name, mass, x-position, y-position, z-position, x-velocity, y-velocity, z-velocity.


Data on the simulation is automatically printed to pre-named files.

Images of the orbits of particles in the simulation can be toggled to be shwon after the simulation.

In certain situations, for example using the Sun-Earth-Moon initial conditions given, information on specific attricutes, such as the Earth's orbital period in the simulation can be printed to stdout.

# Log

Initial version uploaded to GitHub.
