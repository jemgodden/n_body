/*
 Title:   N-Body Code
 Author:  Jeremy Godden
*/

static const char * VERSION  = "1.0.1";
static const char * REV_DATE = "13-Dec-2019";

/*
 This program, 'n_body.c', has the ability to perform a simulation of N number of point masses interacting via
 Newton's universal law of gravitation, in a 3D space.
 Two different integration methods can be used during the simulation, Velocity Verlet and Runge-Kutta. Velocity
 Verlet is written using original code, whereas the Runge-Kutta method is written using the GSL library.

 An example of the command line arguments would be:
    ./a.out -v SunEarthMoon.txt 3600 72000000
 This will read in bodies in the file SunEarthMoon.txt and calculate each body's path for the next 72000000 second,
 at time steps of 3600 seconds.
 Checks will be made on all command line argument values to make sure they are suitable.
 More information on these command line arguments can be found in the help() function below.

 The input file is expected to be in the same form as seen in SunEarthMoon.txt. If the file is not as expected,
 an error message will be displayed. However, due to the use of strtok, the expected structure of the input file
 can be altered easily.
 Input files will be read in a way to ignore any blank lines and anything after a '#' character.

 The position of each body at each time step is written to a file called 'simulation_path_data.txt'. This file
 is then used to plot the paths of each body during the simulation, through GNUplot. GNUplot is accessed remotely
 through the terminal in this program.

 There is the option, within the program, to calculate the kinetic, potential and total energy of the simulation
 at each time step. This is written to a file called 'simulation_energy_data.txt' and is plotted through GNUplot
 in the same way as body paths are. The cumulative error of the simulation is also found when calculating energy.
 This option is activated in this code in main().

 There is also an option within the program to calculate the average orbital period of a body around the origin
 of the simulation. This is useful to compare calculated values to expected values, when moddeling a system such
 as the Earth orbiting aorund the Sun, when the Sun is roughly at the origin. Due to this being very specific,
 the option is not activated in main().
*/

/* Packages used throughout the program. */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_matrix.h>

/* Defined values for command line arguments and reading in from a file. */
#define FIRST_CHAR 0
#define SECOND_CHAR 1
#define OPTION_ARG_LEN 2
#define OPTION_ARG 1
#define INPUT_FILE_ARG 2
#define STEP_ARG 3
#define TIME_LENGTH_ARG 4
#define NO_ARGS 5
#define MAX_LINE_LEN 1000
#define MAX_TOK_LEN 50
#define TOKEN_SEPARATORS " \t\r\n" /* All the separators expected in the file. */
#define MAX_NAME_LEN 20

/* Deifned values for the simulation. */
#define G 6.67e-11
#define NO_COORDS 3
#define DAY (60*60*24) /* Day in seconds. */
#define MIN_STEP 1e-10
#define MAX_STEP 4.32e5
#define MIN_TIME_LENGTH 1e-6
#define MAX_TIME_LENGTH 5e15
#define FIRST_STEP 0
#define STEP_NO (i+1)
#define MAX_CROSS 50 /* Maximum number of axis crossings that can be recorded in each direction
                        for finding orbital period of a specified body. */

/* Defined values used in conjuncture with the GSL library */
#define NO_INPUT_VALS 6 /* Number of input values for each body into the ODE setup function. */
#define ABS_ERR 1e-4
#define REL_ERR 1e-5
#define GSL_SUCC 0 /* Redefining GSL_SUCCESS within the code to avoid error in use with system. */

/* Defined values for accessing GNUplot. */
#define GNUPLOT "gnuplot -persist"


/* All error codes used within the program. */
typedef enum error_code {
    NO_ERROR = 0,
    INVALID_ARGS = 1,
    MEMORY_ERROR = 2,
    FILE_OPEN_ERROR = 3,
    PIPE_ERROR = 4,
    INVALID_FILE = 5,
    ERROR_EOF = 6,
    GSL_USE_ERROR = 7,
} Error_code;

typedef int Error;


/* ===== Structures used in program. ===== */

/* Structure for each file that contains information to give when an error occurs. */
typedef struct context {
    FILE *file;
    const char *file_name;
    char token[MAX_TOK_LEN + 1];
    int line_number;
} Context;

/* Structure of information of a body within the simulation. */
typedef struct body {
    char name[MAX_NAME_LEN];
    double mass;
    double xyz[NO_COORDS];
    double v_xyz[NO_COORDS];
    double a_xyz[NO_COORDS];
    struct body *next; /* Contains a pointer to the next body to make a linked list of all bodies. */
} Body;

/* Structure for the parameters of the simulation. */
typedef struct parameters {
    double step;
    double time_length;
    long no_steps;
} Parameters;

/* Structure containing variables used in calculating energy during the simulation. */
typedef struct energy_info {
    bool energy_calc;
    double E_0;
    double old_ke;
    double old_pe;
    double cumulative_err;
} Energy_info;

/* Structure containing variables used in calculating orbital period of a body during the simulation. */
typedef struct orbit_info {
    bool orbit_calc;
    char orbit_body[MAX_NAME_LEN];
    double old_pos[NO_COORDS];
    int list_count[NO_COORDS];
    double step_list[NO_COORDS][MAX_CROSS];
} Orbit_info;


/* Function to print help with command line arguments to the console for the user. */
void help() {
    fprintf(stderr, "\nPlease enter in command lines like this:\n"
            "./n_body OPTION INPUT_FILE TIME_STEP TOTAL_TIME\n");
    fprintf(stderr, "Options:\n"
            "\t-h: Shows help with command line arguments.\n"
            "\t-v: Uses Velocity Verlet integration scheme.\n"
            "\t-r: Uses Runge-Kutta integration scheme.\n");
    fprintf(stderr, "INPUT_FILE is the file that contains the initial conditions for all the bodies that will be "
            "included in the simulation.\n");
    fprintf(stderr, "TIME_STEP is the step between each new position calculation, in seconds.\n");
    fprintf(stderr, "TOTAL_TIME is the total time the simulation is run for, in seconds.\n");
}


/* ===== Functions for printing errors. ===== */

/* Function to print an error message due to invalid command line arguments. */
Error invalid_args_print(const char *message) {
    fprintf(stderr, "Invalid command line arguments were given. %s\n", message);
    help(); /* Prints help to user after invalid command line argument. */
    return INVALID_ARGS;
}

/* Function to print an error message due to inability to allocate memory. */
Error memory_error_print() {
    fprintf(stderr, "Memory could not be allocated.\n");
    return MEMORY_ERROR;
}

/* Function to print an error message due to inability to open a file. */
Error file_open_error_print(const Context *context) {
    fprintf(stderr, "File %s could not be opened.\n", context->file_name);
    return FILE_OPEN_ERROR;
}

/* Function to print an error message due to inability to pipe to GNUplot. */
Error pipe_error_print() {
    fprintf(stderr, "Pipe to GNUplot could not be opened.\n");
    return PIPE_ERROR;
}

/* Function to print an error message due to invalid an invalid input file. */
Error invalid_file_print(const Context *context, const char *message) {
    fprintf(stderr, "%s is an invalid input file. %s\n", context->file_name, message);
    if (context->token[FIRST_CHAR] == '\0') {
        /* Prints error if the next token cannot be found. */
        fprintf(stderr, "The error is in line %d of the file.\n", context->line_number);
    }
    else {
        /* Prints error if the next token is invalid. */
        fprintf(stderr, "The invalid string in line %d of the file is: %s\n", context->line_number, context->token);
    }
    fclose(context->file);
    return INVALID_FILE;
}

/* Function to print an error message due to an error with a GSL library function.
   Outlines the error given by GSL function. */
Error gsl_error_print(const char *message, const int error) {
    fprintf(stderr, "There was an error using the GSL library. %s\n", message);
    fprintf(stderr, "The driver returned error code %d.\n", error); /* Outlines error code from gsl library for the user. */
    return GSL_USE_ERROR;
}


/* Function to free all bodies in the linked list. */
void free_bodies(Body **bodies) {
    while (*bodies != NULL) {
        Body *body = *bodies;
        *bodies = body->next; /* Sets pointer to next body. */
        free(body); /* Frees previous body. */
    }
}


/* ===== Functions used for reading command line arguments. ===== */

/* Function to convert command line argument string to a double. */
Error get_arg_double(const char *arg_value, double *value) {
    char *end_ptr;
    *value = strtod(arg_value, &end_ptr);

    if (*end_ptr != '\0') {
        return invalid_args_print("Step or total time argument is invalid.");
    }

    return NO_ERROR;
}

/* Function to get all information from the command line arguments. */
Error arg_info(char *argv[], Context *input_file_context, Parameters *simulation) {
    Error error;

    input_file_context->file_name = argv[INPUT_FILE_ARG];
    /* Gets value size of step in simulation. */
    error = get_arg_double(argv[STEP_ARG], &simulation->step);
    if (error != NO_ERROR) {
        return error;
    }
    if (simulation->step > MAX_STEP || simulation->step < MIN_STEP){
        /* Checks that step size is within set limits, giving an error if not. */
        return invalid_args_print("Step time is too small or big.");
    }
    /* Get value of total length of time the simulation is run for. */
    error = get_arg_double(argv[TIME_LENGTH_ARG], &simulation->time_length);
    if (error != NO_ERROR) {
        return error;
    }
    if (simulation->time_length > MAX_TIME_LENGTH || simulation->time_length < MIN_TIME_LENGTH){
        /* Checks that time length of simulation is within set limits, giving an error if not. */
        return invalid_args_print("Total time value is too small or big.");
    }

    simulation->no_steps = (long)floor(simulation->time_length / simulation->step);
    /* Finds maximum number of time steps that can occur in the time length with given step size. */

    if (fmod(simulation->time_length, simulation->step) != 0) {
        /* Informs user if not all the time length is able to be used with the given step size. */
        printf("The total time of the simulation was not an exact multiple of the step time.\n"
                       "Instead, %ld steps were used for a total time of %lf seconds during the simulation.\n\n",
               simulation->no_steps, simulation->no_steps * simulation->step);
    }

    return NO_ERROR;
}


/* ===== Functions used for reading infromation from a file. ===== */

/* Function to read a line from the input file. */
Error read_line(char *line, Context *context) {
    char *token;
    /* Using the while loop is more efficient when tackling recursion.
       This loop keeps getting a new line until it encounters a valid line.
       It ignores any lines starting with a #. */
    while (1) {
        context->token[FIRST_CHAR] = '\0';
        ++context->line_number;

        if (fgets(line, MAX_LINE_LEN + 2, context->file) == NULL) {
            if (feof(context->file)) {
                /* Checks for end of file if fgets returns NULL. Returns end of file error if encountered. */
                return ERROR_EOF;
            }
            /* If not end of file, error is printed to console. */
            return invalid_file_print(context, "Failed to read line from file.");
        }
        if (strlen(line) > MAX_LINE_LEN) {
            return invalid_file_print(context, "Line too long.");
        }
        /* Retrieves first token from line. */
        token = strtok(line, TOKEN_SEPARATORS);
        if (token == NULL) {
            continue;
        }
        if (strlen(token) > MAX_TOK_LEN) {
            return invalid_file_print(context, "Token too long.");
        }

        strcpy(context->token, token);
        if (context->token[FIRST_CHAR] != '#') {
            /* Returns no error once valid token found from a valid line. */
            return NO_ERROR;
        }
    }
}

/* Function to get the next token from a line. */
Error get_token(Context *context) {
    char *token;
    context->token[FIRST_CHAR] = '\0';

    token = strtok(NULL, TOKEN_SEPARATORS);
    if (token == NULL) {
        /* Checks there is a token there. */
        return invalid_file_print(context, "Token not found.");
    }
    if (strlen(token) > MAX_TOK_LEN) {
        return invalid_file_print(context, "Token too long.");
    }
    strcpy(context->token, token);

    return NO_ERROR;
}

/* Function to convert a string in the file, to a double. */
Error get_double(double *value, const Context *context) {
    char *end_ptr;
    /* Using strtod to turn string to a double. */
    *value = strtod(context->token, &end_ptr);

    if (*end_ptr != '\0') {
        /* Gives error if whole of string cannot be turned into a double. */
        return invalid_file_print(context, "Body information is invalid.");
    }

    return NO_ERROR;
}

/* Function that checks for the end of the line. */
Error check_line_end(Context *context) {
    char *token;
    context->token[FIRST_CHAR] = '\0';

    token = strtok(NULL, TOKEN_SEPARATORS);
    if (token == NULL || token[FIRST_CHAR] == '#') {
        /* Checking that there is no next token. If there is, checks that it is a '#', indicating a comment */
        return NO_ERROR;
    }
    if (strlen(token) > MAX_TOK_LEN) {
        return invalid_file_print(context, "Token too long.");
    }
    strcpy(context->token, token);
    /* If a token is found, gives error saying that it could not find the expected end of the line. */
    return invalid_file_print(context, "End of line not found.");
}

/* Function to read the x, y and z positions of a body from the input file. */
Error read_positions(Body *body, Context *context) {
    Error error;

    for (int i=0; i<NO_COORDS; i++) {
        /* Loops for each cooridnate, obtaining each value as a string. */
        error = get_token(context);
        if (error != NO_ERROR) {
            return error;
        }
        /* Then converts string to a double and gves it to the body structure. */
        error = get_double(&body->xyz[i], context);
        if (error != NO_ERROR) {
            return error;
        }
    }
    return NO_ERROR;
}

/* Function to read the x, y and z velocities of a body from the input file. */
Error read_velocities(Body *body, Context *context) {
    Error error;

    for (int i=0; i<NO_COORDS; i++) {
        /* Loops for each cooridnate, obtaining each value as a string. */
        error = get_token(context);
        if (error != NO_ERROR) {
            return error;
        }
        /* Then converts string to a double and gves it to the body structure. */
        error = get_double(&body->v_xyz[i], context);
        if (error != NO_ERROR) {
            return error;
        }
    }

    return NO_ERROR;
}

/* Function to read the mass of a body from the input file. */
Error read_mass(Body *body, Context *context) {
    Error error;
    /* Obtains the value as a string. */
    error = get_token(context);
    if (error != NO_ERROR) {
        return error;
    }
    /* Then converts string to a double and gves it to the body structure. */
    error = get_double(&body->mass, context);
    if (error != NO_ERROR) {
        return error;
    }

    return NO_ERROR;
}

/* Function to read each numerical value of a body, from the input file. */
Error read_body_values(Body *body, Context *context) {
    Error error;

    error = read_mass(body, context);
    if (error != NO_ERROR) {
        return error;
    }

    error = read_positions(body, context);
    if (error != NO_ERROR) {
        return error;
    }

    error = read_velocities(body, context);
    if (error != NO_ERROR) {
        return error;
    }

    return NO_ERROR;
}

/* Function to read all information about a body, from the input file. */
Error read_body(Body **body, char *line, Context *context) {
    Error error;
    /* Reads the line of the body. */
    error = read_line(line, context);
    if (error != NO_ERROR) {
        return error;
    }
    /* Allocates memory for the body being read in. */
    *body = malloc(sizeof(Body));
    if (*body == NULL) {
        return memory_error_print();
    }
    /* Copies the name of the body from the file into the structure. */
    strcpy((*body)->name, context->token);
    /* Reads in all numerical values of body. */
    error = read_body_values(*body, context);
    if (error != NO_ERROR) {
        free(*body);
        return error;
    }
    /* Checks for end of line after all information has been read. */
    error = check_line_end(context);
    if (error != NO_ERROR) {
        free(*body);
        return error;
    }

    return NO_ERROR;
}

/* Function to read all bodies from the input file. */
Error read_bodies(Body **bodies, Context *context) {
    char line[MAX_LINE_LEN + 2]; /* Extra character to allow detection of long lines. */
    Error error;
    context->line_number = 0;

    context->file = fopen(context->file_name, "r");
    if (context->file==NULL) {
        return file_open_error_print(context);
    }
    /* Due to not knowing how many bodies there are in the input file, a while loop is used.
       Means that the program will continue to look for new lines/bodies until the end of file (eof) is found. */
    while (1) {
        Body *body;
        /* Sets pointer to new body and reads in all its values. */
        error = read_body(&body, line, context);
        if (error != NO_ERROR) {
            if (error == ERROR_EOF) {
                /* If the end of file is found, while loop is exited. */
                return NO_ERROR;
            }
            /* If any other error is found, bodies are freed and error is returned. */
            free_bodies(bodies);
            return error;
        }
        /* Sets the newly made body to point at the next body to be made, or NULL if it's the first body.
           Reads bodies in file backwards into code. Whichever it finds first becomes the end of the list. */
        body->next = *bodies;
        *bodies = body;
    }

}


/* ===== Functions for printing and plotting data of paths of bodies in simulation. ===== */

/* Prints positions of all bodies to file. */
void print_body_pos(Body *bodies, const Context *context) {
    fprintf(context->file, "\n");
    for (Body *body_i=bodies; body_i!=NULL; body_i=body_i->next) {
        for (int j=0; j<NO_COORDS; j++) {
            fprintf(context->file, "%.4e\t", body_i->xyz[j]);
        }
    }
}

/* Prints inital information to file about how it was made and it's structure, for bodies' positions. */
Error initial_body_output(Body *bodies, const Context *input_file_context, Context *output_file_context) {
    output_file_context->file = fopen(output_file_context->file_name, "w+");
    if (output_file_context->file==NULL){
        return file_open_error_print(output_file_context);
    }

    fprintf(output_file_context->file, "# Version = %s, Revision date = %s\n", VERSION, REV_DATE);
    fprintf(output_file_context->file, "# Position of each body read from file %s, at each time step.\n",
            input_file_context->file_name);
    fprintf(output_file_context->file, "# ");
    for (Body *body_i=bodies; body_i!=NULL; body_i=body_i->next) {
        /* Prints each position of each body in simulation in a commented row at top of file. */
        fprintf(output_file_context->file, "%s_x\t%s_y\t%s_z\t", body_i->name, body_i->name, body_i->name);
    }

    print_body_pos(bodies, output_file_context);

    return NO_ERROR;
}

/* Pipes to GNUplot through the terminal to plot paths of all bodies in simulation. */
Error plot_path(Body *bodies, const Context *output_file_context) {
    FILE *gp;
    int i=1;
    const char *prefix = "splot "; /* Splot plots images in 3D. */

    gp = popen(GNUPLOT, "w");
    if (gp==NULL) {
        return pipe_error_print();
    }

    /* Setting visual aspects of plot to make it clear. */
    fprintf(gp, "set title \"Paths of bodies during simulation\" font \"Arial, 10\"\n");
    fprintf(gp, "set xlabel \"x(m)\" font \"Arial, 10\"\n");
    fprintf(gp, "set ylabel \"y(m)\" font \"Arial, 10\"\n");
    fprintf(gp, "set zlabel \"z(m)\" font \"Arial, 10\"\n");
    fprintf(gp, "set xrange [ * : * ] noreverse writeback\n");
    fprintf(gp, "set yrange [ * : * ] noreverse writeback\n");
    fprintf(gp, "set zrange [ -1 : 1 ] noreverse writeback\n");
    fprintf(gp, "set tics font \"Arial, 8\"\n");
    for (Body *body_i=bodies; body_i!=NULL; body_i=body_i->next) {
        /* Loops plotting of each body in simulation. */
        fprintf(gp, "%s \"%s\" using %d:%d:%d title \'%s\'", prefix, output_file_context->file_name, i, i+1, i+2, body_i->name);
        i = i + NO_COORDS;
        prefix = ", ";
    }
    fprintf(gp, "\n");
    /* Prevents program from finishing so the user can zoom in on a specific part of the plot. */
    fprintf(gp, "pause mouse close\n");
    /* Replots the data and outputs it to a .png file, so it can be viewed after the program has finished. */
    fprintf(gp, "set terminal png\n");
    fprintf(gp, "set output 'simulation_path.png'\n");
    fprintf(gp, "replot\n");

    pclose(gp);
    return NO_ERROR;
}


/* ===== Functions used for calculating, printing and plotting data for energy in the simulation. ===== */

/* Loops through all bodies, finding each's kinetic energy,
   to find the total kinetic energy in the system at a given time. */
double calc_ke(Body *bodies) {
    double ke = 0;
    double v_squared = 0;

    for (Body *body_i=bodies; body_i!=NULL; body_i=body_i->next) {
        for (int j=0; j<NO_COORDS; j++) {
            v_squared += pow(body_i->v_xyz[j], 2);
        }
        ke += 0.5 * body_i->mass * v_squared;
        v_squared = 0;
    }

    return ke;
}

/* Loops through all bodies, finding each's potential energy via it's interaction with all other bodies,
   to find the total potential energy in the system at a given time. */
double calc_pe(Body *bodies) {
    double pe = 0;
    double r_squared = 0;

    for (Body *body_i=bodies; body_i!=NULL; body_i=body_i->next) {
        for (Body *body_j=bodies; body_j!=NULL; body_j=body_j->next) {
            if (body_i == body_j){
                continue;
            }
            for (int k=0; k<NO_COORDS; k++) {
                r_squared += pow(body_i->xyz[k] - body_j->xyz[k], 2);
            }
            pe += - (G * body_i->mass * body_j->mass) / pow(r_squared, 0.5);
            r_squared = 0;
        }
    }

    return pe;
}

/* Finds values of energy at a given time step and prints them to a file. */
void calc_energies(Body *bodies, const Context *context, const int step_no, Energy_info *energy_info) {
    double ke = calc_ke(bodies);
    double pe = calc_pe(bodies);

    if (step_no != 0) {
        /* Adds to running total of cumulative error. */
        energy_info->cumulative_err += fabs(pe - energy_info->old_pe + ke - energy_info->old_ke) / energy_info->E_0;
    }

    energy_info->old_ke = ke;
    energy_info->old_pe = pe;
    /* Prints energies to a file, along with current step. */
    fprintf(context->file, "%d\t%lf\t%lf\t%lf\n", step_no, pe, ke, pe + ke);
}

/* Prints inital information to file about how it was made and it's structure, for energy in the system. */
Error initial_energy_output(const Context *input_file_context, Context *energy_file_context) {
    energy_file_context->file = fopen(energy_file_context->file_name, "w+");
    if (energy_file_context->file==NULL){
        return file_open_error_print(energy_file_context);
    }

    fprintf(energy_file_context->file, "# Version = %s, Revision date = %s\n", VERSION, REV_DATE);
    fprintf(energy_file_context->file, "# Energies of bodies read from file %s, at each time step.\n",
            input_file_context->file_name);
    fprintf(energy_file_context->file, "# Time_step\tPotential_energy\tKinetic_energy\tTotal_energy\n");

    return NO_ERROR;
}

/* Pipes to GNUplot through the terminal to plot values of energy during simulation. */
Error plot_energy(const Context *energy_file_context) {
    FILE *gp;

    gp = popen(GNUPLOT, "w");
    if (gp==NULL) {
        return pipe_error_print();
    }

    /* Setting visual aspects of plot to make it clear. */
    fprintf(gp, "set title \"Energies during simulation\" font \"Arial, 10\"\n");
    fprintf(gp, "set xlabel \"Time step\" font \"Arial, 10\"\n");
    fprintf(gp, "set ylabel \"Energy (J)\" font \"Arial, 10\"\n");
    fprintf(gp, "set xrange [ * : * ] noreverse writeback\n");
    fprintf(gp, "set yrange [ * : * ] noreverse writeback\n");
    fprintf(gp, "set tics font \"Arial, 8\"\n");
    /* Plots each value of energy using correct pieces of data from file. */
    fprintf(gp, "plot \"%s\" using 1:2 title \'Potential Energy\' with lines lw 3, "
            "\"%s\" using 1:3 title \'Kinetic Energy\' with lines lw 3, "
            "\"%s\" using 1:4 title \'Total Energy\' with lines lw 3\n",
            energy_file_context->file_name, energy_file_context->file_name, energy_file_context->file_name);
    /* Prevents program from finishing so the user can zoom in on a specific part of the plot. */
    fprintf(gp, "pause mouse close\n");
    /* Replots the data and outputs it to a .png file, so it can be viewed after the program has finished. */
    fprintf(gp, "set terminal png\n");
    fprintf(gp, "set output 'simulation_energy.png'\n");
    fprintf(gp, "replot\n");

    pclose(gp);
    return NO_ERROR;
}


/* ===== Functions used for calculating the average orbital period of specified body during the simulation. ===== */

/* Function to find if a specified body has crossed any axis planes during a step. */
void find_axis_crossing(Body *bodies, const int step_no, Orbit_info *orbit_info) {
    for (Body *body_i=bodies; body_i!=NULL; body_i=body_i->next) {
        /* Loops through all bodies looking for specified body by name. */
        if (strcmp(body_i->name, orbit_info->orbit_body) == 0) {
            for (int i=0; i<NO_COORDS; i++) {
                /* Loops for all axis in 3D space.. */
                if (fabs(orbit_info->old_pos[i]) + fabs(body_i->xyz[i]) > fabs(orbit_info->old_pos[i] + body_i->xyz[i])) {
                    /* Checks for a change of sign between old position and new position. */
                    if (orbit_info->list_count[i] > MAX_CROSS) {
                        /* Exits function if already found max amount of crosses of an axis. */
                        return;
                    }
                    /* Appends to list of time step that a crossing occurs.
                       Adds to a tally of how many there have been. */
                    orbit_info->step_list[i][orbit_info->list_count[i]] = step_no;
                    ++orbit_info->list_count[i];
                }
            }
        }
    }
}

/* Calculates average orbital period of a specific period during an interaction. */
void find_orbit(const Parameters *simulation, Orbit_info *orbit_info) {
    long half_orbit_tot[NO_COORDS] = {0, 0, 0};

    for (int i=0; i<NO_COORDS; i++) {
        for (int j=1; j<orbit_info->list_count[i]; j++){
            /* Creates list of number of time steps between each half orbit in each axis. */
            half_orbit_tot[i] += orbit_info->step_list[i][j] - orbit_info->step_list[i][j-1];
        }
    }

    double half_orbit_avg[NO_COORDS];
    double orbit_avg = 0;
    for (int k=0; k<NO_COORDS; k++) {
        /* Calculates average number of time steps for a half orbit in each direction. */
        half_orbit_avg[k] = half_orbit_tot[k] / (orbit_info->list_count[k] - 1);
        /* Finds average time in seconds for a full orbit in each direction and adds to a running total. */
        orbit_avg += (2 * half_orbit_avg[k] * simulation->step);
    }

    int zero_cross_axis = 0;
    for (int n=0; n<NO_COORDS; n++) {
        if (orbit_info->list_count[n] < 2) {
            /* Adds to a running total that discounts any axis corsses that have a total crossing of less than 2. */
            zero_cross_axis++;
        }
    }
    /* Calculates mean average orbital period, in seconds, for a full orbit using all relevant axes. */
    orbit_avg = orbit_avg / (NO_COORDS - zero_cross_axis);

    printf("\nThe average time of %s's orbit around the Sun is %lf seconds.\n", orbit_info->orbit_body, orbit_avg);
    printf("This is equivalent to %lf days.\n", orbit_avg / DAY);
}

/* Function that prints initial data to files in simulation. */
Error initial_file_outputs(Body *bodies, const Context *input_file_context, Context *output_file_context, Context *energy_file_context, const Energy_info *energy_info) {
    Error error;

    error = initial_body_output(bodies, input_file_context, output_file_context);
    if (error != NO_ERROR) {
        return error;
    }

    if (energy_info->energy_calc) {
        error = initial_energy_output(input_file_context, energy_file_context);
        if (error != NO_ERROR) {
            return error;
        }
    }

    return NO_ERROR;
}

/* Function that plots data from simulation. */
Error plot_data(Body *bodies, const Context *output_file_context, const Context *energy_file_context, const Energy_info *energy_info) {
    Error error;

    printf("\nPlotting data...\n\n");
    error = plot_path(bodies, output_file_context);
    if (error != NO_ERROR) {
        return error;
    }
    printf("Image of paths taken by bodies has been saved to 'simulation_path.png'.\n");

    if (energy_info->energy_calc) {
        error = plot_energy(energy_file_context);
        if (error != NO_ERROR) {
            return error;
        }
        printf("Image of energy during the simulation has been saved to 'simulation_energy.png'.\n");
    }

    return NO_ERROR;
}

/* Function that closes files opened during simulation. */
void close_files(Context *output_file_context, Context *energy_file_context, const Energy_info *energy_info) {
    fclose(output_file_context->file);
    printf("Data of paths taken by bodies has been printed to '%s'.\n", output_file_context->file_name);
    if (energy_info->energy_calc) {
        fclose(energy_file_context->file);
        printf("Data of energy during the simulation has been printed to '%s'.\n", energy_file_context->file_name);
    }
}


/* ===== Functions used to find the new accelerations in simulation. ===== */

/* Alternative, and slower, method for calculating the acceleration of all the bodies in the system. */
void find_new_accels_alt(Body *bodies) {
    for (Body *body_i=bodies; body_i!=NULL; body_i=body_i->next) {
        for(int i=0; i<NO_COORDS; i++) {
            /* Sets current acceleration to 0. */
            body_i->a_xyz[i] = 0;
        }
        for (Body *body_j=bodies; body_j!=NULL; body_j=body_j->next) {
            if (body_i == body_j) {
                /* Prevents calculating the force of a body on itself. */
                continue;
            }
            double r_xyz[NO_COORDS];
            for(int j=0; j<NO_COORDS; j++) {
                /* Finds distance between the two bodies ine ach direction. */
                r_xyz[j] = body_i->xyz[j] - body_j->xyz[j];
            }
            /* Finds absolute distance between the two bodies. */
            double r = sqrt(pow(r_xyz[0], 2) + pow(r_xyz[1], 2) + pow(r_xyz[2], 2));

            for (int k=0; k<NO_COORDS; k++) {
                /* Calculates acceleration for the body, due to the other, in each direction. */
                body_i->a_xyz[k] += ( - (G * body_j->mass) / pow(r, 2) ) * (r_xyz[k]/r);
            }
        }
    }
}

/* Function that gives an acceleration to two bodies due to the gravitational force between them. */
void calc_accel(Body *body1, Body *body2, const double *f_xyz) {
    for (int i=0; i<NO_COORDS; i++) {
        body1->a_xyz[i] += f_xyz[i] / body1->mass;
        /* Force is calculated on body1 by body2. */
        body2->a_xyz[i] += - f_xyz[i] / body2->mass;
    }
}

/* Function calculates the gravitational force between two bodies. */
void calc_force(Body *body1, Body *body2) {
    double r_xyz[NO_COORDS];
    double f_xyz[NO_COORDS];

    for(int i=0;i<NO_COORDS;i++) {
        /* Finds distance between the two bodies in each direction. */
        r_xyz[i] = body1->xyz[i] - body2->xyz[i];
    }

    double r = sqrt(pow(r_xyz[0], 2) + pow(r_xyz[1], 2) + pow(r_xyz[2], 2));
    /* Newton's universal law of gravitation: F = - G*m1*m2 / r^2 */
    double force = - (G * body1->mass * body2->mass) / pow(r, 2);

    for (int j=0; j<NO_COORDS; j++) {
        /* Finds direction of force using vector of direction. */
        f_xyz[j] = force * (r_xyz[j]/r);
    }

    calc_accel(body1, body2, f_xyz);
}

/* Function to find the new acceleration of each body in the simulation. */
void find_new_accels(Body *bodies) {
    for (Body *body_i=bodies; body_i!=NULL; body_i=body_i->next) {
        for (int k = 0; k < NO_COORDS; k++) {
            /* Sets all bodies accelerations to 0 before finding new values. */
            body_i->a_xyz[k] = 0;
        }
    }
    for (Body *body_i=bodies; body_i!=NULL; body_i=body_i->next) {
        /* Loops through all bodies finding their accelerations. */
        for (Body *body_j=body_i->next; body_j!=NULL; body_j=body_j->next) {
            /* Loops from the next body to ensure force of body not calculated on itself.
               Due to the way acceleration is calculated, forces between two bodies are not calculated twice. */
            calc_force(body_i, body_j);
        }
    }
}


/* ===== Functions used for the Velocity Verlet method. ===== */

/* Function containing everything that occurs during a step in Velocity Verlet method. */
void vv_step(Body *bodies, const Parameters *simulation, const int step_no, Orbit_info *orbit_info) {
    for (Body *body_i=bodies; body_i!=NULL; body_i=body_i->next) {
        /* Loops through all bodies to give them all new positions and velocities. */
        if (orbit_info->orbit_calc) {
            /* Finds specific body for orbital period calculation. */
            if (strcmp(body_i->name, orbit_info->orbit_body) == 0) {
                for (int i=0; i<NO_COORDS; i++) {
                    /* Once found, appends all current positions to old positions, before finding new ones. */
                    orbit_info->old_pos[i] = body_i->xyz[i];
                }
            }
        }

        for (int j=0; j<NO_COORDS; j++) {
            /* Calculates velocity at half step, as per Velocity Verlet method. */
            body_i->v_xyz[j] += body_i->a_xyz[j] * (simulation->step / 2);
            /* Calculates new position using half-step velocity, as per Velocity Verlet method. */
            body_i->xyz[j] += body_i->v_xyz[j] * simulation->step;
        }
    }
    if (orbit_info->orbit_calc){
        /* Checks for orbit crossings using new positions. */
        find_axis_crossing(bodies, step_no, orbit_info);
    }
    /* Finds new values of acceleration. */
    find_new_accels(bodies);

    for (Body *body_p=bodies; body_p!=NULL; body_p=body_p->next) {
        for (int q=0; q<NO_COORDS; q++) {
            /* Calculates velocity at full step, as per Velocity Verlet method. */
            body_p->v_xyz[q] += body_p->a_xyz[q] * (simulation->step / 2);
        }
    }
}

/* Function for looping over every time step during Velocity Verlet method. */
void vv_stepper(Body *bodies, const Parameters *simulation, Context *output_file_context, Context *energy_file_context, Energy_info *energy_info, Orbit_info *orbit_info) {
    /* Calculates normalisation factor for cumulative error using initial values of energy. */
    energy_info->E_0 = fabs(calc_ke(bodies))+fabs(calc_pe(bodies));
    if (energy_info->energy_calc) {
        /* Calculates energy at first time step. */
        calc_energies(bodies, energy_file_context, FIRST_STEP, energy_info);
    }
    /* Calculates initial acceleration of each body in the system. */
    find_new_accels(bodies);

    for (int i=0; i<simulation->no_steps; i++) {
        /* Loops through each step, for as many steps as decided by user. */
        vv_step(bodies, simulation, STEP_NO, orbit_info);
        /* Prints each body's positions to file at each step. */
        print_body_pos(bodies, output_file_context);

        if (energy_info->energy_calc) {
            /* Calculates energies of system at each time step. */
            calc_energies(bodies, energy_file_context, STEP_NO, energy_info);
        }
    }

    if (orbit_info->orbit_calc) {
        /* Finds average orbital period of specific body from simulation. */
        find_orbit(simulation, orbit_info);
    }
    if (energy_info->energy_calc) {
        /* Prints cumulative error from simulation. */
        printf("The cumulative error of the simulation is %lf\n\n", energy_info->cumulative_err);
    }
}

/* Set up for the Velocity Verlet method. */
Error velocity_verlet(char *argv[], Context *input_file_context, Context *output_file_context, Context *energy_file_context, Parameters *simulation, Energy_info *energy_info, Orbit_info *orbit_info) {
    Error error;
    /* Converts command line arguments into usable variables. */
    error = arg_info(argv, input_file_context, simulation);
    if (error != NO_ERROR) {
        return error;
    }

    Body *bodies = NULL;
    printf("Reading file...\n");
    /* Reads bodies from files and allocates them memory. */
    error = read_bodies(&bodies, input_file_context);
    if (error != NO_ERROR) {
        free_bodies(&bodies);
        return error;
    }

    error = initial_file_outputs(bodies, input_file_context, output_file_context, energy_file_context, energy_info);
    if (error != NO_ERROR) {
        free_bodies(&bodies);
        return error;
    }

    printf("Simulating...\n");
    /* Conducts the simulation using the Velocity Verlet method. */
    vv_stepper(bodies, simulation, output_file_context, energy_file_context, energy_info, orbit_info);
    printf("Simulation complete.\n\n");

    close_files(output_file_context, energy_file_context, energy_info);
    /* Plots data from simulation. */
    error = plot_data(bodies, output_file_context, energy_file_context, energy_info);
    if (error != NO_ERROR) {
        free_bodies(&bodies);
        return error;
    }

    free_bodies(&bodies);
    return NO_ERROR;
}


/* ===== Functions used for the Runge-Kutta method. ===== */

/* Function to find the total number of bodies in the simulation. */
int find_no_bodies(Body *bodies) {
    int body_no = 0;

    for (Body *body_i=bodies; body_i!=NULL; body_i=body_i->next) {
        body_no++;
    }

    return body_no;
}

/* Function to set up the ODE's of the. */
int ODE_setup(double t, const double y[], double f[], void *params) {
    (void)(t); /* Avoids unused parameter warning. */
    Body *bodies = (Body *)params; /* Brings list of bodies in using params variable. */
    int body_no = 0;

    find_new_accels(bodies);

    for (Body *body_i=bodies; body_i!=NULL; body_i=body_i->next) {
        /* Loops for all bodies, setting each up with the correct derivatives and variables for the ODEs. */
        for (int i=0; i<NO_COORDS; i++) {
            body_i->xyz[i] = y[body_no*NO_INPUT_VALS + i];
        }
        for (int j=0; j<NO_COORDS; j++) {
            f[body_no*NO_INPUT_VALS + j] = y[body_no*NO_INPUT_VALS + NO_COORDS + j];
        }
        for (int k=0; k<NO_COORDS; k++) {
            f[body_no*NO_INPUT_VALS + k+NO_COORDS] = body_i->a_xyz[k];
        }
        /* Adds one to body number each time a body's ODE has been set up, to put values in correct position. */
        body_no++;
    }

    return GSL_SUCCESS;
}

/* Function to create the input array, for the GSL driver, of the current positions and velocities of each body. */
void create_input_array(Body *bodies, double *input_array) {
    int body_no = 0; /* Body number used to give each value correct position in array. */

    for (Body *body_i=bodies; body_i!=NULL; body_i=body_i->next) {
        /* Loops through all bodies. */
        for (int j=0; j<NO_COORDS; j++){
            /* Loops through all coordinates to give body positions to array. */
            input_array[body_no*NO_INPUT_VALS + j] = body_i->xyz[j];
        }
        for (int k=0; k<NO_COORDS; k++){
            /* Loops through all coordinates to give body velocities to array. */
            input_array[body_no*NO_INPUT_VALS + NO_COORDS + k] = body_i->v_xyz[k];
        }
        /* Adds one to body number after each body's values has been put in array. */
        body_no++;
    }
}

/* Function to give new positions to each body from the input array, used in the GSL driver. */
void give_new_values(Body *bodies, const double *input_array) {
    int body_no = 0; /* Body number used to obtain the correct position of each value for each body. */

    for (Body *body_i=bodies; body_i!=NULL; body_i=body_i->next) {
        /* Loops through all bodies. */
        for (int j=0; j<NO_COORDS; j++){
            /* Loops through all coordinates to obtain new body positions from array. */
            body_i->xyz[j] = input_array[body_no*NO_INPUT_VALS + j];
        }
        for (int k=0; k<NO_COORDS; k++){
            /* Loops through all coordinates to obtain new body velocities from array. */
            body_i->v_xyz[k] = input_array[body_no*NO_INPUT_VALS + NO_COORDS + k];
        }
        /* Adds one to body number after each body's values has been obtained from array. */
        body_no++;
    }
}

/* Function containing everything that occurs during a step, using the Runge-Kutta method. */
Error rk_step(Body *bodies, const int no_bodies, const Parameters *simulation, gsl_odeiv2_driver *d, double current_time, const int step_no, Orbit_info *orbit_info) {
    Error gsl_error;
    /* Defines size on input array depending on how many bodies there are in the simulation. */
    double input_array[no_bodies * NO_INPUT_VALS];

    if (orbit_info->orbit_calc) {
        for (Body *body_i=bodies; body_i!=NULL; body_i=body_i->next) {
            /* Loops through bodies to find specific body for orbital period calculation. */
            if (strcmp(body_i->name, orbit_info->orbit_body) == 0) {
                for (int i=0; i<NO_COORDS; i++) {
                    /* Once found the correct body, holds onto it's current positions. */
                    orbit_info->old_pos[i] = body_i->xyz[i];
                }
            }
        }
    }
    /* Creates input array for the GSL driver. */
    create_input_array(bodies, input_array);
    /* Applies the driver to the system, moving it forwards a step. New positions given in input array */
    gsl_error = gsl_odeiv2_driver_apply(d, &current_time, (current_time + simulation->step), input_array);
    if (gsl_error != NO_ERROR){
        return gsl_error_print("Step could not be applied.", gsl_error);
    }
    /* Gives new positions and velocities to bodies via the input array, now containing new positions and velocities. */
    give_new_values(bodies, input_array);

    if (orbit_info->orbit_calc) {
        /* Checks for crossing of axis using new positions. */
        find_axis_crossing(bodies, step_no, orbit_info);
    }

    return NO_ERROR;
}

/* Function for looping over every time step when using the Runge-Kutta method. */
Error rk_stepper(Body *bodies, const Parameters *simulation, Context *output_file_context, Context *energy_file_context, Energy_info *energy_info, Orbit_info *orbit_info) {
    Error error;
    double current_time = 0;
    /* Finds number of bodies in simulation, to be used for defining sizes of array and values used with gsl functions. */
    int no_bodies = find_no_bodies(bodies);
    /* Calculates normalisation factor used in find the cumulative error of the simulation. */
    energy_info->E_0 = fabs(calc_ke(bodies))+fabs(calc_pe(bodies));
    if (energy_info->energy_calc) {
        /* Calculates initial energy of the system. */
        calc_energies(bodies, energy_file_context, FIRST_STEP, energy_info);
    }
    /* Defining the GSL system used in the simulation.
       No Jacobian is used so GSL_SUCCESS is implemented straight into the system. */
    gsl_odeiv2_system sys = {ODE_setup, GSL_SUCC, (size_t)no_bodies * NO_INPUT_VALS, bodies};
    /* Defines, and allocates memory for, the driver used to eveole the system forwards in time.
       The step is defined in the driver as the rk4 step. */
    gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk4, simulation->step, ABS_ERR, REL_ERR);

    for (int i=0; i<simulation->no_steps; i++) {
        /* Loops through each step, for as many steps as decided by user. */
        error = rk_step(bodies, no_bodies, simulation, d, current_time, STEP_NO, orbit_info);
        if (error != NO_ERROR){
            /* Frees driver and returns error if GSL driver cannot apply next step. */
            gsl_odeiv2_driver_free(d);
            return error;
        }
        /* Prints current positions of each body to file at each time step. */
        print_body_pos(bodies, output_file_context);

        if (energy_info->energy_calc) {
            /* Calculates energy of system at time step. */
            calc_energies(bodies, energy_file_context, STEP_NO, energy_info);
        }
    }

    if (orbit_info->orbit_calc) {
        /* Finds average orbital period of a specified body during simulation. */
        find_orbit(simulation, orbit_info);
    }
    if (energy_info->energy_calc) {
        printf("The cumulative error of the simulation is %lf\n\n", energy_info->cumulative_err);
    }
    /* Frees gsl driver before leaving function. */
    gsl_odeiv2_driver_free(d);
    return NO_ERROR;
}

/* Set up for the Runge-Kutta method. */
Error runge_kutta(char *argv[], Context *input_file_context, Context *output_file_context, Context *energy_file_context, Parameters *simulation, Energy_info *energy_info, Orbit_info *orbit_info) {
    Error error;
    /* Converts command line arguments into usable variables. */
    error = arg_info(argv, input_file_context, simulation);
    if (error != NO_ERROR) {
        return error;
    }

    Body *bodies = NULL;
    printf("Reading file...\n");
    /* Reads bodies from files and allocates them memory. */
    error = read_bodies(&bodies, input_file_context);
    if (error != NO_ERROR) {
        free_bodies(&bodies);
        return error;
    }

    error = initial_file_outputs(bodies, input_file_context, output_file_context, energy_file_context, energy_info);
    if (error != NO_ERROR) {
        free_bodies(&bodies);
        return error;
    }

    printf("Simulating...\n");
    /* Conducts the simulation using the Runge-Kutta method. */
    rk_stepper(bodies, simulation, output_file_context, energy_file_context, energy_info, orbit_info);
    printf("Simulation complete.\n\n");

    close_files(output_file_context, energy_file_context, energy_info);
    /* Plots data from simulation. */
    error = plot_data(bodies, output_file_context, energy_file_context, energy_info);
    if (error != NO_ERROR) {
        free_bodies(&bodies);
        return error;
    }

    free_bodies(&bodies);
    return NO_ERROR;
}


int main(int argc, char *argv[]) {
    Error error;
    /* Creates parameter structure to store information on simulation. */
    Parameters simulation;
    /* Creates energy information structure for use when calculating energies during simulation. */
    Energy_info energy_info;
    energy_info.energy_calc = true; /* Option as to whether to calculate energy during simulation. */
    energy_info.cumulative_err = 0; /* Sets cumulative error to 0. */
    /* Creates orbit information structure for use when calculating average orbital period during simulation. */
    Orbit_info orbit_info;
    orbit_info.orbit_calc = false; /* Option as to whether to average orbital period of a body during simulation. */
    if (orbit_info.orbit_calc) {
        strcpy(orbit_info.orbit_body, "Earth"); /* Specified body whose average orbital period is calculated. */
        for (int i=0; i<NO_COORDS; i++) {
            orbit_info.list_count[i] = 0; /* Sets counter for number of crossing on each axis to 0. */
        }
    }

    /* Creates context structures for all files used in program. */
    Context input_file_context;
    Context output_file_context;
    output_file_context.file_name = "simulation_path_data.txt"; /* Sets name of output file for body position data. */
    Context energy_file_context;
    energy_file_context.file_name = "simulation_energy_data.txt"; /* Sets name of output file for energy data. */

    /* Checks that option command line argument is in correct form. */
    if (argv[OPTION_ARG][FIRST_CHAR] != '-' || strlen(argv[OPTION_ARG]) != OPTION_ARG_LEN) {
        return invalid_args_print("");
    }
    char option = argv[OPTION_ARG][SECOND_CHAR];

    switch (option){
        case 'h':
            /* Option to print help with command line arguments. */
            help();
            break;
        case 'v':
            /* Option for Velocity Verlet method. */
            if (argc != NO_ARGS) {
                /* Checks number of command line arguments is correct. */
                return invalid_args_print("Incorrect number of arguments.\n");
            }
            error = velocity_verlet(argv, &input_file_context, &output_file_context, &energy_file_context, &simulation, &energy_info, &orbit_info);
            if (error != NO_ERROR) {
                return error;
            }
            break;
        case 'r':
            /* Option for Runge-Kutta method. */
            if (argc != NO_ARGS) {
                /* Checks number of command line arguments is correct. */
                return invalid_args_print("Incorrect number of arguments.\n");
            }
            error = runge_kutta(argv, &input_file_context, &output_file_context, &energy_file_context, &simulation, &energy_info, &orbit_info);
            if (error != NO_ERROR) {
                return error;
            }
            break;
        default:
            /* Prints error if invalid option selected. */
            return invalid_args_print("The choice of integration scheme was invalid.");
    }

    return NO_ERROR;
}

/*
    Example terminal output when running Velocity Verlet method and calculating the orbital period of the
    Earth around the Sun:


jemgodden2@Jems-Air n_body % ./a.out -v SunEarthMoon.txt 3600 72000000
Reading file...
Simulating...

The average time of Earth's orbit around the Sun is 31629600.000000 seconds.
This is equivalent to 366.083333 days.
The cumulative error of the simulation is 0.004954

Simulation complete.

Data of paths taken by bodies has been printed to 'simulation_path_data.txt'.
Data of energy during the simulation has been printed to 'simulation_energy_data.txt'.

Plotting data...

Image of paths taken by bodies has been saved to 'simulation_path.png'.
Image of energy during the simulation has been saved to 'simulation_energy.png'.


    Example terminal output when running Runge-Kutta method:


jemgodden2@Jems-Air n_body % ./a.out -r SunEarthMoon.txt 3600 80000000
The total time of the simulation was not an exact multiple of the step time.
Instead, 22222 steps were used for a total time of 79999200.000000 seconds during the simulation.

Reading file...
Simulating...
The cumulative error of the simulation is 0.005335

Simulation complete.

Data of paths taken by bodies has been printed to 'simulation_path_data.txt'.
Data of energy during the simulation has been printed to 'simulation_energy_data.txt'.

Plotting data...

Image of paths taken by bodies has been saved to 'simulation_path.png'.
Image of energy during the simulation has been saved to 'simulation_energy.png'.


    The saved files alongside the code are for the Runge-Kutta simulation.

*/