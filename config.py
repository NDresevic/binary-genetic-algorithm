# Configuration file that contains input parameters for genetic algorithm.

from projekat1 import loss_goldstein_price, encode, decode, tournament_selection, inversion_mutation, \
    two_point_crossover

# the function that needs to be optimized
loss_function = loss_goldstein_price
left_X = -2
right_X = 2
left_Y = -2
right_Y = 2
min_f = 3
# the precision represented as number of decimals for parameters of the function
decimal_precision = 3
encode_function = encode
decode_function = decode
crossover_function = two_point_crossover
selection_function = tournament_selection
mutation_function = inversion_mutation
population_size = 150
next_population_size = 150
# maximum number of iterations for each run of algorithm
max_iterations = 500
# probability of mutation for each chromosome
mutation_rate = 0.4
# number that defines how many times the algorithm is run with the same parameters
number_of_runs = 5

parameters = {
    "loss_f": loss_function,
    "left_X": left_X,
    "right_X": right_X,
    "left_Y": left_Y,
    "right_Y": right_Y,
    "min_f": min_f,
    "precision": decimal_precision,
    "enc_f": encode_function,
    "dec_f": decode_function,
    "crossover_f": crossover_function,
    "selection_f": selection_function,
    "mut_f": mutation_function,
    "pop_size": population_size,
    "next_pop_size": next_population_size,
    "max_iter": max_iterations,
    "mut_rate": mutation_rate,
    "runs": number_of_runs
}
