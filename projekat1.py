from math import trunc
import config
import random
import matplotlib.pyplot as plt


def loss_goldstein_price(x, y):
    """
    Calculates Goldstein-Price function with given two parameters x and y.
    """
    f = (1 + ((x + y + 1)**2) * (19 - 14*x + 3*(x**2) - 14*y + 6*x*y + 3*(y**2))) * \
        (30 + ((2*x - 3*y)**2) * (18 - 32*x + 12*(x**2) + 48*y - 36*x*y + 27*(y**2)))
    return f


def required_bits(dec_num, begin, end):
    """
    Helper function that calculates required number of bits to represent all numbers in interval with given decimal
    precision.
    """
    numbers = (end - begin) * (10 ** dec_num) + 1           # number of possible numbers in interval
    bits_num = 0                                            # number of bits needed to represent all numbers binary
    while 2 ** bits_num < numbers:
        bits_num += 1
    return bits_num


def encode(chromosome, dec_num, bits_num, begin, end):
    """
    Encodes real number given as the input to chromosome binary. Besides number, input parameters are also the decimal
    precision and possible interval for the input number. Encoding is done by representing left bound of the interval
    as all 0s and right end of the interval as all 1s.
    For example, if interval is [-5, 1] and precision is 3 decimal places, the interval is transformed to [-5000, 1000].
    Now, -5000 is encoded as 13 zeroes, and 1000 is encoded as 13 ones. So, if the given number for encoding is
    -4.99799093423, it becomes -4997 and is encoded to 0000000000011 (-4997 = -5000 + 3 = 0000000000000 + 0000000000011
    = 0000000000011).

    :param chromosome: input number
    :param dec_num: precision, number of decimals
    :param bits_num: number of bits required to represent all numbers from interval
    :param begin: beginning of interval
    :param end: end of interval
    :return: binary representation of chromosome
    """
    left_bound = trunc(begin * 10 ** dec_num)               # left bound of interval for encoding, represented as all 0s
    chromosome = trunc(chromosome * 10 ** dec_num)
    binary = int(bin(chromosome - left_bound)[2:])          # binary value of chromosome based on left bound
    encoded_ch = []                                         # encoded chromosome - result of the function
    while bits_num > 0:
        if binary > 0:
            encoded_ch.insert(0, binary % 10)               # insert binary value into list
            binary //= 10
        else:
            encoded_ch.insert(0, 0)                         # add 0s at the beginning of the list
        bits_num -= 1
    return encoded_ch


def decode(chromosome, dec_num, begin, end):
    """
    Inversion of encode, takes a binary number and decodes it to real number.
    For example, if interval is [-5, 1] and precision is 3 decimal places, the number for decoding represented in
    decimal is 2157. The result number is -2.843 (decoding is done with formula -5 + 0.0001 * 2157 = -2.843).

    :param chromosome: input binary number
    :param dec_num: precision, number of decimals
    :param begin: beginning of interval
    :param end: end of interval
    :return: real number
    """
    num = 0
    degree = 0
    for i in range(len(chromosome) - 1, -1, -1):            # converts binary to decimal
        num += chromosome[i] * 2 ** degree
        degree += 1
    decoded_ch = begin + num * 1 / (10 ** dec_num)
    decoded_ch = trunc(decoded_ch * 10 ** dec_num) / 10 ** dec_num
    if decoded_ch > end:                                    # if number is out of range, it becomes end of the interval
        decoded_ch = end
    return decoded_ch


def two_point_crossover(h1, h2):
    """
    Creates two child chromosomes by doing two-point crossover. On the given chromosomes it picks two random points
    and swaps the bits in between the two positions between the parent chromosomes.

    :param h1: first parent chromosome
    :param h2: second parent chromosome
    :return: two new child chromosomes
    """
    r1 = random.randrange(1, len(h1) - 1)
    while True:
        r2 = random.randrange(1, len(h1) - 1)
        if r1 != r2:
            break
    h3 = h1[:min(r1, r2)] + h2[min(r1, r2):max(r1, r2)] + h1[max(r1, r2):]
    h4 = h2[:min(r1, r2)] + h1[min(r1, r2):max(r1, r2)] + h2[max(r1, r2):]
    return h3, h4


def inversion_mutation(chromosome, probability):
    """
    Creates a mutated chromosome by doing inversion with given probability. It is done by picking two random points on
    chromosome and flipping its bits in between the two points.

    :param chromosome: chromosome for mutation
    :param probability: mutation probability level
    :return: mutated chromosome
    """
    if random.random() <= probability:
        r1 = random.randrange(1, len(chromosome) - 1)
        while True:
            r2 = random.randrange(1, len(chromosome) - 1)
            if r1 != r2:
                break
        for i in range(min(r1, r2), max(r1, r2)):
            chromosome[i] ^= 1
    return chromosome


def tournament_selection(f_loss, population, t_size):
    """
    Returns parameters of chromosome that won the competition (has the best loss function) from specified number of
    chromosomes in population, where each of them have equal chances to participate in competition.

    :param f_loss: loss function
    :param population: chromosomes population
    :param t_size: size of tournament for selection
    :return: parameters with best loss
    """
    chosen_x = []
    chosen_y = []
    while len(chosen_x) < t_size:                               # chooses random t_size chromosomes from population
        r = random.randrange(len(population))
        chosen_x.append(population[r][0])
        chosen_y.append(population[r][1])
    best_x, best_y = None, None
    best_loss = None
    for h1, h2 in zip(chosen_x, chosen_y):                      # finds the best loss among them
        curr_loss = f_loss(h1, h2)
        if best_x is None or curr_loss < best_loss:
            best_loss = curr_loss
            best_x = h1
            best_y = h2
    return best_x, best_y


def genetic_algorithm(params):
    f_loss = params["loss_f"]
    left_x = params["left_X"]
    right_x = params["right_X"]
    left_y = params["left_Y"]
    right_y = params["right_Y"]
    min_f = params["min_f"]
    dec_num = params["precision"]
    enc_f = params["enc_f"]
    dec_f = params["dec_f"]
    crossover_f = params["crossover_f"]
    selection_f = params["selection_f"]
    mut_f = params["mut_f"]
    pop_size = params["pop_size"]
    next_pop_size = params["next_pop_size"]
    max_iter = params["max_iter"]
    mut_rate = params["mut_rate"]
    runs = params["runs"]

    bits_x = required_bits(dec_num, left_x, right_x)
    bits_y = required_bits(dec_num, left_y, right_y)
    min_f = trunc(min_f * 10 ** dec_num) / 10 ** dec_num

    print("GA population: ", pop_size)
    draw_values_x = {}                                          # x values for plotting
    draw_values_y = {}                                          # y values for plotting
    avg_values = {}
    for i in range(runs):
        best = None
        best_loss = None
        population = []
        t = 0
        sum_loss = 0
        s_iter = 0
        best_ever_loss = None
        best_ever = None
        draw_values_x[i] = []
        draw_values_y[i] = []
        avg_values[i] = []

        for j in range(pop_size):                               # generate random population
            x = random.uniform(left_x, right_x)
            y = random.uniform(left_y, right_y)
            population.append((x, y))

        while t < max_iter:
            n_pop = population[:]
            while len(n_pop) < pop_size + next_pop_size:
                # tournament selection
                h1 = selection_f(f_loss, population, 3)
                h2 = selection_f(f_loss, population, 3)
                # encoding
                h1_x = enc_f(h1[0], dec_num, bits_x, left_x, right_x)
                h1_y = enc_f(h1[1], dec_num, bits_y, left_y, right_y)
                h2_x = enc_f(h2[0], dec_num, bits_x, left_x, right_x)
                h2_y = enc_f(h2[1], dec_num, bits_y, left_y, right_y)
                # crossover
                h3, h4 = crossover_f(h1_x + h1_y, h2_x + h2_y)
                # mutation
                h3 = mut_f(h3, mut_rate)
                h4 = mut_f(h4, mut_rate)
                # decoding
                h3_x = dec_f(h3[:bits_x], dec_num, left_x, right_x)
                h3_y = dec_f(h3[bits_x:], dec_num, left_y, right_y)
                h4_x = dec_f(h4[:bits_x], dec_num, left_x, right_x)
                h4_y = dec_f(h4[bits_x:], dec_num, left_y, right_y)
                # adding new chromosomes to population
                n_pop.append((h3_x, h3_y))
                n_pop.append((h4_x, h4_y))

            avg_loss = 0
            for tup in n_pop:
                avg_loss += f_loss(tup[0], tup[1])
            avg_loss /= len(n_pop)
            population = sorted(n_pop, key=lambda tup: f_loss(tup[0], tup[1]))[:pop_size]
            curr_loss = f_loss(population[0][0], population[0][1])

            if best_loss is None or curr_loss < best_loss:
                best_loss = curr_loss
                best = population[0]
            draw_values_x[i].append(t)
            draw_values_y[i].append(best_loss)
            avg_values[i].append(avg_loss)
            t += 1

            if best_loss <= min_f:                              # break if algorithm reaches minimum
                break

        sum_loss += best_loss
        s_iter += t
        if best_ever_loss is None or best_loss < best_ever_loss:
            best_ever_loss = best_loss
            best_ever = best

        print(i, "RUN -", t, "iterations")
        print("Best solution:", best_loss, "\nAverage solution:", sum_loss / max_iter)

    x = encode(best_ever[0], dec_num, bits_x, left_x, right_x)
    y = encode(best_ever[1], dec_num, bits_y, left_y, right_y)
    print("\nBest chromosome:", best_ever, "\nx:", x, "\ny:", y, "\n")
    draw(draw_values_x, draw_values_y, "Best loss", 3, 10)
    draw(draw_values_x, avg_values, "Average loss", 3, 30000)


def draw(x, y, title, begin_y, end_y):
    plt.plot(x[0], y[0], color='red', label='run 0')
    plt.plot(x[1], y[1], color='green', label='run 1')
    plt.plot(x[2], y[2], color='blue', label='run 2')
    plt.plot(x[3], y[3], color='yellow', label='run 3')
    plt.plot(x[4], y[4], color='purple', label='run 4')
    plt.axis([0, 500, begin_y, end_y])
    plt.xlabel("Number of generations")
    plt.ylabel("Loss function")
    plt.legend()
    plt.title(title)
    plt.show()


def main():
    genetic_algorithm(config.parameters)


if __name__ == '__main__':
    main()
