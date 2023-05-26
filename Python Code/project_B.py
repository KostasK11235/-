import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def create_population(num_chromosomes):
    population = []

    for i in range(num_chromosomes):
        chromosome = ''
        for j in range(12):
            # generate a random number between [0,1]
            number = random.uniform(0, 1)
            number = round(number, 4)
            # print(number)
            # encode the number using binary encoding using 14 bits
            binary = format(int(number * (10 ** 4)), '014b')
            # print(binary)
            # concatenate the binary number to the chromosome
            chromosome += binary
        population.append(chromosome)

    return population


def check_outliers(population_list):
    for chromosome in population_list:
        chromosome_list = list(chromosome)

        for k in range(0, len(chromosome_list), 14):
            gene = "".join(chromosome_list[k:k+14])
            # print(gene, "->", int(gene, 2), "->", int(gene, 2)>10000)
            if int(gene, 2) > 10000:
                new_gene = random.uniform(0, 1)
                new_gene = round(new_gene, 4)
                bin_gene = format(int(new_gene * (10 ** 4)), '014b')
                chromosome_list[k:k+14] = list(bin_gene)
                # print(list(bin_gene))
                # print(chromosome_list)

        chromosome_index = population_list.index(chromosome)
        temp = "".join(chromosome_list)
        population_list[chromosome_index] = temp
        # print(population_list)
    return population_list


def score_function(chromosome, bin_group_means):
    # cosine similarity sum for all other 4 classes between them and the given chromosome
    cos_sum = 0
    array1 = np.array(list(chromosome), dtype=int)
    norm1 = np.linalg.norm(array1)

    array_sit = np.array(list(bin_group_means[4]), dtype=int)
    norm_sit = np.linalg.norm(array_sit)

    for i in range(0, 4):
        array2 = np.array(list(bin_group_means[i]), dtype=int)
        norm2 = np.linalg.norm(array2)
        cos_sum += (np.dot(array1, array2))/(norm1*norm2)

    f_v = round(((4*(np.dot(array1, array_sit)/(norm1*norm_sit))) + (1-0.25*cos_sum))/5, 4)

    return f_v

def tournament_selection(population, bin_group_means):
    group_size = 2
    next_gen = []

    for i in range(len(population)):
        # select the chromosomes for the group
        best_score = -1
        best_chromosome = -1
        chosen_chromosome = random.sample(range(len(population)), group_size)

        # find the best chromosome from the selected and pass it to next generation
        for k in chosen_chromosome:
            chromosome_score = score_function(population[k], bin_group_means)
            if chromosome_score > best_score:
                best_score = chromosome_score
                best_chromosome = k

        next_gen.append(population[best_chromosome])

    # checkpoint code
    print("Tournament population: ")
    for person in next_gen:
        print(person)

    return next_gen

def uniform_crossover(population, crossover_probability):
    new_population = []
    crossover_list = []

    # generate crossover chances for each chromosome and move it to next gen or crossover it accordingly
    crossover_chances = [round(random.uniform(0, 1), 3) for i in range(len(population))]
    print("chances:", crossover_chances)
    for i in crossover_chances:
        if i < crossover_probability:
            crossover_list.append(population[crossover_chances.index(i)])
        else:
            new_population.append(population[crossover_chances.index(i)])

    # checkpoint code
    print("cross list len", len(crossover_list))
    print("cross 1: ")
    for person in new_population:
        print(person)

    if len(crossover_list) % 2 == 1:
        new_population.append(crossover_list[len(crossover_list)-1])
        del crossover_list[len(crossover_list)-1]

    print("Cross 2: ")
    for person in new_population:
        print(person)

    # uniformly crossover the selected chromosomes
    for k in range(0, len(crossover_list), 2):
        parent1 = crossover_list[k]
        parent2 = crossover_list[k+1]

        # create child chromosomes
        child1 = ''
        child2 = ''
        for j in range(len(parent1)):
            if random.choice([0, 1]) == 1:
                child1 += parent1[j]
                child2 += parent2[j]
            else:
                child1 += parent2[j]
                child2 += parent1[j]

        new_population.append(child1)
        new_population.append(child2)

    # checkpoint code
    print("Crossed population: ")
    for person in new_population:
        print(person)

    print("************************************")
    return new_population


def chromosome_mutation(population, mutation_probability, bin_group_mean):
    mutated_generation = []
    chromosome_scores = []

    cloned_population = population

    # get the scores for all chromosomes
    for chromosome in cloned_population:
        chromosome_scores.append(score_function(chromosome, bin_group_mean))

    # get the index of the elite chromosomes
    max_score = max(chromosome_scores)
    max_indexes = [index for index, value in enumerate(chromosome_scores) if value == max_score]

    # get the elite chromosomes directly to the next generation and
    # remove them from the to be mutated population
    for i in max_indexes:
        mutated_generation.append(cloned_population[i])

    # error here idk...
    for k in max_indexes:
        del cloned_population[k]

    # mutate the remaining population
    for person in cloned_population:
        person_list = list(person)
        for bit in range(len(person)):
            bit_mutation_chance = round(random.uniform(0, 1), 3)
            if bit_mutation_chance < mutation_probability:
                person_list[bit] = '0' if person_list[bit] == '1' else '1'

        person_index = cloned_population.index(person)
        temp = "".join(person_list)
        mutated_generation.append(temp)

    # checkpoint code
    print("Mutated population: ")
    for person in mutated_generation:
        print(person)

    print("************************************")
    return mutated_generation

def get_int_values(chromosome):
    int_values = []
    chromosome_list = list(chromosome)

    for i in range(0, len(chromosome_list), 14):
        gene = "".join(chromosome_list[i:i+14])
        int_gene = int(gene, 2)
        int_values.append(int_gene)

    return int_values


def main():
    # reading/standardizing and normalizing the data
    data = pd.read_csv("new_data_1.csv")

    data = data.drop(['gender', 'age', 'height', 'weight', 'body_mass_index'], axis=1)
    # print(data.shape)

    positions = data.iloc[:, 1:].values
    stand_data = StandardScaler().fit_transform(X=positions)
    norm_data = MinMaxScaler(copy=False).fit_transform(X=positions)
    # print(stand_data.shape)

    # get unique values and group indices
    unique_values, group_indices = np.unique(norm_data[:, 12], return_inverse=True)

    # group the original array
    grouped_positions = [norm_data[group_indices == i] for i in range(len(unique_values))]

    # get the mean values of each column in every group
    mean_values = []
    for group in grouped_positions:
        group_mean = np.mean(group[:, :-1], axis=0)
        mean_values.append(group_mean)

    # mean values are grouped from 1 to 5, so mean_values[0] has means of class sitting_down...
    # mean_values[4] has mean values of class sitting
    # round mean values to 4 decimals and get their binary representation
    binary_means = []
    for mean in mean_values:
        bin_mean = ''
        for i in range(len(mean)):
            mean[i] = round(mean[i], 4)
            bin_form = format(int(mean[i] * (10 ** 4)), '014b')
            bin_mean += bin_form
        binary_means.append(bin_mean)
        # print(mean)

    print("Values of sitting class: ", mean_values[4])

    # create initial population
    crossover_chance = 0.6
    mutation_chance = 00.1
    max_generation = 1000
    generation = 0
    population = create_population(8)

    # print initial population
    for person in population:
        print(person)

    for i in range(0, 1):
        cloned_population = population
        while generation < 20:
            generation += 1
            tournament_population = tournament_selection(cloned_population, binary_means)
            crossed_population = uniform_crossover(tournament_population, crossover_chance)
            crossed_population = check_outliers(crossed_population)
            mutated_population = chromosome_mutation(crossed_population, mutation_chance, binary_means)
            next_generation = check_outliers(mutated_population)
            cloned_population = next_generation

            scores = []
            for person in cloned_population:
                scores.append(score_function(person, binary_means))
                print(person)

            max_index = [index for index, value in enumerate(scores) if value == max(scores)]
            print("Best chromosome values: ")

            max_chromosome = []
            for j in max_index:
                max_chromosome.append(get_int_values(cloned_population[j]))

            print(max_chromosome)
            print("*****************************************")


main()
