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

    f_v = ((4*(np.dot(array1, array_sit)/(norm1*norm_sit))) + (1-0.25*cos_sum))/5

    return f_v

def tournament_selection(population, bin_group_means):
    group_size = 4
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

    return next_gen

def uniform_crossover(population, crossover_probability):
    new_population = []
    crossover_list = []

    # generate crossover chances for each chromosome and move it to next gen or crossover it accordingly
    crossover_chances = [round(random.uniform(0, 1), 3) for i in range(4)]
    for i in crossover_chances:
        if i < crossover_probability:
            crossover_list.append(population[crossover_chances.index(i)])
        else:
            new_population.append(population[crossover_chances.index(i)])

    # αν έχουμε μονό αριθμό επιλεγμένων χρωμοσωμάτων?
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

    return new_population


def chromosome_mutation(population, mutation_probability, bin_group_mean):
    mutated_generation = []
    chromosome_scores = []

    # get the scores for all chromosomes
    for chromosome in population:
        chromosome_scores.append(score_function(chromosome, bin_group_mean))

    # get the index of the elite chromosomes
    max_score = max(chromosome_scores)
    max_indexes = [index for index, value in enumerate(chromosome_scores) if value == max_score]

    # get the elite chromosomes directly to the next generation and
    # remove them from the to be mutated population
    for i in max_indexes:
        mutated_generation.append(population[i])
        del population[i]

    # mutate the remaining population
    for person in population:
        person_list = list(person)
        for bit in range(len(person)):
            bit_mutation_chance = round(random.uniform(0, 1), 3)
            if bit_mutation_chance < mutation_probability:
                person_list[bit] = '0' if person_list[bit] == '1' else '1'

        person_index = population.index(person)
        temp = "".join(person_list)
        population[person_index] = temp

    return mutated_generation


def main():
    # reading/standardizing and normalizing the data
    data = pd.read_csv("new_data_1.csv")

    data = data.drop(['gender', 'age', 'height', 'weight', 'body_mass_index'], axis=1)
    print(data.shape)

    positions = data.iloc[:, 1:].values
    stand_data = StandardScaler().fit_transform(X=positions)
    norm_data = MinMaxScaler(copy=False).fit_transform(X=positions)
    print(stand_data.shape)

    # get unique values and group indices
    unique_values, group_indices = np.unique(norm_data[:, 12], return_inverse=True)

    # group the original array
    grouped_positions = [norm_data[group_indices == i] for i in range(len(unique_values))]

    # get the mean values of each column in every group
    mean_values = []
    for group in grouped_positions:
        group_mean = np.mean(group[:, :-1], axis=0)
        mean_values.append(group_mean)

    # mean values are grouped from 1 to 5, so mean_values[0] has means of class sitting_down...mean_values[4] has
    # mean values of class sitting
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

    # print the binary form of every mean
    # for item in binary_means:
    #    print(item)

    pop = create_population(5)

    for person in pop:
        print(person)

    pop[0] = pop[0][0:14] + format(int(16000), '014b') + pop[0][28:]

    print("altered: ")
    for person in pop:
        print(person)

    new_population = check_outliers(pop)
    print("new")
    for k in new_population:
        print(k)
    # grouped_data = np.split(stand_data, np.where(stand_data[:, 13][1:] != stand_data[:, 13][:-1])[0]+1)
    # print(grouped_data)
    # new = stand_data.groupby('class,,')
    # print(new)
    # mean_values = data.groupby('class,,')[['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4']].mean()
    # print(mean_values)


main()
