import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from bitstring import BitArray

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
                print(list(bin_gene))
                print(chromosome_list)

        population_list = "".join(chromosome_list)
        print(population_list)
    return population_list


def main():
    # reading/standardizing and normalizing the data
    data = pd.read_csv("new_data_1.csv")

    data = data.drop(['gender', 'age', 'height', 'weight', 'body_mass_index'], axis=1)
    print(data.shape)

    used = data.iloc[:, 1:-1].values
    y = data.iloc[:, 13:14].values

    stand_data = StandardScaler().fit_transform(X=used)
    norm_data = MinMaxScaler(copy=False).fit_transform(X=stand_data)
    # print(norm_data)

    # grouped_data = np.split(stand_data, np.where(stand_data[:, 18][1:] != stand_data[:, 18][:-1])[0]+1)
    # print(grouped_data)
    # new = stand_data.groupby('class,,')
    # print(new)
    # mean_values = data.groupby('class,,')[['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4']].mean()
    # print(mean_values)


