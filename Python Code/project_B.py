import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from keras.callbacks import EarlyStopping

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

population = create_population(5)

print(population[0])
print(population[1])

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


