import numpy as np

from os import listdir
import os
from os.path import isfile, join
from random import shuffle

import csv


test = os.listdir("datasets")

for item in test:
    if item.endswith(".npy"):
        os.remove(os.path.join("datasets", item))

datasets = [f for f in listdir("datasets") if isfile(join("datasets", f))]
ratings = ['C', 'CC', 'CCC', 'B', 'BB', 'BBB', 'A', 'AA', 'AAA']

train_data = []
test_data = []

with open('data.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=",")
    row_count = 0
    for row in csv_reader:
        if row_count == 0:
            print(row)
            row_count += 1
        elif row_count < 5000:
            row_values = row[8:]
            row_values[2] = ratings.index(row_values[2])
            row_values[:] = [x.replace(",","") if type(x) is str else x for x in row_values]
            row_values[0],row_values[2] = row_values[2],row_values[0]
            row_values[0:] = [ 0 if x == '' else float(x) for x in row_values]
            row_values[0] = int(row_values[0])
            train_data.append(row_values)
            row_count +=1
        elif row_count < 15000:
            row_values = row[8:]
            row_values[2] = ratings.index(row_values[2])
            row_values[:] = [x.replace(",","") if type(x) is str else x for x in row_values]
            row_values[0],row_values[2] = row_values[2],row_values[0]
            row_values[0:] = [ 0 if x == '' else float(x) for x in row_values]
            row_values[0] = int(row_values[0])
            test_data.append(row_values)
            print(row_values)
            row_count +=1
        else:
            break
test_set = np.array(test_data)
train_set = np.array(train_data)

np.save("datasets/test_data", test_set)
np.save("datasets/train_data", train_set)
