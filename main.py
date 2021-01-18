from utils import generate_all_cutting_configuration, val2index, conf2mat, sort_length, covering_model
from knapsack import knapsack
import numpy as np
import sys

sys.setrecursionlimit(int(1e4))
L = 120
num = 240
demand = [10, 11, 11, 12, 12, 12, 10, 11, 12, 10]
lengths = [92, 59, 97, 32, 38, 55, 80, 75, 108, 57]

len_demand = dict(zip(range(len(demand)), demand))

configurations = generate_all_cutting_configuration(L, sorted(lengths))

configurations = [configuration for configuration in configurations if configuration is not None]

len_dict = val2index(lengths)

conf_mat = conf2mat(configurations, len_dict, lengths)
num_conf, num_shape = conf_mat.shape

weight = np.array([3, 5, 9])
value = np.array([1 / 5, 1 / 3, 1])
demand = np.array([25, 20, 15])
capacity = 17
n = len(weight)

lengths = np.array([92, 59, 97, 32, 38, 55, 80, 75, 108, 57])
demand = np.array([10, 11, 11, 12, 12, 12, 10, 11, 12, 10])

lengths, demand = sort_length(lengths, demand)
len_demand = dict(zip(range(len(demand)), demand))
capacity = 120

from column_generation import generate_optimal_column

configuration = generate_optimal_column(lengths, capacity, demand)
covering_model(configuration, len_demand)
