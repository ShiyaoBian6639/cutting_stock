from utils import generate_all_cutting_configuration, val2index, conf2mat
from knapsack import knapsack
import numpy as np
import sys

sys.setrecursionlimit(int(1e9))
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

weight = [3, 5, 9]
value = [1/5, 1/3, 1]
n = len(weight)
index = np.argsort(weight)
weight_arr = np.zeros(n)
value_arr = np.zeros(n)
for i in range(n):
    weight_arr[i] = weight[index[i]]
    value_arr[i] = value[index[i]]
configuration = np.zeros(n, dtype=int)
print(weight_arr)
print(value_arr)
gain, configuration = knapsack(weight_arr, value_arr, 17, n, configuration)

# from column_generation import generate_basis
#
# basis, configuration = generate_basis(np.array(lengths), L)
