from utils import generate_all_cutting_configuration, val2index, conf2mat, sort_length, covering_model
from knapsack import knapsack
from column_generation import generate_optimal_column
import numpy as np
import sys
import time

sys.setrecursionlimit(int(1e4))
L = 120
num = 240
demand = [10, 11, 11, 12, 12, 12, 10, 11, 12, 10, 1000, 800]
lengths = [92, 59, 97, 32, 38, 55, 80, 75, 108, 57, 1, 2]

# len_demand = dict(zip(range(len(demand)), demand))

configurations = generate_all_cutting_configuration(L, lengths)

len_dict = val2index(lengths)
len_demand = dict(zip(range(len(demand)), demand))
conf_mat = conf2mat(configurations, len_dict, lengths)
res = covering_model(conf_mat, len_demand)


demand = np.array([10, 11, 11, 12, 12, 12, 10, 11, 12, 10, 1000, 800])
lengths = np.array([92, 59, 97, 32, 38, 55, 80, 75, 108, 57, 1, 2])


lengths, demand = sort_length(lengths, demand)
len_demand = dict(zip(range(len(demand)), demand))
capacity = 120
begin = time.perf_counter()
configuration, num_iter = generate_optimal_column(lengths, capacity, demand)
elapsed = time.perf_counter() - begin
print(f"Column generation takes {elapsed} seconds")
cg_res = covering_model(configuration, len_demand)
