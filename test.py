from knapsack import knapsack
import numpy as np


k = 10000
s = 600
weight = np.arange(s, k + s)
value = np.random.randint(1, 100, k)
capacity = 2000
n = len(weight)
configuration = np.zeros(n, dtype=int)
table = -np.ones(capacity + 1)
optimal_conf = np.zeros((capacity + 1, n), dtype=int)
 res = knapsack(weight, value, capacity, n, configuration, table, optimal_conf)