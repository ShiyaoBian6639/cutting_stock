import numpy as np
from numba import njit, int32


@njit()
def knapsack(weight, value, capacity, n, configuration):
    if capacity < weight[0]:
        return 0, configuration
    gain = 0
    item = 0
    final_conf = configuration
    for i in range(n):
        remain = capacity - weight[i]
        if remain >= 0:
            cur_gain, configuration = knapsack(weight, value, remain, n, np.zeros(n, dtype=int32))
            cur_gain += value[i]
            if cur_gain > gain:
                gain = cur_gain
                item = i
                final_conf = configuration
        else:
            break
    final_conf[item] += 1
    return gain, final_conf
