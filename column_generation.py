"""
Book: Introduction to Mathematical Programming
Author: Wayne L. Winston
Chapter 10: Advanced topics in Linear Programming
Part3: Using Column Generation to solve large-scale LPs

lumber length: 3-ft, 5-ft and 9-ft
demand:        25    20       15
board: 17-ft

table of configurations: (* indicates initial basis)
configuration  3-ft   5-ft   9-ft
      1*        5      0      0
      2         4      1      0
      3         2      2      0
      4         2      0      1
      5         1      1      1
      6*        0      3      0
      7*        0      0      1

decision variable: x_i = the number of 17-ft board cut according to combination i
objective: minimize x1 +  x2 +  x3 +  x4 + x5 +  x6 + x7
st:
                   5x1 + 4x2 + 2x3 + 2x4 + x5            >= 25
                          x2 + 2x3 +       x5 + 3x6      >= 20
                                       x4 + x5       + x7 >= 15
"""
import numpy as np
from knapsack import knapsack


def generate_basis(lengths, max_length):
    num_conf = lengths.shape[0]
    basis = np.zeros((num_conf, num_conf))
    configuration = np.zeros((num_conf, num_conf), dtype=int)
    diag = (max_length / lengths).astype(int)
    np.fill_diagonal(basis, 1 / diag)
    np.fill_diagonal(configuration, diag)
    return basis, configuration


def get_col(basis, weight, max_len):
    value = basis.sum(axis=0)
    n = len(weight)
    configuration = np.zeros(n, dtype=int)
    table = -np.ones(max_len + 1)
    optimal_conf = np.zeros((max_len + 1, n), dtype=int)
    reduced_cost, column = knapsack(weight, value, max_len, n, configuration, table, optimal_conf)
    return reduced_cost, column


def ratio_test(inv_b, column, demand):
    num = np.inner(inv_b, column)
    den = np.inner(inv_b, demand)
    ratio = den / num
    leaving = -1
    min_value = np.inf
    for i in range(len(ratio)):
        value = ratio[i]
        if min_value > value > 0:
            min_value = value
            leaving = i
    return leaving, num


def prod_inv(num, n, r):
    res = np.eye(n)
    temp = num[r]
    ero = - num / temp
    ero[r] = 1 / temp
    res[:, r] = ero
    return res


def update_inv_b(mat_e, inv_b):
    return mat_e.dot(inv_b)


def generate_optimal_column(weight, capacity, demand):
    n = len(weight)
    # generate initial configuration
    inv_b, configuration = generate_basis(weight, capacity)
    reduced_cost, column = get_col(inv_b, weight, capacity)
    count = n
    while reduced_cost - 1 > 1e-4:
        leaving, num = ratio_test(inv_b, column, demand)
        # configuration = np.vstack((configuration, column))
        configuration[leaving] = column  # update configuration
        mat_e = prod_inv(num, n, leaving)
        inv_b = update_inv_b(mat_e, inv_b)
        reduced_cost, column = get_col(inv_b, weight, capacity)
        print(reduced_cost)
    return configuration
