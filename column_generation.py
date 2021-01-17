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


def generate_basis(lengths, max_length):
    num_conf = lengths.shape[0]
    basis = np.zeros((num_conf, num_conf))
    configuration = np.zeros((num_conf, num_conf), dtype=int)
    diag = (max_length / lengths).astype(int)
    np.fill_diagonal(basis, 1 / diag)
    np.fill_diagonal(configuration, diag)
    return basis, configuration


def generate_column():
    pass
