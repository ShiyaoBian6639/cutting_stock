import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, lpSum, CPLEX_CMD, SCIP_CMD


def generate_all_cutting_configuration(L, lengths):
    res = []
    add([], L, L, lengths, 0, res)
    return res


def add(cur, remain, L, lengths, index, res):
    if remain < lengths[0]:
        return cur
    for i in range(index, len(lengths)):
        val = lengths[i]
        temp = cur.copy()
        if cur not in res and len(cur) > 0 and cur is not None:
            res.append(cur)
        if remain - val >= 0:
            temp.append(val)
            new = add(temp, remain - val, L, lengths, i, res)
            if new not in res and cur is not None:
                res.append(new)
        else:
            return cur


def val2index(lengths):
    res = {}
    for index, val in enumerate(lengths):
        res[val] = index
    return res


def conf2mat(configurations, len_dict, lengths):
    res = np.zeros((len(configurations), len(lengths)), dtype=int)
    for conf_ind, configuration in enumerate(configurations):
        for length in configuration:
            index = len_dict[length]
            res[conf_ind, index] += 1
    return res


def sort_length(length, demand):
    n = len(length)
    index = np.argsort(length)
    length_arr = np.zeros(n, dtype=int)
    demand_arr = np.zeros(n)
    for i in range(n):
        length_arr[i] = length[index[i]]
        demand_arr[i] = demand[index[i]]
    return length_arr, demand_arr


def covering_model(conf_mat, len_demand):
    """
    :param conf_mat:
    :param len_demand:
    :param solver: CBC, GUROBI, CPLEX, GLPK
    :return:
    """
    num_conf, num_shape = conf_mat.shape
    prob = LpProblem("cutting_stock", LpMinimize)  # build model
    y = LpVariable.dict("configurations", range(num_conf), lowBound=0, cat=LpInteger)  # decision variable
    prob += lpSum(y[i] for i in range(num_conf))  # objective
    # add constraints
    for j in range(num_shape):
        prob += lpSum(y[i] * conf_mat[i, j] for i in range(num_conf)) >= len_demand[j]
    prob.writeLP("cutting_stock.lp")

    prob.solve()
