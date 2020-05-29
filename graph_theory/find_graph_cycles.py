#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy

sys.path.append("../combinatorics/")
import different_combinations as combinations

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

import numpy as np

from utils_graph_theory import get_cycles_of_1_directed_graph

# simple tests
# 1 -> 3, 2 -> 3, 3 -> 2, etc.
edges_directed = [(1, 3), (2, 3), (3, 2), (5, 4), (4, 5), (6, 6)]
list_of_cycles_1_calc = get_cycles_of_1_directed_graph(edges_directed)
list_of_cycles_1_ref = [[6], [2, 3], [4, 5]]
assert list_of_cycles_1_calc==list_of_cycles_1_ref

edges_directed = [(1, 3), (3, 2), (2, 1)]
list_of_cycles_2_calc = get_cycles_of_1_directed_graph(edges_directed)
list_of_cycles_2_ref = [[1, 3, 2]]
assert list_of_cycles_2_calc==list_of_cycles_2_ref

if __name__ == "__main__":
    pass
