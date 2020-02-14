#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import random
import sys

import numpy as np

from copy import deepcopy
from functools import reduce
from time import time

from PIL import Image

from pysat.solvers import Glucose3

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

def get_func_generate_new_variables(n, start_count=0):
    d = dict(count=start_count)
    def generate_new_variables():
        l_vars = [i for i in range(d['count'], d['count']+n)]
        d['count'] += n
        return l_vars
    return generate_new_variables


if __name__=='__main__':
    # first define the cnfs for e.g. 9x9 sudoku!
    n_small = 3
    n = n_small**2

    cnfs = []
    generate_new_variables = get_func_generate_new_variables(n=n, start_count=1)

    d_field_cells = {(j, i): generate_new_variables() for j in range(0, n) for i in range(0, n)}
    d_var_num_to_cell_num = {v: i for l_vars in d_field_cells.values() for i, v in enumerate(l_vars, 1)}
    d_cell_pos_num_to_var_num = {(j+1, i+1, num): var_num for (j, i), l_vars in d_field_cells.items() for num, var_num in enumerate(l_vars, 1)}

    def extend_cnfs(cnfs, l_vars):
        cnfs.append(l_vars)
        cnfs.extend([[-v1, -v2] for i, v1 in enumerate(l_vars, 1) for v2 in l_vars[i:]])

    # generate constraints for each cell
    for l_vars in d_field_cells.values():
        extend_cnfs(cnfs, l_vars)
        # cnfs.append(l_vars)
        # cnfs.extend([[-v1, -v2] for i, v1 in enumerate(l_vars, 1) for v2 in l_vars[i:]])
    
    # generate constraints for each row and column, where each number can only exists
    # once in each row or column! Also it must come only once!
    for row in range(0, n):
        for num in range(0, n):
            l_vars_row = [d_field_cells[(row, column)][num] for column in range(0, n)]
            extend_cnfs(cnfs=cnfs, l_vars=l_vars_row)
            # cnfs.append(l_vars_row)
            # cnfs.extend([[-v1, -v2] for i, v1 in enumerate(l_vars_row, 1) for v2 in l_vars_row[i:]])

    for column in range(0, n):
        for num in range(0, n):
            l_vars_column = [d_field_cells[(row, column)][num] for row in range(0, n)]
            extend_cnfs(cnfs=cnfs, l_vars=l_vars_column)
            # cnfs.append(l_vars_column)
            # cnfs.extend([[-v1, -v2] for i, v1 in enumerate(l_vars_column, 1) for v2 in l_vars_column[i:]])

    for row_outer in range(0, n_small):
        row1 = row_outer*n_small
        for column_outer in range(0, n_small):
            column1 = column_outer*n_small

            for num in range(0, n):
                l_vars_square = [d_field_cells[(row1+row_inner, column1+column_inner)][num] for row_inner in range(0, n_small) for column_inner in range(0, n_small)]
                extend_cnfs(cnfs=cnfs, l_vars=l_vars_square)
                # cnfs.append(l_vars_square)
                # cnfs.extend([[-v1, -v2] for i, v1 in enumerate(l_vars_square, 1) for v2 in l_vars_square[i:]])

    # example of a field
    prepared_field = "9x9:"+\
    "3,4,7,0,0,0,5,2,1;"+\
    "0,9,0,0,0,0,0,3,0;"+\
    "0,0,2,0,0,0,0,0,4;"+\
    "0,1,0,0,5,0,0,6,0;"+\
    "6,5,4,0,0,0,0,0,2;"+\
    "0,0,0,0,8,0,7,0,0;"+\
    "0,0,0,9,1,0,0,0,6;"+\
    "0,2,0,0,4,0,0,7,0;"+\
    "1,0,5,0,6,7,9,0,3"

    assert prepared_field.count(':')==1
    l_info = prepared_field.split(':')
    assert l_info[0].count('x')==1
    l_size = l_info[0].split('x')
    size_y, size_x = list(map(int, l_size))
    print("size_y: {}, size_x: {}".format(size_y, size_x))
    assert size_y==size_x
    n_size = size_y
    assert l_info[1].count(';')==n_size-1
    assert l_info[1].count(',')==(n_size-1)*n_size

    l_field = [list(map(int, l.split(','))) for l in l_info[1].split(';')]
    print("l_field: {}".format(l_field))

    arr_field = np.array(l_field)
    print("np.sum(arr_field!=0): {}".format(np.sum(arr_field!=0)))

    for j, l_row in enumerate(l_field, 1):
        for i, num in enumerate(l_row, 1):
            if num!=0:
                cnfs.append([d_cell_pos_num_to_var_num[(j, i, num)]])

    # then solve the cnfs form
    with Glucose3(bootstrap_with=cnfs) as m:
        print(m.solve())
        # print(list(m.get_model()))
        models = [m for m, _ in zip(m.enum_models(), range(0, 10000))]
    
    # then convert back the solved solution to a normal sudoku field (e.g. 9x9)!
    arr_solutions_idxs = np.array([[d_var_num_to_cell_num[i] for i in m if i>0] for m in models]).reshape((-1, n, n))
    print("arr_solutions_idxs[0]:\n{}".format(arr_solutions_idxs[0]))
    print("len(models): {}".format(len(models)))
