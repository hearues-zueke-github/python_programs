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


def generate_objs_for_sudoku(n_small=3):
    # first define the cnfs for e.g. 9x9 sudoku!
    n_small = 3
    n = n_small**2

    cnfs_base = []
    generate_new_variables = get_func_generate_new_variables(n=n, start_count=1)

    d_field_cells = {(j, i): generate_new_variables() for j in range(0, n) for i in range(0, n)}
    d_var_num_to_cell_num = {v: i for l_vars in d_field_cells.values() for i, v in enumerate(l_vars, 1)}
    d_cell_pos_num_to_var_num = {(j+1, i+1, num): var_num for (j, i), l_vars in d_field_cells.items() for num, var_num in enumerate(l_vars, 1)}

    def extend_cnfs(cnfs, l_vars):
        cnfs.append(l_vars)
        cnfs.extend([[-v1, -v2] for i, v1 in enumerate(l_vars, 1) for v2 in l_vars[i:]])

    # generate constraints for each cell
    for l_vars in d_field_cells.values():
        extend_cnfs(cnfs=cnfs_base, l_vars=l_vars)
    
    # generate constraints for each row and column, where each number can only exists
    # once in each row or column! Also it must come only once!
    for row in range(0, n):
        for num in range(0, n):
            l_vars_row = [d_field_cells[(row, column)][num] for column in range(0, n)]
            extend_cnfs(cnfs=cnfs_base, l_vars=l_vars_row)

    for column in range(0, n):
        for num in range(0, n):
            l_vars_column = [d_field_cells[(row, column)][num] for row in range(0, n)]
            extend_cnfs(cnfs=cnfs_base, l_vars=l_vars_column)

    for row_outer in range(0, n_small):
        row1 = row_outer*n_small
        for column_outer in range(0, n_small):
            column1 = column_outer*n_small

            for num in range(0, n):
                l_vars_square = [d_field_cells[(row1+row_inner, column1+column_inner)][num] for row_inner in range(0, n_small) for column_inner in range(0, n_small)]
                extend_cnfs(cnfs=cnfs_base, l_vars=l_vars_square)

    return {
        'cnfs_base': cnfs_base,
        'd_field_cells': d_field_cells,
        'd_var_num_to_cell_num': d_var_num_to_cell_num,
        'd_cell_pos_num_to_var_num': d_cell_pos_num_to_var_num,
    }


if __name__=='__main__':
    PATH_DIR_TXT = PATH_ROOT_DIR+'txt_files/'
    if not os.path.exists(PATH_DIR_TXT):
        os.makedirs(PATH_DIR_TXT)

    n_small = 3
    n = n_small**2
    objs = generate_objs_for_sudoku(n_small=n_small)

    cnfs_base = objs['cnfs_base']
    d_field_cells = objs['d_field_cells']
    d_var_num_to_cell_num = objs['d_var_num_to_cell_num']
    d_cell_pos_num_to_var_num = objs['d_cell_pos_num_to_var_num']


    FILE_SUFFIX = '_nr_2'

    with open(PATH_DIR_TXT+'unsolved_sudokus{}.txt'.format(FILE_SUFFIX), 'r') as f:
        lines = [l.replace('\n', '') for l in f.readlines()]

    l_unsolved_fields = []

    for num_field, prepared_field_str in enumerate(lines, 0):
        print("num_field: {}".format(num_field))
        try:
            assert prepared_field_str.count(':')==1
            l_info = prepared_field_str.split(':')
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

            l_unsolved_fields.append(('OK', num_field, prepared_field_str, l_field))
        except:
            l_unsolved_fields.append(('INVALID INPUT', num_field, prepared_field_str))

    l_solved_fields = []
    for t in l_unsolved_fields:
        if t[0]!='OK':
            l_solved_fields.append(t)
            continue

        try:
            cnfs = deepcopy(cnfs_base)

            l_field = t[3]
            for j, l_row in enumerate(l_field, 1):
                for i, num in enumerate(l_row, 1):
                    if num!=0:
                        cnfs.append([d_cell_pos_num_to_var_num[(j, i, num)]])

            # then solve the cnfs form
            with Glucose3(bootstrap_with=cnfs) as m:
                assert m.solve()
                models = [m for m, _ in zip(m.enum_models(), range(0, 1000))]
            l_solution_field = np.array([[d_var_num_to_cell_num[i] for i in m if i>0] for m in models[:min(10, len(models))]]).reshape((-1, n, n)).tolist()

            l_field_str = [';'.join([','.join(map(str, l_row)) for l_row in l_field]) for l_field in l_solution_field]
            l_solved_fields.append(('OK', t[1], t[2], l_field_str, len(models)))
        except:
            l_solved_fields.append(('NOT SOLVABLE', )+t[1:3])
    
    with open(PATH_DIR_TXT+'solved_sudokus{}.txt'.format(FILE_SUFFIX), 'w') as f:
        for t in l_solved_fields:
            num_field = t[1]
            prepared_field_str = t[2]
            f.write('f{}|{}\n'.format(num_field, prepared_field_str))
            
            info = t[0]
            print("num_field: {}, info: {}".format(num_field, info))
            if info!='OK':
                f.write('{}\n'.format(info))
                continue

            for solve_num, solved_field_str in enumerate(t[3], 0):
                f.write('f{}s{}|{}\n'.format(num_field, solve_num, solved_field_str))
            f.write('Found {} so far.\n'.format(t[4]))
