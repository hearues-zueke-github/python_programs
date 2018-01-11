#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import os
import re
import sys

import numpy as np

from math import factorial as fac

from PIL import Image

from math_functions_utils import SequenceFunctions

sys.path.insert(0, os.path.abspath('../'))

from LangtonsAnt import Ant, Field

np.set_printoptions(threshold=np.nan)

def get_increment_numbers_starting_zero_diff_n(ns):
    # should return e.g. for ns = [2, 3]
    # [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]

    def increment_by_one(l):
        i = len(l)-1

        while i >= 0:
            l[i] += 1
            if l[i] >= ns[i]:
                l[i] = 0
                i -= 1
            else:
                break

        return l

    length = len(ns)
    l = [0 for _ in xrange(0, length)]
    ls = []
    times = ns[0]
    for i in ns[1:]:
        times *= i

    for i in xrange(0, times):
        l = increment_by_one(list(l))
        ls.append(l)

    return ls

def get_specific_function_powers(ls):
    def get_x_mults(l):
        expr = ""
        for i, j in enumerate(l):
            if j > 0:
                if expr != "":
                    expr += "*"
                expr += "x["+str(i)+"]"
                if j > 1:
                    expr += "**"+str(j)

        return expr

    str_expr = "lambda x, m: ("

    mult_exprs = [get_x_mults(l) for l in ls]

    str_expr += mult_exprs[0]
    for mult_expr in mult_exprs[1:]:
        str_expr += "+"+mult_expr

    str_expr += ") % m"

    return str_expr, eval(str_expr)

def create_picture_2d_2_parameters_random_params(amount_args, modulo, max_power):
    ls = np.random.randint(0, max_power, (np.random.randint(5, 15), amount_args))
    ls = np.delete(ls, np.where(np.sum(ls, axis=1)==0)[0], axis=0)

    ls_eliminated = np.array(list(set(list(map(tuple, ls)))))

    ls_sorted = np.array(sorted(ls_eliminated, key=lambda x: np.sum(x*max_power**np.arange(0, x.shape[0])[::-1])))
    print("ls_sorted:\n{}".format(ls_sorted))
    
    def get_fixed_valued_2_parameter_function(str_f_expr, params_idx, fixed_values):
        # # define the two used params
        # mix_idx = np.random.permutation(np.arange(0, amount_args))
        param_names = ["a", "b"]
        
        params_idx = sorted(params_idx)
        # params_idx = np.sort(mix_idx[:2])
        fixed_idx = sorted(list(set(np.arange(0, amount_args)).difference(params_idx))) # np.sort(mix_idx[2:])
        # fixed_values = np.random.randint(0, modulo, (amount_args-2, ))

        # print("params_idx: {}".format(params_idx))
        # print("fixed_idx: {}".format(fixed_idx))
        # print("fixed_values: {}".format(fixed_values))

        # redefine the function with two params

        str_new_expr = str(str_f_expr)
        for idx, val in zip(fixed_idx, fixed_values):
            str_new_expr = str_new_expr.replace("x["+str(idx)+"]", str(val))

        for idx, param in zip(params_idx, param_names):
            str_new_expr = str_new_expr.replace("x["+str(idx)+"]", param)

        str_new_expr = str_new_expr.replace("**", "p")

        print("str_new_expr: {}".format(str_new_expr))

        # now simpliefy the new expr as much as possible (parse the num*num or num**num expressions)
        expr_calc = re.search(".*: \((.*)\) % m", str_new_expr).group(1)
        print("expr_calc: {}".format(expr_calc))

        exprs_plus_split = expr_calc.split("+")

        str_expr_replaced_vals = ""

        for expr_mult in exprs_plus_split:
            expr_mult_split = expr_mult.split("*")

            variable_list = []
            numbers_list = []

            for expr_pow in expr_mult_split:
                if any(param in expr_pow for param in param_names):
                    variable_list.append(expr_pow)
                else:
                    numbers_list.append(expr_pow)

            combined_new_list = [s.replace("p", "**") for s in variable_list]

            for i, expr_str in enumerate(numbers_list):
                if "p" in expr_str:
                    l_nums = [int(s) for s in expr_str.split("p")]
                    numbers_list[i] = l_nums[0]**l_nums[1]
                else:
                    numbers_list[i] = int(expr_str)

            prod = 1
            if len(numbers_list) > 0:
                prod = numbers_list[0]
                for i in numbers_list[1:]:
                    prod *= i
                prod %= modulo

            if prod == 0:
                combined_new_list = []
            else:
                combined_new_list += [str(prod)]

            str_expr_mult = ""
            if len(combined_new_list) > 0:
                for expr in combined_new_list:
                    if str_expr_mult != "":
                        str_expr_mult += "*"
                    str_expr_mult += expr

            if str_expr_mult != "":
                if str_expr_replaced_vals != "":
                    str_expr_replaced_vals += "+"
                str_expr_replaced_vals += str_expr_mult

        print("str_expr_replaced_vals: {}\n".format(str_expr_replaced_vals))
            # numbers_list = [str(prod)]

        return eval("lambda a, b, m: ("+str_expr_replaced_vals+") % m")
    
    color_table = get_increment_numbers_starting_zero_diff_n([256, 4, 1])
    color_table = np.array(color_table)
    color_table[:, 1] *= (256//4)
    color_table[:, 2] = color_table[:, 0]
    # color_table = color_table[:modulo]

    empty_table = np.zeros((modulo, modulo)).astype(np.int)
    pix = np.zeros((modulo, modulo, 3)).astype(np.uint8)

    x_vals = empty_table.copy()
    x_vals[:] = np.arange(0, modulo)

    y_vals = x_vals.T.copy()

    m_vals = empty_table.copy()
    m_vals[:] = modulo

    params_idx = np.sort(np.random.permutation(np.arange(0, amount_args))[:2])
    # fixed_values = np.random.randint(0, modulo, (amount_args-2, ))
    # fixed_values[:] = 1

    # create the needed folder(s) and change the current directory
    home = os.path.expanduser("~")
    full_path = home+"/Pictures/sequences_pictures/parameters_2d_picture_args_{}_modulo_{}_max_power_{}/params_idx_{}".format(amount_args, modulo, max_power, "_".join(list(map(str, params_idx))))
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    os.chdir(full_path)

    file_name = "f_str.txt"

    if os.path.isfile(file_name):
        with open(file_name, "r") as fin:
            line = fin.readline()[:-1]
            print("line: {}".format(line))
            str_expr = line
    else:
        str_expr, f_stand = get_specific_function_powers(ls_sorted)

        with open(file_name, "w") as fout:
            fout.write(str_expr+"\n")

    print("str_expr: {}".format(str_expr))

    # for j in xrange(0, modulo):
    #     fixed_values[-2] = j
    #     for i in xrange(0, modulo):
    #         fixed_values[-1] = i
    for _ in xrange(0, 10):
        fixed_values = np.random.randint(0, modulo, (amount_args-2, ))

        print("params_idx: {}".format(params_idx))
        print("fixed_values: {}".format(fixed_values))

        f_new = get_fixed_valued_2_parameter_function(str_expr, params_idx, fixed_values)

        table_2d = np.vectorize(f_new)(x_vals, y_vals, m_vals)#//4

        pix[:, :] = color_table[table_2d]
        # pix[:, :] = table_2d.reshape((modulo, modulo, 1))

        img = Image.fromarray(pix)
        img.save("param_idx_{}_{}_values_{}.png".format(params_idx[0], params_idx[1], "_".join(list(map(str, fixed_values)))))

def main():
    amount_args = 4
    # modulo = 256
    # modulo = 512
    modulo = 1024
    max_power = 2

    # TODO: fix amount_args = 3, not working
    create_picture_2d_2_parameters_random_params(amount_args, modulo, max_power)

if __name__ == "__main__":
    # main()
    amount_args = 10
    modulo = 32
    max_power = 3
    D = np.random.randint(0, modulo, (amount_args, amount_args, amount_args, amount_args))
    C = np.random.randint(0, modulo, (amount_args, amount_args, amount_args))
    A = np.random.randint(0, modulo, (amount_args, amount_args))
    B = np.random.randint(0, modulo, (amount_args, ))

    nextX_D = lambda x1, x2, x3: (np.dot(np.dot(np.dot(D, x1)%modulo, x2)%modulo, x3)%modulo+B)%modulo
    nextX_CAB = lambda x: (np.dot(np.dot(x, C)%modulo, x)%modulo+np.dot(A, x)%modulo+B)%modulo
    nextX_AB = lambda x: (np.dot(A, x)%modulo+B)%modulo
    sequfuncs = SequenceFunctions(amount_args, modulo, max_power)
    sequfuncs.define_new_funcitons()

    x = np.random.randint(0, modulo, (amount_args, ))
    x2 = np.random.randint(0, modulo, (amount_args, ))
    x3 = np.random.randint(0, modulo, (amount_args, ))
    # x = tuple(x)

    # for func_str in sequfuncs.func_strs:
    #     print("func_str: {}".format(func_str))

    # for func in sequfuncs.funcs:
    #     print("func: {}".format(func))

    width = 500
    height = 500
    moves = [0,0,1,0]
    # moves = [0]*4
    field = Field(width, height, moves)

    print("at start: x: {}".format(x))
    xs = [x]
    i = 0
    for _ in xrange(0, 150000):
        xn = nextX_D(x, x2, x3)
        x3 = x2
        x2 = x
        x = xn
        # x = nextX_AB(x)
        # x = sequfuncs.nextX()

        x_mod = tuple(i % 4 for i in x)
        # print("i: {}".format(i))
        i += 1

        field.do_many_other_steps(x_mod)
        if i % 5000 == 0:
            print("i: {}, x: {}, x_mod: {}".format(i, x, x_mod))
        if i % 500 == 0:
            field.save_field_as_png("langtons_ant_functions")
            # print("printing image!")
