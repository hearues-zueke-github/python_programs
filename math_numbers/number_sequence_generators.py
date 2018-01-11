#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import os
import re
import sys

import numpy as np

from math import factorial as fac

from PIL import Image

np.set_printoptions(threshold=np.nan)

def binomial_cofficient(m, n):
    return fac(m)//fac(n)//fac(m-n)

def get_increment_numbers_starting_zero(length, n, other_n=None):
    # should return e.g. for n = 3 and length = 2:
    # [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]

    def increment_by_one(l):
        i = len(l)-1

        while i >= 0:
            l[i] += 1
            if l[i] >= n:
                l[i] = 0
                i -= 1
            else:
                break

        return l

    l = [0 for _ in xrange(0, length)]
    ls = []
    times = n**length-1

    for i in xrange(0, times):
        l = increment_by_one(list(l))
        ls.append(l)

    return ls

def get_increment_numbers_starting_zero_diff_n(ns):
    # should return e.g. ns = [2, 3]
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

def get_increment_numbers_starting_increasing(n):
    # should return e.g. for n = 3:
    # [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]

    def increment_by_one(l):
        length = len(l)
        j = length
        l[-1] += 1
        while l[-1] >= n:
            j -= 1
            if j < 0:
                return np.arange(length).tolist()

            l[j] += 1
            for i in xrange(j+1, length):
                l[i] = l[i-1]+1

        return l

    def get_all_incements(length):
        l = np.arange(length).tolist()
        ls = [l]

        times = binomial_cofficient(n, length)-1

        for _ in xrange(0, times):
            l = increment_by_one(list(l))
            ls.append(l)

        return ls

    ls = [get_all_incements(i) for i in xrange(1, n+1)]
    ls = [j for i in ls for j in i]

    return ls

def get_random_function(amount_args):
    def get_random_x_mults(n):
        sorted_nums = np.sort(np.random.permutation(np.arange(amount_args))[:n])

        expr = "x["+str(sorted_nums[0])+"]"
        for j in sorted_nums[1:]:
            expr += "*x["+str(j)+"]"

        return expr


    str_expr = "lambda x, m: ("

    num_mult_expr = np.random.randint(1, 2**amount_args)
    mult_exprs = [get_random_x_mults(np.random.randint(1, amount_args+1)) for _ in xrange(0, num_mult_expr)]
    mult_exprs = list(set(mult_exprs))

    str_expr += mult_exprs[0]
    for mult_expr in mult_exprs[1:]:
        str_expr += "+"+mult_expr

    str_expr += ") % m"

    return str_expr, eval(str_expr)

def get_specific_function(amount_args, ls):
    def get_x_mults(sorted_nums):
        expr = "x["+str(sorted_nums[0])+"]"
        for j in sorted_nums[1:]:
            expr += "*x["+str(j)+"]"

        return expr

    str_expr = "lambda x, m: ("

    num_mult_expr = np.random.randint(1, 2**amount_args)
    mult_exprs = [get_x_mults(l) for l in ls]

    # print("mult_exprs: {}".format(mult_exprs))
    str_expr += mult_exprs[0]
    for mult_expr in mult_exprs[1:]:
        str_expr += "+"+mult_expr

    str_expr += ") % m"

    return str_expr, eval(str_expr)

def get_specific_function_powers(amount_args, ls):
    def get_x_mults(l):
        expr = ""
        for i, j in enumerate(l):
            if j != 0:
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

def find_pattern(l, amount_args):
    arr = np.array(l[::-1])

    positions = np.where(arr[1:]==arr[0])[0]+1

    # print("positions: {}".format(positions))

    length = len(l)

    for p in positions:
        if p*2 > length:
            break

        # print("arr[:p]: {}".format(arr[:p]))

        if np.sum(arr[:p]!=arr[p:2*p]) == 0:
            # print("found sequence!")
            # print("length: "+str(p))
            # print("arr[:p]: {}".format(arr[:p]))

            return arr[:p][::-1]

    return np.array([0]).astype(np.int)

def try_sequence_function(amount_args, modulo, l=None, str_expr=None, f=None):
    if l is None:
        l = np.random.randint(0, modulo, (amount_args, )).tolist()
    if not str_expr is None and f is None:
        f = eval(str_expr)
    elif str_expr is None and f is None:
        str_expr, f = get_random_function(amount_args)

    for _ in range(0, 20000):
        l.append(f(l[-amount_args:], modulo))

    pattern = find_pattern(l, amount_args)

    return str_expr, pattern, pattern.shape[0]

def move_pattern_max_first(pattern):
    pattern_concat = pattern.tolist()+pattern[:amount_args].tolist()

    pattern_num_matrix = np.zeros((amount_args, len(pattern))).astype(np.int)

    for i in xrange(0, amount_args):
        pattern_num_matrix[i] = pattern_concat[i:-amount_args+i]

    pattern_num_matrix = pattern_num_matrix.T

    multiply = pattern_num_matrix*modulo**(np.arange(0, amount_args)[::-1])
    sum_row = np.sum(multiply, axis=1)
    arg_max = np.argmax(sum_row)

    if arg_max != 0:
        return np.hstack((pattern[arg_max:], pattern[:arg_max]))

    return pattern

def create_picture_with_sequences(amount_args, modulo, l):
    factors_idx = get_increment_numbers_starting_increasing(amount_args)
    factors_idx = np.array(factors_idx)
    ls = get_increment_numbers_starting_zero(2**amount_args-1, 2)

    # sort by amount of 1's
    ls_arr = np.array(ls)
    amount_ones = np.sum(ls_arr, axis=1).reshape((-1, 1))

    arr = np.hstack((ls_arr, amount_ones))
    arr = arr[arr[:, -1].argsort()]

    for i in xrange(1, 8):
        pos_same_amount_ones = arr[:, -1]==i
        one_part = arr[pos_same_amount_ones]
        arr[pos_same_amount_ones] = one_part[np.lexsort(one_part.T)]

    lss = arr[:, :-1].tolist()

    print("lss: {}".format(lss))

    tried_sequences = []

    for ls in lss:
        choosen_factors = factors_idx[np.array(ls)==1].tolist()
        str_expr, f = get_specific_function(amount_args, choosen_factors)

        _, pattern, pattern_length = try_sequence_function(amount_args, modulo, l=list(l), str_expr=str_expr, f=f)
        
        if pattern.shape[0] > amount_args:
            pattern = move_pattern_max_first(pattern)

        tried_sequences.append((ls, str_expr, pattern, pattern_length))

    lss_new, str_exprs, patterns, patterns_length = list(zip(*tried_sequences))

    max_length = np.max(patterns_length)
    print("max_length: {}".format(max_length))
    
    new_patterns = []
    for pat in patterns:
        new_patterns.append((pat.tolist()*(max_length//len(pat)+1))[:max_length])
    print("len(new_patterns): {}".format(len(new_patterns)))
    print("len(new_patterns[0]): {}".format(len(new_patterns[0])))

    tried_sequences_new = list(zip(*(lss_new, str_exprs, new_patterns, patterns_length)))

    tried_sequences_sorted = sorted(tried_sequences_new, key=lambda x: x[3])[::-1]

    _, _, _, patterns_length = list(zip(*tried_sequences_sorted))

    patterns_lengths_unique = sorted(list(set(patterns_length)))[::-1]
    print("patterns_lengths_unique: {}".format(patterns_lengths_unique))

    tried_sequens_arr = np.array(tried_sequences_sorted)

    for pattern_legngth in patterns_lengths_unique:
        idx = np.where(tried_sequens_arr[:, 3]==pattern_length)[0]
        lst = tried_sequens_arr[idx].tolist()
        tried_sequens_arr[idx] = sorted(lst, key=lambda x: "".join(list(map(str, x[2]))))[::-1]
    
    lss_new, str_exprs, patterns, patterns_length = list(zip(*(tried_sequens_arr.tolist())))

    pix = np.zeros((len(lss), len(lss[0])+1+max_length, 3)).astype(np.uint8)
    pix[:, :2**amount_args-1] = np.array(lss_new).reshape((len(lss_new), len(lss_new[0]), 1))*255
    pix[:, 2**amount_args-1] = (255, 102, 255)

    color_table = get_increment_numbers_starting_zero(3, 8)
    color_table = np.array(color_table)*32
    color_table = color_table[:modulo]

    # def map_num_to_other_color_range(nums):
    #     pass

    for i, pattern in enumerate(patterns):
        # TODO: make a function for mapping other color ranges
        pix[i, 2**amount_args:] = color_table[pattern]
        # np.array(pattern).reshape((-1 ,1))

    # create the neeed folder
    home = os.path.expanduser("~")
    full_path = home+"/Pictures/sequences_pictures"
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    os.chdir(full_path)

    img = Image.fromarray(pix)
    img.save("sequence_amount_args_{}_nums_{}.png".format(amount_args, "_".join(list(map(str, l)))))

def create_picture_with_sequences_random_factors(amount_args, modulo, l):
    factors_idx = get_increment_numbers_starting_increasing(amount_args)
    factors_idx = np.array(factors_idx)

    lss = []

    get_random_factor_idx = lambda: np.random.randint(0, 2, (2**amount_args-1, )).tolist()
    for i in xrange(0, 500):
    # for i in xrange(0, 50):
        factors = get_random_factor_idx()
        while np.sum(factors) == 0:
            factors = get_random_factor_idx()
        while factors in lss:
            factors = get_random_factor_idx()
            while np.sum(factors) == 0:
                factors = get_random_factor_idx()

        lss.append(factors)

    print("lss: {}".format(lss))

    tried_sequences = []

    for ls in lss:
        choosen_factors = factors_idx[np.array(ls)==1].tolist()
        str_expr, f = get_specific_function(amount_args, choosen_factors)

        _, pattern, pattern_length = try_sequence_function(amount_args, modulo, l=list(l), str_expr=str_expr, f=f)
        
        if pattern.shape[0] > amount_args:
            pattern = move_pattern_max_first(pattern)

        tried_sequences.append((ls, str_expr, pattern, pattern_length))

    lss_new, str_exprs, patterns, patterns_length = list(zip(*tried_sequences))

    max_length = np.max(patterns_length)
    print("max_length: {}".format(max_length))
    
    new_patterns = []
    for pat in patterns:
        new_patterns.append((pat.tolist()*(max_length//len(pat)+1))[:max_length])
    print("len(new_patterns): {}".format(len(new_patterns)))
    print("len(new_patterns[0]): {}".format(len(new_patterns[0])))

    tried_sequences_new = list(zip(*(lss_new, str_exprs, new_patterns, patterns_length)))

    tried_sequences_sorted = sorted(tried_sequences_new, key=lambda x: x[3])[::-1]

    _, _, _, patterns_length = list(zip(*tried_sequences_sorted))

    patterns_lengths_unique = sorted(list(set(patterns_length)))[::-1]
    print("patterns_lengths_unique: {}".format(patterns_lengths_unique))

    tried_sequens_arr = np.array(tried_sequences_sorted)

    for pattern_legngth in patterns_lengths_unique:
        idx = np.where(tried_sequens_arr[:, 3]==pattern_length)[0]
        lst = tried_sequens_arr[idx].tolist()
        tried_sequens_arr[idx] = sorted(lst, key=lambda x: "".join(list(map(str, x[2]))))[::-1]
    
    lss_new, str_exprs, patterns, patterns_length = list(zip(*(tried_sequens_arr.tolist())))

    pix = np.zeros((len(lss), len(lss[0])+1+max_length, 3)).astype(np.uint8)
    pix[:, :2**amount_args-1] = np.array(lss_new).reshape((len(lss_new), len(lss_new[0]), 1))*255
    pix[:, 2**amount_args-1] = (255, 102, 255)

    color_table = get_increment_numbers_starting_zero(3, 9)
    color_table = np.array(color_table)*28
    color_table = color_table[:modulo]

    # def map_num_to_other_color_range(nums):
    #     pass

    for i, pattern in enumerate(patterns):
        # TODO: make a function for mapping other color ranges
        pix[i, 2**amount_args:] = color_table[pattern]
        # np.array(pattern).reshape((-1 ,1))

    # create the neeed folder
    home = os.path.expanduser("~")
    full_path = home+"/Pictures/sequences_pictures"
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    os.chdir(full_path)

    img = Image.fromarray(pix)
    img.save("sequence_amount_args_{}_modulo_{}_nums_{}.png".format(amount_args, modulo, "_".join(list(map(str, l)))))

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

        print("str_expr_replaced_vals: {}".format(str_expr_replaced_vals))
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
    fixed_values = np.random.randint(0, modulo, (amount_args-2, ))
    fixed_values[:] = 1

    # create the needed folder(s) and change the current directory
    home = os.path.expanduser("~")
    full_path = home+"/Pictures/sequences_pictures/parameters_2d_picture_args_{}_modulo_{}_max_power_{}/params_idx_{}".format(amount_args, modulo, max_power, "_".join(list(map(str, params_idx))))
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    os.chdir(full_path)

    file_name = "f_str.txt"

    if os.path.isfile(file_name):
        with open(file_name, "r") as fin:
            line = fin.readline()
            print("line: {}".format(line))
            str_expr = line
    else:
        str_expr, f_stand = get_specific_function_powers(amount_args, ls_sorted)

        with open(file_name, "w") as fout:
            fout.write(str_expr+"\n")

    print("str_expr: {}".format(str_expr))

    # for j in xrange(0, modulo):
    #     fixed_values[-2] = j
    #     for i in xrange(0, modulo):
    #         fixed_values[-1] = i
    for _ in xrange(0, 100):
        fixed_values = np.random.randint(0, modulo, (amount_args-2, ))

        print("params_idx: {}".format(params_idx))
        print("fixed_values: {}".format(fixed_values))

        f_new = get_fixed_valued_2_parameter_function(str_expr, params_idx, fixed_values)

        table_2d = np.vectorize(f_new)(x_vals, y_vals, m_vals)#//4

        pix[:, :] = color_table[table_2d]
        # pix[:, :] = table_2d.reshape((modulo, modulo, 1))

        img = Image.fromarray(pix)
        img.save("param_idx_{}_{}_values_{}.png".format(params_idx[0], params_idx[1], "_".join(list(map(str, fixed_values)))))


# ls = get_increment_numbers_starting_increasing(5)
# ls = np.array(ls)
# print("ls: {}".format(ls))
# print("len(ls): {}".format(len(ls)))

# choosen_factors = np.random.randint(0, 2, (len(ls), ))
# print("choosen_factors: {}".format(choosen_factors))

# choosen_terms = ls[choosen_factors==1].tolist()
# print("choosen_terms: {}".format(choosen_terms))

# str_expr, f = get_specific_function(5, choosen_terms)

# color_table = get_increment_numbers_starting_zero(3, 16)
# color_table = np.array(color_table)*16
# # color_table[color_table==256] = 255

# print("color_table: {}".format(color_table))

# sys.exit(0)

if __name__ == "__main__":
    amount_args = 4
    # modulo = 256
    # modulo = 512
    modulo = 1024
    max_power = 2

    # TODO: fix amount_args = 3, not working
    create_picture_2d_2_parameters_random_params(amount_args, modulo, max_power)

    sys.exit(0)

    amount_args = 4
    modulo = 256

    l = [0 for _ in xrange(0, amount_args-1)]+[1]
    # l = [0, 0, 1]
    # l = [0, 0, 0, 1]

    for i in xrange(1, 128):
        print("i: {}".format(i))
        li = list(l)
        li[-1] = i
        create_picture_with_sequences_random_factors(amount_args, modulo, li)
        # create_picture_with_sequences(amount_args, modulo, li)

    # for i in xrange(1, 256):
    #     li = list(l)
    #     li[-1] = i
    #     for j in xrange(0, 256):
    #         lj = list(li)
    #         lj[-2] = j
    #         create_picture_with_sequences(amount_args, modulo, lj)

    sys.exit(0)

    ls = get_increment_numbers_starting_increasing(amount_args)
    ls = np.array(ls)

    choosen_factors = np.random.randint(0, 2, (len(ls), ))
    choosen_terms = ls[choosen_factors==1].tolist()

    str_expr, f = get_specific_function(amount_args, choosen_terms)

    tried_sequences = [try_sequence_function(amount_args, modulo, str_expr=str_expr, f=f) for _ in xrange(0, 100)]

    tried_sequences_sorted = sorted(tried_sequences, key=lambda x: x[-1])

    for i, (str_expr, pattern, pattern_length) in enumerate(tried_sequences_sorted):
        print("\nnr: {}".format(i))
        print("  str_expr: {}".format(str_expr))
        print("  pattern: {}".format(pattern))
        print("  pattern_length: {}".format(pattern_length))
