#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import pdb
import sys

import numpy as np

from dotmap import DotMap
from PIL import Image


def create_lambda_functions_with_matrices(path_lambda_functions):
    # With this you can create any size for the window!
    # ft...frame thickness
    ft = 1
    params_arr = np.empty((ft*2+1, ft*2+1), dtype=np.object)

    params_arr[ft, ft] = "p"
    
    for j in range(1, ft+1):
        params_arr[ft-j, ft] = "u"*j
        params_arr[ft+j, ft] = "d"*j
        params_arr[ft, ft-j] = "l"*j
        params_arr[ft, ft+j] = "r"*j
        for i in range(1, ft+1):
            params_arr[ft-j, ft-i] = "u"*j+"l"*i
            params_arr[ft-j, ft+i] = "u"*j+"r"*i
            params_arr[ft+j, ft-i] = "d"*j+"l"*i
            params_arr[ft+j, ft+i] = "d"*j+"r"*i
    # print("params_arr: \n{}".format(params_arr))

    params_1 = params_arr.reshape((-1))
    params_0 = np.array(["inv({})".format(param) for param in params_1])

    params = np.vstack((params_1, params_0)).T

    # print("params:\n{}".format(params))
    # return

    # print("params: {}".format(params))

    # TODO: make matrices for 1, 2 to 9 elements!
    # BUT! for 1 e.g.:
    # [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    # for 2:
    # [[0, 1], [0, 2], ... , [0, 9], [1, 2], [1, 3], ... [7, 8]]
    # for 3:
    # [[0, 1, 2], [0, 1, 3], ...]

    sys.path.append("../combinatorics/")
    import different_combinations

    m = 2
    n = params.shape[0]
    idx = np.array([0, 1])
    for i in range(0, n):
        idx = np.hstack((idx, [0]))+np.hstack(([0], idx))
        # print("i: {}".format(i))
        # print("  idx: {}".format(idx))
    idx = np.cumsum(idx)
    # print("idx: {}".format(idx))
    # return

    arr = different_combinations.get_all_combinations_repeat(m, n)
    arr = np.hstack((np.sum(arr, axis=1).reshape((-1, 1)), arr))
    arr2 = arr.astype(np.uint8).reshape((-1, )).view(",".join(["u1" for _ in range(0, n+1)]))
    arr2 = np.sort(arr2, order=("f0", )).view(np.uint8).reshape((-1, n+1))[:, 1:]
    # arr2 = arr2.view(np.uint8).reshape((-1, n+1))[:, :n]
    # arr_combs = different_combinations.get_all_combinations_increment(m, n)
    # print("m: {}".format(m))
    # print("n: {}".format(n))
    # print("arr:\n{}".format(arr))
    # print("arr2:\n{}".format(arr2))

    groups = np.array([arr2[i1:i2] for i1, i2 in zip(idx[1:-1], idx[2:])])
    # print("groups:\n{}".format(groups))
    groups_2 = [np.where(group==1)[1].reshape((-1, i)) for i, group in enumerate(groups, 1)]
    # print("groups_2:\n{}".format(groups_2))

    group_num_amount = {i: group for i, group in enumerate(groups, 1)}
    # group_num_amount = {i: group for i, group in enumerate(groups_2, 1)}

    globals()["group_num_amount"] = group_num_amount
    # globals()["groups"] = groups
    # globals()["groups_2"] = groups_2
    # globals()["arr2"] = arr2

    def get_random_and(params, group_num_amount, min_n=1, max_n=3):
        # First get random num_amount
        key = np.random.choice(np.arange(min_n, max_n+1))
        group_num = group_num_amount[key]

        # choose one row of group_num!
        idx_choosen_param = group_num[np.random.randint(0, group_num.shape[0])]

        # Now invert some of the params if needed!
        idx_inv_param = np.random.randint(0, 2, params.shape[0])
        idx_arr_inv_param = np.zeros(params.shape, dtype=np.int)
        idx_arr_inv_param[np.arange(0, params.shape[0]), idx_inv_param] = 1
        choosen_inv_params = params[idx_arr_inv_param==1]

        # print("idx_inv_param:\n{}".format(idx_inv_param))
        # print("idx_arr_inv_param:\n{}".format(idx_arr_inv_param))

        # print("choosen_inv_params: {}".format(choosen_inv_params))
        # print("idx_choosen_param: {}".format(idx_choosen_param))

        and_params = choosen_inv_params[idx_choosen_param==1]
        # and_params = choosen_inv_params[idx_choosen_param]
        # print("  and_params: {}".format(and_params))
        and_str = "&".join(and_params)
        # print("and_str: {}".format(and_str))

        return and_str, idx_choosen_param, idx_inv_param

    def get_random_or(params, group_num_amount, min_and=1, max_and=4, min_n=1, max_n=3):
        amount_and = np.random.randint(min_and, max_and+1)
        # print("amount_and: {}".format(amount_and))
        and_values = [get_random_and(params, group_num_amount, min_n=min_n, max_n=max_n) for _ in range(0, amount_and)]

        and_lst, idx_choosen_params, idx_inv_params = list(zip(*and_values))

        or_str = "lambda: "+"|".join(and_lst)

        return or_str, np.array(idx_choosen_params), np.array(idx_inv_params)

    def get_random_booleans_str(params, group_num_amount, min_or=1, max_or=8, min_and=1, max_and=4, min_n=1, max_n=3):
        amount_or = np.random.randint(min_or, max_or+1)
        or_lst = [get_random_or(params, group_num_amount, min_and=min_and, max_and=max_and, min_n=min_n, max_n=max_n)
        for _ in range(0, amount_or)]

        return or_lst

    # get_random_and(params, group_num_amount)
    # or_str = get_random_or(params, group_num_amount)
    function_str_values = get_random_booleans_str(params, group_num_amount,
        min_or=1, max_or=2,
        min_and=1, max_and=3,
        min_n=3, max_n=9)
    function_str_lst, idx_choosen_params_lst, idx_inv_params_lst = list(zip(*function_str_values))
    idx_choosen_params_lst = np.array(idx_choosen_params_lst)
    idx_inv_params_lst = np.array(idx_inv_params_lst)

    # for j, (func_str, idx_choosen, idx_inv) in enumerate(zip(function_str_lst, idx_choosen_params_lst, idx_inv_params_lst), 1):
    #     print("\n  j: {}, func_str: {}".format(j, func_str))
    #     print("  idx_choosen:\n{}".format(idx_choosen))
    #     print("  idx_inv:\n{}".format(idx_inv))

    dm = DotMap()
    dm.params = params
    dm.function_str_lst = function_str_lst
    dm.idx_choosen_params_lst = idx_choosen_params_lst
    dm.idx_inv_params_lst = idx_inv_params_lst

    with gzip.open(path_lambda_functions+"dm.pkl.gz", "wb") as fout:
        dill.dump(dm, fout)

    with open(path_lambda_functions+"lambdas.txt", "w") as fout:
        for line in function_str_lst:
            fout.write("{}\n".format(line))

    return dm


def simplest_lambda_functions(path_lambda_functions):
    function_str_lst = [
        "((u+1)%2)|((d+1)%2)",
        "r|((l+1)%2)",
    ]

    # function_str_lst = ["lambda: "+func_str for func_str in function_str_lst]

    with open(path_lambda_functions, "w") as fout:
        for func_str in function_str_lst:
            fout.write("lambda: {}\n".format(func_str))

    return function_str_lst


def conway_game_of_life_functions(path_lambda_functions):
    params = ["u", "d", "l", "r", "ur", "ul", "dr", "dl"]
    params_negative = ["(({}+1)%2)".format(param) for param in params]
    # print("params:\n{}".format(params))
    # print("params_negative:\n{}".format(params_negative))

    # First create the 2 and 3 live cells as neighbor logic!

    and_parts_1 = []
    for i1 in range(0, 7):
        for i2 in range(i1+1, 8):
            and_part = "{}&{}".format(params[i1], params[i2])
            for i in range(0, 8):
                if i == i1 or i == i2:
                    continue
                and_part += "&{}".format(params_negative[i])
            and_parts_1.append(and_part)
    # print("and_parts_1: {}".format(and_parts_1))
    two_cells_alive_neighbor_func = "|".join(and_parts_1)

    and_parts_2 = []
    for i1 in range(0, 6):
        for i2 in range(i1+1, 7):
            for i3 in range(i2+1, 8):
                and_part = "{}&{}&{}".format(params[i1], params[i2], params[i3])
                for i in range(0, 8):
                    if i == i1 or i == i2 or i == i3:
                        continue
                    and_part += "&{}".format(params_negative[i])
                and_parts_2.append(and_part)
    # print("and_parts_2: {}".format(and_parts_2))
    three_cells_alive_neighbor_func = "|".join(and_parts_2)

    # This is only one function, but a very huge one!
    function_str_lst = [
        # "p&("+two_cells_alive_neighbor_func+"|"+three_cells_alive_neighbor_func+")|"+
        # "((p+1)%2)&("+three_cells_alive_neighbor_func+")",
        # "((lambda x: np.logical_or.reduce((np.logical_and.reduce((p==1, np.logical_or.reduce((x==2, x==3)))), np.logical_and.reduce((p==0, x==3)))) )(u+d+r+l+dr+dl+ur+ul)+0).astype(np.uint8)",
"""
def a():
    x = u+d+r+l+ur+ul+dr
    t1 = np.logical_or.reduce((x==2, x==4, x==5))
    p1 = np.logical_and.reduce((p==1, t1))
    p2 = np.logical_and.reduce((p==0, x==3))

    return np.logical_or.reduce((p1, p2)).astype(np.uint8)
    # return (u|inv(d)).astype(np.uint8)
"""[1:-1],
    ]

    print("function_str_lst: {}".format(function_str_lst))

    with open(path_lambda_functions, "w") as fout:
        for func_str in function_str_lst:
            if "def " in func_str:
                fout.write("{}\n".format(func_str))
            else:
                fout.write("lambda: {}\n".format(func_str))

    return function_str_lst


def simple_random_lambda_creation(function_amount=1, path_lambda_functions_file=None):
    params = ["p", "u", "d", "l", "r", "ul", "ur", "dl", "dr"]
    params += ["inv({})".format(param) for param in params]
    # print("params: {}".format(params))

    params = np.array(params)

    len_params = len(params)

    def and_part():
        random_params = params[np.random.randint(0, len_params, (np.random.randint(2, 6), ))]
        return "&".join(random_params)
    
    def or_part():
        random_params = [and_part() for _ in range(0, np.random.randint(2, 5))]
        return "|".join(random_params)

    def func_part():
        random_funcs = ["lambda: "+or_part() for _ in range(0, np.random.randint(1, 2))]
        return random_funcs

    function_str_lst = func_part()

    if path_lambda_functions_file is not None:
        with open(path_lambda_functions_file, "w") as fout:
            for func_str in function_str_lst:
                fout.writelines(func_str+"\n")
            # for func_str in function_str_lst:
            #     fout.write(func_str+"\n")

    return function_str_lst


def create_lambda_functions_2(path_lambda_functions):
    max_moves = 2 # e.g. uu, dd, ll, rr are possible for max_moves == 2
    n = 30
    m = 45
    print("n: {}, m: {}".format(n, m))
    used_u = n
    used_d = n
    used_l = m
    used_r = m

    # first create random samples with each side
    moves_u = []
    moves_d = []
    moves_l = []
    moves_r = []

    while used_u > 0:
        amount = np.random.randint(0, 2 if used_u > 2 else used_u)+1
        used_u -= amount
        moves_u.append("u"*amount)
    while used_d > 0:
        amount = np.random.randint(0, 2 if used_d > 2 else used_d)+1
        used_d -= amount
        moves_d.append("d"*amount)
    while used_l > 0:
        amount = np.random.randint(0, 2 if used_l > 2 else used_l)+1
        used_l -= amount
        moves_l.append("l"*amount)
    while used_r > 0:
        amount = np.random.randint(0, 2 if used_r > 2 else used_r)+1
        used_r -= amount
        moves_r.append("r"*amount)

    print("moves_u: {}".format(moves_u))
    print("moves_d: {}".format(moves_d))
    print("moves_l: {}".format(moves_l))
    print("moves_r: {}".format(moves_r))

    # second combine u and d, also l and r, also mix them up
    moves_ud = np.random.permutation(moves_u+moves_d).tolist()
    moves_lr = np.random.permutation(moves_l+moves_r).tolist()

    print("moves_ud: {}".format(moves_ud))
    print("moves_lr: {}".format(moves_lr))

    # third get all moves into one list
    used_moves = []
    while len(moves_ud) or len(moves_lr):
        use_ud = np.random.randint(0, 2) == 1
        use_lr = np.random.randint(0, 2) == 1

        move = ""
        if (use_ud or len(moves_lr) == 0) and len(moves_ud) > 0:
            move += moves_ud.pop()
        if (not use_ud or use_lr or len(moves_ud) == 0) and len(moves_lr) > 0:
            move += moves_lr.pop()

        used_moves.append(move)

    print("used_moves: {}".format(used_moves))

    print("len(used_moves): {}".format(len(used_moves)))

    with open(path_lambda_functions+"lambdas_2.txt", "w") as fout:
        for use_move in used_moves:
            fout.write("lambda: {}\n".format(use_move))

    # get max up, down, left, right moved pixels
    
    y_min = 0
    y_max = 0
    x_min = 0
    x_max = 0
    
    moved_y = 0
    moved_x = 0
    for use_move in used_moves:
        moved_y += -use_move.count("u")+use_move.count("d")
        moved_x += -use_move.count("l")+use_move.count("r")

        if y_min > moved_y:
            y_min = moved_y
        if y_max < moved_y:
            y_max = moved_y
        if x_min > moved_x:
            x_min = moved_x
        if x_max < moved_x:
            x_max = moved_x

    moved_u = np.abs(y_min)
    moved_d = y_max
    moved_l = np.abs(x_min)
    moved_r = x_max

    print("Picture will move:")
    print("  moved_u: {}".format(moved_u))
    print("  moved_d: {}".format(moved_d))
    print("  moved_l: {}".format(moved_l))
    print("  moved_r: {}".format(moved_r))

    dm = DotMap()
    dm.resize_params = [moved_u, moved_d, moved_l, moved_r]
    dm.max_iterations = len(used_moves)

    with open(path_lambda_functions+"resize_params_2.pkl", "wb") as fout:
        dill.dump(dm, fout)


def create_lambda_functions_3(path_lambda_functions, tf=2):
    n = np.random.randint(5, 31)
    moves_operations = []
    def get_rnd_move():
        moves = ""
        used_ud = np.random.randint(0, 2) == 0
        used_lr = np.random.randint(0, 2) == 0
        if used_ud:
            moves += ("u" if np.random.randint(0, 2) == 0 else "d")*np.random.randint(1, tf+1)
        if not used_ud or used_lr:
            moves += ("l" if np.random.randint(0, 2) == 0 else "r")*np.random.randint(1, tf+1)
        return moves

    def get_and_concat():
        return "&".join([get_rnd_move() for _ in range(0, np.random.randint(2, 5))]+([] if np.random.randint(0, (tf*2+1)**2) != 0 else ["p"]))

    def get_or_concat():
        return "|".join([get_and_concat() for _ in range(0, np.random.randint(4, 12))])

    for i in range(0, n):
        moves = get_or_concat()
        moves_operations.append(moves)

    with open(path_lambda_functions+"lambdas_3.txt", "w") as fout:
        for moves_operation in moves_operations:
            print("moves_operation: {}".format(moves_operation))
            fout.write("lambda: {}\n".format(moves_operation))

def create_lambda_functions_4(path_lambda_functions, tf=2):
    n = np.random.randint(20, 51)
    # create a list of all moves
    all_moves = ["u"*i for i in range(1, tf+1)]+\
                ["d"*i for i in range(1, tf+1)]+\
                ["l"*i for i in range(1, tf+1)]+\
                ["r"*i for i in range(1, tf+1)]+\
                ["u"*j+"l"*i for j in range(1, tf+1) for i in range(1, tf+1)]+\
                ["u"*j+"r"*i for j in range(1, tf+1) for i in range(1, tf+1)]+\
                ["d"*j+"l"*i for j in range(1, tf+1) for i in range(1, tf+1)]+\
                ["d"*j+"r"*i for j in range(1, tf+1) for i in range(1, tf+1)]+\
                ["p"]
    all_moves = np.array(all_moves)
    print("all_moves: {}".format(all_moves))
    print("all_moves.shape: {}".format(all_moves.shape))

    moves_operations = []
    moves_operations_bits = []
    moves_operations_sizes = []

    def get_permutated_1_0():
        arr = np.zeros((all_moves.shape[0], ), dtype=np.int)
        and_concats = np.random.randint(3, 6)
        arr[:and_concats] = 1
        return np.random.permutation(arr)

    for i in range(0, n):
        # this is a matrix with 1's and 0's but with each row containing 2-4 1's e.g.
        # also there can be 2-6 lines e.g., sooo this matrix has the dimension
        # in this case (6, all_moves.shape[0])
        or_concats = np.random.randint(2, 9)
        moves_bits_arr = np.vstack((get_permutated_1_0() for _ in range(0, or_concats)))
        moves_operations_bits.append(moves_bits_arr)
        moves_operations_sizes.append(moves_bits_arr.shape[0])
        moves = "|".join(["&".join(all_moves[bits_row==1]) for bits_row in moves_bits_arr])
        moves_operations.append(moves)

        print("\ni: {}".format(i))
        print("moves_bits_arr:\n{}".format(moves_bits_arr))
        print("moves: {}".format(moves))

    print("\nmoves_operations_sizes: {}".format(moves_operations_sizes))

    with open(path_lambda_functions+"lambdas_4.txt", "w") as fout:
        for moves_operation in moves_operations:
            # print("moves_operation: {}".format(moves_operation))
            fout.write("lambda: {}\n".format(moves_operation))

    # globals()["mdoves_operations"] = moves_operations
    # globals()["moves_operations_bits"] = moves_operations_bits
    # globals()["moves_operations_sizes"] = moves_operations_sizes

def create_lambda_functions_5(path_lambda_functions, tf=2):
    n = np.random.randint(10, 26)
    # create a list of all moves
    all_moves = ( ["u"*i for i in range(1, tf+1)]+
                  ["d"*i for i in range(1, tf+1)]+
                  ["l"*i for i in range(1, tf+1)]+
                  ["r"*i for i in range(1, tf+1)]+
                  ["u"*j+"l"*i for j in range(1, tf+1) for i in range(1, tf+1)]+
                  ["u"*j+"r"*i for j in range(1, tf+1) for i in range(1, tf+1)]+
                  ["d"*j+"l"*i for j in range(1, tf+1) for i in range(1, tf+1)]+
                  ["d"*j+"r"*i for j in range(1, tf+1) for i in range(1, tf+1)]+
                  ["p"] )
    all_moves = np.array(all_moves)
    print("all_moves: {}".format(all_moves))
    print("all_moves.shape: {}".format(all_moves.shape))

    moves_operations = []
    moves_operations_bits = []
    moves_operations_sizes = []

    def get_permutated_1_0():
        arr = np.zeros((all_moves.shape[0], ), dtype=np.int)
        and_concats = np.random.randint(3, 7)
        arr[:and_concats] = 1
        return np.random.permutation(arr)

    for i in range(0, n):
        # this is a matrix with 1's and 0's but with each row containing 2-4 1's e.g.
        # also there can be 2-6 lines e.g., sooo this matrix has the dimension
        # in this case (6, all_moves.shape[0])
        or_concats = np.random.randint(2, 8)
        moves_bits_arr = np.vstack((get_permutated_1_0() for _ in range(0, or_concats)))
        moves_operations_bits.append(moves_bits_arr)
        moves_operations_sizes.append(moves_bits_arr.shape[0])
        
    # find duplicates in moves_operations_bits

    k = 0
    check_if_equal = lambda x, y: np.sum(x!=y)==0 if x.shape == y.shape else False
    check_if_in_lst = lambda k, xs: (lambda x: np.sum([check_if_equal(x, y) for i, y in enumerate(xs) if i != k]) != 0)(xs[k])
    print("len(moves_operations_bits): {}".format(len(moves_operations_bits)))
    for k in range(len(moves_operations_bits)-1, 0, -1):
        if check_if_in_lst(k, moves_operations_bits):
            print("moves_operations_bits[k]: {}".format(moves_operations_bits[k]))
            moves_operations_bits.pop(k)
    print("len(moves_operations_bits): {}".format(len(moves_operations_bits)))

    for moves_bits_arr in moves_operations_bits:
        def get_moves_and_lst(bits_row):
            moves_lst = all_moves[bits_row==1]
            return [move if np.random.randint(0, 3) != 0 else "("+move+"==0)" for move in moves_lst]
        moves = "|".join(["&".join(get_moves_and_lst(bits_row)) for bits_row in moves_bits_arr])
        moves_operations.append(moves)

        print("\ni: {}".format(i))
        print("moves_bits_arr:\n{}".format(moves_bits_arr))
        print("moves: {}".format(moves))

    print("\nmoves_operations_sizes: {}".format(moves_operations_sizes))

    print("\nCalculated moves (functions):")

    for i, move in enumerate(moves_operations):
        print("i: {}, move: {}".format(i, move))

    with open(path_lambda_functions+"lambdas_5.txt", "w") as fout:
        for moves_operation in moves_operations:
            # print("moves_operation: {}".format(moves_operation))
            fout.write("lambda: {}\n".format(moves_operation))

    # globals()["mdoves_operations"] = moves_operations
    # globals()["moves_operations_bits"] = moves_operations_bits
    # globals()["moves_operations_sizes"] = moves_operations_sizes

if __name__ == "__main__":
    path_lambda_functions = "lambda_functions/"

    if not os.path.exists(path_lambda_functions):
        os.makedirs(path_lambda_functions)

    create_lambda_functions_with_matrices(path_lambda_functions)

    # create lambda for shaking images
    # simplest_lambda_functions(path_lambda_functions)
    # simple_random_lambda_creation(path_lambda_functions)
    # create_lambda_functions_2(path_lambda_functions)
    
    # create_lambda_functions_3(path_lambda_functions)
    # create_lambda_functions_4(path_lambda_functions)
    # create_lambda_functions_5(path_lambda_functions)
