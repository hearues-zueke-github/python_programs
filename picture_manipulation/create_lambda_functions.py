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

path_dir_root = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
sys.path.append(path_dir_root+"../combinatorics/")
import different_combinations
# from ..combinatorics import different_combinations

def write_dm_obj_txt(dm):
    print("Now in def 'write_dm_obj_txt'")
    
    assert dm.path_dir
    assert dm.save_data
    assert dm.function_str_lst
    assert dm.file_name_dm
    assert dm.file_name_txt

    path_dir = dm.path_dir
    save_data = dm.save_data
    function_str_lst = dm.function_str_lst
    file_name_dm = dm.file_name_dm
    file_name_txt = dm.file_name_txt

    print("file_name_dm: {}".format(file_name_dm))
    print("file_name_txt: {}".format(file_name_txt))

    if path_dir == None:
        path_dir = "./"
    if path_dir[-1] != "/":
        path_dir += "/"

    if save_data:
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        with gzip.open(path_dir+file_name_dm, "wb") as fout:
            dill.dump(dm, fout)

        with open(path_dir+file_name_txt, "w") as fout:
            for line in function_str_lst:
                fout.write("{}\n".format(line))


def get_params_arr(ft):
    # ft...frame thickness
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

    return params_arr


def create_lambda_functions_with_matrices(
    path_dir=None,
    save_data=False,
    file_name_dm="dm.pkl.gz",
    file_name_txt="lambdas.txt",
    ft=1,
    min_or=1, max_or=3,
    min_and=3, max_and=5,
    min_n=3, max_n=6):
    # With this you can create any size for the window!
    print("Now in def 'create_lambda_functions_with_matrices'")
    
    params_arr = get_params_arr(ft)
    # print("params_arr: \n{}".format(params_arr))

    params_1 = params_arr.reshape((-1))
    params_0 = np.array(["i({})".format(param) for param in params_1])

    params = np.vstack((params_1, params_0)).T

    globals()['params'] = params

    # TODO: make matrices for 1, 2... to 9 elements!
    # BUT! for 1 e.g.:
    # [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    # for 2:
    # [[0, 1], [0, 2], ... , [0, 9], [1, 2], [1, 3], ... [7, 8]]
    # for 3:
    # [[0, 1, 2], [0, 1, 3], ...]

    # TODO: maybe can be removed!
    # m = 2
    # n = params.shape[0]
    # idx = np.array([0, 1]) # do the pascal triangle!
    # for i in range(0, n):
    #     idx = np.hstack((idx, [0]))+np.hstack(([0], idx))
    # idx = np.cumsum(idx) # get all absolute idx from-to for the arr

    # arr = different_combinations.get_all_combinations_repeat(m, n)
    # arr = np.hstack((np.sum(arr, axis=1).reshape((-1, 1)), arr))
    # arr2 = arr.astype(np.uint8).reshape((-1, )).view(",".join(["u1" for _ in range(0, n+1)]))
    # arr2 = np.sort(arr2, order=("f0", )).view(np.uint8).reshape((-1, n+1))[:, 1:]

    # groups = np.array([arr2[i1:i2] for i1, i2 in zip(idx[1:-1], idx[2:])])
    # group_num_amount = {i: group for i, group in enumerate(groups, 1)}

    # globals()["groups"] = groups
    # globals()["group_num_amount"] = group_num_amount

    # sys.exit()

    def get_random_and(params, min_n=1, max_n=3):
        # First get random num_amount
        key = np.random.choice(np.arange(min_n, max_n+1))
        
        # choose one row of group_num!
        # group_num = group_num_amount[key]
        # idx_choosen_param = group_num[np.random.randint(0, group_num.shape[0])] 

        # print("key: {}".format(key))
        # print("group_num: {}".format(group_num))
        # print("idx_choosen_param: {}".format(idx_choosen_param))
        # globals()['idx_choosen_param'] = idx_choosen_param

        group_num = np.zeros((params.shape[0], ), dtype=np.uint8)
        group_num[:key] = 1
        idx_choosen_param = np.random.permutation(group_num)
        # print("key: {}".format(key))
        # print("group_num: {}".format(group_num))
        # print("idx_choosen_param: {}".format(idx_choosen_param))
        # sys.exit(-1)

        # Now invert some of the params if needed!
        idx_inv_param = np.random.randint(0, 2, params.shape[0])
        idx_arr_inv_param = np.zeros(params.shape, dtype=np.int)
        idx_arr_inv_param[np.arange(0, params.shape[0]), idx_inv_param] = 1
        choosen_inv_params = params[idx_arr_inv_param==1]

        and_params = choosen_inv_params[idx_choosen_param==1]
        # and_params = choosen_inv_params[idx_choosen_param]
        # print("  and_params: {}".format(and_params))
        and_str = "&".join(and_params)
        # print("and_str: {}".format(and_str))

        return and_str, idx_choosen_param, idx_inv_param

    def get_random_or(params, min_and=1, max_and=4, min_n=1, max_n=3):
        amount_and = np.random.randint(min_and, max_and+1)
        # print("amount_and: {}".format(amount_and))
        and_values = [get_random_and(params, min_n=min_n, max_n=max_n) for _ in range(0, amount_and)]

        and_lst, idx_choosen_params, idx_inv_params = list(zip(*and_values))

        or_str = "lambda: "+"|".join(and_lst)

        return or_str, np.array(idx_choosen_params), np.array(idx_inv_params)

    def get_random_booleans_str(params, min_or=1, max_or=8, min_and=1, max_and=4, min_n=1, max_n=3):
        amount_or = np.random.randint(min_or, max_or+1)
        or_lst = [get_random_or(params, min_and=min_and, max_and=max_and, min_n=min_n, max_n=max_n)
        for _ in range(0, amount_or)]

        return or_lst

    # get_random_and(params, group_num_amount)
    # or_str = get_random_or(params, group_num_amount)
    function_str_values = get_random_booleans_str(params,
        min_or=min_or, max_or=max_or,
        min_and=min_and, max_and=max_and,
        min_n=min_n, max_n=max_n)
    function_str_lst, idx_choosen_params_lst, idx_inv_params_lst = list(zip(*function_str_values))
    idx_choosen_params_lst = np.array(idx_choosen_params_lst)
    idx_inv_params_lst = np.array(idx_inv_params_lst)

    # for j, (func_str, idx_choosen, idx_inv) in enumerate(zip(function_str_lst, idx_choosen_params_lst, idx_inv_params_lst), 1):
    #     print("\n  j: {}, func_str: {}".format(j, func_str))
    #     print("  idx_choosen:\n{}".format(idx_choosen))
    #     print("  idx_inv:\n{}".format(idx_inv))

    dm = DotMap()

    func_params = DotMap({
        # 'path_dir': path_dir,
        # 'file_name_dm': file_name_dm,
        # 'file_name_txt': file_name_txt,
        # 'save_data': save_data,
        # 'ft': ft,
        'min_or': min_or,
        'max_or': max_or,
        'min_and': min_and,
        'max_and': max_and,
        'min_n': min_n,
        'max_n': max_n,
    })

    dm.path_dir = path_dir
    dm.save_data = save_data
    dm.file_name_dm = file_name_dm
    dm.file_name_txt = file_name_txt
    dm.used_method = "create_lambda_functions_with_matrices"
    dm.function_str_lst = function_str_lst
    dm.func_params = func_params
    dm.params = params
    dm.idx_choosen_params_lst = idx_choosen_params_lst
    dm.idx_inv_params_lst = idx_inv_params_lst
    dm.ft = ft

    print("2nd save_data: {}".format(save_data))
    # if save_data:
    #     if path_dir == None:
    #         path_dir = "./"
    #     if path_dir[-1] != "/":
    #         path_dir += "/"
    #     if not os.path.exists(path_dir):
    #         os.makedirs(path_dir)

    #     with gzip.open(path_dir+file_name_dm, "wb") as fout:
    #         dill.dump(dm, fout)

    #     with open(path_dir+file_name_txt, "w") as fout:
    #         for line in function_str_lst:
    #             fout.write("{}\n".format(line))

    # write_dm_obj_txt(dm)

    return dm


def simplest_lambda_functions(path_dir):
    function_str_lst = [
        "((u+1)%2)|((d+1)%2)",
        "r|((l+1)%2)",
    ]

    # function_str_lst = ["lambda: "+func_str for func_str in function_str_lst]

    with open(path_dir, "w") as fout:
        for func_str in function_str_lst:
            fout.write("lambda: {}\n".format(func_str))

    return function_str_lst


def conway_game_of_life_functions(path_dir=None, save_data=True):
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
"""[1:-1],
    ]

    print("function_str_lst: {}".format(function_str_lst))

    if save_data:
        with open(path_dir, "w") as fout:
            for func_str in function_str_lst:
                if "def " in func_str:
                    fout.write("{}\n".format(func_str))
                else:
                    fout.write("lambda: {}\n".format(func_str))

    dm = DotMap()
    dm.params = params
    dm.function_str_lst = function_str_lst
    dm.idx_choosen_params_lst = [np.random.randint(0, 2, (3, 3), dtype=np.uint8)] # idx_choosen_params_lst
    dm.idx_inv_params_lst = [np.random.randint(0, 2, (3, 3), dtype=np.uint8)]

    dm.ft = 1

    return dm
    # return function_str_lst


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


def create_lambda_functions_2(path_dir):
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

    with open(path_dir+"lambdas_2.txt", "w") as fout:
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

    with open(path_dir+"resize_params_2.pkl", "wb") as fout:
        dill.dump(dm, fout)


def create_lambda_functions_3(path_dir, tf=2):
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

    with open(path_dir+"lambdas_3.txt", "w") as fout:
        for moves_operation in moves_operations:
            print("moves_operation: {}".format(moves_operation))
            fout.write("lambda: {}\n".format(moves_operation))

def create_lambda_functions_4(path_dir, tf=2):
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

    with open(path_dir+"lambdas_4.txt", "w") as fout:
        for moves_operation in moves_operations:
            # print("moves_operation: {}".format(moves_operation))
            fout.write("lambda: {}\n".format(moves_operation))

    # globals()["mdoves_operations"] = moves_operations
    # globals()["moves_operations_bits"] = moves_operations_bits
    # globals()["moves_operations_sizes"] = moves_operations_sizes

def create_lambda_functions_5(path_dir, tf=2):
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

    with open(path_dir+"lambdas_5.txt", "w") as fout:
        for moves_operation in moves_operations:
            # print("moves_operation: {}".format(moves_operation))
            fout.write("lambda: {}\n".format(moves_operation))

    # globals()["mdoves_operations"] = moves_operations
    # globals()["moves_operations_bits"] = moves_operations_bits
    # globals()["moves_operations_sizes"] = moves_operations_sizes

if __name__ == "__main__":
    argv = sys.argv
    value_str = ""
    if len(argv) > 1:
        value_str = "".join(argv[1:]).lstrip(",").rstrip(",")
    print("value_str: {}".format(value_str))
    value_str_split = value_str.split(",")
    # print("value_str_split: {}".format(value_str_split))
    value_str_split = list(filter(lambda x: x.count("=") == 1, value_str_split))
    # print("value_str_split: {}".format(value_str_split))
    var_val_lst = list(map(lambda x: x.split("="), value_str_split))
    print("var_val_lst:\n{}".format(var_val_lst))

    ft = 1
    max_n = 5
    path_dir = "lambda_functions/"
    file_name_dm = "dm.pkl.gz"
    file_name_txt = "lambdas.txt"
    save_data = True
    min_or=4
    max_or=4
    min_and=3
    max_and=3
    min_n=2
    max_n=2

    using_vars_type = {
        'ft': int,
        'max_n': int,
        'path_dir': str,
        'save_data': bool,
        'min_and' : int,
        'max_and' : int,
        'min_or' : int,
        'max_or' : int,
        'min_n' : int,
        'max_n' : int,
    }

    print("Values for variables before input:")
    print(" - ft: {}".format(ft))
    print(" - min_and: {}".format(min_and))
    print(" - max_and: {}".format(max_and))
    print(" - min_or: {}".format(min_or))
    print(" - max_or: {}".format(max_or))
    print(" - min_n: {}".format(min_n))
    print(" - max_n: {}".format(max_n))
    print(" - path_dir: {}".format(path_dir))
    print(" - file_name_dm: {}".format(file_name_dm))
    print(" - file_name_txt: {}".format(file_name_txt))
    print(" - save_data: {}".format(save_data))

    # Do the check and convert the variable for the given input!
    for var, val in var_val_lst:
        if var in using_vars_type:
            try:
                type_var = using_vars_type[var]
                val_converted = type_var(val)
                if type_var == int:
                    exec("{var} = {val}".format(var=var, val=val_converted), globals())
                elif type_var == str:                    
                    exec("{var} = '{val}'".format(var=var, val=val_converted), globals())
                elif type_var == bool:                    
                    if val == "0" or val == "False" or val == "false":
                        save_data = False
                    elif val == "1" or val == "True" or val == "true":
                        save_data = True
                    else:
                        raise
                    # exec("{var} = bool('{val}')".format(var=var, val=val_converted), globals())
                print("var: {}, val: {}, type_var: {}, val_converted: {}".format(
                    var, val, type_var, val_converted))
            except:
                print("For var '{var}' could not convert to type '{type_var}' of val '{val}'!".format(
                    var=var, type_var=type_var, val=val))

    print("Values for variables after input:")
    print(" - ft: {}".format(ft))
    print(" - min_and: {}".format(min_and))
    print(" - max_and: {}".format(max_and))
    print(" - min_or: {}".format(min_or))
    print(" - max_or: {}".format(max_or))
    print(" - min_n: {}".format(min_n))
    print(" - max_n: {}".format(max_n))
    print(" - path_dir: {}".format(path_dir))
    print(" - file_name_dm: {}".format(file_name_dm))
    print(" - file_name_txt: {}".format(file_name_txt))
    print(" - save_data: {}".format(save_data))

    # if not os.path.exists(path_dir):
    #     os.makedirs(path_dir)

    dm = create_lambda_functions_with_matrices(path_dir=path_dir, ft=ft,
        min_or=min_or, max_or=max_or,
        min_and=min_and, max_and=max_and,
        min_n=min_n, max_n=max_n, save_data=save_data)

    write_dm_obj_txt(dm)

    # create lambda for shaking images
    # simplest_lambda_functions(path_dir)
    # simple_random_lambda_creation(path_dir)
    # create_lambda_functions_2(path_dir)
    
    # create_lambda_functions_3(path_dir)
    # create_lambda_functions_4(path_dir)
    # create_lambda_functions_5(path_dir)
