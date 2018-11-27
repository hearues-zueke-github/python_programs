#! /usr/bin/python3.6

import dill
import os
import pdb

import numpy as np

from dotmap import DotMap
from PIL import Image

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

    # create lambda for shaking images
    # create_lambda_functions_2(path_lambda_functions)
    
    # create_lambda_functions_3(path_lambda_functions)
    # create_lambda_functions_4(path_lambda_functions)
    create_lambda_functions_5(path_lambda_functions)
