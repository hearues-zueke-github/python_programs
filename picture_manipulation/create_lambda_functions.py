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


def create_lambda_functions_3(path_lambda_functions):
    n = np.random.randint(5, 31)
    moves_operations = []
    def get_rnd_move():
        moves = ""
        used_ud = np.random.randint(0, 2) == 0
        used_lr = np.random.randint(0, 2) == 0
        if used_ud:
            moves += ("u" if np.random.randint(0, 2) == 0 else "d")*np.random.randint(1, 2)
        if not used_ud or used_lr:
            moves += ("l" if np.random.randint(0, 2) == 0 else "r")*np.random.randint(1, 2)
        return moves

    for i in range(0, n):
        # print("i: {}".format(i))
        moves = "|".join(["&".join([get_rnd_move() for _ in range(0, np.random.randint(2, 5))]) for _ in range(0, np.random.randint(4, 12))])
        moves_operations.append(moves)

    with open(path_lambda_functions+"lambdas_3.txt", "w") as fout:
        for moves_operation in moves_operations:
            print("moves_operation: {}".format(moves_operation))
            fout.write("lambda: {}\n".format(moves_operation))


if __name__ == "__main__":
    path_lambda_functions = "lambda_functions/"

    if not os.path.exists(path_lambda_functions):
        os.makedirs(path_lambda_functions)

    # create lambda for shaking images
    # create_lambda_functions_2(path_lambda_functions)
    
    create_lambda_functions_3(path_lambda_functions)
