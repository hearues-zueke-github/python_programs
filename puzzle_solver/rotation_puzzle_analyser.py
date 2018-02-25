#! /usr/bin/python2.7

import sys

import numpy as np

import matplotlib.pyplot as plt

from colorama import Fore, Style
from copy import deepcopy

def get_rotation_functions(field):
    rot_idx_1 = np.array([[-1, -1, -1, 0, 1, 1, 1, 0], [-1, 0, 1, 1, 1, 0, -1, -1]])
    rot_idx_2 = np.array([[0, -1, -1, -1, 0, 1, 1, 1], [-1, -1, 0, 1, 1, 1, 0, -1]])

    def get_rotate_func(field, y, x):
        coord = ((y, ), (x, ))
        idx_1 = (rot_idx_1+coord).tolist()
        idx_2 = (rot_idx_2+coord).tolist()

        def rotate_cw(i=1):
            for _ in xrange(0, i):
                field[idx_1] = field[idx_2]
        def rotate_cc(i=1):
            for _ in xrange(0, i):
                field[idx_2] = field[idx_1]

        return rotate_cw, rotate_cc

    y, x = field.shape
    rot_funcs = [get_rotate_func(field, j, i) for j in xrange(1, y-1) for i in xrange(1, x-1)]

    return rot_funcs

def mix_field(field):
    rot_funcs = get_rotation_functions(field)

    for _ in xrange(0, 100):
        i_f = np.random.randint(0, 4)
        i_c = np.random.randint(0, 2)
        i_t = np.random.randint(1, 5)
        print("{}, {}, {}".format(i_f, i_c, i_t))
        # print("i_f: {}, i_c: {}, i_t: {}".format(i_f, i_c, i_t))
        rot_funcs[i_f][i_c](i_t)

def find_solving_pattern(field):
    field_orig = field.copy()
    rot_funcs = get_rotation_functions(field)

    def get_anti_moves(moves):
        anti_moves = []

        for f, c, r in moves[::-1]:
            anti_moves.append((f, (c+1)%2, r))

        return anti_moves

    solutions_for_field = {}
    
    for c1 in xrange(0, 2):
        for r1 in xrange(1, 5):
            moves_1 = [(0, c1, r1)]
            anti_moves_1 = get_anti_moves(moves_1)
            moves_2 = [(1, c1, r1)]
            anti_moves_2 = get_anti_moves(moves_2)

            field[:] = field_orig
            for f, c, r in moves_1:
                rot_funcs[f][c](r)
            nums_1 = tuple(field.reshape((-1, )).tolist())
            if not nums_1 in solutions_for_field:
                solutions_for_field[nums_1] = anti_moves_1
            else:
                print("already a solution for: {}, moves used: {}".format(nums_1, moves_1))
            
            field[:] = field_orig
            for f, c, r in moves_2:
                rot_funcs[f][c](r)
            nums_2 = tuple(field.reshape((-1, )).tolist())
            if not nums_2 in solutions_for_field:
                solutions_for_field[nums_2] = anti_moves_2
            else:
                print("already a solution for: {}, moves used: {}".format(nums_2, moves_2))

    for c1 in xrange(0, 2):
     for c2 in xrange(0, 2):
        for r1 in xrange(1, 5):
         for r2 in xrange(1, 5):
            moves_1 = [(0, c1, r1), (1, c2, r2)]
            anti_moves_1 = get_anti_moves(moves_1)
            moves_2 = [(1, c1, r1), (0, c2, r2)]
            anti_moves_2 = get_anti_moves(moves_2)

            field[:] = field_orig
            for f, c, r in moves_1:
                rot_funcs[f][c](r)
            nums_1 = tuple(field.reshape((-1, )).tolist())
            if not nums_1 in solutions_for_field:
                solutions_for_field[nums_1] = anti_moves_1
            else:
                print("already a solution for: {}, moves used: {}".format(nums_1, moves_1))
            
            field[:] = field_orig
            for f, c, r in moves_2:
                rot_funcs[f][c](r)
            nums_2 = tuple(field.reshape((-1, )).tolist())
            if not nums_2 in solutions_for_field:
                solutions_for_field[nums_2] = anti_moves_2
            else:
                print("already a solution for: {}, moves used: {}".format(nums_2, moves_2))

    for c1 in xrange(0, 2):
     for c2 in xrange(0, 2):
      for c3 in xrange(0, 2):
        for r1 in xrange(1, 5):
         for r2 in xrange(1, 5):
          for r3 in xrange(1, 5):
            moves_1 = [(0, c1, r1), (1, c2, r2), (0, c3, r3)]
            anti_moves_1 = get_anti_moves(moves_1)
            moves_2 = [(1, c1, r1), (0, c2, r2), (1, c3, r3)]
            anti_moves_2 = get_anti_moves(moves_2)

            field[:] = field_orig
            for f, c, r in moves_1:
                rot_funcs[f][c](r)
            nums_1 = tuple(field.reshape((-1, )).tolist())
            if not nums_1 in solutions_for_field:
                solutions_for_field[nums_1] = anti_moves_1
            else:
                print("already a solution for: {}, moves used: {}".format(nums_1, moves_1))
            
            field[:] = field_orig
            for f, c, r in moves_2:
                rot_funcs[f][c](r)
            nums_2 = tuple(field.reshape((-1, )).tolist())
            if not nums_2 in solutions_for_field:
                solutions_for_field[nums_2] = anti_moves_2
            else:
                print("already a solution for: {}, moves used: {}".format(nums_2, moves_2))

    for c1 in xrange(0, 2):
     for c2 in xrange(0, 2):
      for c3 in xrange(0, 2):
       for c4 in xrange(0, 2):
        for r1 in xrange(1, 5):
         for r2 in xrange(1, 5):
          for r3 in xrange(1, 5):
           for r4 in xrange(1, 5):
            moves_1 = [(0, c1, r1), (1, c2, r2), (0, c3, r3), (1, c4, r4)]
            anti_moves_1 = get_anti_moves(moves_1)
            moves_2 = [(1, c1, r1), (0, c2, r2), (1, c3, r3), (0, c4, r4)]
            anti_moves_2 = get_anti_moves(moves_2)

            field[:] = field_orig
            for f, c, r in moves_1:
                rot_funcs[f][c](r)
            nums_1 = tuple(field.reshape((-1, )).tolist())
            if not nums_1 in solutions_for_field:
                solutions_for_field[nums_1] = anti_moves_1
            else:
                print("already a solution for: {}, moves used: {}".format(nums_1, moves_1))
            
            field[:] = field_orig
            for f, c, r in moves_2:
                rot_funcs[f][c](r)
            nums_2 = tuple(field.reshape((-1, )).tolist())
            if not nums_2 in solutions_for_field:
                solutions_for_field[nums_2] = anti_moves_2
            else:
                print("already a solution for: {}, moves used: {}".format(nums_2, moves_2))

    for c1 in xrange(0, 2):
     for c2 in xrange(0, 2):
      for c3 in xrange(0, 2):
       for c4 in xrange(0, 2):
        for c5 in xrange(0, 2):
            for r1 in xrange(1, 5):
             for r2 in xrange(1, 5):
              for r3 in xrange(1, 5):
               for r4 in xrange(1, 5):
                for r5 in xrange(1, 5):
                    moves_1 = [(0, c1, r1), (1, c2, r2), (0, c3, r3), (1, c4, r4), (0, c5, r5)]
                    anti_moves_1 = get_anti_moves(moves_1)
                    moves_2 = [(1, c1, r1), (0, c2, r2), (1, c3, r3), (0, c4, r4), (1, c5, r5)]
                    anti_moves_2 = get_anti_moves(moves_2)

                    field[:] = field_orig
                    for f, c, r in moves_1:
                        rot_funcs[f][c](r)
                    nums_1 = tuple(field.reshape((-1, )).tolist())
                    if not nums_1 in solutions_for_field:
                        solutions_for_field[nums_1] = anti_moves_1
                    else:
                        print("already a solution for: {}, moves used: {}".format(nums_1, moves_1))
                    
                    field[:] = field_orig
                    for f, c, r in moves_2:
                        rot_funcs[f][c](r)
                    nums_2 = tuple(field.reshape((-1, )).tolist())
                    if not nums_2 in solutions_for_field:
                        solutions_for_field[nums_2] = anti_moves_2
                    else:
                        print("already a solution for: {}, moves used: {}".format(nums_2, moves_2))

    # moves = [(1, 0, 2), (0, 1, 3)]
    # anti_moves = get_anti_moves(moves)
    # print("in find solving pattern")
    # field[:] = field_orig
    # print("field:\n{}".format(field))
    # for f, c, r in moves:
    #     rot_funcs[f][c](r)
    # print("field:\n{}".format(field))

    # nums = tuple(field.reshape((-1, )).tolist())
    # print("nums: {}".format(nums))
    # solutions_for_field[nums] = anti_moves

    # not_correct_fields = field!=field_orig
    # not_correct_idx = np.where((not_correct_fields).reshape((-1, )))[0]
    # not_correct_nums = field[not_correct_fields]
    # print("not_correct_idx: {}".format(not_correct_idx))
    # print("not_correct_nums: {}".format(not_correct_nums))
    # for f, c, r in anti_moves:
    #     rot_funcs[f][c](r)
    # print("field:\n{}".format(field))

    for key in solutions_for_field:
        print("key: {}, moves: {}".format(key, solutions_for_field[key]))

    print("len(solutions_for_field): {}".format(len(solutions_for_field)))

if __name__ == "__main__":
    # field = np.arange(0, 16).reshape((4, 4))
    field = np.arange(0, 12).reshape((3, 4))
    field_orig = field.copy()
    print("field_orig:\n{}".format(field_orig))

    # print("rot_idx_1:\n{}".format(rot_idx_1))
    # print("rot_idx_2:\n{}".format(rot_idx_2))

    # rot_funcs = get_rotation_functions(field)
    # for j in xrange(0, 1000):
    #     for i in xrange(0, 4):
    #         rot_funcs[i][0](2)
    #     print("\nj: {}, field:\n{}".format(j, field))

    #     if np.sum(field!=field_orig) == 0:
    #         break

    find_solving_pattern(field)

    # field[:] = field_orig
    # print("\nfield:\n{}".format(field))
    # mix_field(field)
    # print("field mixed:\n{}".format(field))

    # rot_1_cw()
    # print("field:\n{}".format(field))
    # rot_2_cw()
    # print("field:\n{}".format(field))
    # rot_3_cw()
    # print("field:\n{}".format(field))
