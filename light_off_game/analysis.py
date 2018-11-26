#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import sys

import numpy as np

modulo = 2

def generate_field_position_crosses(n):
    positions = np.zeros((n+2, n+2), dtype=object)

    for y in range(1, n+1):
        for x in range(1, n+1):
            positions[y, x] = np.array([[y, x], [y, x+1], [y, x-1], [y+1, x], [y-1, x]], dtype=np.int8)

    positions = positions[1:n+1, 1:n+1]-1

    # now calc -1 and eliminate everything which contains 0 or n as a value
    y_x_coords = \
    [(y, 0) for y in range(0, n)] + \
    [(y, n-1) for y in range(0, n)] + \
    [(0, x) for x in range(1, n-1)] + \
    [(n-1, x) for x in range(1, n-1)]

    for y, x in y_x_coords:
        coordinates = positions[y, x]
        remove_idx = np.where(np.logical_or.reduce((coordinates==-1) | (coordinates==n), axis=1))[0]
        positions[y, x] = np.delete(coordinates, remove_idx, axis=0)

    # print("positions:\n{}".format(positions))
    for y in range(0, n):
        for x in range(0, n):
            positions[y, x] = tuple(positions[y, x].T.tolist())

    # print("positions:\n{}".format(positions))
    return positions

def apply_on_field(field, positions, y_x_coords):
    for y, x in y_x_coords:
        coordinates = positions[y, x]
        field[coordinates] = (field[coordinates]+1) % modulo

def apply_on_field_once(field, positions, y, x):
    coordinates = positions[y, x]
    # print("type(coordinates): {}".format(type(coordinates)))
    field[coordinates] = (field[coordinates]+1) % modulo

def mix_field(field, positions):
    n = field.shape[0]

    move_random_arr = np.random.randint(0, 2, (n, n))
    # print("move_random_arr:\n{}".format(move_random_arr))
    y_x_coords = np.array(np.where(move_random_arr == 1)).T
    apply_on_field(field, positions, y_x_coords)

def solve_field(field, positions):
    n = field.shape[0]
    field_orig = field.copy()

    # first do something on the first row
    # second solve from the frist  until the n-1 row
    # third check if the last row n is solved, if now, repeat from fist step
    #   until it is done

    y_x_idx = np.zeros((n-1, n, 2), dtype=np.uint8)
    y_x_idx[:, :, 0] = np.arange(n-2, -1, -1).reshape((-1, 1))
    y_x_idx[:, :, 1] = np.arange(0, n).reshape((1, -1))
    y_x_idx = y_x_idx.reshape((-1, 2))

    tries = 1
    is_not_solved = True
    while is_not_solved:
        field = field_orig.copy()
        moves_done = np.zeros((n, n), dtype=np.int8)

        random_moves = np.random.randint(0, 2, (n, )) # for the last row (n-th row)
        for i in np.where(random_moves == 1)[0]:
            apply_on_field_once(field, positions, n-1, i)

        moves_done[-1] = random_moves

        # for y in range(n-2, -1, -1):
        #     for x in range(0, n):
        for y, x in y_x_idx:
            if field[y+1, x] == 1:
                apply_on_field_once(field, positions, y, x)
                moves_done[y, x] = 1

        if np.sum(field[0]) == 0:
            is_not_solved = False

        tries += 1

    print("last try at tries: {}".format(tries))

    # print("moves_done:\n{}".format(moves_done))

    return moves_done

def test_shuffle_and_solve(n):
    field = np.zeros((n, n), dtype=np.int8)
    positions = generate_field_position_crosses(n)

    print("\nbefore mixing:")
    print("field:\n{}".format(field))
    
    mix_field(field, positions)

    print("\nafter mixing:")
    print("field:\n{}".format(field))

    moves_done = solve_field(field, positions)
    print("\nmoves_done:\n{}".format(moves_done))

    apply_on_field(field, positions, np.array(np.where(moves_done == 1)).T)

    print("\nafter moving:")
    print("field:\n{}".format(field))

# This one line is the last (bottom) line, so everything below is solved except the first row
# of the varibale field. If there are all possible solutions for the last line, then the
# solving of the light game is much much easier!
def get_all_possible_1_liner_solutions(n):
    all_1_liner = []
    all_1_liner_with_solution = []

    field = np.zeros((n, n), dtype=np.int8)
    positions = generate_field_position_crosses(n)

    y_x_idx = np.zeros((n-1, n, 2), dtype=np.uint8)
    y_x_idx[:, :, 0] = np.arange(1, n).reshape((-1, 1))
    y_x_idx[:, :, 1] = np.arange(0, n).reshape((1, -1))
    y_x_idx = y_x_idx.reshape((-1, 2))

    def get_moves_solve_except_last_line(field):
        for y, x in y_x_idx:
            if field[y-1, x] != 0:
                apply_on_field_once(field, positions, y, x)

    for _ in range(0, 2500):
        mix_field(field, positions)
        get_moves_solve_except_last_line(field)
        last_line = field[-1].copy().tolist()

        if not last_line in all_1_liner:
            all_1_liner.append(last_line)
            all_1_liner_with_solution.append((last_line.copy(), solve_field(field, positions)))

    return all_1_liner_with_solution

if __name__ == "__main__":
    n = 6

    # test_shuffle_and_solve(n)

    all_1_liner_with_solution = get_all_possible_1_liner_solutions(n)

    print("all_1_liner_with_solution:\n{}".format(all_1_liner_with_solution))
    # print("len(all_1_liner_with_solution): {}".format(len(all_1_liner_with_solution)))

    for idx, (last_line, moves) in enumerate(all_1_liner_with_solution):
        print("\nidx: {}, last_line: {}".format(idx, last_line))
        print("moves:\n{}".format(moves))
