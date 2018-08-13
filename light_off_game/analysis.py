#! /usr/bin/python3.5

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

    for y in range(0, n):
        for x in range(0, n):
            positions[y, x] = list(map(tuple, positions[y, x].T.tolist()))

    return positions

def apply_on_field(field, positions, y_x_coords):
    for y, x in y_x_coords:
        coordinates = positions[y, x]
        field[coordinates] = (field[coordinates]+1) % modulo

def apply_on_field_once(field, positions, y, x):
    coordinates = positions[y, x]
    field[coordinates] = (field[coordinates]+1) % modulo

def mix_field(field, positions):
    n = field.shape[0]

    mix_field_arr = np.random.randint(0, 2, (n, n))
    print("mix_field_arr:\n{}".format(mix_field_arr))
    y_x_coords = np.array(np.where(mix_field_arr == 1)).T
    apply_on_field(field, positions, y_x_coords)

def solve_field(field, positions):
    n = field.shape[0]
    field_orig = field.copy()

    # first do something on the first row
    # second solve from the frist  until the n-1 row
    # third check if the last row n is solved, if now, repeat from fist step
    #   until it is done

    tries = 1
    is_not_solved = True
    while is_not_solved:
        # print("tries: {}".format(tries))
        field = field_orig.copy()
        moves_done = np.zeros((n, n), dtype=np.int8)

        # print("before solving:")
        # print("field:\n{}".format(field))
        
        random_moves = np.random.randint(0, 2, (n, )) # for the last row (n-th row)
        for i in np.where(random_moves == 1)[0]:
            apply_on_field_once(field, positions, n-1, i)

        moves_done[-1] = random_moves

        # print("after last row finish")
        # print("field:\n{}".format(field))

        for y in range(n-2, -1, -1):
            for x in range(0, n):
                if field[y+1, x] == 1:
                    apply_on_field_once(field, positions, y, x)
                    moves_done[y, x] = 1

            # print("after row #: {}".format(y))
            # print("field:\n{}".format(field))

        if np.sum(field[0]) == 0:
            is_not_solved = False

        # sys.exit(0)

        tries += 1

    print("last try at tries: {}".format(tries))

    # print("moves_done:\n{}".format(moves_done))

    return moves_done

if __name__ == "__main__":
    n = 5

    field = np.zeros((n, n), dtype=np.uint8)

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
