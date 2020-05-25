#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import functools
import itertools
import os
import random
import sys
import time

from copy import deepcopy

import numpy as np

from PIL import Image, ImageTk

# Simple enough, just import everything from tkinter.
# from tkinter import *
import tkinter as tk
import tkinter.ttk as ttk

import multiprocessing as mp
from multiprocessing import Process, Pipe # , Lock
from recordclass import recordclass, RecordClass

import matplotlib.pyplot as plt

from threading import Lock

import base64
import json

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()
# import tempfile

import platform
print("platform.system(): {}".format(platform.system()))
os.system('xset r off')

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

class TetrisGeneticAlgorithm(Exception):
    def __init__(self, field_columns, field_rows, field_add_rows):
        self.field_columns = field_columns
        self.field_rows = field_rows

        self.l_piece_name = ['O', 'I', 'S', 'Z', 'T', 'J', 'L']
        self.l_piece_direction = ['W', 'N', 'E', 'S']
        self.piece_name_str = ''.join(sorted(self.l_piece_name))

        self.field_add_rows = field_add_rows
        self.field = np.zeros((self.field_rows+self.field_add_rows, self.field_columns), dtype=np.uint8)
        self.field_y = self.field.shape[0]
        self.field_x = self.field.shape[1]
        self.need_field_reset = False
        
        self.with_garbage = False
        self.garbage_lines = 5

        self.d_pieces = {
            'Z': {
                'pos': {
                    'S': ([[0, -1], [0, 0], [1, 0], [1, 1]], [1, self.field_x-1]),
                    'E': ([[-1, 1], [0, 1], [0, 0], [1, 0]], [0, self.field_x-1]),
                    'N': ([[-1, -1], [-1, 0], [0, 0], [0, 1]], [1, self.field_x-1]),
                    'W': ([[-1, 0], [0, 0], [0, -1], [1, -1]], [1, self.field_x]),
                },
                'idx': 1
            },
            'S': {
                'pos': {
                    'S': ([[0, -1], [0, 0], [-1, 0], [-1, 1]], [1, self.field_x-1]),
                    'E': ([[-1, 0], [0, 0], [0, 1], [1, 1]], [0, self.field_x-1]),
                    'N': ([[1, -1], [1, 0], [0, 0], [0, 1]], [1, self.field_x-1]),
                    'W': ([[-1, -1], [0, -1], [0, 0], [1, 0]], [1, self.field_x]),
                },
                'idx': 2
            },
            'O': {
                'pos': {
                    'S': ([[0, 0], [0, 1], [1, 0], [1, 1]], [0, self.field_x-1]),
                    'E': ([[0, 0], [0, 1], [1, 0], [1, 1]], [0, self.field_x-1]),
                    'N': ([[0, 0], [0, 1], [1, 0], [1, 1]], [0, self.field_x-1]),
                    'W': ([[0, 0], [0, 1], [1, 0], [1, 1]], [0, self.field_x-1]),
                },
                'idx': 3
            },
            'T': {
                'pos': {
                    'S': ([[0, 0], [0, 1], [0, -1], [-1, 0]], [1, self.field_x-1]),
                    'E': ([[0, 0], [1, 0], [-1, 0], [0, -1]], [1, self.field_x]),
                    'N': ([[0, 0], [0, 1], [0, -1], [1, 0]], [1, self.field_x-1]),
                    'W': ([[0, 0], [1, 0], [-1, 0], [0, 1]], [0, self.field_x-1]),
                },
                'idx': 4
            },
            'L': {
                'pos': {
                    'S': ([[0, 0], [0, -1], [0, 1], [-1, 1]], [1, self.field_x-1]),
                    'E': ([[0, 0], [-1, 0], [1, 0], [-1, -1]], [1, self.field_x]),
                    'N': ([[0, 0], [0, -1], [0, 1], [1, -1]], [1, self.field_x-1]),
                    'W': ([[0, 0], [-1, 0], [1, 0], [1, 1]], [0, self.field_x-1]),
                },
                'idx': 5
            },
            'J': {
                'pos': {
                    'S': ([[0, 0], [0, -1], [0, 1], [1, 1]], [1, self.field_x-1]),
                    'E': ([[0, 0], [-1, 0], [1, 0], [-1, 1]], [0, self.field_x-1]),
                    'N': ([[0, 0], [0, -1], [0, 1], [-1, -1]], [1, self.field_x-1]),
                    'W': ([[0, 0], [-1, 0], [1, 0], [1, -1]], [1, self.field_x]),
                },
                'idx': 6
            },
            'I': {
                'pos': {
                    'S': ([[-1, 0], [-1, -1], [-1, 1], [-1, 2]], [1, self.field_x-2]),
                    'E': ([[0, 1], [1, 1], [2, 1], [-1, 1]], [-1, self.field_x-1]),
                    'N': ([[0, 0], [0, -1], [0, 1], [0, 2]], [1, self.field_x-2]),
                    'W': ([[0, 0], [1, 0], [2, 0], [-1, 0]], [0, self.field_x]),
                },
                'idx': 7
            },
        }

        self.d_piece_name_to_piece_idx = {piece_name: d['idx'] for piece_name, d in self.d_pieces.items()}

        self.d_piece_name_counter = {piece_name: 0 for piece_name in self.l_piece_name}
        self.clear_lines_total = 0
        self.clear_lines = 0
        self.max_height = 0

        # the amount of boxes in fields + 7 nodes for currecnt piece + 7 nodes for next piece
        self.input_nodes = self.field_columns*self.field_rows+7+7
        self.d_piece_name_to_output_node_to_direction_pos = {}
        self.d_piece_name_to_needed_directions = {
            'O': set(['S']),
            'I': set(['S', 'E']),
            'S': set(['S', 'E']),
            'Z': set(['S', 'E']),
            'T': set(['S', 'E', 'N', 'W']),
            'J': set(['S', 'E', 'N', 'W']),
            'L': set(['S', 'E', 'N', 'W']),
        }
        for piece_name, d in self.d_pieces.items():
            s_needed_directions = self.d_piece_name_to_needed_directions[piece_name]
            d_output_node_to_direction_pos = {}
            output_node_nr = 0
            for direction, (_, idx_ranges) in d['pos'].items():
                if direction not in s_needed_directions:
                    continue

                for i in range(idx_ranges[0], idx_ranges[1]):
                    d_output_node_to_direction_pos[output_node_nr] = (direction, i)
                    output_node_nr += 1
            self.d_piece_name_to_output_node_to_direction_pos[piece_name] = d_output_node_to_direction_pos
        # self.output_nodes = len(d_output_node_to_direction_pos)


    def get_input_vector(self):
        input_vector = np.zeros((self.input_nodes, ))-1
        input_vector[:-14][self.field[-self.field_rows:].reshape((-1, ))!=0] = 1
        input_vector[-14:-7][self.d_piece_name_to_piece_idx[self.piece_next_name]-1] = 1
        input_vector[-7:][self.d_piece_name_to_piece_idx[self.piece_name]-1] = 1
        return input_vector


    def define_next_piece(self, piece_name, output_node):
        self.piece_name = piece_name
        self.piece_idx = self.d_piece_name_to_piece_idx[self.piece_name]
        self.piece_posi_y = 1
        
        d_output_node_to_direction_pos = self.d_piece_name_to_output_node_to_direction_pos[self.piece_name]
        # for future stuff, here comes the neural network calculation for the argmax of output_node!
        piece_direction, piece_posi_x = d_output_node_to_direction_pos[output_node]

        self.piece_direction = piece_direction
        self.piece_posi_x = piece_posi_x
        self.piece_positions = self.d_pieces[self.piece_name]['pos'][self.piece_direction][0]


    def reset_field(self):
        self.need_field_reset = False
        self.field[:] = 0
        
        if self.with_garbage:
            self.field[-self.garbage_lines:] = 1
            ys = np.arange(self.field_rows+self.field_add_rows-1, self.field_rows+self.field_add_rows-1-self.garbage_lines, -1)
            xs = np.random.randint(0, self.field_columns, (self.garbage_lines, ))
            self.field[(ys, xs)] = 0
        
        for k in self.d_piece_name_counter:
            self.d_piece_name_counter[k] = 0
        
        self.clear_lines_total = 0
        self.clear_lines = 0
        self.max_height = 0


    def do_move_piece_down_instant(self):
        is_move_down_possible = True
        s_pos_now = set()
        for yi_, xi_ in self.piece_positions:
            yi = self.piece_posi_y+yi_
            xi = self.piece_posi_x+xi_
            if yi+1>=self.field_y:
                is_move_down_possible = False
                break
            s_pos_now.add((yi, xi))
        
        if not is_move_down_possible:
            return False

        s_pos_down = set()
        for yi, xi in s_pos_now:
            s_pos_down.add((yi+1, xi))

        s_next_pos = s_pos_down-s_pos_now

        if not all([self.field[yi, xi]==0 for yi, xi in s_next_pos]):
            is_move_down_possible = False
            if self.piece_posi_y < 5:
                self.need_field_reset = True
            return False

        s_pos_down_prev = s_pos_down
        i = 2
        while True:
            s_pos_down = set()
            for yi, xi in s_pos_now:
                s_pos_down.add((yi+i, xi))

            s_next_pos = s_pos_down-s_pos_now

            if any([yi>=self.field_y for yi, _ in s_next_pos]) or any([self.field[yi, xi]!=0 for yi, xi in s_next_pos]):
                break

            s_pos_down_prev = s_pos_down
            i += 1

        s_pos_down = s_pos_down_prev

        s_next_pos = s_pos_down-s_pos_now
        s_prev_pos = s_pos_now-s_pos_down

        for yi, xi in s_prev_pos:
            self.field[yi, xi] = 0
           
        for yi, xi in s_next_pos:
            self.field[yi, xi] = self.piece_idx

        self.piece_posi_y += i-1

        return True
    
    def do_next_piece(self, piece_name, output_node):
        self.define_next_piece(piece_name, output_node)
        is_piece_moved = self.do_move_piece_down_instant()
        self.clear_lines = 0

        # print("self.field:\n{}".format(self.field))

        if not is_piece_moved and self.need_field_reset:
            return
            # self.reset_field()

        idxs_row_full = np.all(self.field!=0, axis=1)
        if np.any(idxs_row_full):
            self.field[idxs_row_full] = 0

            self.clear_lines_total += np.sum(idxs_row_full)
            self.clear_lines = np.sum(idxs_row_full)

            all_rows_full = np.where(idxs_row_full)[0]
            all_rows_empty = np.where(np.all(self.field==0, axis=1))[0]
            all_rows = np.arange(self.field_y-1, -1, -1)

            all_rows_rest = all_rows[(~np.isin(all_rows, all_rows_full))&(~np.isin(all_rows, all_rows_empty))]

            all_rows_needed = all_rows[:all_rows_rest.shape[0]]
            idxs_same_rows = np.where(all_rows_rest!=all_rows_needed)[0]
            
            if idxs_same_rows.shape[0]>0:
                idx_first_row = idxs_same_rows[0]

                rows_from = all_rows_rest[idx_first_row:]
                rows_to = all_rows_needed[idx_first_row:]

                for row_from, row_to in zip(rows_from, rows_to):
                    self.field[row_to] = self.field[row_from]
                    self.field[row_from] = 0

        idxs_pos = np.where(np.flip(np.any(self.field!=0, axis=1)))[0]
        if idxs_pos.shape[0]==0:
            self.max_height = 0
        else:
            self.max_height = idxs_pos[-1]+1


if __name__ == "__main__":
    amount_pieces = 500

    field_columns = 5
    field_rows = 30

    field_add_rows = 4
    field_rows_total = field_rows+field_add_rows
    tetris = TetrisGeneticAlgorithm(field_columns, field_rows, field_add_rows)

    field = tetris.field

    # create garbage!
    field[-5:] = 8
    field[-5:][(np.arange(0, 5), np.random.randint(0, field.shape[1], (5, )))] = 0
    # field[-5:][np.random.randint(0, 2, (5, field.shape[1]))==1] = 8

    d_piece_name_to_amount_output_nodes = {}
    for piece_name in tetris.l_piece_name:
        d_output_node_to_direction_pos = tetris.d_piece_name_to_output_node_to_direction_pos[piece_name]
        amount_output_nodes = len(d_output_node_to_direction_pos)
        d_piece_name_to_amount_output_nodes[piece_name] = amount_output_nodes
    
    minimum_val = min(field_columns, field_rows_total)
    maxiumu_val = max(field_columns, field_rows_total)
    field_weights = tetris.field.copy().astype(dtype=np.int)
    col_vals = np.flip(np.arange(1, field_rows_total+1))
    for i in range(0, field_columns):
        field_weights[:, i] = col_vals+i
    field_weights += np.flip(np.cumsum(np.arange(1, field_rows_total+1)).reshape((-1, 1)))
    print("field_weights:\n{}".format(field_weights))

    amount_next_pieces = 1
    l_best_sequence_piece_name_idx = []
    l_sequence_piece_name = [tetris.l_piece_name[i] for i in np.random.randint(0, 7, (amount_pieces+amount_next_pieces, ))]
    print("l_sequence_piece_name: {}".format(l_sequence_piece_name))

    for piece_nr in range(0, len(l_sequence_piece_name)-amount_next_pieces-1):
        l_piece_name_order_first = l_sequence_piece_name[piece_nr:piece_nr+amount_next_pieces+1]
        
        prev_field = tetris.field.copy()

        d_t_idxs_to_weights_sum_clear_lines_max_height = {}
        for piece_name_last in tetris.l_piece_name:
            def calc_weights(l_piece_name_order):
                max_i = len(l_piece_name_order)-1

                def calc_weights_intern(i, t_idxs, clear_lines):
                    piece_name = l_piece_name_order[i]
                    amount_output_nodes = d_piece_name_to_amount_output_nodes[piece_name]
                    for output_node in range(0, amount_output_nodes):
                        prev_field = tetris.field.copy()
                        tetris.do_next_piece(piece_name=piece_name, output_node=output_node)
                        
                        clear_lines_new = clear_lines+tetris.clear_lines

                        t_idxs_now = t_idxs+(output_node, )
                        
                        if i==max_i:
                            weights_sum = np.sum(field_weights[tetris.field!=0])
                            max_height = tetris.max_height
                            d_t_idxs_to_weights_sum_clear_lines_max_height[t_idxs_now] = (weights_sum, clear_lines_new, max_height)
                        else:
                            calc_weights_intern(i+1, t_idxs_now, clear_lines_new)

                        tetris.field[:] = prev_field
                calc_weights_intern(0, (piece_name_last, ), clear_lines=0)

            l_piece_name_order = l_piece_name_order_first+[piece_name_last]
            calc_weights(l_piece_name_order)
        

        # pn1 = l_piece_name_order_first[0]
        # pn2 = l_piece_name_order_first[1]
        l_t_idxs_combined = list(itertools.product(*[range(0, d_piece_name_to_amount_output_nodes[pn]) for pn in l_piece_name_order_first]))
        
        l_idxs = []
        l_weights_average = []

        # for idx1 in range(0, d_piece_name_to_amount_output_nodes[pn1]):
        #     for idx2 in range(0, d_piece_name_to_amount_output_nodes[pn2]):
        for t_idxs_combined in l_t_idxs_combined:
                max_height_sum = 0
                clear_lines_sum = 0
                weight_sum = 0
                amount_values = 0

                for pn_last in tetris.l_piece_name:
                    max_clear_lines = 0
                    min_weight = np.sum(field_weights)
                    min_max_height = field_rows
                    for idx_last in range(0, d_piece_name_to_amount_output_nodes[pn_last]):
                        t_combined = (pn_last, )+t_idxs_combined+(idx_last, )
                        weight, clear_lines, max_height = d_t_idxs_to_weights_sum_clear_lines_max_height[t_combined]
                        # weight, clear_lines, max_height = d_t_idxs_to_weights_sum_clear_lines_max_height[(pn_last, idx1, idx2, idx_last)]

                        if min_max_height > max_height:
                            min_max_height = max_height
                            max_clear_lines = clear_lines
                            min_weight = weight
                        elif max_clear_lines < clear_lines:
                            max_clear_lines = clear_lines
                            min_weight = weight
                        elif min_weight > weight:
                            min_weight = weight

                    max_height_sum += min_max_height
                    clear_lines_sum += max_clear_lines
                    weight_sum += min_weight
                    amount_values += 1

                l_idxs.append(t_idxs_combined)
                # l_idxs.append([idx1, idx2])
                l_weights_average.append(weight_sum/amount_values)

        best_t_idxs = l_idxs[np.argmin(l_weights_average)]

        t = (l_piece_name_order_first[0], best_t_idxs[0])
        tetris.do_next_piece(piece_name=t[0], output_node=t[1])
        column_between = np.empty((prev_field.shape[0], 1), dtype=object)
        column_between[:] = '|'
        combined_field = np.hstack((prev_field.astype(object), column_between, tetris.field.astype(object)))
        print("combined_field:\n{}".format(combined_field))
        # print("tetris.field:\n{}".format(tetris.field))
        print("l_piece_name_order_first: {}".format(l_piece_name_order_first))
        print("piece_nr: {}, t: {}, max_height: {}".format(piece_nr, t, tetris.max_height))

        l_best_sequence_piece_name_idx.append(t)


    prev_field = tetris.field.copy()
    tetris.reset_field()
    print("\n")
    l_max_height = []
    for step_i, (piece_name, output_node) in enumerate(l_best_sequence_piece_name_idx, 1):
        tetris.do_next_piece(piece_name=piece_name, output_node=output_node)
        idxs_pos = np.where(np.flip(np.any(tetris.field!=0, axis=1)))[0]
        if idxs_pos.shape[0]==0:
            max_height = 0
        else:
            max_height = idxs_pos[-1]+1
        l_max_height.append(max_height)
        print("piece_name: {}, output_node: {}".format(piece_name, output_node))
        print("step_i: {}, tetris.field:\n{}".format(step_i, tetris.field))
    tetris.field[:] = prev_field

    print("l_max_height: {}".format(l_max_height))

    # print a simple graph, where the max height of each step is shown!
    plt.figure()

    plt.plot(np.arange(1, len(l_max_height)+1), l_max_height, 'b.-')
    plt.title('Max Height per next piece nr.')
    plt.xlabel('Piece Nr.')
    plt.ylabel('Max Height')

    plt.show()

    # TODO: next, add a conversion from given field, piece_currenct and piece_next
    # into input_vector and output_vector for the neural network learning datas!
    # TODO: save the given input_vector and output_vector for the piece_current and piece_next
    # into files for the givene (rows, cols) of the field!
