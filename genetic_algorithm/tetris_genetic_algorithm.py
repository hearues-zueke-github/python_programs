#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import functools
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

from threading import Lock

import matplotlib.pyplot as plt

import base64
import json

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()
# import tempfile

from utils_tetris import parse_tetris_game_data

import platform
print("platform.system(): {}".format(platform.system()))
# os.system('xset r off')

from SimpleNeuralNetwork import SimpleNeuralNetwork

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

class TetrisGameField(Exception):
    def __init__(self, d_basic_data_info, l_snn, max_piece_nr=1000, using_pieces=3):
        self.rows = d_basic_data_info['rows']
        self.cols = d_basic_data_info['cols']
        self.amount_pieces = d_basic_data_info['amount_pieces']
        self.l_amount_group_pieces = d_basic_data_info['l_amount_group_pieces']
        self.l_group_pieces = d_basic_data_info['l_group_pieces']

        self.arr_pcs_idx = np.arange(0, self.amount_pieces)
        self.arr_pcs_one_hot = np.diag(np.ones((self.amount_pieces, )))

        self.l_group_piece_arr_pos = d_basic_data_info['l_group_piece_arr_pos']
        self.l_group_piece_max_x = d_basic_data_info['l_group_piece_max_x']
        self.l_pcs_idx_l_index_to_group_idx_pos = d_basic_data_info['l_pcs_idx_l_index_to_group_idx_pos']

        self.using_pieces = using_pieces
        self.max_piece_nr = max_piece_nr

        self.l_snn = l_snn

        self.arr_x = np.empty((self.rows*self.cols+self.amount_pieces*self.using_pieces, ))

        self.garbage_lines = 5
        self.with_garbage = False

        self.field_add_rows = 4
        self.field = np.zeros((self.rows+self.field_add_rows, self.cols), dtype=np.uint8)
        self.field_y = self.field.shape[0]
        self.field_x = self.field.shape[1]

        # self.reset_values()


    def __repr__(self):
        return f'TetrisGameField(rows={self.rows}, cols={self.cols}, amount_pieces={self.amount_pieces})'


    def reset_values(self):
        self.field[:] = 0

        self.l_clear_lines = []
        self.l_pieces_used = []

        self.need_field_reset = False
        self.piece_nr = 0
        self.clear_lines = 0

        self.l_next_pcs = np.random.choice(self.arr_pcs_idx, size=(self.using_pieces, )).tolist()
        self.pcs_now = random.choice(self.arr_pcs_idx)

        if self.with_garbage:
            self.add_garbage_lines()


    def add_garbage_lines(self):
        self.field[-self.garbage_lines:] = 1
        ys = np.arange(self.rows+self.field_add_rows-1, self.rows+self.field_add_rows-1-self.garbage_lines, -1)
        xs = np.random.randint(0, self.cols, (self.garbage_lines, ))
        self.field[(ys, xs)] = 0


    def define_next_piece(self):
        self.pcs_now = self.l_next_pcs.pop(0)
        self.l_next_pcs.append(random.choice(self.arr_pcs_idx))
        
        self.piece_posi_y = 1
        
        snn = self.l_snn[self.pcs_now]
        self.arr_x[:] = 0
        self.arr_x[:self.rows*self.cols] = (self.field[self.field_add_rows:].reshape((-1, ))!=0)+0
        self.arr_x[self.rows*self.cols:] = self.arr_pcs_one_hot[[self.pcs_now]+self.l_next_pcs[:-1]].reshape((-1, ))
        self.arr_x[self.arr_x==0] = -1

        # print("self.arr_x: {}".format(self.arr_x))
        arr_field = self.arr_x[:self.rows*self.cols].reshape((-1, self.cols)).astype(object)
        arr_field[arr_field==1.] = '0'
        arr_field[arr_field==-1.] = ' '
        print("self.arr_x:\n{}".format(arr_field))
        input("ENTER...")
        argmax = snn.calc_output_argmax(self.arr_x)

        group_idx, piece_posi_x = self.l_pcs_idx_l_index_to_group_idx_pos[self.pcs_now][argmax]

        self.group_idx = group_idx
        self.piece_posi_x = piece_posi_x

        self.piece_positions = self.l_group_piece_arr_pos[self.pcs_now][self.group_idx]


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
            self.field[yi, xi] = self.pcs_now+1

        self.piece_posi_y += i-1

        return True


    def main_game_loop(self):
        self.reset_values()

        while True:
            self.define_next_piece()
            is_piece_moved = self.do_move_piece_down_instant()

            if not is_piece_moved and self.need_field_reset:
                return False
            elif self.piece_nr>self.max_piece_nr:
                return True

            idxs_row_full = np.all(self.field!=0, axis=1)
            if np.any(idxs_row_full):
                self.field[idxs_row_full] = 0

                self.clear_lines += np.sum(idxs_row_full)

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

            self.piece_nr += 1


class GeneticAlgorithmTetris(Exception):
    def __init__(self, population_size, Xs, Ts):
        self.population_size = population_size

        self.input_nodes = 5
        self.hidden_nodes = [7, 6]
        self.output_nodes = 3
        
        self.l_snn = [self.generate_new_simple_nn() for _ in range(0, population_size)]

        self.using_parent_amount_percent = 0.4
        self.using_parent_amount = int(self.population_size*self.using_parent_amount_percent)
        self.crossover_percent = 0.40
        self.mutation_percent = 0.1
        self.mutation_add_percent = 0.05
        self.mutation_sub_percent = 0.05

        self.Xs = Xs
        self.Ts = Ts

        sys.exit()

        # data only for testing the performance of the snn!
        # self.amount = 2000

        # X = np.random.randint(0, 2, (self.amount, self.input_nodes)).T
        # T = np.vstack(((X[0]^X[3]+X[4])%2, (X[1]^X[2])&(X[2]|X[3]), (X[3]|X[4])&X[0])).T
        # X = X.T

        # X[X==0] = -1

        # self.X = X.astype(np.float)
        # self.T = T.astype(np.float)

        # # add some noise to the X matrix!
        # self.X += np.random.uniform(-1./100, 1./100, self.X.shape)

        # split_n = int(X.shape[0]*0.7)
        # self.X_train = X[:split_n]
        # self.X_test = X[split_n:]
        # self.T_train = T[:split_n]
        # self.T_test = T[split_n:]


        input_nodes = Xs_full[0].shape[1]
        hidden_nodes = [50]
        def create_new_snn(output_nodes):
            return SimpleNeuralNetwork(input_nodes=input_nodes, hidden_nodes=hidden_nodes, output_nodes=output_nodes)

        max_piece_nr = 1000
        using_pieces = 3

        l_snn = [create_new_snn(output_nodes=T.shape[1]) for T in Ts_full]
        tgm = TetrisGameField(d_basic_data_info=d_basic_data_info, l_snn=l_snn, max_piece_nr=max_piece_nr, using_pieces=using_pieces)



    def generate_new_simple_nn(self):
        return SimpleNeuralNetwork(input_nodes=self.input_nodes, hidden_nodes=self.hidden_nodes, output_nodes=self.output_nodes)


    def simple_genetic_algorithm_training(self, epochs=100):
        l_cecf_train_all = []

        l_cecf_train = [snn.f_cecf(snn.calc_feed_forward(self.X_train), self.T_train)/self.T_train.shape[0]/self.T_train.shape[1]
            for snn in self.l_snn
        ]
        l_cecf_train_all.append(l_cecf_train)

        print("first: l_cecf_train: {}".format(l_cecf_train))

        for epoch in range(1, epochs):
            print("epoch: {}".format(epoch))

            l_sorted = sorted([(i, v) for i, v in enumerate(l_cecf_train, 0)], key=lambda x: x[1])
            l_idx = [idx for idx, cecf_train in l_sorted]
            
            l_snn_new_parents = [self.l_snn[i] for i in l_idx[:self.using_parent_amount]]

            # do the back_propagation only for the one best parent!
            snn = l_snn_new_parents[0]
            bwsd = snn.calc_backprop(self.X_train, self.T_train, snn.bws)
            eta = 0.001
            snn.bws = [w-wg*eta for w, wg in zip(snn.bws, bwsd)]

            amount_parents = len(l_snn_new_parents)
            amount_childs = self.population_size-amount_parents
            
            using_idx_delta = np.random.randint(1, amount_parents, (amount_childs, ))
            using_idx_delta[0] = np.random.randint(0, amount_parents)
            using_idx = np.cumsum(using_idx_delta)%amount_parents
            l_snn_new_childs = [self.generate_new_simple_nn() for _ in using_idx]

            for i1, i2, snn_child in zip(using_idx, np.roll(using_idx, 1), l_snn_new_childs):
                snn1, snn2 = l_snn_new_parents[i1], l_snn_new_parents[i2]

                for w1, w2, w_c in zip(snn1.bws, snn2.bws, snn_child.bws):
                    shape = w1.shape
                    a1 = w1.reshape((-1, ))
                    a2 = w2.reshape((-1, ))
                    a_c = w_c.reshape((-1, ))

                    length = a1.shape[0]

                    # crossover step
                    idx_crossover = np.zeros((length, ), dtype=np.uint8)
                    idx_crossover[:int(length*self.crossover_percent)] = 1

                    a_c[:] = a1

                    # idx_crossover_1 = np.random.permutation(idx_crossover)==1
                    # assert np.all(a_c[idx_crossover_1]==a1[idx_crossover_1])
                    # a_c[idx_crossover_1] = a1[idx_crossover_1]
                    
                    idx_crossover_2 = np.random.permutation(idx_crossover)==1
                    a_c[idx_crossover_2] = a2[idx_crossover_2]

                    # mutation step
                    idx_mutation = np.zeros((length, ), dtype=np.uint8)
                    idx_mutation[:int(length*self.mutation_percent)] = 1

                    idx_mutation_c = np.random.permutation(idx_mutation)==1
                    amount_mutation_vals = np.sum(idx_mutation)
                    a_c[np.random.permutation(idx_mutation_c)] = (np.random.random((amount_mutation_vals, ))*2.-1.)/10.

                    # mutation step add
                    idx_mutation_add = np.zeros((length, ), dtype=np.uint8)
                    idx_mutation_add[:int(length*self.mutation_add_percent)] = 1

                    idx_mutation_add_c = np.random.permutation(idx_mutation_add)==1
                    amount_mutation_add_vals = np.sum(idx_mutation_add)
                    a_c[np.random.permutation(idx_mutation_add_c)] += (np.random.random((amount_mutation_add_vals, ))*2.-1.)/10000.

                    # mutation step sub
                    idx_mutation_sub = np.zeros((length, ), dtype=np.uint8)
                    idx_mutation_sub[:int(length*self.mutation_sub_percent)] = 1

                    idx_mutation_sub_c = np.random.permutation(idx_mutation_sub)==1
                    amount_mutation_sub_vals = np.sum(idx_mutation_sub)
                    a_c[np.random.permutation(idx_mutation_sub_c)] -= (np.random.random((amount_mutation_sub_vals, ))*2.-1.)/10000.

            l_snn_new = l_snn_new_parents+l_snn_new_childs
            assert len(l_snn_new)==self.population_size
            self.l_snn = l_snn_new

            l_cecf_train = [snn.f_cecf(snn.calc_feed_forward(self.X_train), self.T_train)/self.T_train.shape[0]/self.T_train.shape[1]
                for snn in self.l_snn
            ]

            print("np.min(l_cecf_train): {}".format(np.min(l_cecf_train)))

            l_cecf_train_all.append(l_cecf_train)

        return l_cecf_train_all


# TODO: next step: take the already played games and learn the nn from it!
def load_Xs_Ts_from_tetris_data(d_data, using_pieces=3):
    rows = d_data['rows']
    cols = d_data['cols']
    amount_pieces = d_data['amount_pieces']
    l_amount_group_pieces = d_data['l_amount_group_pieces']
    l_group_pieces = d_data['l_group_pieces']

    arr_fields = d_data['arr_fields'][1::2]
    arr_pcs_idx_pos = d_data['arr_pcs_idx_pos']
    # could be used instead of the arr_fields, but lets see
    arr_max_height_per_column = d_data['arr_max_height_per_column']

    # calculate first all possible tuple combinations for the piece + direction + pos!
    l_pcs_group = []
    l_pcs_group_idx = []
    l_pcs_group_max_x = []
    l_pcs_idx_l_tpl_pcs_group_pos = []
    group_idx_acc = 0
    for pcs_idx, amount in enumerate(l_amount_group_pieces, 0):
        l = []
        l_idx = []
        l_max_x = []
        for j in range(0, amount):
            l_pos = l_group_pieces[group_idx_acc]
            max_x = max(l_pos[1::2])

            l.append(l_pos)
            l_idx.append(group_idx_acc)
            l_max_x.append(max_x)

            group_idx_acc += 1
        l_pcs_group.append(l)
        l_pcs_group_idx.append(l_idx)
        l_pcs_group_max_x.append(l_max_x)

        l_pcs_idx_l_tpl_pcs_group_pos.append([(pcs_idx, group_idx, pos) for group_idx, max_x in zip(l_idx, l_max_x) for pos in range(0, cols-max_x)])

    d_pcl_idx_d_tpl_pcs_group_pos_to_index = {
        pcs_idx: {t: i for i, t in enumerate(l_tpl_pcs_group_pos, 0)} for pcs_idx, l_tpl_pcs_group_pos in enumerate(l_pcs_idx_l_tpl_pcs_group_pos, 0)
    }

    s_used_pcs_group_pos = set([tuple(l) for l in arr_pcs_idx_pos.tolist()]) 
    s_all_pcs_group_pos = set([t for l in l_pcs_idx_l_tpl_pcs_group_pos for t in l])

    s_diff = s_all_pcs_group_pos - s_used_pcs_group_pos

    # convert the fields + other data into the learning X and T Matrices!
    Xs = [[] for _ in range(0, amount_pieces)]
    Ts = [[] for _ in range(0, amount_pieces)]

    arr_pcs_one_hot = np.diag(np.ones((amount_pieces, ), dtype=np.int8))

    x_len = rows*cols + using_pieces*amount_pieces
    l_t_len = [len(l) for l in l_pcs_idx_l_tpl_pcs_group_pos]
    for i in range(0, d_data['length']-using_pieces+1):
        arr_field = arr_fields[i]
        arr_pcs_idx_pos_part = arr_pcs_idx_pos[i:i+using_pieces]
        arr_pcs_idx, arr_group_idx, arr_pos = arr_pcs_idx_pos_part.T

        pcs_idx = arr_pcs_idx[0]
        X = Xs[pcs_idx]
        T = Ts[pcs_idx]

        # vector for x
        arr_x = np.zeros((x_len, ), dtype=np.int8)
        arr_t = np.zeros((l_t_len[pcs_idx], ), dtype=np.int8)

        arr_x[:rows*cols] = (arr_field.reshape((-1, ))!=0)+0
        arr_x[rows*cols:] = arr_pcs_one_hot[arr_pcs_idx].reshape((-1, ))

        tpl = tuple(arr_pcs_idx_pos_part[0].tolist())
        # l_tpl_pcs_group_pos = l_pcs_idx_l_tpl_pcs_group_pos[pcs_idx]
        d_tpl_pcs_group_pos = d_pcl_idx_d_tpl_pcs_group_pos_to_index[pcs_idx]
        t_idx = d_tpl_pcs_group_pos[tpl]

        arr_t[t_idx] = 1

        arr_x[arr_x==0] = -1

        X.append(arr_x)
        T.append(arr_t)
        # break

    l_len_Xs = [len(X) for X in Xs]
    # print("l_len_Xs: {}".format(l_len_Xs))


    Xs = [np.array(X, dtype=np.float) for X in Xs]
    Ts = [np.array(T, dtype=np.float) for T in Ts]

    return Xs, Ts


def load_Xs_Ts_full(using_pieces=3):
    l_suffix = ['{:03}_{}'.format(i, j) for i in range(101, 108) for j in range(1, 2)]

    file_name_template = 'tetris_game_data/data_fields_{suffix}.ttrsfields'
    data_file_path_template = PATH_ROOT_DIR+file_name_template

    print("suffix: {}".format(l_suffix[0]))
    data_file_path = data_file_path_template.format(suffix=l_suffix[0])
    d_data = parse_tetris_game_data(file_path=data_file_path)

    d_basic_data_info = {
        'rows': d_data['rows'],
        'cols': d_data['cols'],
        'amount_pieces': d_data['amount_pieces'],
        'l_amount_group_pieces': d_data['l_amount_group_pieces'],
        'l_group_pieces': d_data['l_group_pieces'],

        'l_group_piece_arr_pos': d_data['l_group_piece_arr_pos'],
        'l_group_piece_max_x': d_data['l_group_piece_max_x'],
        'l_pcs_idx_l_index_to_group_idx_pos': d_data['l_pcs_idx_l_index_to_group_idx_pos'],
    }

    Xs, Ts = load_Xs_Ts_from_tetris_data(d_data=d_data, using_pieces=3)

    l_Xs = [[X] for X in Xs]
    l_Ts = [[T] for T in Ts]

    for suffix in l_suffix[1:]:
        print("suffix: {}".format(suffix))
        data_file_path = data_file_path_template.format(suffix=suffix)
        d_data = parse_tetris_game_data(file_path=data_file_path)

        d_basic_data_info_new = {
            'rows': d_data['rows'],
            'cols': d_data['cols'],
            'amount_pieces': d_data['amount_pieces'],
            'l_amount_group_pieces': d_data['l_amount_group_pieces'],
            'l_group_pieces': d_data['l_group_pieces'],

            'l_group_piece_arr_pos': d_data['l_group_piece_arr_pos'],
            'l_group_piece_max_x': d_data['l_group_piece_max_x'],
            'l_pcs_idx_l_index_to_group_idx_pos': d_data['l_pcs_idx_l_index_to_group_idx_pos'],
        }

        assert d_basic_data_info==d_basic_data_info_new

        Xs, Ts = load_Xs_Ts_from_tetris_data(d_data=d_data)

        for l_X, l_T, X, T in zip(l_Xs, l_Ts, Xs, Ts):
            l_X.append(X)
            l_T.append(T)

    Xs_full = [np.vstack(l_X) for l_X in l_Xs]
    Ts_full = [np.vstack(l_T) for l_T in l_Ts]

    l_len_Xs_full = [len(X) for X in Xs_full]
    print("l_len_Xs_full: {}".format(l_len_Xs_full))

    return Xs_full, Ts_full, d_basic_data_info


if __name__ == "__main__":
    Xs_full, Ts_full, d_basic_data_info = load_Xs_Ts_full()

    ga = GeneticAlgorithmTetris(population_size=10, Xs=Xs_full, Ts=Ts_full)
