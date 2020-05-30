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

import base64
import json

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()
# import tempfile

import platform
print("platform.system(): {}".format(platform.system()))
os.system('xset r off')

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

class SimpleNeuralNetwork(Exception):
    def __init__(self, input_nodes=None, hidden_nodes=None, output_nodes=None, bws=None):
        if bws is not None:
            self.bws = deepcopy(bws)
            return

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.eta = 0.001
        self.learning_epochs = 100

        self.layers = [input_nodes]+hidden_nodes+[output_nodes]
        self.nl = self.layers

        self.bws = self.get_random_bws()
        # self.bws = self.get_random_bws_old()


    # def get_random_bws_old(self):
    #     return [(np.random.random((i2, i1+1))*2.-1.)/10. for i1, i2 in zip(self.layers[:-1], self.layers[1:])]


    def get_random_bws(self):
        return np.array([np.random.uniform(-1./np.sqrt(n), 1./np.sqrt(n), (m+1, n)) for m, n in zip(self.nl[:-1], self.nl[1:])])
    

    def __repr__(self):
        return f"SimpleNeuralNetwork(ni={self.input_nodes}, nh={self.hidden_nodes}, on={self.output_nodes}, "+ \
            f"eta={self.eta}, learning_epochs={self.learning_epochs})"


    def calc_output_argmax(self, input_vec):
        x = np.hstack((input_vec, [1]))
        for w in self.bws[:-1]:
            z = np.dot(w, x)
            x = self.f1(z)
            x = np.hstack((x, [1]))
        z = np.dot(self.bws[-1], x)
        y = self.f2(z)
        return np.argmax(y)


    def calc_grad_numeric(self, X, T, epsilon=0.0001, softmax=False):
        ws_grad = [np.empty(w.shape) for w in self.bws]

        for bw, bw_g in zip(self.bws, ws_grad):
            rows, cols = bw.shape
            for j in range(0, rows):
                for i in range(0, cols):
                    v = bw[j, i]

                    bw[j, i] = v+epsilon
                    Y = self.calc_Y(X, softmax=softmax)
                    cecf_plus = self.f_cecf(Y, T) # /T.shape[0]/T.shape[1]

                    bw[j, i] = v-epsilon
                    Y = self.calc_Y(X, softmax=softmax)
                    cecf_minus = self.f_cecf(Y, T) # /T.shape[0]/T.shape[1]

                    v_grad = (cecf_plus-cecf_minus) / 2. / epsilon
                    bw[j, i] = v
                    bw_g[j, i] = v_grad

        return ws_grad


    def calc_Y(self, X, softmax=False):
        ones = np.ones((X.shape[0], 1))

        # Y = X
        X = np.hstack((ones, X))
        for bw in self.bws[:-1]:
            Z = np.dot(X, bw)
            Y = self.f1(Z)
            X = np.hstack((ones, Y))
        Z = np.dot(X, self.bws[-1])
        Y = self.f2(Z)
        if softmax:
            Y = Y / np.sum(Y, axis=1).reshape((-1, 1))
        return Y


    def calc_feed_forward(self, X):
        ones = np.ones((X.shape[0], 1))
        Y = X
        for i, bw in zip(range(len(self.bws), 0, -1), self.bws[:-1]):
            Y = self.f1(np.hstack((ones, Y)).dot(bw)) # *self.hidden_multiplier**i)
        Y = self.f2(np.hstack((ones, Y)).dot(self.bws[-1]))
        return Y


    def calc_backprop(self, X, T, bws, softmax=False):
        Xs = []
        Ys = [X]
        ones = np.ones((X.shape[0], 1))
        Y = X
        for bw in bws[:-1]:
            A = np.hstack((ones, Y)).dot(bw)
            Xs.append(A)
            Y = self.f1(A)
            Ys.append(Y)
        A = np.hstack((ones, Y)).dot(bws[-1]); Xs.append(A)
        Y = self.f2(A)
        if softmax:
            Y = Y / np.sum(Y, axis=1).reshape((-1, 1))
        Ys.append(Y)

        d = (Y-T)
        bwds = np.array([np.zeros(bwsdi.shape) for bwsdi in bws])
        if softmax:
            bwds[-1] = np.hstack((ones, Ys[-2])).T.dot(d)
        else:
            bwds[-1] = np.hstack((ones, Ys[-2])).T.dot(d)

        length = len(bws)
        for i in range(2, length+1):
            d = d.dot(bws[-i+1][1:].T)*self.f1_d(Xs[-i])
            bwds[-i] = np.hstack((ones, Ys[-i-1])).T.dot(d)

        return np.array(bwds)


    def f_rmse(self, Y, T):
        return np.sqrt(np.mean(np.sum((Y-T)**2, axis=1)))
    def f_cecf(self, Y, T):
        return np.sum(np.sum(np.vectorize(lambda y, t: -np.log(y) if t==1. else -np.log(1-y))(Y, T), axis=1))


    # def f_cecf(self, Ys, Ts):
    #     diff = Ys-Ts
    #     return np.sum(diff**2)/diff.shape[0]/diff.shape[1]


    # def calc_logistic_direct(self, Ys, Ts):
    #     diff = Ys-Ts
    #     return np.sum(diff**2)/diff.shape[0]/diff.shape[1]


    # def calc_mse_direct(self, Xs, Ts):
    #     Ys = self.calc_Y(Xs)
    #     diff = Ys-Ts
    #     return np.sum(diff**2)/diff.shape[0]/diff.shape[1]


    def f1(self, x):
        return np.tanh(x)


    def f1_d(self, x):
        # return 1/(x**2+1)
        return 1 - np.tanh(x)**2


    def f2(self, x):
        return 1 / (1 + np.exp(-x))


    def f2_d(self, x):
        return self.f2(x)*(1-self.f2(x))


# do some testing for the backpropagation function!

input_nodes = 5
output_nodes = 3

snn = SimpleNeuralNetwork(input_nodes=input_nodes, hidden_nodes=[7, 6], output_nodes=output_nodes)

amount = 2000

X = np.random.randint(0, 2, (amount, input_nodes)).T

T = np.vstack(((X[0]^X[3]+X[4])%2, (X[1]^X[2])&(X[2]|X[3]), (X[3]|X[4])&X[0])).T
X = X.T

snn.ws = snn.get_random_bws()

X[X==0] = -1

X = X.astype(np.float)
T = T.astype(np.float)

X_part = X[:40]
T_part = T[:40]

bws_prev = deepcopy(snn.bws)

# bwds_grad_numeric_softmax = snn.calc_grad_numeric(X_part, T_part, softmax=True)
bwds_grad_numeric_nosoftmax = snn.calc_grad_numeric(X_part, T_part, softmax=False)

bws_now = deepcopy(snn.bws)
assert all([np.all(w_p==w_n) for w_p, w_n in zip(bws_prev, bws_now)])

# bwds_grad_true_softmax = snn.calc_backprop(X_part, T_part, snn.bws, softmax=True)
bwds_grad_true_nosoftmax = snn.calc_backprop(X_part, T_part, snn.bws, softmax=False)

bwd_n_soft = bwds_grad_numeric_softmax[-1]
bwd_n_nosoft = bwds_grad_numeric_nosoftmax[-1]

# bwd_t_soft = bwds_grad_true_softmax[-1]
# bwd_t_nosoft = bwds_grad_true_nosoftmax[-1]

# print("bwd_n_nosoft:\n{}".format(bwd_n_nosoft))
# print("bwd_t_nosoft:\n{}".format(bwd_t_nosoft))

# print("bwd_n_soft:\n{}".format(bwd_n_soft))
# print("bwd_t_soft:\n{}".format(bwd_t_soft))

assert np.sum([np.sum(np.abs(bwd_n-bwd_t)<0.001) for bwd_n, bwd_t in zip(bwds_grad_numeric_nosoftmax, bwds_grad_true_nosoftmax)])==np.sum([bw.shape[0]*bw.shape[1] for bw in snn.bws])


# test for a simple backpropagation example!
# add some noise to the X matrix!
X += np.random.uniform(-1./100, 1./100, X.shape)

split_n = int(X.shape[0]*0.7)
X_train = X[:split_n]
X_test = X[split_n:]
T_train = T[:split_n]
T_test = T[split_n:]

l_cecf_train = [snn.f_cecf(snn.calc_feed_forward(X_train), T_train)/T_train.shape[0]/T_train.shape[1]]

eta = 0.0001
for i in range(1, 2001):
    ws_grad = snn.calc_backprop(X_train, T_train, snn.bws)
    # ws_grad = snn.calc_grad_numeric(X_train, T_train)
    ws_new = [w-wg*eta for w, wg in zip(snn.bws, ws_grad)]
    snn.bws = ws_new
    Y_train = snn.calc_feed_forward(X_train)
    cecf_train = snn.f_cecf(Y_train, T_train)/T_train.shape[0]/T_train.shape[1]
    print("i: {:4}, cecf_train: {:06f}".format(i, cecf_train))
    if cecf_train >= l_cecf_train[-1]:
        l_cecf_train.append(cecf_train)
        print("Stop!! cecf_train >= l_cecf_train[-1]!")
        break
    l_cecf_train.append(cecf_train)

print("l_cecf_train: {}".format(l_cecf_train))

Y_test = snn.calc_feed_forward(X_test)
cecf_test = snn.f_cecf(Y_test, T_test)/T_test.shape[0]/T_test.shape[1]

print("cecf_test: {}".format(cecf_test))

sys.exit()


class GeneticAlgorithm(Exception):
    pass


class TetrisGeneticAlgorithm(Exception):
    def __init__(self):
        self.field_columns = 10
        self.field_rows = 20

        self.l_piece_name = ['Z', 'S', 'O', 'T', 'L', 'J', 'I']
        self.l_piece_direction = ['W', 'N', 'E', 'S']
        self.piece_name_str = ''.join(sorted(self.l_piece_name))

        self.l_clear_lines = []
        self.l_pieces_used = []

        self.POPULATION_SIZE = 100
        self.crossover_percent = 0.40
        self.mutation_percent = 0.10
        self.mutation_add_percent = 0.08
        self.mutation_sub_percent = 0.08

        self.using_parent_amount_percent = 0.6
        self.using_parent_amount = int(self.POPULATION_SIZE*self.using_parent_amount_percent)

        self.hidden_layers = [100]

        self.change_hash_at_module = 1

        self.generation_nr_acc = 0
        self.generation_nr = 0

        self.max_piece_nr = self.field_columns*200
        self.piece_nr = 0
        self.max_clear_lines = 0
        self.max_pieces = 0

        self.field_add_rows = 4
        self.field = np.zeros((self.field_rows+self.field_add_rows, self.field_columns), dtype=np.uint8)
        self.field_y = self.field.shape[0]
        self.field_x = self.field.shape[1]
        self.need_field_reset = False

        print("self.field.shape: {}".format(self.field.shape))

        d_pieces = {
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

        d_piece_name_to_piece_idx = {piece_name: d['idx'] for piece_name, d in d_pieces.items()}
        d_pieces_show_pos = {
            'Z': [0, 1],
            'S': [1, 1],
            'O': [0, 1],
            'T': [1, 1],
            'L': [1, 1],
            'J': [0, 1],
            'I': [1, 1],
        }

        d_piece_name_counter = {piece_name: 0 for piece_name in self.l_piece_name}
        self.clear_lines = 0

        # the amount of boxes in fields + 7 nodes for currecnt piece + 7 nodes for next piece
        input_nodes = self.field_columns*self.field_rows+7+7
        d_piece_name_to_output_node_to_direction_pos = {}
        for piece_name, d in d_pieces.items():
            d_output_node_to_direction_pos = {}
            output_node_nr = 0
            for direction, (_, idx_ranges) in d['pos'].items():
                for i in range(idx_ranges[0], idx_ranges[1]):
                    d_output_node_to_direction_pos[output_node_nr] = (direction, i)
                    output_node_nr += 1
            d_piece_name_to_output_node_to_direction_pos[piece_name] = d_output_node_to_direction_pos
        output_nodes = len(d_output_node_to_direction_pos)
        
        # print("input_nodes: {}".format(input_nodes))
        # print("output_nodes: {}".format(output_nodes))
        # input('ENTER...')

        def generate_new_d_simple_nn(d_simple_nn=None):
            if d_simple_nn is None:
                d_simple_nn = {
                    piece_name: SimpleNeuralNetwork(
                        input_nodes=input_nodes,
                        hidden_nodes=self.hidden_layers,
                        output_nodes=output_nodes,
                    ) for piece_name, d_output_node_to_direction_pos in d_piece_name_to_output_node_to_direction_pos.items()
                }
            else:
                d_simple_nn = {
                    piece_name: SimpleNeuralNetwork(
                        bws=simple_nn.bws,
                    ) for piece_name, simple_nn in d_simple_nn.items()
                }

            # ws_orig = d_simple_nn['J'].bws
            # for piece_name in self.l_piece_name:
            #     if piece_name=='J':
            #         continue
            #     d_simple_nn[piece_name].bws = ws_orig
            
            return d_simple_nn

        # TODO: write the learning part in a separate function or make a separate class for the learning only!
        self.l_d_simple_nn = [generate_new_d_simple_nn() for _ in range(0, self.POPULATION_SIZE)]

        self.simple_nn_idx = 0
        self.using_d_piece_name_simple_nn = self.l_d_simple_nn[0]
        self.l_statistics_of_simple_nn = [None for _ in range(0, self.POPULATION_SIZE)]

        def get_input_vector():
            input_vector = np.zeros((input_nodes, ))-1
            input_vector[:-14][self.field[-self.field_rows:].reshape((-1, ))!=0] = 1
            input_vector[-14:-7][d_piece_name_to_piece_idx[self.piece_next_name]-1] = 1
            input_vector[-7:][d_piece_name_to_piece_idx[self.piece_name]-1] = 1
            return input_vector


        self.piece_next_name = random.choice(self.l_piece_name)
        self.piece_name = random.choice(self.l_piece_name)

        def define_next_random_piece():
            piece_name = random.choice(self.l_piece_name)
            # piece_name = self.l_piece_name[(self.hash_val+self.piece_nr*123456789)%len(self.l_piece_name)]

            d_piece = d_pieces[piece_name]

            self.piece_next_name = piece_name
            self.piece_next_idx = d_piece['idx']
            
            self.piece_next_posi_y = 1


        def define_next_piece():
            self.piece_name = self.piece_next_name
            
            d_piece_name_counter[self.piece_name] += 1

            self.piece_idx = self.piece_next_idx
            self.piece_posi_y = self.piece_next_posi_y
            
            simple_nn = self.using_d_piece_name_simple_nn[self.piece_name]
            input_vector = get_input_vector()
            argmax = simple_nn.calc_output_argmax(input_vector)

            d_output_node_to_direction_pos = d_piece_name_to_output_node_to_direction_pos[self.piece_name]
            piece_direction, piece_posi_x = d_output_node_to_direction_pos[argmax]

            self.piece_direction = piece_direction
            self.piece_posi_x = piece_posi_x
            self.piece_positions = d_pieces[self.piece_name]['pos'][self.piece_direction][0]

            define_next_random_piece()


        def show_next_piece():
            y_s, x_s = d_pieces_show_pos[self.piece_next_name]
            self.canv_next.create_rectangle(0, 0, self.canv_next_w, self.canv_next_h, fill=self.canv_next_bg, width=0)
            for yi_, xi_ in self.piece_next_show_positions:
                y = self.box_h*(y_s+yi_)
                x = self.box_w*(x_s+xi_)
                self.canv_next.create_rectangle(x, y, x+self.box_w, y+self.box_h, fill=self.piece_next_color, width=0)


        def show_start_piece_position():
            for yi_, xi_ in self.piece_positions:
                self.field[self.piece_posi_y+yi_, self.piece_posi_x+xi_] = self.piece_idx
                y = self.pos_y_start+self.box_h*(self.piece_posi_y+yi_)
                x = self.pos_x_start+self.box_w*(self.piece_posi_x+xi_)
                self.canvas.create_rectangle(x, y, x+self.box_w, y+self.box_h, fill=self.piece_color, width=0)


        def reset_field():
            self.need_field_reset = False
            self.field[:] = 0
            # set some garbage lines at the bottom last 5 lines e.g.
            garbage_lines = 5
            self.field[-garbage_lines:] = 1
            ys = np.arange(self.field_rows+self.field_add_rows-1, self.field_rows+self.field_add_rows-1-garbage_lines, -1)
            xs = np.random.randint(0, self.field_columns, (garbage_lines, ))
            self.field[(ys, xs)] = 0
            # print("self.field:\n{}".format(self.field))
            # print("ys: {}".format(ys))
            # print("xs: {}".format(xs))
            # input('ENTER..')

            for k in d_piece_name_counter:
                d_piece_name_counter[k] = 0
            
            self.clear_lines = 0
            self.piece_nr = -1
            define_next_random_piece()
            self.piece_nr = 0


        def do_move_piece_down_instant():
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


        # TODO: write a separate function for the running the game and a seperate function for calculting the new offsprings
        def get_new_statistics():
            t = (self.simple_nn_idx, self.piece_nr, self.clear_lines)
            self.l_statistics_of_simple_nn[self.simple_nn_idx] = t
            if self.max_clear_lines<self.clear_lines:
                self.max_clear_lines = self.clear_lines
            if self.max_pieces<self.piece_nr:
                self.max_pieces = self.piece_nr
            
            self.calculate_new_hash_val()

            self.simple_nn_idx = (self.simple_nn_idx+1)%len(self.l_d_simple_nn)
            if self.simple_nn_idx==0:
                if self.generation_nr_acc%self.change_hash_at_module == 0:
                    self.calculate_new_hash_val()
                    self.generation_nr = 0

                l_sorted = sorted(self.l_statistics_of_simple_nn, key=lambda x: (x[2], x[1]), reverse=True)
                l_sorted_filtered = [(i2, i1) for _, i1, i2 in l_sorted]
                print("self.generation_nr_acc: {}".format(self.generation_nr_acc))
                print("self.generation_nr: {}".format(self.generation_nr))

                self.l_clear_lines.append(l_sorted_filtered[0])
                self.l_pieces_used.append(l_sorted_filtered[1])

                print("self.max_clear_lines: {}, self.max_pieces: {}".format(self.max_clear_lines, self.max_pieces))
                print("self.field_rows: {}, self.field_columns: {}".format(self.field_rows, self.field_columns))
                print("self.l_piece_name: {}".format(self.l_piece_name))
                elements_in_row = 20
                l_sorted_filtered_str = [':'.join(['{:2},{:2}'.format(i1, i2) for i1, i2 in l_sorted_filtered[i:i+elements_in_row]]) for i in range(0, len(l_sorted_filtered), elements_in_row)]
                print("l_sorted_filtered_str:\n{}".format(':\n'.join(l_sorted_filtered_str)))

                l_idx = [idx for idx, amount_pieces, clear_lines in l_sorted]
                l_d_simple_nn_new_parents = [self.l_d_simple_nn[i] for i in l_idx[:self.using_parent_amount]]
                amount_parents = len(l_d_simple_nn_new_parents)
                amount_childs = self.POPULATION_SIZE-amount_parents
                
                using_idx_delta = np.random.randint(1, amount_parents, (amount_childs, ))
                using_idx_delta[0] = np.random.randint(0, amount_parents)
                using_idx = np.cumsum(using_idx_delta)%amount_parents
                l_d_simple_nn_new_childs = [generate_new_d_simple_nn(d_simple_nn=l_d_simple_nn_new_parents[i]) for i in using_idx]
                # l_d_simple_nn_new_childs = [generate_new_d_simple_nn() for _ in range(0, self.POPULATION_SIZE-len(l_d_simple_nn_new_parents))]

                for i1, i2, d_snn_child in zip(using_idx, np.roll(using_idx, 1), l_d_simple_nn_new_childs):
                    d_snn1, d_snn2 = l_d_simple_nn_new_parents[i1], l_d_simple_nn_new_parents[i2]
                    for piece_name in self.l_piece_name:
                        snn1 = d_snn1[piece_name]
                        snn2 = d_snn2[piece_name]
                        snn_child = d_snn_child[piece_name]

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

                l_d_simple_nn_new = l_d_simple_nn_new_parents+l_d_simple_nn_new_childs
                assert len(l_d_simple_nn_new)==self.POPULATION_SIZE
                self.l_d_simple_nn = l_d_simple_nn_new

                if (self.generation_nr_acc)%500==0:
                    obj = {
                        'field_columns': self.field_columns,
                        'field_rows': self.field_rows,
                        'crossover_percent': self.crossover_percent,
                        'mutation_percent': self.mutation_percent,
                        'mutation_add_percent': self.mutation_add_percent,
                        'mutation_sub_percent': self.mutation_sub_percent,
                        'l_d_simple_nn_best': self.l_d_simple_nn[:5],
                        'l_clear_lines': self.l_clear_lines,
                        'l_pieces_used': self.l_pieces_used,
                    }

                    directory = '/run/user/1000/genetic_algorithm/rows_{}_cols_{}/using_pieces_{}/'.format(
                        self.field_rows, self.field_columns, self.piece_name_str)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    t = tempfile.NamedTemporaryFile(suffix='_genetic_alg_tetris_pop_size_{}_gen_{}_clear_rows_{}_pcs_{}.pkl'.format(
                            self.POPULATION_SIZE,
                            self.generation_nr_acc,
                            self.clear_lines,
                            self.piece_nr,
                        ),
                        delete=False,
                        dir=directory,
                    )
                    print("Saving file: t.name: {}".format(t.name))

                    dill.dump(obj, t)

                    t.close()

                self.generation_nr_acc += 1
                self.generation_nr += 1

            self.using_d_piece_name_simple_nn = self.l_d_simple_nn[self.simple_nn_idx]


        self.calculate_new_hash_val()
        reset_field()

        def main_game_loop():  
            while True:
                define_next_piece()
                is_piece_moved = do_move_piece_down_instant()

                if not is_piece_moved and self.need_field_reset or self.piece_nr>self.max_piece_nr:
                    get_new_statistics()
                    reset_field()

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
        
        main_game_loop()


    def calculate_new_hash_val(self):
        self.str_txt = '0123456789'
        self.hash_val = hash(''.join([self.str_txt[i] for i in np.random.randint(0, len(self.str_txt), (10, ))]))


    def client_exit_menu_btn(self):
        print("Pressed the EXIT button or X!")
        self.master.destroy()


if __name__ == "__main__":
    TetrisGeneticAlgorithm()
