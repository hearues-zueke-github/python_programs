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

import platform
print("platform.system(): {}".format(platform.system()))
os.system('xset r off')

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

class SimpleNeuralNetwork(Exception):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        layers = [input_nodes]+hidden_nodes+[output_nodes]
        self.ws = [np.random.random((i2, i1))*2.-1. for i1, i2 in zip(layers[:-1], layers[1:])]


    def calc_output_argmax(self, input_vec):
        x = input_vec
        for w in self.ws[:-1]:
            z = np.dot(w, x)
            x = np.tanh(z)
        z = np.dot(self.ws[-1], x)
        y = self.sig(z)
        return np.argmax(y)


    def sig(self, x):
        return 1 / (1 + np.exp(-x))


class GeneticAlgorithm(Exception):
    pass


class MainWindow(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master, class_='MainWindow')
        self.master = master

        self.width = 620
        self.height = 580

        w = self.master.winfo_screenwidth()
        h = self.master.winfo_screenheight()
        size = (self.width, self.height)

        x = w/2 - size[0]/2
        y = h/2 - size[1]/2
        self.master.geometry("%dx%d+%d+%d" % (size + (x, y)))

        self.init_window()


    def init_window(self):
        self.pixel = tk.PhotoImage(master=self, name='photo_pixel_1x1', width=1, height=1)
        self.master.protocol("WM_DELETE_WINDOW", self.client_exit_menu_btn)

        self.master.title("Simple Tetris Game")

        # allowing the widget to take the full space of the root window
        self.pack(fill=tk.BOTH, expand=1)

        self.menu = tk.Menu(self.master)
        self.master.config(menu=self.menu)

        self.file = tk.Menu(self.menu, tearoff=False)
        self.file.add_command(label="Exit", command=self.client_exit_menu_btn)

        self.menu.add_cascade(label="File", menu=self.file)

        self.edit = tk.Menu(self.menu, tearoff=False)
        self.edit.add_command(label="Undo")

        self.menu.add_cascade(label="Edit", menu=self.edit)

        self.box_w = 20
        self.box_h = self.box_w
        self.box_x = 0
        self.box_y = 0

        self.canv_column = 8
        self.canv_rows = 22

        self.canv_x = 10
        self.canv_y = 20
        self.canv_w = self.box_w*self.canv_column
        self.canv_h = self.box_h*self.canv_rows
        self.canv_bg = '#000000'
        
        self.canv_outer_border = 1
        self.canv_outer_bg = '#FFFFFF'
        self.canv_outer = tk.Canvas(self, background=self.canv_outer_bg, borderwidth=0, highlightthickness=0, width=self.canv_w+self.canv_outer_border*2, height=self.canv_h+self.canv_outer_border*2)
        self.canv_outer.place(x=self.canv_x-self.canv_outer_border, y=self.canv_y-self.canv_outer_border)
        
        self.canvas = tk.Canvas(self.canv_outer, background=self.canv_bg, borderwidth=0, highlightthickness=0, width=self.canv_w, height=self.canv_h)
        self.canvas.place(x=self.canv_outer_border, y=self.canv_outer_border)

        self.time_begin = time.time()

        self.lbl_time_text = tk.StringVar()
        self.lbl_time_text.set('time: {}'.format(0))

        self.lbl_txt_next = tk.Label(self, text='Next Piece:', bg='#A0A0A0')
        self.lbl_txt_next.place(x=270, y=20)

        self.canv_next_x = 270
        self.canv_next_y = 50
        self.canv_next_w = self.box_w*4
        self.canv_next_h = self.box_h*2
        self.canv_next_bg = '#000000'
        self.canv_next = tk.Canvas(self, background=self.canv_next_bg, borderwidth=0, highlightthickness=0, width=self.canv_next_w, height=self.canv_next_h)
        self.canv_next.place(x=self.canv_next_x, y=self.canv_next_y)

        y_start_lbl = 100

        self.lbl_time = tk.Label(self, textvariable=self.lbl_time_text, bg='#A0A0A0')
        self.lbl_time.place(x=270, y=y_start_lbl+30*0)
        
        self.lbl_tick_text = tk.StringVar()
        self.lbl_tick_text.set('tick')

        self.lbl_tick = tk.Label(self, textvariable=self.lbl_tick_text, bg='#A0A0A0')
        self.lbl_tick.place(x=270, y=y_start_lbl+30*1)

        self.lbl_fps_text = tk.StringVar()
        self.lbl_fps_text.set('fps')

        self.lbl_fps = tk.Label(self, textvariable=self.lbl_fps_text, bg='#A0A0A0')
        self.lbl_fps.place(x=270, y=y_start_lbl+30*2)

        l_piece_name = ['Z', 'S', 'O', 'T', 'L', 'J', 'I']
        l_piece_direction = ['W', 'N', 'E', 'S']

        y_start = y_start_lbl+30*3
        y_diff = 25
        x1 = 270
        x2 = 320
        self.d_strvar_pcs_counter = {}
        self.d_lbl_pcs = {}
        self.d_lbl_pcs_counter = {}
        for i_num, piece_name in enumerate(l_piece_name, 0):
            self.d_strvar_pcs_counter[piece_name] = tk.StringVar()
            self.d_strvar_pcs_counter[piece_name].set('0')

            self.d_lbl_pcs[piece_name] = tk.Label(self, text='pcs {}: '.format(piece_name), bg='#B090C0')
            self.d_lbl_pcs[piece_name].place(x=x1, y=y_start+y_diff*i_num)
            
            self.d_lbl_pcs_counter[piece_name] = tk.Label(self, textvariable=self.d_strvar_pcs_counter[piece_name], bg='#A0A0A0')
            self.d_lbl_pcs_counter[piece_name].place(x=x2, y=y_start+y_diff*i_num)


        y_start_2 = y_start+y_diff*7
        
        self.lbl_clear_lines_txt = tk.StringVar()
        self.lbl_clear_lines_txt.set('fps')

        self.lbl_clear_lines = tk.Label(self, text='clear lines:', bg='#A0A0A0')
        self.lbl_clear_lines.place(x=270, y=y_start_2+30*0)

        self.lbl_clear_lines_counter = tk.Label(self, textvariable=self.lbl_clear_lines_txt, bg='#A0A0A0')
        self.lbl_clear_lines_counter.place(x=340, y=y_start_2+30*0)


        self.lock = Lock()

        self.d_keycode_keyname = {
            111: 'Up',
            114: 'Right',
            116: 'Down',
            113: 'Left',
        }
        self.d_keyname_keycode = {keyname: keycode for keycode, keyname in self.d_keycode_keyname.items()}

        self.d_keycode_pressed = {keycode: [False] for keycode in self.d_keycode_keyname.keys()}
        self.d_keyname_pressed = {keyname: self.d_keycode_pressed[keycode] for keycode, keyname in self.d_keycode_keyname.items()}

        self.crossover_percent = 0.40
        self.mutation_percent = 0.05

        self.tick = 0
        self.piece_nr = 0
        self.main_loop_fps_needed = 300
        # self.main_loop_fps_needed = 60
        self.main_loop_ms_pause = 1000//self.main_loop_fps_needed

        self.field_add_rows = 4
        self.field = np.zeros((self.canv_rows+self.field_add_rows, self.canv_column), dtype=np.uint8)
        self.field_y = self.field.shape[0]
        self.field_x = self.field.shape[1]
        self.need_field_reset = False

        self.pos_y_start = -self.box_w*self.field_add_rows
        self.pos_x_start = 0

        print("self.field.shape: {}".format(self.field.shape))

        d_pieces = {
            'Z': {
                'pos': {
                    'S': ([[0, -1], [0, 0], [1, 0], [1, 1]], [1, self.field_x-1]),
                    'E': ([[-1, 1], [0, 1], [0, 0], [1, 0]], [0, self.field_x-1]),
                    'N': ([[-1, -1], [-1, 0], [0, 0], [0, 1]], [1, self.field_x-1]),
                    'W': ([[-1, 0], [0, 0], [0, -1], [1, -1]], [1, self.field_x]),
                },
                'color': '#000080',
                'idx': 1
            },
            'S': {
                'pos': {
                    'S': ([[0, -1], [0, 0], [-1, 0], [-1, 1]], [1, self.field_x-1]),
                    'E': ([[-1, 0], [0, 0], [0, 1], [1, 1]], [0, self.field_x-1]),
                    'N': ([[1, -1], [1, 0], [0, 0], [0, 1]], [1, self.field_x-1]),
                    'W': ([[-1, -1], [0, -1], [0, 0], [1, 0]], [1, self.field_x]),
                },
                'color': '#008000',
                'idx': 2
            },
            'O': {
                'pos': {
                    'S': ([[0, 0], [0, 1], [1, 0], [1, 1]], [0, self.field_x-1]),
                    'E': ([[0, 0], [0, 1], [1, 0], [1, 1]], [0, self.field_x-1]),
                    'N': ([[0, 0], [0, 1], [1, 0], [1, 1]], [0, self.field_x-1]),
                    'W': ([[0, 0], [0, 1], [1, 0], [1, 1]], [0, self.field_x-1]),
                },
                'color': '#808000',
                'idx': 3
            },
            'T': {
                'pos': {
                    'S': ([[0, 0], [0, 1], [0, -1], [-1, 0]], [1, self.field_x-1]),
                    'E': ([[0, 0], [1, 0], [-1, 0], [0, -1]], [1, self.field_x]),
                    'N': ([[0, 0], [0, 1], [0, -1], [1, 0]], [1, self.field_x-1]),
                    'W': ([[0, 0], [1, 0], [-1, 0], [0, 1]], [0, self.field_x-1]),
                },
                'color': '#800080',
                'idx': 4
            },
            'L': {
                'pos': {
                    'S': ([[0, 0], [0, -1], [0, 1], [-1, 1]], [1, self.field_x-1]),
                    'E': ([[0, 0], [-1, 0], [1, 0], [-1, -1]], [1, self.field_x]),
                    'N': ([[0, 0], [0, -1], [0, 1], [1, -1]], [1, self.field_x-1]),
                    'W': ([[0, 0], [-1, 0], [1, 0], [1, 1]], [0, self.field_x-1]),
                },
                'color': '#800000',
                'idx': 5
            },
            'J': {
                'pos': {
                    'S': ([[0, 0], [0, -1], [0, 1], [1, 1]], [1, self.field_x-1]),
                    'E': ([[0, 0], [-1, 0], [1, 0], [-1, 1]], [0, self.field_x-1]),
                    'N': ([[0, 0], [0, -1], [0, 1], [-1, -1]], [1, self.field_x-1]),
                    'W': ([[0, 0], [-1, 0], [1, 0], [1, -1]], [1, self.field_x]),
                },
                'color': '#008080',
                'idx': 6
            },
            'I': {
                'pos': {
                    'S': ([[-1, 0], [-1, -1], [-1, 1], [-1, 2]], [1, self.field_x-2]),
                    'E': ([[0, 1], [1, 1], [2, 1], [-1, 1]], [-1, self.field_x-1]),
                    'N': ([[0, 0], [0, -1], [0, 1], [0, 2]], [1, self.field_x-2]),
                    'W': ([[0, 0], [1, 0], [2, 0], [-1, 0]], [0, self.field_x]),
                },
                'color': '#808080',
                'idx': 7
            },
        }
        # globals()['d_pieces'] = d_pieces
        # import pdb
        # pdb.set_trace()

        d_piece_idx_to_color = {d['idx']: d['color'] for d in d_pieces.values()}
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

        d_piece_name_counter = {piece_name: 0 for piece_name in l_piece_name}
        self.clear_lines = 0

        # def define_random_piece():
        #     piece_name = random.choice(l_piece_name)
        #     piece_direction = random.choice(l_piece_direction)
        #     d_piece = d_pieces[piece_name]

        #     d_piece_name_counter[piece_name] += 1
        #     self.d_strvar_pcs_counter[piece_name].set('{}'.format(d_piece_name_counter[piece_name]))
            
        #     self.piece_color = d_piece['color']
        #     self.piece_positions, self.piece_x_range = d_piece['pos'][piece_direction]
        #     self.piece_idx = d_piece['idx']

        #     self.piece_posi_x = random.randrange(self.piece_x_range[0], self.piece_x_range[1])
        #     self.piece_posi_y = 1

        # self.l_piece_list = [
        #     ('I', 'W', 0), ('I', 'W', 1), ('I', 'W', 2),
        #     ('I', 'W', 3),
        #     ('I', 'W', 0), ('I', 'W', 1), ('I', 'W', 2),
        #     ('I', 'W', 0), ('I', 'W', 1), ('I', 'W', 2),
        #     ('I', 'W', 3),
        #     ('I', 'W', 3),
        # ]

        # self.l_piece_list = [
        #     ('T', 'W', 0), ('T', 'W', 2), ('T', 'W', 4), ('T', 'W', 6), ('T', 'W', 8),
        #     ('T', 'E', 1), ('T', 'E', 3), ('T', 'E', 5), ('T', 'E', 7), ('T', 'E', 9),
        # ]

        # self.l_piece_list = [
        #     'T', 'T', 'T', 'T', 'T', 'I'
        # ]

        # define a super set of all possible positions and directions of the piece!

        # the amount of boxes in fields + 7 nodes for currecnt piece + 7 nodes for next piece
        input_nodes = self.canv_column*self.canv_rows+7+7
        d_piece_name_to_output_node_to_direction_pos = {}
        for piece_name, d in d_pieces.items():
            d_output_node_to_direction_pos = {}
            output_node_nr = 0
            for direction, (_, idx_ranges) in d['pos'].items():
                for i in range(idx_ranges[0], idx_ranges[1]):
                    d_output_node_to_direction_pos[output_node_nr] = (direction, i)
                    output_node_nr += 1
            d_piece_name_to_output_node_to_direction_pos[piece_name] = d_output_node_to_direction_pos

        hidden_layers = [100]

        def generate_new_d_simple_nn():
            return {
                piece_name: SimpleNeuralNetwork(
                    input_nodes=input_nodes,
                    hidden_nodes=hidden_layers,
                    output_nodes=len(d_output_node_to_direction_pos),
                ) for piece_name, d_output_node_to_direction_pos in d_piece_name_to_output_node_to_direction_pos.items()
            }

        # TODO: write the learning part in a separate function or make a separate class for the learning only!
        self.POPULATION_SIZE = 50
        self.l_d_simple_nn = [generate_new_d_simple_nn() for _ in range(0, self.POPULATION_SIZE)]

        self.simple_nn_idx = 0
        self.using_d_piece_name_simple_nn = self.l_d_simple_nn[0]
        self.l_statistics_of_simple_nn = [None for _ in range(0, self.POPULATION_SIZE)]

        def get_input_vector():
            input_vector = np.zeros((input_nodes, ))-1
            input_vector[:-14][self.field[-self.canv_rows:].reshape((-1, ))!=0] = 1
            input_vector[-14:-7][d_piece_name_to_piece_idx[self.piece_next_name]-1] = 1
            input_vector[-7:][d_piece_name_to_piece_idx[self.piece_name]-1] = 1
            return input_vector


        self.piece_next_name = random.choice(l_piece_name)
        self.piece_name = random.choice(l_piece_name)

        # import pdb
        # pdb.set_trace()

        def define_next_random_piece():
            piece_name = l_piece_name[(self.hash_val+self.piece_nr*123456789)%len(l_piece_name)]
            # piece_name = self.l_piece_list.pop(0)
            # self.l_piece_list.append(piece_name)

            # piece_name, piece_direction, piece_posi_x = t

            # piece_name = random.choice(l_piece_name)

            # piece_direction = random.choice(l_piece_direction)

            d_piece = d_pieces[piece_name]

            # d_piece_name_counter[piece_name] += 1
            # self.d_strvar_pcs_counter[piece_name].set('{}'.format(d_piece_name_counter[piece_name]))
            
            self.piece_next_name = piece_name
            self.piece_next_color = d_piece['color']
            # self.piece_next_direction = piece_direction
            # self.piece_next_positions, self.piece_next_x_range = d_piece['pos'][piece_direction]
            self.piece_next_idx = d_piece['idx']
            
            self.piece_next_show_positions, _ = d_piece['pos']['S']

            # self.piece_next_posi_x = piece_posi_x
            # self.piece_next_posi_x = random.randrange(self.piece_next_x_range[0], self.piece_next_x_range[1])
            self.piece_next_posi_y = 1


        def define_next_piece():
            self.piece_name = self.piece_next_name
            
            d_piece_name_counter[self.piece_name] += 1
            self.d_strvar_pcs_counter[self.piece_name].set('{}'.format(d_piece_name_counter[self.piece_name]))
            
            self.piece_color = self.piece_next_color
            self.piece_idx = self.piece_next_idx
            self.piece_posi_y = self.piece_next_posi_y
            
            simple_nn = self.using_d_piece_name_simple_nn[self.piece_name]
            # simple_nn = self.d_piece_name_to_simple_nn[self.piece_name]
            input_vector = get_input_vector()
            argmax = simple_nn.calc_output_argmax(input_vector)

            d_output_node_to_direction_pos = d_piece_name_to_output_node_to_direction_pos[self.piece_name]
            piece_direction, piece_posi_x = d_output_node_to_direction_pos[argmax]

            self.piece_direction = piece_direction
            self.piece_posi_x = piece_posi_x
            self.piece_positions = d_pieces[self.piece_name]['pos'][self.piece_direction][0]

            # self.piece_direction = self.piece_next_direction
            # self.piece_posi_x = self.piece_next_posi_x

            define_next_random_piece()
            # show_next_piece()


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
            for k in d_piece_name_counter:
                d_piece_name_counter[k] = 0
                self.d_strvar_pcs_counter[k].set('0')
            # self.canvas.create_rectangle(0, 0, self.canv_w, self.canv_h, fill=self.canv_bg, width=0)
            
            self.piece_nr = 0

            self.clear_lines = 0
            self.lbl_clear_lines_txt.set('{}'.format(self.clear_lines))

            define_next_random_piece()
            # define_next_piece()
            # show_start_piece_position()

            # define_next_piece()
            # show_start_piece_position()


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
            #     y = self.pos_y_start+self.box_h*yi
            #     x = self.pos_x_start+self.box_w*xi
            #     self.canvas.create_rectangle(x, y, x+self.box_w, y+self.box_h, fill=self.canv_bg, width=0)
               
            for yi, xi in s_next_pos:
                self.field[yi, xi] = self.piece_idx
            #     y = self.pos_y_start+self.box_h*yi
            #     x = self.pos_x_start+self.box_w*xi
            #     self.canvas.create_rectangle(x, y, x+self.box_w, y+self.box_h, fill=self.piece_color, width=0)

            self.piece_posi_y += i-1

            return True


        def get_new_statistics():
            t = (self.simple_nn_idx, self.piece_nr, self.clear_lines)
            self.l_statistics_of_simple_nn[self.simple_nn_idx] = t
            
            # print("self.simple_nn_idx: {}".format(self.simple_nn_idx))
            self.simple_nn_idx = (self.simple_nn_idx+1)%len(self.l_d_simple_nn)
            if self.simple_nn_idx==0:
                l_sorted = sorted(self.l_statistics_of_simple_nn, key=lambda x: (x[2], x[1]), reverse=True)
                print("l_sorted:\n{}".format(l_sorted))

                # import pdb
                # pdb.set_trace()

                l_idx = [idx for idx, amount_pieces, clear_lines in l_sorted]
                l_d_simple_nn_new_parents = [self.l_d_simple_nn[i] for i in l_idx[:self.POPULATION_SIZE//2]]
                l_d_simple_nn_new_child = [generate_new_d_simple_nn() for _ in range(0, self.POPULATION_SIZE//2)]
                for d_snn1, d_snn2, d_snn_child in zip(l_d_simple_nn_new_parents[:-1], l_d_simple_nn_new_parents[1:], l_d_simple_nn_new_child):
                    for piece_name in l_piece_name:
                        snn1 = d_snn1[piece_name]
                        snn2 = d_snn2[piece_name]
                        snn_child = d_snn_child[piece_name]

                        for w1, w2, w_c in zip(snn1.ws, snn2.ws, snn_child.ws):
                            shape = w1.shape
                            a1 = w1.reshape((-1, ))
                            a2 = w2.reshape((-1, ))
                            a_c = w_c.reshape((-1, ))

                            length = a1.shape[0]

                            # crossover step
                            idx_crossover = np.zeros((length, ), dtype=np.uint8)
                            idx_crossover[:int(length*self.crossover_percent)] = 1

                            a_c[:] = a1

                            idx_crossover_1 = np.random.permutation(idx_crossover)==True
                            # idx_crossover_2 = np.random.permutation(idx_crossover)==True
                            a_c[idx_crossover_1] = a1[idx_crossover_1]
                            # a_c[idx_crossover_2] = a2[idx_crossover_2]

                            # mutation step
                            idx_mutation = np.zeros((length, ), dtype=np.uint8)
                            idx_mutation[:int(length*self.mutation_percent)] = 1

                            idx_mutation_c = np.random.permutation(idx_mutation)==True
                            amount_mutation_vals = np.sum(idx_mutation)
                            a_c[np.random.permutation(idx_mutation_c)] = np.random.random((amount_mutation_vals, ))*2.-1.

                l_d_simple_nn_new = l_d_simple_nn_new_parents+l_d_simple_nn_new_child
                assert len(l_d_simple_nn_new)==self.POPULATION_SIZE
                self.l_d_simple_nn = l_d_simple_nn_new

            self.calculate_new_hash_val()

            self.using_d_piece_name_simple_nn = self.l_d_simple_nn[self.simple_nn_idx]


        self.calculate_new_hash_val()
        reset_field()

        def main_game_loop():  
            # is_piece_moved = do_move_piece_down_instant()
            # # is_piece_moved = do_move_piece_down()

            # if not is_piece_moved:
            #     if self.need_field_reset:
            #         reset_field()
            #     else:
            #         define_next_piece()
            #         show_start_piece_position()

            define_next_piece()
            # show_start_piece_position()
            is_piece_moved = do_move_piece_down_instant()

            if not is_piece_moved:
                if self.need_field_reset:
                    get_new_statistics()
                    reset_field()
                    self.master.after(self.main_loop_ms_pause, main_game_loop)
                    return


            # print("self.field:\n{}".format(self.field))
            # check, if lines can be cleared or not!?
            idxs_row_full = np.all(self.field!=0, axis=1)
            if np.any(idxs_row_full):
                self.field[idxs_row_full] = 0

                self.clear_lines += np.sum(idxs_row_full)
                self.lbl_clear_lines_txt.set('{}'.format(self.clear_lines))

                all_rows_full = np.where(idxs_row_full)[0]
                all_rows_empty = np.where(np.all(self.field==0, axis=1))[0]
                all_rows = np.arange(self.field_y-1, -1, -1)

                all_rows_rest = all_rows[(~np.isin(all_rows, all_rows_full))&(~np.isin(all_rows, all_rows_empty))]

                # print("all_rows_full: {}".format(all_rows_full))
                # print("all_rows_empty: {}".format(all_rows_empty))
                # print("all_rows: {}".format(all_rows))
                # print("all_rows_rest: {}".format(all_rows_rest))

                # for row in all_rows_full:
                #     y = self.pos_y_start+self.box_h*row
                #     x = self.pos_x_start
                #     self.canvas.create_rectangle(x, y, x+self.canv_w, y+self.box_h, fill=self.canv_bg, width=0)
                

                all_rows_needed = all_rows[:all_rows_rest.shape[0]]
                idxs_same_rows = np.where(all_rows_rest!=all_rows_needed)[0]
                
                # idxs_row_to_move = np.where(~idxs_same_rows)[0]
                # print("idxs_same_rows: {}".format(idxs_same_rows))
                # import pdb
                # pdb.set_trace()

                if idxs_same_rows.shape[0]>0:
                    idx_first_row = idxs_same_rows[0]

                    rows_from = all_rows_rest[idx_first_row:]
                    rows_to = all_rows_needed[idx_first_row:]

                    # print("rows_from: {}".format(rows_from))
                    # print("rows_to: {}".format(rows_to))

                    # import pdb
                    # pdb.set_trace()

                    for row_from, row_to in zip(rows_from, rows_to):
                        self.field[row_to] = self.field[row_from]
                        self.field[row_from] = 0

                        # y = self.pos_y_start+self.box_h*row_from
                        # x = self.pos_x_start
                        # self.canvas.create_rectangle(x, y, x+self.canv_w, y+self.box_h, fill=self.canv_bg, width=0)
                        
                        # y = self.pos_y_start+self.box_h*row_to
                        # for xi, v in enumerate(self.field[row_to], 0):
                        #     if v==0:
                        #         continue
                        #     x = self.pos_x_start+self.box_w*xi
                        #     self.canvas.create_rectangle(x, y, x+self.box_w, y+self.box_h, fill=d_piece_idx_to_color[v], width=0)
                        
            self.tick += 1
            self.piece_nr += 1
            # print("self.piece_nr: {}".format(self.piece_nr))

            time_diff = time.time()-self.time_begin

            self.lbl_time_text.set('time_diff: {:.04f}'.format(time_diff))
            self.lbl_tick_text.set('tick: {}'.format(self.tick))
            fps = self.tick/time_diff
            self.lbl_fps_text.set('fps: {:.04f}'.format(fps))

            fps_needed_diff = fps-self.main_loop_fps_needed
            if abs(fps_needed_diff) >= 0.01:
                if fps_needed_diff < 0:
                    self.main_loop_ms_pause -= 1
                elif fps_needed_diff > 0:
                    self.main_loop_ms_pause += 1

            self.master.after(self.main_loop_ms_pause, main_game_loop)
        
        main_game_loop()


    def calculate_new_hash_val(self):
        self.str_txt = '0123456789'
        self.hash_val = hash(''.join([self.str_txt[i] for i in np.random.randint(0, len(self.str_txt), (10, ))]))


    def client_exit_menu_btn(self):
        print("Pressed the EXIT button or X!")
        self.master.destroy()


if __name__ == "__main__":
    # try:
    root = tk.Tk()
    app = MainWindow(master=root)
    root.mainloop()
    # except:
    #     import pdb
    #     pdb.set_trace()
