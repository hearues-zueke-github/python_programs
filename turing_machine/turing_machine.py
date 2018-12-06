#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import colorsys
import datetime
import dill
import gzip
import os
import shutil
import string
import subprocess
import sys
import time
import traceback

import numpy as np

import matplotlib.pyplot as plt

from dotmap import DotMap

from indexed import IndexedOrderedDict
from collections import OrderedDict
from dotmap import DotMap
from PIL import Image, ImageDraw, ImageFont

import utils

# TODO: create field, and instructions!
# TODO: create next move instruction! (or do n moves also!)
class TuringMachine(Exception):
    def __init__(self):
        pass


if __name__ == "__main__":
    turing_machine = TuringMachine()

    # state now -> found symbol idx on tape -> (new symbol, move, new state)

    # Using the instructions for the busy beaver from wikipedia:
    # https://en.wikipedia.org/wiki/Busy_beaver

    # instructions for the bussy beaver with 3 states
    instructions_3_states = IndexedOrderedDict([
        ('A', IndexedOrderedDict([
            ('0', ('1', 'R', 'B')),
            ('1', ('1', 'R', 'HALT')),
        ])),

        ('B', IndexedOrderedDict([
            ('0', ('0', 'R', 'C')),
            ('1', ('1', 'R', 'B')),
        ])),

        ('C', IndexedOrderedDict([
            ('0', ('1', 'L', 'C')),
            ('1', ('1', 'L', 'A')),
        ])),
    ])


    # Is also a busy beaver but with 4 states
    instructions_4_states = IndexedOrderedDict([
        ('A', IndexedOrderedDict([
            ('0', ('1', 'R', 'B')),
            ('1', ('1', 'L', 'B')),
        ])),

        ('B', IndexedOrderedDict([
            ('0', ('1', 'L', 'A')),
            ('1', ('0', 'L', 'C')),
        ])),

        ('C', IndexedOrderedDict([
            ('0', ('1', 'R', 'HALT')),
            ('1', ('1', 'L', 'D')),
        ])),

        ('D', IndexedOrderedDict([
            ('0', ('1', 'R', 'D')),
            ('1', ('0', 'R', 'A')),
        ])),
    ])

    # Also a busy beaver but with 5 states
    instructions_5_states = IndexedOrderedDict([
        ('A', IndexedOrderedDict([
            ('0', ('1', 'R', 'B')),
            ('1', ('1', 'L', 'C')),
        ])),

        ('B', IndexedOrderedDict([
            ('0', ('1', 'R', 'C')),
            ('1', ('1', 'R', 'B')),
        ])),

        ('C', IndexedOrderedDict([
            ('0', ('1', 'R', 'D')),
            ('1', ('0', 'L', 'E')),
        ])),

        ('D', IndexedOrderedDict([
            ('0', ('1', 'L', 'A')),
            ('1', ('1', 'L', 'D')),
        ])),

        ('E', IndexedOrderedDict([
            ('0', ('1', 'R', 'HALT')),
            ('1', ('0', 'L', 'A')),
        ])),
    ])

    # to calculate the different possible turing machines for each busy beaver
    # it is possible to use the following formula: beavers(n) = (4*(n+1))**(n*2)
    # Which is O(n**(2*n))

    # instructions = instructions_3_states
    # instructions = instructions_4_states
    instructions = instructions_5_states

    print("instructions:")
    for state, other in instructions.items():
        print("\nState: {}".format(state))

        for symbol, tupl in other.items():
            print("  symbol: {} -> (New Symbol: {}, Move: {}, New State: {})".format(symbol, tupl[0], tupl[1], tupl[2]))

    tape = IndexedOrderedDict()
    position = 0

    state_now = "A"
    moves_total = 0

    symbols_lst = []
    moves_lst = []
    states_lst = [state_now]

    history = DotMap()
    history.symbols_lst = symbols_lst
    history.moves_lst = moves_lst
    history.states_lst = states_lst

    while state_now != "HALT":
        dct = instructions[state_now]

        if not position in tape:
            tape[position] = "0"

        symobl_now = tape[position]
        new_symbol, move, new_state = dct[symobl_now]

        print("pos: {}, new_symbol: {}, move: {}, new_state: {}".format(position, new_symbol, move, new_state))

        # symbols_lst.append(new_symbol)
        # moves_lst.append(move)
        # states_lst.append(new_state)

        tape[position] = new_symbol

        moves_total += 1

        print("moves_total: {}".format(moves_total))
        # print("moves_total: {}, tape: {}".format(moves_total, tape))

        if move == "R":
            position += 1
        elif move == "L":
            position -= 1
        else:
            sys.exit("ERROR!")

        state_now = new_state
