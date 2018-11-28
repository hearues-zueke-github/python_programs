#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import colorsys
import datetime
import dill
import os
import string
import sys
import traceback

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont


class Node(Exception):
    pass


class Operator(Node):
    operator = None

    def __init__(self):
        pass


class Tree(Exception):
    prev: "Tree" = None
    branches: "Tree" = None
    value: str = ""
    values: list = []
    values_args: list = []
    operators: list = []
    func_str: str

    def __init__(self, deep, max_deep, max_length_values, prev=None, value=""):
        self.prev = prev
        self.value = value

        self.deep = deep
        self.max_deep = max_deep
        self.max_length_values = max_length_values

        self.possible_values = {
            # const is a complex(const_num, const_num) number!
            0: np.array(["z", "z.real", "z.imag", "const"]),
            1: np.array(["cos({})", "sin({})"]),
            2: np.array(["pow({},{})", "complex({},{})"])
        }
        self.possible_operators = np.array(["+", "-", "*"]) # will not use '/' still yet
        self.combine_str = ""


    def generate_new_z_function(self):
        intendation = "  "*self.deep
        
        if self.deep < self.max_deep:
            self.generate_new_values()
        else:
            self.generate_new_values(const_only=True)
            print("{}const only".format(intendation))

        self.values_strip = [v.strip("({},)") for v in self.values]
        self.branches = [None for _ in range(0, len(self.values_strip))]

        print("{}self.deep: {}".format(intendation, self.deep))
        print("{}self.values_strip: {}".format(intendation, self.values_strip))
        print("{}self.operators: {}".format(intendation, self.operators))

        self.values_args = []

        for i, v in enumerate(self.values_strip, 0):
            if v in ["z", "z.real", "z.imag", "const"]:
                if v == "z":
                    self.values_args.append("z")
                    self.values.append("z")
                elif v == "z.real":
                    self.values_args.append("z.real")
                    self.values.append("z.real")
                elif v == "z.imag":
                    self.values_args.append("z.imag")
                    self.values.append("z.imag")
                elif v == "const":
                    args = "{}".format(np.random.randint(0, 1000)/100.)
                    self.values_args.append(args)
                    self.values[i] = args
                continue

            if v == "cos":
                bt = Tree(self.max_deep, self.max_deep, self.max_length_values, prev=self, value=v)
                # bt = Tree(self.deep+1, self.max_deep, self.max_length_values, prev=self, value=v)
                print("{}func cos".format(intendation))
                func_str = bt.generate_new_z_function()
                self.branches[i] = (bt, )
                self.values[i] = self.values[i].replace("cos", "np.cos").format(func_str)

            elif v == "sin":
                bt = Tree(self.max_deep, self.max_deep, self.max_length_values, prev=self, value=v)
                # bt = Tree(self.deep+1, self.max_deep, self.max_length_values, prev=self, value=v)
                print("{}func sin".format(intendation))
                func_str = bt.generate_new_z_function()
                self.branches[i] = (bt, )
                self.values[i] = self.values[i].replace("sin", "np.sin").format(func_str)

            elif v == "pow":
                bt_1 = Tree(self.deep+1, self.max_deep, self.max_length_values, prev=self, value=v)
                bt_2 = Tree(self.max_deep, self.max_deep, self.max_length_values, prev=self, value=v)
                print("{}func pow".format(intendation))
                func_str_1 = bt_1.generate_new_z_function()
                func_str_2 = bt_2.generate_new_z_function()
                self.branches[i] = (bt_1, bt_2)
                self.values[i] = self.values[i].format(func_str_1, func_str_2)

            elif v == "complex":
                bt_1 = Tree(self.deep+1, self.max_deep, self.max_length_values, prev=self, value=v)
                bt_2 = Tree(self.deep+1, self.max_deep, self.max_length_values, prev=self, value=v)
                print("{}func complex".format(intendation))
                func_str_1 = bt_1.generate_new_z_function()
                func_str_2 = bt_2.generate_new_z_function()
                self.branches[i] = (bt_1, bt_2)
                self.values[i] = self.values[i].format(func_str_1, func_str_2)

        print("{}self.values_args: {}".format(intendation, self.values_args))

        # Now build the func_str!
        self.func_str = self.values[0]

        for o, v in zip(self.operators, self.values[1:]):
            self.func_str += o+v

        print("{}self.func_str: {}".format(intendation, self.func_str))

        return self.func_str

    def generate_new_values(self, const_only=False):
        length_values = np.random.randint(1, self.max_length_values+1)

        self.values = self.get_random_values(length_values, const_only=const_only)
        self.operators = self.get_random_operators(length_values-1)
        self.combine_str = self.values[0]+"".join([o+v for o, v in zip(self.operators, self.values[1:])])


    def get_random_values(self, n, const_only=False):
        values = []
        if const_only:
            for _ in range(0, n):
                values.append(np.random.choice(self.possible_values[0]))
            
            if self.value in ["cos", "sin"]:
                # replace z with other values, but not z! TODO: add better explanation
                for i, v in enumerate(values, 0):
                    if v == "z":
                        values[i] = np.random.choice(self.possible_values[0][1:])
        else:
            for _ in range(0, n):
                values.append(np.random.choice(
                    self.possible_values[np.random.choice(list(self.possible_values.keys()))]
                ))

        return values


    def get_random_operators(self, n):
        operators = []
        for _ in range(0, n):
            operators.append(np.random.choice(self.possible_operators))
        return operators


if __name__ == "__main__":
    max_deep = 2
    max_length_values = 4
    
    bt_root = Tree(0, max_deep, max_length_values)

    func_str = bt_root.generate_new_z_function()

    print("\nfinal func_str: {}".format(func_str))

    with open("z_func.txt", "w") as fout:
        fout.write(func_str)
