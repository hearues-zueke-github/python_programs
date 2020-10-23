#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import datetime
import dill
import gzip
import inspect
import os
import pdb
import shutil
import string
import sys

from copy import deepcopy

import matplotlib.pyplot as plt

import numpy as np
from array2gif import write_gif

from dotmap import DotMap
from PIL import Image, ImageFont, ImageDraw

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

sys.path.append(PATH_ROOT_DIR+"/..")
import utils_all
import utils_serialization

class TreeNode(Exception):
    def __init__(self, parent, value):
        self.parent = parent
        if parent is not None:
            parent.nodes.append(self)
        self.value = value
        self.nodes = []


    def __str__(self):
        # return '(parent_val: {}, val: {}, node_vals: {})'.format(
        return '(p: {}, v: {}, n: {})'.format(
            None if self.parent is None else self.parent.value,
            self.value,
            [node.value for node in self.nodes]
        )


    def get_values_top_down_breadth(self):
        l_vals = [self.value]
        l_nodes = self.nodes
        while len(l_nodes)>0:
            l_nodes_new = []
            for node_now in l_nodes:
                l_vals.append(node_now.value)
                # for node_child in node_now.nodes:
                #     l_nodes_new.append(node_child)
                l_nodes_new.extend(node_now.nodes)
            l_nodes = l_nodes_new

        return l_vals


if __name__ == "__main__":
    n1 = TreeNode(None, 1)    
    
    n2 = TreeNode(n1, 2)
    # n1.nodes.append(n2)
    
    s_vals = set([1, 2])
    l_nodes = [n2]

    for _ in range(2, 10):
        l_nodes_new = []
        for node in l_nodes:
            # get all top values first!
            l_add_vals = []
            v1 = node.value
            node_ = node.parent
            while node_ is not None:
                l_add_vals.append(node_.value)
                node_ = node_.parent
            for v2 in reversed(l_add_vals):
                v = v1+v2
                if not v in s_vals:
                    s_vals.add(v)
                    node_new = TreeNode(node, v)
                    l_nodes_new.append(node_new)
        l_nodes = l_nodes_new

    l_values = n1.get_values_top_down_breadth()
    print("l_values:\n{}".format(l_values))
    # print("sorted(l_values): {}".format(sorted(l_values)))
    
    # index_first_diff_gt_1 = np.where(np.diff(np.array(sorted(l_values)))>1)[0][0]
    # print("index_first_diff_gt_1: {}".format(index_first_diff_gt_1))
