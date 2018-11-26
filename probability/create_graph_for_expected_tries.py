#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

from collections import namedtuple
from dotmap import DotMap

# n ... amount of numbers
# r ... amount of tries max
def create_graph_exptected_tries(n=3, r=4):
    node_settings = {
        "white": 'node [margin=0 fillcolor=white fontcolor=black fixedsize=true fontsize=12 width=0.5 shape=circle style=filled];',
        "gray":  'node [margin=0 fillcolor=gray fontcolor=blue fixedsize=true fontsize=12 width=0.5 shape=circle style=filled];',
        "green": 'node [margin=0 fillcolor="#00AA22" fontcolor=yellow fixedsize=true fontsize=12 width=0.5 shape=circle style=filled];',
        "red": 'node [margin=0 fillcolor="#FF1100" fontcolor=white fixedsize=true fontsize=12 width=0.5 shape=circle style=filled];'
    }

    Node = DotMap({k: None for k in ["name", "label", "color", "numbers", "tries", "left", "right"]})

    # n = 4
    # r = 8

    root = Node.copy()
    root.name="R"
    root.label="1, 1"
    root.numbers=1
    root.tries=1
    root.color="white"

    nodes = {
        "white": [root],
        "gray": [],
        "green": [],
        "red": []
    }
    nodes_connection = []

    def create_recursively_tree(node):
        if node.tries < r and node.numbers < n:
            numbers = node.numbers
            tries = node.tries+1
            name = node.name

            is_left_finished = True
            if numbers+1 == n:
                color = "green"
            else:
                color = "gray"
                is_left_finished = False

            left = Node.copy()
            left.name=name+"L"
            left.numbers=numbers+1
            left.label="{}, {}".format(numbers+1, tries)
            left.color=color
            left.tries=tries

            if not is_left_finished:
                create_recursively_tree(left)
            node.left = left

            right = Node.copy()
            right.name=name+"R"
            right.numbers=numbers
            right.label="{}, {}".format(numbers, tries)
            right.color="red"
            right.tries=tries
            create_recursively_tree(right)
            node.right = right

            nodes[left.color].append(left)
            nodes[right.color].append(right)

            nodes_connection.append([node.name, left.name, right.name])

    create_recursively_tree(root)

    s = "digraph graphname {"
    s += "  ranksep = 0.9;\n"
    s += "  nodesep = 0.0;\n\n"

    s += "  ordering=out;\n\n"

    for key, value in node_settings.items():
        s += "  // {} node\n".format(key)
        s += "  {\n"
        s += "    {}\n".format(value)
        nodes_of_color = nodes[key]
        for node in nodes_of_color:
            s += "    {} [label=\"{}\"];\n".format(node.name, node.label)
        s += "  }\n\n"

    for node_connect in nodes_connection:
        s += "  {} -> {};\n".format(node_connect[0], ", ".join(node_connect[1:]))

    s += "}\n"
    
    path_dir = "dot_files/"
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    file_name = path_dir+"expected_tries_n_{}_r_{}.dot".format(n, r)
    with open(file_name, "w") as fout:
        fout.write(s)

if __name__ == "__main__":
    for n in range(2, 8):
        print("n: {}".format(n))
        for r in range(7, 15):
            print("  r: {}".format(r))
            create_graph_exptected_tries(n=n, r=r)
