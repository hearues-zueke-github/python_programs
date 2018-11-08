#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import sys

import numpy as np

from collections import namedtuple
from dotmap import DotMap

if __name__ == "__main__":
    file_name = "test_graph.dot"

    node_settings = {
        "white": 'node [margin=0 fillcolor=white fontcolor=black fixedsize=true fontsize=12 width=0.5 shape=circle style=filled];',
        "gray":  'node [margin=0 fillcolor=gray fontcolor=blue fixedsize=true fontsize=12 width=0.5 shape=circle style=filled];',
        "green": 'node [margin=0 fillcolor="#00AA22" fontcolor=yellow fixedsize=true fontsize=12 width=0.5 shape=circle style=filled];',
        "red": 'node [margin=0 fillcolor="#FF1100" fontcolor=white fixedsize=true fontsize=12 width=0.5 shape=circle style=filled];'
    }

    # Node = namedtuple("Node", ["name", "label", "color", "numbers", "tries", "left", "right"])
    # Node.__new__.__defaults__ = (None,) * len(Node._fields)
    Node = DotMap({k: None for k in ["name", "label", "color", "numbers", "tries", "left", "right"]})
    # Node = DotMap()
    # Node.name = None
    # Node.label = None
    # Node.color = None
    # Node.numbers = None
    # Node.tries = None
    # Node.left = None
    # Node.right = None

    # Tree = namedtuple("Tree", ["value", "left", "right"])

    n = 4
    r = 6
    # first create the Tree with the Nodes!
    # TODO: create a graph with 

    # root = Tree(value=Node(name="root", label="1, 1", color="white"), left=None, right=None)
    # root = Node(name="R", label="1, 1", numbers=1, tries=1, color="white")
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

            # left = Node(name=name+"L",
            #              numbers=numbers+1,
            #              label="{}, {}".format(numbers+1, tries),
            #              color=color,
            #              tries=tries)
            left = Node.copy()
            left.name=name+"L"
            left.numbers=numbers+1
            left.label="{}, {}".format(numbers+1, tries)
            left.color=color
            left.tries=tries

            if not is_left_finished:
                create_recursively_tree(left)
            node.left = left

            # right = Node(name=name+"R",
            #              numbers=numbers,
            #              label="{}, {}".format(numbers, tries),
            #              color="red",
            #              tries=tries)

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
        # return node

    # nodes = {
    #     "white": [],
    #     "gray": [],
    #     "green": [],
    #     "red": []
    # }

    # nodes_connection = []


    # nodes = {
    #     "white": [
    #         Node(name="n1", label="n11"),
    #         Node(name="n2", label="n12")
    #     ],
    #     "gray": [
    #         Node(name="n3", label="n21"),
    #         Node(name="n4", label="n22")
    #     ],
    #     "green": [
    #         Node(name="n5", label="n31"),
    #         Node(name="n6", label="n32")
    #     ],
    #     "red": [
    #         Node(name="n7", label="n41"),
    #         Node(name="n8", label="n42")
    #     ]
    # }

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


    # nodes_connection = [
    #     ["n1", "n2", "n3"],
    #     ["n2", "n4", "n6"],
    #     ["n3", "n7", "n8"],
    #     ["n6", "n5"]
    # ]

    for node_connect in nodes_connection:
        s += "  {} -> {};\n".format(node_connect[0], ", ".join(node_connect[1:]))

    s += "}\n"
    
    with open(file_name, "w") as fout:
        fout.write(s)
