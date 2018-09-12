#! /usr/bin/python3.6

import os

import numpy as np

# from PIL import Image
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont

def create_dots_connection(rows, cols):
    connections = {}

    for y in range(0, rows):
        for x in range(0, cols):
            next_node = []
            if y > 0:
                next_node.append((y-1, x))
            if y < rows-1:
                next_node.append((y+1, x))
            if x > 0:
                next_node.append((y, x-1))
            if x < cols-1:
                next_node.append((y, x+1))
            connections[(y, x)] = next_node

    return connections

# TODO: add first some points inbetween (needed for bigger fields)
def find_way_start_finish(rows, cols):
    connections = create_dots_connection(rows, cols)

    node_now = (0, 0)
    nodes = [node_now]
    tries = 0
    # probab_tbl = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    while node_now != (rows-1, cols-1):
        next_nodes = connections[node_now]
        idxs = np.random.permutation(np.arange(0, len(next_nodes)))
        # print("nodes: {}".format(nodes))
        # print("next_nodes: {}".format(next_nodes))
        # input()

        # next_node = (lambda x: x[np.random.randint(0, len(x))])(connections[node_now])
        is_found = False
        for idx in idxs:
        # for next_node in next_nodes:
            next_node = next_nodes[idx]
            if not next_node in nodes:
                nodes.append(next_node)
                is_found = True
                break

        if not is_found:
            tries += 1
            if tries % 1000 == 0:
                print("tries: {}".format(tries))
                print("len(nodes): {}".format(len(nodes)))
            # for _ in range(1, np.random.randint(1, 6)):
            # amount_remove_nodes = probab_tbl[np.random.randint(0, len(probab_tbl))]
            amount_remove_nodes = np.random.randint(0, len(nodes)//2)
            # print("amount_remove_nodes: {}".format(amount_remove_nodes))
            for _ in range(0, amount_remove_nodes):
                if len(nodes) > 1:
                    nodes.pop(-1)

        node_now = nodes[-1]

    rest_connections = deepcopy(connections)
    for node1, node2 in zip(nodes[:-1], nodes[1:]):
        rest_connections[node1].remove(node2)
        rest_connections[node2].remove(node1)

    print("nodes: {}".format(nodes))
    return rest_connections, nodes

def create_labyrinth(rows, cols):
    pix_field = np.zeros((rows*2+1, cols*2+1, 3), dtype=np.uint8)
    
    # fill all in between white pixels
    for y in range(0, rows):
        for x in range(0, cols):
            pix_field[1+y*2, 1+x*2] = (255, 255, 255)

    # mark start and finish
    pix_field[1, 1] = (0, 0, 255)
    pix_field[-2, -2] = (255, 0, 0)
    
    rest_connections, nodes = find_way_start_finish(rows, cols)
    # get position to set the gaps between
    pos_y = []
    pos_x = []
    for (y1, x1), (y2, x2) in zip(nodes[:-1], nodes[1:]):
        if y1 == y2:
            pos_y.append(1+y1*2)
            pos_x.append(2+np.min((x1, x2))*2)
        else:
            pos_y.append(2+np.min((y1, y2))*2)
            pos_x.append(1+x1*2)
    pix_field[(pos_y, pos_x)] = (255, 255, 0)

    img = Image.fromarray(pix_field)
    # img.show()
    if not os.path.exists("images"):
        os.makedirs("images")
    img.save("images/labyrinth.png", "PNG")

    return rest_connections, nodes

if __name__ == "__main__":
    rows = 50
    cols = 85

    rest_connections, nodes = create_labyrinth(rows, cols)
    # set start and end pixels


    # # now add some gaps in between rows and cols
    # rows_gaps = np.random.randint(0, 2, (rows-1, cols))
    # cols_gaps = np.random.randint(0, 2, (rows, cols-1))

    # # TODO: create one path from start to finish!
    # # TODO: add random generated other path from the one finishing path!

    # pix_field[(lambda x: (x[0]*2+2, x[1]*2+1))(np.where(rows_gaps==1))] = (255, 255, 255)
    # pix_field[(lambda x: (x[0]*2+1, x[1]*2+2))(np.where(cols_gaps==1))] = (255, 255, 255)

