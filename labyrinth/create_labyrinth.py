#! /usr/bin/python3.6

import os
import sys

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
# TODO: make finding radnom path from start to finish much better!
def find_way_start_finish(rows, cols, start_node=None, finish_node=None, between_stops=0):
    connections = create_dots_connection(rows, cols)

    if start_node == None:
        start_node = (0, 0)
    if finish_node == None:
        finish_node = (rows-1, cols-1)

    between_nodes = [(np.random.randint(1, rows-1), np.random.randint(1, cols-1)) for _ in range(0, between_stops)]
    print("between_stops: {}".format(between_stops))
    print("between_nodes: {}".format(between_nodes))
    checkpoint_nodes = [start_node]+between_nodes+[finish_node]
    print("checkpoint_nodes: {}".format(checkpoint_nodes))
    path_found = False
    start_overs = 0
    while not path_found:
        all_nodes = []
        do_start_over = False
        rest_connections = deepcopy(connections)

        for node_idx, (temp_start_node, temp_finish_node) in enumerate(zip(checkpoint_nodes[:-1], checkpoint_nodes[1:])):
            print("node_idx: {}".format(node_idx))
            node_now = temp_start_node
            nodes = [node_now]
            get_length = lambda node1, node2: np.sum((np.array(node1)-np.array(node2))**2)
            best_dist_length = get_length(temp_start_node, temp_finish_node)
            best_dist_node = temp_start_node
            
            tries = 0
            while node_now != temp_finish_node:
                next_nodes = rest_connections[node_now]
                idxs = np.random.permutation(np.arange(0, len(next_nodes)))

                is_found = False
                for idx in idxs:
                    next_node = next_nodes[idx]
                    if not next_node in nodes:
                        nodes.append(next_node)
                        is_found = True
                        break

                if not is_found:
                    tries += 1
                    if tries > 50000:
                        do_start_over = True
                        start_overs += 1
                        print("start_overs: {}, node_idx: {}".format(start_overs, node_idx))
                        print("nodes: {}".format(nodes))
                        break

                    amount_remove_nodes = np.random.randint(1, len(nodes)//2)

                    if len(nodes) > amount_remove_nodes:
                        nodes = nodes[:-amount_remove_nodes]
                    else:
                        nodes = nodes[:1]
                
                node_now = nodes[-1]

                temp_dist_length = get_length(node_now, temp_finish_node)
                if temp_dist_length < best_dist_length:
                    best_dist_length = temp_dist_length
                    best_dist_node = node_now
                    print("best_dist_length: {:6.4f}, best_dist_node: {}".format(best_dist_length, best_dist_node))
                # try:
                #     node_now = nodes[-1]
                # except:
                #     print("error: nodes: {}".format(nodes))
                #     sys.exit(0)

            if do_start_over:
                break

            for node1, node2 in zip(nodes[:-1], nodes[1:]):
                rest_connections[node1].remove(node2)
                rest_connections[node2].remove(node1)
            all_nodes += nodes[:-1]

        if not do_start_over:
            all_nodes += [nodes[-1]]
            break

    nodes = all_nodes

    print("nodes: {}".format(nodes))
    return rest_connections, nodes, between_nodes

def create_labyrinth(rows, cols):
    pix_field = np.zeros((rows*2+1, cols*2+1, 3), dtype=np.uint8)
    
    # fill all in between white pixels
    for y in range(0, rows):
        for x in range(0, cols):
            pix_field[1+y*2, 1+x*2] = (255, 255, 255)

    # mark start and finish
    pix_field[1, 1] = (0, 0, 255)
    pix_field[-2, -2] = (255, 0, 0)
    
    # rest_connections, nodes = find_way_start_finish(rows, cols)
    rest_connections, nodes, between_nodes = find_way_start_finish(rows, cols, between_stops=2)
    for y, x in between_nodes:
        pix_field[1+y*2, 1+x*2] = (0, 255, 255)

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
    rows = 20
    cols = 35

    rest_connections, nodes = create_labyrinth(rows, cols)
    # set start and end pixels


    # # now add some gaps in between rows and cols
    # rows_gaps = np.random.randint(0, 2, (rows-1, cols))
    # cols_gaps = np.random.randint(0, 2, (rows, cols-1))

    # # TODO: create one path from start to finish!
    # # TODO: add random generated other path from the one finishing path!

    # pix_field[(lambda x: (x[0]*2+2, x[1]*2+1))(np.where(rows_gaps==1))] = (255, 255, 255)
    # pix_field[(lambda x: (x[0]*2+1, x[1]*2+2))(np.where(cols_gaps==1))] = (255, 255, 255)

