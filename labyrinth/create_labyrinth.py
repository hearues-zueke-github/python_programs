#! /usr/bin/python3.6

import os
import sys

import numpy as np

# from PIL import Image
from copy import deepcopy
from dotmap import DotMap
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt

color_white = (255, 255, 255)
color_yellow = (255, 255, 0)
color_cyan = (0, 255, 255)
color_red = (255, 0, 0)
color_blue = (0, 0, 255)

used_nodes = {
    2: [(1, 0), (0, 1)],
    3: [(0, 1, 2), (0, 2, 1), (1, 2, 0), (1, 0, 2), (2, 1, 0), (2, 0, 1)]
}

# used_nodes = {
#     2: [
#     # (0, ), (1, ),
#         (0, 1), (1, 0)],
#     3: [
#     # (0, ), (1, ), (2, ),
#         (1, 0), (0, 1), (1, 2), (2, 1), (0, 2), (2, 0),
#         (0, 1, 2), (0, 2, 1), (1, 2, 0), (1, 0, 2), (2, 1, 0), (2, 0, 1)],
# }

def get_random_node_idx(n):
    nodes_idx = used_nodes[n]
    return nodes_idx[np.random.randint(0, len(nodes_idx))]


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

    print("connections: {}".format(connections))

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
    print("rest_connections: {}".format(rest_connections))
    return rest_connections, nodes, between_nodes


def create_labyrinth_one_path(rows, cols):
    pix_field = np.zeros((rows*2+1, cols*2+1, 3), dtype=np.uint8)
    
    # fill all in between white pixels
    for y in range(0, rows):
        for x in range(0, cols):
            pix_field[1+y*2, 1+x*2] = (255, 255, 255)

    # mark start and finish
    pix_field[0, 1] = (0, 0, 255)
    pix_field[-1, -2] = (255, 0, 0)
    
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
    pix_field[(pos_y, pos_x)] = (255, 255, 255)
    # pix_field[(pos_y, pos_x)] = (255, 255, 0)

    img = Image.fromarray(pix_field)
    # img.show()
    if not os.path.exists("images"):
        os.makedirs("images")
    img.save("images/labyrinth.png", "PNG")
    img = img.resize((img.width*8, img.height*8))
    img.save("images/resized_labyrinth.png", "PNG")

    return rest_connections, nodes


def create_labyrinth_picture(rows, cols, nodes, show_plot=False):
    pix_field = np.zeros((rows*2+1, cols*2+1, 3), dtype=np.uint8)
    
    # fill all in between white pixels
    for y in range(0, rows):
        for x in range(0, cols):
            pix_field[1+y*2, 1+x*2] = color_white

    # mark start and finish
    pix_field[0, 1] = color_blue
    pix_field[-1, -2] = color_red
    
    try:
        for y, x in between_nodes:
            pix_field[1+y*2, 1+x*2] = color_cyan
    except:
        pass

    # get position to set the gaps between
    pos_y = []
    pos_x = []
    for (y1, x1), (y2, x2) in nodes:
        if y1 == y2:
            pos_y.append(1+y1*2)
            pos_x.append(2+np.min((x1, x2))*2)
        else:
            pos_y.append(2+np.min((y1, y2))*2)
            pos_x.append(1+x1*2)
    pix_field[(pos_y, pos_x)] = color_white

    img = Image.fromarray(pix_field)
    # img.show()
    if not os.path.exists("images"):
        os.makedirs("images")
    img.save("images/labyrinth_complete.png", "PNG")
    img = img.resize((img.width*8, img.height*8))
    img.save("images/resized_labyrinth_complete.png", "PNG")

    if show_plot:
        fig = plt.figure()
        plt.imshow(img)


def labyrinth_picture_used_fields(field, show_plot=False):
    rows, cols = field.shape
    pix_field = np.zeros((rows*2+1, cols*2+1, 3), dtype=np.uint8)
    
    # fill all in between white pixels
    for y in range(0, rows):
        for x in range(0, cols):
            is_used_field = field[y, x] == 1
            if is_used_field:
                pix_field[1+y*2, 1+x*2] = color_yellow
            else:
                pix_field[1+y*2, 1+x*2] = color_white

    # mark start and finish
    pix_field[0, 1] = color_blue
    pix_field[-1, -2] = color_red
    
    img = Image.fromarray(pix_field)

    if not os.path.exists("images"):
        os.makedirs("images")
    img.save("images/labyrinth_complete_used_fields.png", "PNG")
    img = img.resize((img.width*8, img.height*8))
    img.save("images/resized_labyrinth_complete_used_fields.png", "PNG")

    if show_plot:
        fig = plt.figure()
        plt.imshow(img)


def create_labyrinth_picture_with_color(rows, cols, nodes, field, suffix_name="", show_plot=False):
    pix_field = np.zeros((rows*2+1, cols*2+1, 3), dtype=np.uint8)
    
    # fill all in between white pixels
    for y in range(0, rows):
        for x in range(0, cols):
            pix_field[1+y*2, 1+x*2] = color_white

    # mark start and finish
    pix_field[0, 1] = color_blue
    pix_field[-1, -2] = color_red
    
    # get all unique nums from field!
    unique_nums = np.unique(field)

    # create different colors!
    colors = np.random.randint(0, 256, (unique_nums.shape[0], 3), dtype=np.uint8)

    y0, x0 = nodes[0][0]
    pix_field[1, 1] = colors[field[y0, x0]]

    # get position to set the gaps between
    pos_y = []
    pos_x = []
    for (y1, x1), (y2, x2) in nodes:
        # n1 = field[y1, x1]
        n2 = field[y2, x2]
        # c1 = colors[n1]
        c2 = colors[n2]
        
        # c_mix = ((c1.astype(np.float)+c2.astype(np.float))/2.).astype(np.uint8)
        c_mix = c2

        pix_field[1+y2*2, 1+x2*2] = c2
        
        if y1 == y2:
            if x1 > x2:
                pix_field[1+y1*2, 1+x1*2-1] = c_mix
            else:
                pix_field[1+y1*2, 1+x1*2+1] = c_mix
        else:
            if y1 > y2:
                pix_field[1+y1*2-1, 1+x1*2] = c_mix
            else:
                pix_field[1+y1*2+1, 1+x1*2] = c_mix

        # pix_field[1+np.min((y1, y2))*2, 1+np.min((x1, x2))*2] = c

    img = Image.fromarray(pix_field)
    if not os.path.exists("images"):
        os.makedirs("images")

    suffix_name = ("" if suffix_name == "" else "_"+suffix_name)

    img.save("images/labyrinth_complete{}.png".format(suffix_name), "PNG")
    img = img.resize((img.width*8, img.height*8))
    img.save("images/resized_labyrinth_complete{}.png".format(suffix_name), "PNG")

    if show_plot:
        fig = plt.figure()
        plt.imshow(img)


def create_labyrinth(rows, cols):
    connections = create_dots_connection(rows, cols)
    field = np.zeros((rows, cols), dtype=np.uint8)
    field_different_paths = np.zeros((rows, cols), dtype=np.int)
    field_same_paths = np.zeros((rows, cols), dtype=np.int)

    nodes = []
    used_connections = {}

    dm = DotMap()
    dm.nodes = nodes
    dm.connections = connections
    dm.field = field
    dm.used_connections = used_connections
    dm.field_different_paths = field_different_paths
    dm.field_same_paths = field_same_paths

    field[0, 0] = 1

    # TODO: create a recursive function for creating the labyrinth!

    def go_the_labyrinth_through(dm, y, x, i, j):
        nodes = dm.nodes
        connections = dm.connections
        field = dm.field
        used_connections = dm.used_connections
        field_different_paths = dm.field_different_paths
        field_same_paths = dm.field_same_paths

        node_now = (y, x)
        to_nodes = connections[node_now]

        field[y, x] = 1

        print("\nnode_now: {}".format(node_now))
        print("i: {}".format(i))

        is_field_set = []
        for iy, ix in to_nodes:
            is_field_set.append(field[iy, ix])

        to_nodes_possible = list(filter(lambda node: field[node[0], node[1]] == 0, to_nodes))

        field_different_paths[y, x] = i
        field_same_paths[y, x] = j
        
        n = len(to_nodes_possible)
        print("n: {}".format(n))
        if n > 1:
            using_nodes_idx = get_random_node_idx(n)
            nodes_chosen = []
            for idx in using_nodes_idx:
                nodes_chosen.append(to_nodes_possible[idx])
        elif n == 0:
            print("No more possible nodes!")
            return
        else:
            nodes_chosen = to_nodes_possible

        # for iy, ix in nodes_chosen:
        #     field[iy, ix] = 1

        for k, node in enumerate(nodes_chosen, 0):
            iy, ix = node
            if field[iy, ix] == 1:
                continue

            # field[iy, ix] = 1

            nodes.append((node_now, node))

            if not node_now in used_connections:
                used_connections[node_now] = []
            if not node in used_connections:
                used_connections[node] = []

            used_connections[node_now].append(node)
            used_connections[node].append(node_now)

            go_the_labyrinth_through(dm, node[0], node[1], i+1, j+k)

        # sys.exit(-1)

    # TODO: need to fix something too! Labyrinth is not created very perfect random!
    # field[0, 0] = 1
    go_the_labyrinth_through(dm, 0, 0, 0, 0)
    create_labyrinth_picture(rows, cols, nodes)
    labyrinth_picture_used_fields(field)

    create_labyrinth_picture_with_color(rows, cols, nodes, dm.field_different_paths, suffix_name="different_paths")
    create_labyrinth_picture_with_color(rows, cols, nodes, dm.field_same_paths, suffix_name="same_paths", show_plot=True)

    plt.show()

    return dm


if __name__ == "__main__":
    rows = 30
    cols = 45

    dm = create_labyrinth(rows, cols)

    # rest_connections, nodes = create_labyrinth(rows, cols)
    
    # rest_connections, nodes = create_labyrinth_one_path(rows, cols)
    # set start and end pixels


    # # now add some gaps in between rows and cols
    # rows_gaps = np.random.randint(0, 2, (rows-1, cols))
    # cols_gaps = np.random.randint(0, 2, (rows, cols-1))

    # # TODO: create one path from start to finish!
    # # TODO: add random generated other path from the one finishing path!

    # pix_field[(lambda x: (x[0]*2+2, x[1]*2+1))(np.where(rows_gaps==1))] = (255, 255, 255)
    # pix_field[(lambda x: (x[0]*2+1, x[1]*2+2))(np.where(cols_gaps==1))] = (255, 255, 255)

