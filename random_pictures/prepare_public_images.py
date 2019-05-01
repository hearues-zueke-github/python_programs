#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import re
import sys
import time

import itertools
import multiprocessing

import matplotlib
import matplotlib.pyplot as plt

import filehash
import subprocess

cpu_amount = multiprocessing.cpu_count()

import numpy as np

from PIL import Image, ImageFile

hasher = filehash.FileHash('sha512')

# import decimal
# prec = 50
# decimal.getcontext().prec = prec

# from decimal import Decimal as Dec

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

def get_image_size_from_path(file_path):
    ImPar=ImageFile.Parser()
    with open(file_path, "rb") as f:
        ImPar=ImageFile.Parser()
        chunk = f.read(2048)
        count=2048
        while chunk != "":
            ImPar.feed(chunk)
            if ImPar.image:
                break
            chunk = f.read(2048)
            count+=2048
        # print(ImPar.image.size)
        # print(count)
    return ImPar.image.size

"""
public internet pictures can be see at:
https://pixabay.com/images/search/
https://www.publicdomainpictures.net/en/latest-pictures.php?page=0
https://isorepublic.com/
etc.

general:
https://en.99designs.at/blog/resources/public-domain-image-resources/
"""

if __name__ == "__main__":
    print("Hello World!")

    images_path = ROOT_PATH+'images/pixabay_com/'

    images_path_pngs = images_path+'pngs/'
    if not os.path.exists(images_path_pngs):
        os.makedirs(images_path_pngs)

    obj_path = images_path+"picture_dict_objs/"
    if not os.path.exists(obj_path):
        os.makedirs(obj_path)

    root_path_dir, dir_names, file_names = next(os.walk(images_path), range(0, 1))
    print("root_path_dir: {}".format(root_path_dir))
    print("len(dir_names): {}".format(len(dir_names)))
    print("len(file_names): {}".format(len(file_names)))

    extensions_counter = {}
    shapes_counter = {}
    # channels_counter = {}

    # heights = [300, 150, 75, 3*20, 3*15, 3*10, 3*8, 3*5, 3*3, 3*2]
    # widths = [400, 200, 100, 4*20, 4*15, 4*10, 4*8, 4*5, 4*3, 4*2]
    
    heights = [75, 3*20, 3*15, 3*10, 3*8, 3*5, 3*3, 3*2]
    widths = [100, 4*20, 4*15, 4*10, 4*8, 4*5, 4*3, 4*2]

    paths_sizes_landscape = [(w, h, images_path_pngs+"{w}x{h}/".format(w=w, h=h), obj_path+"{w}x{h}.pkl.gz".format(w=w, h=h)) for w, h in zip(widths, heights)]
    paths_sizes_portaire = [(w, h, images_path_pngs+"{w}x{h}/".format(w=w, h=h), obj_path+"{w}x{h}.pkl.gz".format(w=w, h=h)) for w, h in zip(heights, widths)]

    # for _, _, dir_path in (paths_sizes_landscape+paths_sizes_portaire):
    #     if not os.path.exists(dir_path):
    #         os.makedirs(dir_path)

    objs = {}

    for _, _, _, obj_path in paths_sizes_landscape:
        if not os.path.exists(obj_path):
            print("obj_path '{}' do not exists!".format(obj_path))
            objs[obj_path] = {}            
        else:
            print("obj_path '{}' will be loaded in!".format(obj_path))
            with gzip.open(obj_path, "rb") as f:
                o = dill.load(f)
            objs[obj_path] = o

    for _, _, _, obj_path in paths_sizes_portaire:
        if not os.path.exists(obj_path):
            print("obj_path '{}' do not exists!".format(obj_path))
            objs[obj_path] = {}            
        else:
            print("obj_path '{}' will be loaded in!".format(obj_path))
            with gzip.open(obj_path, "rb") as f:
                o = dill.load(f)
            objs[obj_path] = o

    first_landscape_obj = objs[paths_sizes_landscape[0][3]]
    first_portaire_obj = objs[paths_sizes_portaire[0][3]]

    # paths_sizes = paths_sizes_landscape+paths_sizes_portaire

    file_nr_counter = 0
    for file_nr, file_name in enumerate(file_names, 0):
    # for file_nr, file_name in enumerate(file_names[:1000], 0):
        if not "." in file_name:
            continue
        extension = file_name.split(".")[-1]
        if not extension in extensions_counter:
            extensions_counter[extension] = 0
        extensions_counter[extension] += 1

        # img = Image.open(root_path_dir+file_name)
        # pix = np.array(img)

        img_orig_file_path = root_path_dir+file_name
        shape = get_image_size_from_path(img_orig_file_path)
        # shape = pix.shape
        if not shape in shapes_counter:
            shapes_counter[shape] = 0
        shapes_counter[shape] += 1

        # channels = shape[2]
        # if not channels in channels_counter:
        #     channels_counter[channels] = 0
        # channels_counter[channels] += 1

        if extension == "png":
            continue

        width, height = shape

        width_1 = int(height*4/3)
        width_2 = int(height*3/4)

        img = Image.open(img_orig_file_path)
        pix = np.array(img)

        h, w = pix.shape[:2]
        if len(pix.shape) == 2:
            pix = np.tile(pix.reshape((-1, )), 3).reshape((3, -1)).T.reshape((h, w, 3))
        elif pix.shape[2] == 4:
            pix = pix[:, :, :3]

        h_512 = hasher.hash_file(img_orig_file_path).upper()
        hash_part = h_512[:8]
        new_file_name = "{}.png".format(hash_part)

        if width >= width_1:
            if new_file_name in first_landscape_obj:
                print("skip file '{}' for landscape!".format(new_file_name))
                continue

            x_opt = (width-width_1)//2
            pix = pix[:, x_opt:x_opt+width_1]
            img = Image.fromarray(pix)

            # for w, h, dir_path in paths_sizes_landscape:
            paths_sizes = paths_sizes_landscape
            # for w, h, _, obj_path in paths_sizes_landscape:
            #     img_resize = img.resize((w, h), Image.LANCZOS)
            #     img_resize.save(dir_path+new_file_name)
            #     # img_resize.save(dir_path+'{:05}.png'.format(file_nr_counter))
        elif width >= width_2:
            if new_file_name in first_portaire_obj:
                print("skip file '{}' for portaire!".format(new_file_name))
                continue

            x_opt = (width-width_2)//2
            pix = pix[:, x_opt:x_opt+width_2]
            img = Image.fromarray(pix)

            paths_sizes = paths_sizes_portaire
            # for w, h, dir_path in paths_sizes_portaire:
            #     img_resize = img.resize((w, h), Image.LANCZOS)
            #     img_resize.save(dir_path+new_file_name)
            #     # img_resize.save(dir_path+'{:05}.png'.format(file_nr_counter))
        else:
            continue

        for w, h, _, obj_path in paths_sizes:
            img_resize = img.resize((w, h), Image.LANCZOS)
            objs[obj_path][new_file_name] = np.array(img_resize)
            # img_resize.save(dir_path+new_file_name)
            # img_resize.save(dir_path+'{:05}.png'.format(file_nr_counter))

        print("file_nr: {}, file_nr_counter: {}".format(file_nr, file_nr_counter))

        # img = Image.fromarray(pix)
        # img.save(images_path_pngs+new_file_name)
        # img.save(images_path_pngs+'{:05}.png'.format(file_nr_counter))
        file_nr_counter += 1

        # img.save(images_path_pngs+file_name.replace("."+extension, ".png"))

    for _, _, _, obj_path in paths_sizes_landscape:
        print("Saving object '{}'".format(obj_path))
        with gzip.open(obj_path, "wb") as f:
            dill.dump(objs[obj_path], f)

    for _, _, _, obj_path in paths_sizes_portaire:
        print("Saving object '{}'".format(obj_path))
        with gzip.open(obj_path, "wb") as f:
            dill.dump(objs[obj_path], f)

    print("extensions_counter:\n{}".format(extensions_counter))
    print("shapes_counter:\n{}".format(shapes_counter))
    # print("channels_counter:\n{}".format(channels_counter))
