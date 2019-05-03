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

hasher512 = filehash.FileHash('sha512')
hasher256 = filehash.FileHash('sha256')

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
    class ImagesObjectConversion(Exception):
        def __init__(self, images_path, data_path, meta_data_file_name, object_file_name_template):
            self.images_path = images_path
            self.data_path = data_path

            if not os.path.exists(data_path):
                os.makedirs(data_path)

            self.meta_data_file_name = meta_data_file_name
            self.object_file_name_template = object_file_name_template

            self.meta_path = data_path+meta_data_file_name
            self.obj_template_path = data_path+object_file_name_template

            self.meta_obj = None


        def load_meta_object(self):
            if not os.path.exists(self.meta_path):
                self.meta_obj = {'amount_parts': [0], 'current_counter': [0], 'max_part_size': 2000, 'files_name_saved': {}, 'hashes_256_used': {}}
                self.save_pixs_object({}, self.obj_template_path.format(0))
            else:
                with gzip.open(self.meta_path, "rb") as f:
                    self.meta_obj = dill.load(f)

            self.amount_parts = self.meta_obj['amount_parts']
            self.current_counter = self.meta_obj['current_counter']
            self.max_part_size = self.meta_obj['max_part_size']
            self.files_name_saved = self.meta_obj['files_name_saved']
            self.hashes_256_used = self.meta_obj['hashes_256_used']

            self.pixs_obj = self.load_pixs_object(self.obj_template_path.format(self.amount_parts[0]))


        def save_meta_object(self):
            assert not isinstance(self.meta_obj, type(None))
            with gzip.open(self.meta_path, "wb") as f:
                dill.dump(self.meta_obj, f)


        def load_pixs_object(self, object_path):
            with gzip.open(object_path, "rb") as f:
                obj = dill.load(f)
            return obj


        def save_pixs_object(self, obj, object_path):
            with gzip.open(object_path, "wb") as f:
                dill.dump(obj, f)


        def load_all_files_name(self, excepted_extensions=['jpg']):
            _, _, files_name = next(os.walk(self.images_path))

            def get_extension(path):
                if not "." in path:
                    return ""
                return path.split(".")[-1]

            self.files_name = list(filter(lambda x: get_extension(x) in excepted_extensions, files_name))


        def save_pixs_to_objects(self):
            self.load_meta_object()

            self.load_all_files_name()
            for idx, file_name in enumerate(self.files_name, 0):
            # for idx, file_name in enumerate(self.files_name[:3800], 0):
                full_img_path = self.images_path+file_name
                
                h_256 = hasher256.hash_file(full_img_path).upper()

                if h_256 in self.hashes_256_used:
                # if file_name in self.files_name_saved:
                    print("Skip file '{}'!".format(full_img_path))
                    continue

                self.hashes_256_used[h_256] = 0
                # self.files_name_saved[file_name] = 0


                img = Image.open(full_img_path)
                pix = np.array(img)

                h, w = pix.shape[:2]
                if len(pix.shape) == 2:
                    pix = np.tile(pix.reshape((-1, )), 3).reshape((3, -1)).T.reshape((h, w, 3))
                elif pix.shape[2] == 4:
                    pix = pix[:, :, :3]

                if h*4 <= w*3: # trim horizontally
                    hn = h
                    wn = int(h*4/3)
                    x = int((w-wn)/2)
                    y = 0
                else:          # trim vertically
                    wn = w
                    hn = int(w*3/4)
                    y = int((h-hn)/2)
                    x = 0

                pixn = pix[y:y+hn, x:x+wn]
                imgn = Image.fromarray(pix)

                if idx % 100 == 0:
                    print("idx: {:6}, file_name: '{}'!".format(idx, file_name))
                    print(" - (w, h): {} -> (wn, hn): {}".format((w, h), (wn, hn)))

                # h_512 = hasher512.hash_file(full_img_path).upper()

                # all_orig_images_dict[file_name] = (pix, h_512)
                imgnr = imgn.resize((200, 150), Image.LANCZOS)
                pixnr = np.array(imgnr)
                # self.pixs_obj[file_name] = (pixnr, h_256)
                self.pixs_obj[h_256] = pixnr
                # self.pixs_obj[file_name] = (pixnr, h_512)
                # self.pixs_obj[file_name] = (pix, h_512)
                self.current_counter[0] += 1

                if self.current_counter[0] % self.max_part_size == 0:
                    self.save_pixs_object(self.pixs_obj, self.obj_template_path.format(self.amount_parts[0]))
                    self.amount_parts[0] += 1
                    self.current_counter[0] = 0
                    self.pixs_obj = {}

            self.save_meta_object()
            self.save_pixs_object(self.pixs_obj, self.obj_template_path.format(self.amount_parts[0]))


    images_path = ROOT_PATH+'images/pixabay_com_2/'

    # images_path_pngs = images_path+'pngs/'
    # if not os.path.exists(images_path_pngs):
    #     os.makedirs(images_path_pngs)

    obj_path = ROOT_PATH+"datas/"
    # obj_path = images_path+"picture_dict_objs/"
    if not os.path.exists(obj_path):
        os.makedirs(obj_path)

    root_path_dir, dir_names, file_names = next(os.walk(images_path), range(0, 1))
    print("root_path_dir: {}".format(root_path_dir))
    print("len(dir_names): {}".format(len(dir_names)))
    print("len(file_names): {}".format(len(file_names)))

    # first save all jpg images in a .pkl.gz object!

    img_obj_conv = ImagesObjectConversion(images_path, obj_path, "pixabay_com_meta_data.pkl.gz", "pixabay_com_obj_data_{:03}.pkl.gz")
    img_obj_conv.save_pixs_to_objects()

    # TODO: create a function/class for creating the mosaic pixses!

    sys.exit(-1)

    path_file_all_orig_images_main_dict = obj_path+"all_orig_images_main_dict.pkl.gz"
    path_file_all_orig_images_dict = obj_path+"all_orig_images_dict.pkl.gz"
    if not os.path.exists(path_file_all_orig_images_dict):
        all_orig_images_dict = {}
    else:
        with gzip.open(path_file_all_orig_images_dict, "rb") as f:
            all_orig_images_dict = dill.load(f)

    file_nr_counter = 0
    for file_nr, file_name in enumerate(file_names[:1000], 0):
        if not "." in file_name:
            continue
        
        extension = file_name.split(".")[-1]
        if extension == "png" or extension != "jpg":
            continue

        img_orig_file_path = root_path_dir+file_name
        
        if file_name in all_orig_images_dict:
            print("File '{}' already in dict!".format(img_orig_file_path))
            continue

        img = Image.open(img_orig_file_path)
        pix = np.array(img)

        h, w = pix.shape[:2]
        if len(pix.shape) == 2:
            pix = np.tile(pix.reshape((-1, )), 3).reshape((3, -1)).T.reshape((h, w, 3))
        elif pix.shape[2] == 4:
            pix = pix[:, :, :3]

        h_512 = hasher512.hash_file(img_orig_file_path).upper()
        all_orig_images_dict[file_name] = (pix, h_512)

        print("file_nr: {}, file_nr_counter: {}".format(file_nr, file_nr_counter))

        file_nr_counter += 1

    print("Saving 'all_orig_images_dict' to '{}'!".format(path_file_all_orig_images_dict))
    with gzip.open(path_file_all_orig_images_dict, "wb") as f:
        dill.dump(all_orig_images_dict, f)

    sys.exit(-1)

    extensions_counter = {}
    shapes_counter = {}
    # channels_counter = {}

    # heights = [300, 150, 75, 3*20, 3*15, 3*10, 3*8, 3*5, 3*3, 3*2]
    # widths = [400, 200, 100, 4*20, 4*15, 4*10, 4*8, 4*5, 4*3, 4*2]
    
    heights = [75, 3*20, 3*15, 3*10, 3*8, 3*5, 3*3, 3*2]
    widths = [100, 4*20, 4*15, 4*10, 4*8, 4*5, 4*3, 4*2]

    paths_sizes_landscape = [(w, h, obj_path+"{w}x{h}.pkl.gz".format(w=w, h=h)) for w, h in zip(widths, heights)]
    paths_sizes_portaire = [(w, h, obj_path+"{w}x{h}.pkl.gz".format(w=w, h=h)) for w, h in zip(heights, widths)]
    # paths_sizes_landscape = [(w, h, images_path_pngs+"{w}x{h}/".format(w=w, h=h), obj_path+"{w}x{h}.pkl.gz".format(w=w, h=h)) for w, h in zip(widths, heights)]
    # paths_sizes_portaire = [(w, h, images_path_pngs+"{w}x{h}/".format(w=w, h=h), obj_path+"{w}x{h}.pkl.gz".format(w=w, h=h)) for w, h in zip(heights, widths)]

    # for _, _, dir_path in (paths_sizes_landscape+paths_sizes_portaire):
    #     if not os.path.exists(dir_path):
    #         os.makedirs(dir_path)

    objs = {}

    for _, _, obj_path in paths_sizes_landscape:
        if not os.path.exists(obj_path):
            print("obj_path '{}' do not exists!".format(obj_path))
            objs[obj_path] = {}            
        else:
            print("obj_path '{}' will be loaded in!".format(obj_path))
            with gzip.open(obj_path, "rb") as f:
                o = dill.load(f)
            objs[obj_path] = o

    for _, _, obj_path in paths_sizes_portaire:
        if not os.path.exists(obj_path):
            print("obj_path '{}' do not exists!".format(obj_path))
            objs[obj_path] = {}            
        else:
            print("obj_path '{}' will be loaded in!".format(obj_path))
            with gzip.open(obj_path, "rb") as f:
                o = dill.load(f)
            objs[obj_path] = o

    first_landscape_obj = objs[paths_sizes_landscape[0][2]]
    first_portaire_obj = objs[paths_sizes_portaire[0][2]]

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

        h_512 = hasher512.hash_file(img_orig_file_path).upper()
        hash_part = h_512[:8]
        new_file_name = "{}.png".format(hash_part)

        if width >= width_1:
            if new_file_name in first_landscape_obj:
                print("skip file '{}' for landscape!".format(new_file_name))
                continue

            x_opt = (width-width_1)//2
            pix = pix[:, x_opt:x_opt+width_1]
            img = Image.fromarray(pix)

            paths_sizes = paths_sizes_landscape
        elif width >= width_2:
            if new_file_name in first_portaire_obj:
                print("skip file '{}' for portaire!".format(new_file_name))
                continue

            x_opt = (width-width_2)//2
            pix = pix[:, x_opt:x_opt+width_2]
            img = Image.fromarray(pix)

            paths_sizes = paths_sizes_portaire
        else:
            continue

        for w, h, obj_path in paths_sizes:
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

    for _, _, obj_path in paths_sizes_landscape:
        print("Saving object '{}'".format(obj_path))
        with gzip.open(obj_path, "wb") as f:
            dill.dump(objs[obj_path], f)

    for _, _, obj_path in paths_sizes_portaire:
        print("Saving object '{}'".format(obj_path))
        with gzip.open(obj_path, "wb") as f:
            dill.dump(objs[obj_path], f)

    print("extensions_counter:\n{}".format(extensions_counter))
    print("shapes_counter:\n{}".format(shapes_counter))
    # print("channels_counter:\n{}".format(channels_counter))
