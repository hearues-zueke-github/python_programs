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

import hashlib
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

class ImagesObjectHashConversion(Exception):
    # TODO: take the first 2 letters of the hash and create 256 object files!
    # TODO: before creating c
    def __init__(self, images_path, data_path, object_file_name_template):
        self.images_path = images_path
        self.data_path = data_path

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        # self.meta_data_file_name = meta_data_file_name
        self.object_file_name_template = object_file_name_template

        # self.meta_path = data_path+meta_data_file_name
        self.obj_template_path = data_path+object_file_name_template

        self.meta_obj = None

        self.parts_amount = 0x20

        for i in range(0, self.parts_amount):
            file_path = self.obj_template_path.format(i)
            if not os.path.exists(file_path):
                self.save_pixs_object({}, file_path)


    def load_all_files_name(self, excepted_extensions=['jpg']):
        _, _, files_name = next(os.walk(self.images_path))

        def get_extension(path):
            if not "." in path:
                return ""
            return path.split(".")[-1]

        self.files_name = list(filter(lambda x: get_extension(x) in excepted_extensions, files_name))
        self.files_name = np.array(self.files_name)


    def load_pixs_object(self, object_path):
        with gzip.open(object_path, "rb") as f:
            obj = dill.load(f)
        return obj


    def save_pixs_object(self, obj, object_path):
        with gzip.open(object_path, "wb") as f:
            dill.dump(obj, f)


    def convert_image_to_smaller_pix(self, full_img_path):
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

        # print("h: {}".format(h))
        # print("w: {}".format(w))
        # print("x: {}".format(x))
        # print("y: {}".format(y))
        # print("wn: {}".format(wn))
        # print("hn: {}".format(hn))

        pixn = pix[y:y+hn, x:x+wn]
        imgn = Image.fromarray(pixn)
        # imgn.show()

        # if idx % 100 == 0:
        #     print("idx: {:6}, file_name: '{}'!".format(idx, file_name))
        #     print(" - (w, h): {} -> (wn, hn): {}".format((w, h), (wn, hn)))

        # h_512 = hasher512.hash_file(full_img_path).upper()

        # all_orig_images_dict[file_name] = (pix, h_512)
        imgnr = imgn.resize((200, 150), Image.LANCZOS)
        pixnr = np.array(imgnr)

        return pixnr


    def get_pixses_rgb_arr(self, files_name):
        pixses_rgb = np.empty((files_name.shape[0], 150, 200, 3), dtype=np.uint8)
        
        for i, file_name in enumerate(files_name, 0):
            full_img_path = self.images_path+file_name
            pix = self.convert_image_to_smaller_pix(full_img_path)
            pixses_rgb[i] = pix
        
        return pixses_rgb


    def save_pixs_to_objects(self):
        self.load_all_files_name()

        files_name = self.files_name
        n = len(files_name)
        part_length = 30000
        n_parts = n//part_length+((n%part_length)>0)

        # for n_p in range(0, n_parts):
        #     print("n_p: {}".format(n_p))
        #     files_name_whole_part = files_name[part_length*n_p:part_length*(n_p+1)]
        for n_p in range(0, 1):
            files_name_whole_part = files_name

            # hashes_sha256 = list(map(lambda file_name: hasher256.hash_file(self.images_path+file_name).upper(), files_name_whole_part))
            print("Calculating hashes of files_name.")
            hashes_sha256 = []
            for idx_num, file_name in enumerate(files_name_whole_part, 0):
                file_name_hash = hashlib.sha256(file_name.encode('utf-8')).hexdigest().upper()
                hashes_sha256.append(int(file_name_hash[:8], 16))
            hashes_sha256 = np.array(hashes_sha256, dtype=np.uint64)
            # hashes_sha256 = np.array([int(x, 16) for x in hashes_sha256], dtype=np.uint64)
            print("Getting 2 digits hesh values.")
            hashes_sha256_2_digits = (hashes_sha256%self.parts_amount).astype(hashes_sha256.dtype)
            # hashes_sha256_2_digits = ((hashes_sha256>>(8*4-7))%(self.parts_amount)).astype(hashes_sha256.dtype)

            print("Do idx_argsort of hashes.")
            idx_argsort = np.argsort(np.vstack((hashes_sha256_2_digits, hashes_sha256)).T.reshape((-1, )).view("u8,u8"))
            # idx_argsort = np.argsort(hashes_sha256)

            files_name_whole_part = files_name_whole_part[idx_argsort]
            hashes_sha256 = hashes_sha256[idx_argsort]
            hashes_sha256_2_digits = hashes_sha256_2_digits[idx_argsort]

            print("Remove duplicates files_name.")
            idxs = np.hstack(((True,), hashes_sha256[:-1]!=hashes_sha256[1:]))

            files_name_whole_part = files_name_whole_part[idxs]
            hashes_sha256 = hashes_sha256[idxs]
            hashes_sha256_2_digits = hashes_sha256_2_digits[idxs]

            # arr = np.array((files_name_whole_part, hashes_sha256, hashes_sha256_2_digits)).T[idx_argsort]
            
            sum_new_pics = 0
            indexes = np.hstack(((0, ), np.where(hashes_sha256_2_digits[:-1]!=hashes_sha256_2_digits[1:])[0]+1, (hashes_sha256_2_digits.shape[0], )))
            for idx1, idx2 in zip(indexes[:-1], indexes[1:]):
                idx_2digits = hashes_sha256_2_digits[idx1]
                print("idx_2digits: {}".format(idx_2digits))
                files_name_part = files_name_whole_part[idx1:idx2]
                hashes_sha256_part = hashes_sha256[idx1:idx2]
                print("files_name_part.shape: {}".format(files_name_part.shape))

                obj = self.load_pixs_object(self.obj_template_path.format(idx_2digits))

                if not 'files_name' in obj:
                    sum_new_pics += files_name_part.shape[0]

                    obj['files_name'] = files_name_part
                    obj['hashes_sha256'] = hashes_sha256_part
                    obj['pixses_rgb'] = self.get_pixses_rgb_arr(files_name_part)
                    self.save_pixs_object(obj, self.obj_template_path.format(idx_2digits))
                    continue

                files_name_all = obj['files_name']
                hashes_sha256_all = obj['hashes_sha256']

                print("Calculating idxs_is_not_found array!")
                idxs_is_not_found = ~np.any(files_name_all.reshape((-1, 1))==files_name_part, axis=0)
                print("Finished Calculating idxs_is_not_found array!")

                if not np.any(idxs_is_not_found):
                    print(" - Only old files_name found in saved files!")
                    continue

                amount_new_pics = np.sum(idxs_is_not_found)
                print(" - amount_new_pics: {}".format(amount_new_pics))
                sum_new_pics += amount_new_pics

                files_name_part_new = files_name_part[idxs_is_not_found]
                hashes_sha256_part_new = hashes_sha256_part[idxs_is_not_found]

                obj['files_name'] = np.hstack((obj['files_name'], files_name_part_new))
                obj['hashes_sha256'] = np.hstack((obj['hashes_sha256'], hashes_sha256_part_new))
                obj['pixses_rgb'] = np.vstack((obj['pixses_rgb'], self.get_pixses_rgb_arr(files_name_part_new)))
                
                print("Available pictures: {}".format(obj['files_name'].shape[0]))
                self.save_pixs_object(obj, self.obj_template_path.format(idx_2digits))

            print(" -- sum_new_pics: {}".format(sum_new_pics))


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
                print("idx: {}, Skip file '{}'!".format(idx, full_img_path))
                continue

            print("Current file: full_img_path: {}".format(full_img_path))

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
            self.hashes_256_used[h_256] = 0
            self.pixs_obj[h_256] = pixnr
            # self.pixs_obj[file_name] = (pixnr, h_512)
            # self.pixs_obj[file_name] = (pix, h_512)
            self.current_counter[0] += 1

            if self.current_counter[0] % self.max_part_size == 0:
                self.save_pixs_object(self.pixs_obj, self.obj_template_path.format(self.amount_parts[0]))
                self.amount_parts[0] += 1
                self.current_counter[0] = 0
                self.pixs_obj = {}
                self.save_pixs_object(self.pixs_obj, self.obj_template_path.format(self.amount_parts[0]))
                self.save_meta_object

        self.save_meta_object()
        self.save_pixs_object(self.pixs_obj, self.obj_template_path.format(self.amount_parts[0]))


if __name__ == "__main__":
    # images_path = ROOT_PATH+'images/pixabay_com_3/'

    # images_path_pngs = images_path+'pngs/'
    # if not os.path.exists(images_path_pngs):
    #     os.makedirs(images_path_pngs)

    # root_path_dir, dir_names, file_names = next(os.walk(images_path), range(0, 1))
    # print("root_path_dir: {}".format(root_path_dir))
    # print("len(dir_names): {}".format(len(dir_names)))
    # print("len(file_names): {}".format(len(file_names)))

    # first save all jpg images in a .pkl.gz object!

    # img_obj_conv = ImagesObjectConversion(images_path, obj_path, "pixabay_com_meta_data.pkl.gz", "pixabay_com_obj_data_{:03}.pkl.gz")
    # img_obj_conv.save_pixs_to_objects()

    print("images_path: {}".format(images_path))
    print("obj_path: {}".format(obj_path))
    # sys.exit(-2)

    def get_all_jpg_files_name(images_path, excepted_extensions=['jpg']):
        _, _, files_name = next(os.walk(images_path))

        def get_extension(path):
            if not "." in path:
                return ""
            return path.split(".")[-1]

        files_name = list(filter(lambda x: get_extension(x) in excepted_extensions, files_name))
        return np.array(files_name)

    # TODO: get all files_name and paths! remove duplicate all file_name!
    images_path_template = ROOT_PATH+'images/pixabay_com{}/'
    root_dirs = [images_path_template.format("")]+[images_path_template.format("_{}".format(i)) for i in range(2, 7)]
    root_dirs = np.array(root_dirs)
    print("root_dirs: {}".format(root_dirs))

    lst_files_name = []
    lst_dirs_num = []
    for i, root_dir in enumerate(root_dirs, 0):
        # _, _, files_name = next(os.walk(root_dir), range(0, 1))
        # print("len(files_name): {}".format(len(files_name)))
        files_name = get_all_jpg_files_name(root_dir)
        dir_num = np.empty((files_name.shape[0]), dtype=np.uint32)
        dir_num[:] = i
        
        lst_files_name.append(files_name)
        lst_dirs_num.append(dir_num)
        # lst_files_name.append(np.vstack((files_name, dir_num)).T)

    files_name = np.hstack(lst_files_name)
    dirs_num = np.hstack(lst_dirs_num)

    print("files_name.shape: {}".format(files_name.shape))

    idxs = np.argsort(files_name)

    files_name = files_name[idxs]
    dirs_num = dirs_num[idxs]

    idxs2 = np.hstack(((True, ), files_name[:-1]!=files_name[1:]))

    files_name = files_name[idxs2]
    dirs_num = dirs_num[idxs2]

    # idxs_random = np.random.permutation(np.arange(0, files_name.shape[0]))

    # files_name_rnd = files_name[idxs_random]
    # dirs_num_rnd = dirs_num[idxs_random]

    files_name_rnd = files_name
    dirs_num_rnd = dirs_num

    print("files_name.shape: {}".format(files_name.shape))

    def convert_image_to_smaller_pix(full_img_path):
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
        imgn = Image.fromarray(pixn)

        imgnr = imgn.resize((60, 45), Image.LANCZOS)
        # imgnr = imgn.resize((200, 150), Image.LANCZOS)
        pixnr = np.array(imgnr)

        return pixnr

    parts_amount = 30
    n = files_name_rnd.shape[0]
    length_part = n//parts_amount+((n%parts_amount)>0)

    # TODO: write a cpp program for faster resizing images!
    for k in range(0, parts_amount):
        files_name_part = files_name[length_part*k:length_part*(k+1)]
        dirs_num_part = dirs_num[length_part*k:length_part*(k+1)]

        pixses_rgb = np.empty((files_name_part.shape[0], 45, 60, 3), dtype=np.uint8)

        print("pixses_rgb.shape: {}".format(pixses_rgb.shape))
        # input("ENTER...")
        for i in range(0, files_name_part.shape[0]):
            print("i: {}".format(i))
            image_path = root_dirs[dirs_num_part[i]]+files_name_part[i]
            pix = convert_image_to_smaller_pix(image_path)
            pixses_rgb[i] = pix

        with gzip.open("datas/all_pixabay_com_pixses_45_60_part_{:02}.pkl.gz".format(k), "wb") as f:
            dill.dump(pixses_rgb, f)

    sys.exit(-2)

    obj_path = ROOT_PATH+"datas/objects_2_hex_digits/"
    # obj_path = images_path+"picture_dict_objs/"
    if not os.path.exists(obj_path):
        os.makedirs(obj_path)
    
    img_obj_hash_conv = ImagesObjectHashConversion(images_path, obj_path, "pixabay_com_2hex_obj_data_{:02X}.pkl.gz")
    img_obj_hash_conv.save_pixs_to_objects()
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
