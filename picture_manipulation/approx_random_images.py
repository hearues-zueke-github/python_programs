#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

# TODO: need to be python convension confirm! (coding standard as much as possible!)

import datetime
import dill
import gzip
import inspect
import os
import pdb
import shutil
import string
import sys

import numpy as np

from dotmap import DotMap
from PIL import Image, ImageFont, ImageDraw

# from . import create_lambda_functions

import create_lambda_functions

path_root_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

sys.path.append(path_root_dir+"/..")
import utils_all
# from .. import utils_all

class BitFieldBWConverter(Exception):
    possible_bits = [1, 8, 24]

    def __init__(self, bits):
        assert isinstance(bits, int)
        assert bits in self.possible_bits
        
        if bits == 1:
            self._convert_bws_to_pix = self._convert_1_bit_field_to_pix
        if bits == 8:
            self._convert_bws_to_pix = self._convert_8_bit_field_to_pix
        if bits == 24:
            self._convert_bws_to_pix = self._convert_24_bit_field_to_pix


    def convert_bws_to_pix(self, bws):
        return self._convert_bws_to_pix(bws)


    def _convert_1_bit_field_to_pix(self, bws):
        assert isinstance(bws, list)
        assert len(bws) == 1

        pix_bw = bws[0]
        pix = np.dstack((pix_bw, pix_bw, pix_bw)).astype(np.uint8)*255
        return pix


    def _convert_8_bit_field_to_pix(self, bws):
        assert isinstance(bws, list)
        assert len(bws) == 8

        pix_bw = np.zeros(bws[0].shape, dtype=np.uint8)
        for i, p in zip(range(7, -1, -1), bws):
            pix_bw += p<<i
        pix = np.dstack((pix_bw, pix_bw, pix_bw)).astype(np.uint8)
        return pix


    def _convert_24_bit_field_to_pix(self, bws):
        assert isinstance(bws, list)
        assert len(bws) == 24

        pix_bw_r = np.zeros(bws[0].shape, dtype=np.uint8)
        for i, p in zip(range(7, -1, -1), bws[:8]):
            pix_bw_r += p<<i
        pix_bw_g = np.zeros(bws[0].shape, dtype=np.uint8)
        for i, p in zip(range(7, -1, -1), bws[8:16]):
            pix_bw_g += p<<i
        pix_bw_b = np.zeros(bws[0].shape, dtype=np.uint8)
        for i, p in zip(range(7, -1, -1), bws[16:24]):
            pix_bw_b += p<<i
        pix = np.dstack((pix_bw_r, pix_bw_g, pix_bw_b)).astype(np.uint8)
        return pix


class BitNeighborManipulation(Exception):
    def __init__(self, ft=2, with_frame=True, path_dir=None, lambda_str_funcs_lst=None):
        self.ft = ft
        if with_frame:
            self.add_frame = self._get_add_frame_function() # function
            self.remove_frame = self._get_remove_frame_function() # function
        else:
            self.add_frame = None
            self.remove_frame = None

        self.max_bit_operators = 4
        self.bit_operators_idx = [0, 1, 2, 3]
        
        self.get_pixs = self._generate_pixs_function() # function
        self.bit_operations = self._generate_lambda_functions(path_dir, lambda_str_funcs_lst) # list of lambdas
        self.it1 = 0 # for the iterator variable (1st)
        self.it2 = 0 # for the iterator variable (2nd)

        self.np = np
        self.lror = np.logical_or.reduce
        self.lrand = np.logical_and.reduce


    def _get_add_frame_function(self):
        ft = self.ft
        def add_frame(pix_bw):
            t = np.vstack((pix_bw[-ft:], pix_bw, pix_bw[:ft]))
            return np.hstack((t[:, -ft:], t, t[:, :ft]))
        return add_frame


    def _generate_pixs_function(self):
        ft = self.ft

        def generate_pixs(pix_bw):
            height, width = pix_bw.shape[:2]

            zero_row = np.zeros((width, ), dtype=np.uint8)
            zero_col = np.zeros((height, 1), dtype=np.uint8)

            move_arr_u = lambda pix_bw: np.vstack((pix_bw[1:], zero_row))
            move_arr_d = lambda pix_bw: np.vstack((zero_row, pix_bw[:-1]))
            move_arr_l = lambda pix_bw: np.hstack((pix_bw[:, 1:], zero_col))
            move_arr_r = lambda pix_bw: np.hstack((zero_col, pix_bw[:, :-1]))

            pixs = np.zeros((ft*2+1, ft*2+1, height, width), dtype=np.uint8)
            pixs[ft, ft] = pix_bw

            # first set all y pixs (center ones)
            for i in range(ft, 0, -1):
                pixs[i-1, ft] = move_arr_u(pixs[i, ft])
            for i in range(ft, ft*2):
                pixs[i+1, ft] = move_arr_d(pixs[i, ft])

            # then set all x pixs (except the center ones, they are already set)
            for j in range(0, ft*2+1):
                for i in range(ft, 0, -1):
                    pixs[j, i-1] = move_arr_l(pixs[j, i])
                for i in range(ft, ft*2):
                    pixs[j, i+1] = move_arr_r(pixs[j, i])

            return pixs

        return generate_pixs


    def _generate_lambda_functions(self, path_dir, lambda_str_funcs_lst):
        if path_dir != None:
            if not os.path.exists(path_dir):
                print("File path '{}' does not exists!".format(path_dir))
                print("Will use default lambda functions then instead!")
                sys.exit(-1)

            with open(path_dir, "r") as fin:
                # lines = fin.readlines()
                lines = list(filter(lambda x: len(x) > 0, fin.read().splitlines()))
        elif lambda_str_funcs_lst != None:
            lines = lambda_str_funcs_lst
        else:
            print("ERROR! No lambda functions can be found!")
            sys.exit(-2)

        # TODO: add a security function, where each line will be checked up
        
        # print("lines:\n{}".format(lines))

        def inv(l):
            return (l+1)%2
        self.__dict__["i"] = inv

        def return_function(def_func_str):
            local = {}
            exec(def_func_str, self.__dict__, local)
            return local["a"]

        # TODO: maybe make an other function for a better splitting up for defs and lambdas!
        # first find 'def' functions and split it up!
        
        print("read lines:\n{}".format(lines))

        lambdas = []
        lambdas_str = []
        def_func = ""
        is_prev_def = False
        for line in lines:
            if "def " in line or (len(line) >= 6 and "lambda" != line[:6]):
                if is_prev_def == False:
                    is_prev_def = True
                    def_func = ""
                def_func += line+"\n"
            else:
                if is_prev_def == True:
                    is_prev_def = False
                    lambdas.append(return_function(def_func))
                    lambdas_str.append(def_func)
                lambdas.append(eval(line, self.__dict__))
                lambdas_str.append(line)
        if is_prev_def == True:
            lambdas.append(return_function(def_func))
            lambdas_str.append(def_func)

        self.max_bit_operators = len(lambdas)
        print("len(lambdas): {}".format(len(lambdas)))
        self.lambdas_str = lambdas_str

        return lambdas


    def _get_remove_frame_function(self):
        ft = self.ft
        def remove_frame(pix_bw):
            return pix_bw[ft:-ft, ft:-ft]
        return remove_frame


    def apply_neighbor_logic_1_bit(self, pix_bw):
        if self.add_frame != None:
            pix_bw = self.add_frame(pix_bw)
        pixs = self.get_pixs(pix_bw)

        ft = self.ft
        self.__dict__['p'] = pixs[ft, ft]
        for y in range(0, ft*2+1):
            for x in range(0, ft*2+1):
                if y == ft and x == ft:
                    continue
                var_name = ("u"*(ft-y) if y < ft else "d"*(y-ft))+("l"*(ft-x) if x < ft else "r"*(x-ft))
                self.__dict__[var_name] = pixs[y, x]

        # TODO: add later more generic combinations and/or functions for apply_neighbor_logic_1_bit !
        pix_bw1 = self.bit_operations[(self.it1+self.it2)%self.max_bit_operators]()

        pix_bw = pix_bw1

        self.it2 += 1
        
        if self.remove_frame != None:
            return self.remove_frame(pix_bw)
        return pix_bw


    # maybe this function can stay as it is! self.it1 is always increasing by 1!
    # self.it2 starts always with 0
    # add other self.iter_vars = [0, 0,...] variables, which can be used in the apply_neighbor_logic_1_bit function!
    # or show that for it1 and it2 every output function could be mapped to some vectors!
    # e.g. (it1, it2) -> (a0, a1, a2,..., an), where every ai value is used for some self defined function!

    def apply_neighbor_logic(self, pix_bws):

        pix_bws_new = []

        self.it2 = 0
        for i, pix_bw in enumerate(pix_bws, 0):
           pix_bws_new.append(self.apply_neighbor_logic_1_bit(pix_bw))
        self.it1 += 1

        return pix_bws_new


# TODO: in main define specific arguments for creating path etc.
def create_bits_neighbour_pictures(dm_params, dm_params_lambda):
    print("Now in def 'create_bits_neighbour_pictures'")

    assert not ( isinstance(dm_params, DotMap) and
                 isinstance(dm_params, list) and
                 isinstance(dm_params, dict) )

    if isinstance(dm_params, list):
        dm_params = dict(dm_params)
    if isinstance(dm_params, dict):
        dm_params = DotMap(dm_params)

    assert dm_params.height
    assert dm_params.width
    assert dm_params.ft
    assert dm_params.next_folder
    assert dm_params.suffix
    assert isinstance(dm_params.with_frame, bool)
    assert dm_params.return_pix_array
    assert isinstance(dm_params.save_pictures, bool)
    assert isinstance(dm_params.functions_str_lst, list)
    assert dm_params.width_append_frame    
    assert isinstance(dm_params.lambdas_in_picture, bool)
    assert dm_params.max_it    
    assert isinstance(dm_params.with_resize_image, bool)
    assert dm_params.resize_factor
    assert dm_params.bits
    assert not (len(dm_params.temp_path_lambda_file) > 0 and len(dm_params.functions_str_lst) > 0)
    assert isinstance(dm_params.image_by_str, str)

    height = dm_params.height
    width = dm_params.width
    ft = dm_params.ft
    next_folder = dm_params.next_folder
    suffix = dm_params.suffix
    with_frame = dm_params.with_frame
    return_pix_array = dm_params.return_pix_array
    save_pictures = dm_params.save_pictures
    functions_str_lst = dm_params.functions_str_lst
    width_append_frame = dm_params.width_append_frame
    lambdas_in_picture = dm_params.lambdas_in_picture
    max_it = dm_params.max_it
    with_resize_image = dm_params.with_resize_image
    resize_factor = dm_params.resize_factor
    bits = dm_params.bits
    temp_path_lambda_file = dm_params.temp_path_lambda_file
    image_by_str = dm_params.image_by_str

    print("lambdas_in_picture: {}".format(lambdas_in_picture))

    np.random.seed()

    # TODO: maybe should be done with os.join ? or something similar?!
    if isinstance(next_folder, str) and (len(next_folder) > 1 and next_folder[:-1] != "/" or len(next_folder) == 0):
        next_folder += "/"

    if len(suffix) > 0 and suffix[0] != "_":
        suffix = "_" + suffix

    print("save_pictures: {}".format(save_pictures))
    # if save_pictures:
    font_name = "712_serif.ttf"
    font_size = 16
    fnt = ImageFont.truetype('../fonts/{}'.format(font_name), font_size)

    char_width, char_height = fnt.getsize("a")
    print("char_width of 'a': {}".format(char_width))
    print("char_height of 'a': {}".format(char_height))

    if save_pictures:
        path_pictures = path_root_dir+"images/{next_folder}changing_bw_1_bit{suffix}/".format(next_folder=next_folder, suffix=suffix)
        print("path_pictures: {}".format(path_pictures))

        if os.path.exists(path_pictures):
            os.system("rm -rf {}".format(path_pictures))
        if not os.path.exists(path_pictures):
            os.makedirs(path_pictures)

        print("path_pictures:\n{}".format(path_pictures))

        dm_params.path_dir = path_pictures

        dm_params_lambda.path_dir = path_pictures
        dm_params_lambda.save_data = save_pictures
    else:
        dm_params.path_dir = None
        dm_params_lambda.path_dir = None
        dm_params_lambda.save_data = False

    if len(functions_str_lst) == 0:
        if len(temp_path_lambda_file) > 0:
            assert os.path.exists(temp_path_lambda_file)

            # dm_params_lambda.path_dir = path_pictures
            # dm_params_lambda.save_data = True
            dm_params_lambda.used_method = "from_temp_path_file"
            bnm = BitNeighborManipulation(path_dir=temp_path_lambda_file)
            dm_params_lambda.functions_str_lst =  bnm.lambdas_str
            print("dm_params_lambda.functions_str_lst:\n{}".format(dm_params_lambda.functions_str_lst))
            print("len(dm_params_lambda.functions_str_lst): {}".format(len(dm_params_lambda.functions_str_lst)))
        else:
            dm_params_lambda = create_lambda_functions.create_lambda_functions_with_matrices(dm_params_lambda)
            dm_params_lambda.used_method = 'create_new_random_lambdas'
            # dm_params_lambda.functions_str_lst = functions_str_lst
    else:
        # dm_params_lambda.path_dir = path_pictures
        # dm_params_lambda.save_data = True
        dm_params_lambda.used_method = "own_defined_lambdas"
        dm_params_lambda.functions_str_lst = functions_str_lst

    # print("len(dm_params_lambda.functions_str_lst): {}".format(len(dm_params_lambda.functions_str_lst)))
    # sys.exit(-2)

    if dm_params_lambda.save_data:
        create_lambda_functions.write_dm_obj_txt(dm_params_lambda)

    functions_str_lst = dm_params_lambda.functions_str_lst

    functions_str_lst_split = []
    # functions_str_lst_split = [l.split("\n")[:-1] if "def" in "{:2}: ".format(i)+l else l ]

    for i, l in enumerate(functions_str_lst, 1):
        if "def" in l:
            l = l.split("\n")[:-1]
            l[0] = "{:2}: ".format(i)+l[0]
            l[1:] = list(map(lambda x: "  / "+x, l[1:]))
            functions_str_lst_split.extend(l)
        else:
            functions_str_lst_split.append("{:2}: ".format(i)+l)

    functions_str_lst = []
    for l in functions_str_lst_split:
        if isinstance(l, list):
            functions_str_lst.extend(l)
        else:
            functions_str_lst.append(l)

    lines_print = []
    for i, line in enumerate(functions_str_lst, 0):
        lines_print.append("{}".format(line))
        # lines_print.append("i: {}, {}".format(i, line))

    # text_sizes = []
    max_chars = 0
    max_width = 0
    max_char_height = 0
    for line in lines_print:
        size = fnt.getsize(line)
        char_height = size[1]
        if max_char_height < char_height:
            max_char_height = char_height

        text_width = size[0]
        if max_width < text_width:
            max_width = text_width

        # text_sizes.append(size)
        length = len(line)
        if max_chars < length:
            max_chars = length

    # for i, text_size in enumerate(text_sizes):
    #     print("i: {}, text_size: {}".format(i, text_size))

    # print("max_chars: {}".format(max_chars))

    pix2 = np.zeros((len(lines_print)*(max_char_height+2)+2, max_width+2, 3), dtype=np.uint8)
    pix2 += 0x40
    img2 = Image.fromarray(pix2)
    d = ImageDraw.Draw(img2)

    for i, line in enumerate(lines_print):
        d.text((1, 1+i*(max_char_height+2)), line, font=fnt, fill=(255, 255, 255))

    if save_pictures:
        img2.save(path_pictures+"lambdas.png")

    # save the lambda functions into as a image!
    pix2_1 = np.array(img2)

    if pix2_1.shape[0] < height:
        diff = height-pix2_1.shape[0]
        h1 = diff//2
        h2 = h1+diff%2
        pix2_1 = np.vstack((
            np.zeros((h1, pix2_1.shape[1], 3), dtype=np.uint8),
            pix2_1,
            np.zeros((h2, pix2_1.shape[1], 3), dtype=np.uint8)
        ))

    print("pix2_1.shape: {}".format(pix2_1.shape))
    # else:
    # dm_params_lambda.path_dir = None
    # dm_params_lambda.save_pictures = False
    # # dm = create_lambda_functions.create_lambda_functions_with_matrices(path_dir=None, save_data=False)
    # create_lambda_functions.create_lambda_functions_with_matrices(dm_params_lambda)
    # # dm = create_lambda_functions.conway_game_of_life_functions(path_dir=None, save_data=False)

    # function_str_lst = create_lambda_functions.conway_game_of_life_functions(path_pictures)
    # function_str_lst = create_lambda_functions.simplest_lambda_functions(path_pictures)
    # function_str_lst = create_lambda_functions.simple_random_lambda_creation(path_lambda_functions_file=path_pictures)

    # function_str_lst = dm.function_str_lst

    # TODO: could be used later for better printing letters on images!
    # char_sizes = []
    # for c in list(string.ascii_letters+"0123456789-_#'+*/()[]{}?!"):
    #     size = fnt.getsize(c)
    #     print("c: {}, fnt.getsize(c): {}".format(c, size))
    #     char_sizes.append(size)

    # print("function_str_lst:\n\n{}".format("\n".join(function_str_lst)))
    # sys.exit(-12342)
    
    print("bits: {}".format(bits))
    bws_to_pix_converter = BitFieldBWConverter(bits=bits)

    if len(image_by_str) > 0:
        dir_random_images = path_root_dir+"images/random_images_by_name/" #h{height}_w{width}/".format(height=height, width=width)
        # dir_random_images = path_root_dir+"images/random_images_by_name/h{height}_w{width}/".format(height=height, width=width)
        if not os.path.exists(dir_random_images):
            os.makedirs(dir_random_images)
        
        if len(image_by_str) > 4 and ".png" != image_by_str[-4:] or len(image_by_str) <= 4:
            image_by_str += ".png"
            dm_params.image_by_str = image_by_str

        path_rnd_img = dir_random_images+image_by_str
        if os.path.exists(path_rnd_img):
            img = Image.open(path_rnd_img)
            pix = np.array(img)
            print("pix.shape: {}".format(pix.shape))
            if len(pix.shape) > 3:
                pix = pix[:, :, :3]
            is_pix_changed = False

            if pix.shape[0] < height:
                pix = np.vstack((pix, np.random.randint(0, 256, (height-pix.shape[0], pix.shape[1], 3), dtype=np.uint8)))
                is_pix_changed = True
            if pix.shape[1] < width:
                pix = np.hstack((pix, np.random.randint(0, 256, (pix.shape[0], width-pix.shape[1], 3), dtype=np.uint8)))
                is_pix_changed = True
                
            if is_pix_changed:
                img = Image.fromarray(pix)
                img.save(path_rnd_img)
            
            if pix.shape[0] > height:
                pix = pix[:height]
            if pix.shape[1] > width:
                pix = pix[:, :width]
        else:
            pix = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(pix)
            img.save(path_rnd_img)
            # channels = [pix[:, :, 0], pix[:, :, 1], pix[:, :, 2]]

        # make a function for this! (or not xD)
        pix_bws = []
        for c in range(0, 3):
            channel = pix[:, :, c]
            for i in range(7, -1, -1):
                pix_bws.append((channel>>i)&0x1)
    else:
        pix_bws = [np.random.randint(0, 2, (height, width), dtype=np.uint8) for _ in range(0, bits)]

    pix_1 = bws_to_pix_converter.convert_bws_to_pix(pix_bws)
    
    color_frame_image = np.array([0x40, 0x20, 0xFF], dtype=np.uint8)
    # color_frame_image = np.array([0x40, 0x80, 0xFF], dtype=np.uint8)

    def add_left_right_frame(pix, color, width):
        field = np.zeros((pix.shape[0], width, 3), dtype=np.uint8)+color
        return np.hstack((
            field.copy(),
            pix,
            field
        ))

    # if lambdas_in_picture:
    if pix_1.shape[0] < pix2_1.shape[0]:
        diff = pix2_1.shape[0]-pix_1.shape[0]
        h1 = diff//2
        h2 = h1+diff%2
        pix1_1 = np.vstack((
            np.zeros((h1, pix_1.shape[1], 3), dtype=np.uint8)+color_frame_image,
            pix_1,
            np.zeros((h2, pix_1.shape[1], 3), dtype=np.uint8)+color_frame_image,
        ))
    else:
        pix1_1 = pix_1

    pix1_2 = add_left_right_frame(pix1_1, color_frame_image, width_append_frame)

    pix_combined = np.hstack((pix2_1, pix1_2))
    # else:

    pixs = [pix_1]
    pixs_combined = [pix_combined]

    if save_pictures:
        if lambdas_in_picture:
            pix_used = pix_combined
        else:
            pix_used = pix_1
        
        img = Image.fromarray(pix_used)
        if with_resize_image:
            img = img.resize((width*resize_factor, height*resize_factor))
            
        img.save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, 0))
        
    # Image.fromarray(pix_bw*255).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, 0))

    # with_frame = False
    
    print("TEST!!!")
    print("dm_params_lambda.functions_str_lst: {}".format(dm_params_lambda.functions_str_lst))
    print("TEST!!!")
    if save_pictures:
        if len(temp_path_lambda_file) > 0:
            # copy the temp_path_lambda_file as path_pictures+'lambdas.txt' file!
            shutil.copy(temp_path_lambda_file, path_pictures+'lambdas.txt')
        bit_neighbor_manipulation = BitNeighborManipulation(ft=ft, with_frame=with_frame, path_dir=path_pictures+"lambdas.txt")
    else:
        bit_neighbor_manipulation = BitNeighborManipulation(ft=ft, with_frame=with_frame, lambda_str_funcs_lst=dm_params_lambda.functions_str_lst)

    bit_neighbor_manipulation.bws_to_pix_converter = bws_to_pix_converter

    # so long there are white pixels, repeat the elimination_process!
    # it_max = 100
    # it_max = height
    it = 1
    # pix_bw_prev = pix_bw.copy()
    # pixs = [pix_bw.copy()]
    # repeat anything until it is complete blank / black / 0
    while it < max_it:
    # while np.sum(pix_bw == 1) > 0 and it < it_max:
        print("it: {}".format(it))
        
        # TODO: need to be fixed!

        pix_bws = bit_neighbor_manipulation.apply_neighbor_logic(pix_bws)
        pix_1 = bws_to_pix_converter.convert_bws_to_pix(pix_bws)
        # print("type(pix_bw): {}".format(type(pix_bw)))
        # pix_bw = apply_neighbour_logic(pix_bw)

        # pix_1 = np.dstack((pix_bw, pix_bw, pix_bw)).astype(np.uint8)*255

        is_already_available = False
        for pix in pixs:
            if np.all(pix==pix_1):
                is_already_available = True
                break

        if is_already_available:
            print("WAS ALREADY FOUND ONCE AT LEAST!")
            break

        # if lambdas_in_picture:
        # add vertical color_frame_image too!
        if pix_1.shape[0] < pix2_1.shape[0]:
            diff = pix2_1.shape[0]-pix_1.shape[0]
            h1 = diff//2
            h2 = h1+diff%2
            pix1_1 = np.vstack((
                np.zeros((h1, pix_1.shape[1], 3), dtype=np.uint8)+color_frame_image,
                pix_1,
                np.zeros((h2, pix_1.shape[1], 3), dtype=np.uint8)+color_frame_image,
            ))
        else:
            pix1_1 = pix_1
        
        pix1_2 = add_left_right_frame(pix1_1, color_frame_image, width_append_frame)
        pix_combined = np.hstack((pix2_1, pix1_2))
        # else:

        pixs.append(pix_1)
        pixs_combined.append(pix_combined)

        if save_pictures:
            if lambdas_in_picture:
                pix_used = pix_combined
            else:
                pix_used = pix_1
            
            img = Image.fromarray(pix_used)
            if with_resize_image:
                img = img.resize((width*resize_factor, height*resize_factor))

            img.save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, it))

        it += 1
        
    if save_pictures:
        print("new path_pictures: {path_pictures}".format(path_pictures=path_pictures))
        command = "convert -delay 10 -loop 0 {path_pictures}rnd_*.png {path_pictures}myimage.gif".format(path_pictures=path_pictures)
        print("command:\n{}".format(command))
        os.system(command)
        
        path_gifs = path_root_dir+"images/{next_folder}animations/".format(next_folder=next_folder)
        if not os.path.exists(path_gifs):
            os.makedirs(path_gifs)

        command2 = "cp {path_pictures}myimage.gif {path_gifs}1bit{suffix}.gif".format(
            path_pictures=path_pictures, path_gifs=path_gifs, suffix=suffix)
        print("command2: {}".format(command2))
        os.system(command2)

    print("\nPrinting the lambda functions (lines)!")
    for line in lines_print:
        print("{}".format(line))

    dm_params.it = it # save the last it iterator

    # save bit_neighbor_manipulation, pixs and pixs_combined as .pkl.gz objects tooo! (if save_pictures == True)
    if save_pictures:
        with gzip.open(path_pictures+'bit_neighbor_manipulation.pkl.gz', 'wb') as fout:
            dill.dump(bit_neighbor_manipulation, fout)

        with gzip.open(path_pictures+'pixs.pkl.gz', 'wb') as fout:
            dill.dump(pixs, fout)

        with gzip.open(path_pictures+'pixs_combined.pkl.gz', 'wb') as fout:
            dill.dump(pixs, fout)

    if return_pix_array:
        return pixs, pixs_combined, dm_params, dm_params_lambda

    return path_pictures, dm_params, dm_params_lambda


def combine_images_from_folders(paths_pictures):
    suffix = "{}_{}".format(
        utils_all.get_date_time_str_full(),
        utils_all.get_random_str_base_16(4)
    )

    path_pictures_combined = "images/combined_changing_bw_1_bit_{}/".format(suffix)
    if not os.path.exists(path_pictures_combined):
        os.system("rm -rf {}".format(path_pictures_combined))
    if not os.path.exists(path_pictures_combined):
        os.makedirs(path_pictures_combined)

    png_file_paths_lst = []
    for path_pictures in paths_pictures:
        png_file_paths = []
        for root_dir, dirs, files in os.walk(path_pictures):
            for file_name in files:
                if not ".png" in file_name:
                    continue
                png_file_paths.append(root_dir+file_name)
            break

        png_file_paths_lst.append(sorted(png_file_paths))
        print("png_file_paths:\n{}".format(png_file_paths))

        # break

    color_frame_left = np.array([0x80, 0x40, 0xE0], dtype=np.uint8)
    color_frame_top = np.array([0x70, 0x20, 0xE8], dtype=np.uint8)
    color_frame_bottom = np.array([0x90, 0x10, 0xF0], dtype=np.uint8)
    frame_top_bottom = 10

    for i in range(0, len(png_file_paths_lst)):
        print("i: {}".format(i))
        pixs =[np.array(Image.open(paths[i])) for paths in png_file_paths_lst]
        max_width = 0
        for pix in pixs:
            width = pix.shape[1]
            if max_width < width:
                max_width = width
        print("max_width: {}".format(max_width))
        pixs = [pix if pix.shape[1] == max_width else 
            np.vstack((
                np.zeros((
                    frame_top_bottom,
                    max_width,
                    pix.shape[2]
                ), dtype=np.uint8)+color_frame_top,
                # pix,
                np.hstack((
                    np.zeros((
                        pix.shape[0],
                        max_width-pix.shape[1],
                        pix.shape[2]
                    ), dtype=np.uint8)+color_frame_left,
                    pix,
                )),
                np.zeros((
                    frame_top_bottom,
                    max_width,
                    pix.shape[2]
                ), dtype=np.uint8)+color_frame_bottom
            )) for pix in pixs]

        pix_comb = np.vstack(pixs)

        Image.fromarray(pix_comb).save(path_pictures_combined+"comb_pic_nr_{:05}.png".format(i))

    os.system("convert -delay 10 -loop 0 ./{}/*.png ./{}/myimage.gif".format(path_pictures_combined, path_pictures_combined))

    return path_pictures_combined


def create_1_bit_neighbour_pictures_only_good_ones(height, width, amount=10):
    def generate_folder():
        suffix = "{}_{}_{}_{}".format(
            height,
            width,
            utils_all.get_date_time_str_full(),
            utils_all.get_random_str_base_16(4)
        )
        path_pictures = "images/automaton_1_lambda_function/good_1_bit_automaton_{}/".format(suffix)

        if not os.path.exists(path_pictures):
            os.system("rm -rf {}".format(path_pictures))
        if not os.path.exists(path_pictures):
            os.makedirs(path_pictures)

        return suffix, path_pictures

    font_name = "712_serif.ttf"
    font_size = 16

    fnt = ImageFont.truetype('../fonts/{}'.format(font_name), font_size)

    it_amount = 0
    while it_amount < amount:
        pix_bw = np.random.randint(0, 2, (height, width), dtype=np.uint8)
        # Image.fromarray(pix_bw*255).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, 0))
        
        # path_lambda_functions_file = path_pictures+"lambdas.txt"
        function_str_lst = create_lambda_functions.simple_random_lambda_creation() # path_lambda_functions_file=path_lambda_functions_file)

        print("function_str_lst:\n\n{}".format("\n".join(function_str_lst)))
        sys.exit(-12342)

        pix2 = np.zeros((height, 600, 3), dtype=np.uint8)
        img2 = Image.fromarray(pix2)
        d = ImageDraw.Draw(img2)

        for i, line in enumerate(function_str_lst):
            d.text((8, 8+i*(font_size+2)), "i: {}, {}".format(i, line), font=fnt, fill=(255, 255, 255))

        pix2_1 = np.array(img2)

        # path_test_font_images = "images/font_images/"
        # if not os.path.exists(path_test_font_images):
        #     os.makedirs(path_test_font_images)

        # font_names = [
        #     "Graph-35-pix.ttf",

        # #     "novem___.ttf",
        # #     "monofonto.ttf",
        # #     "Monospace.ttf",
        # #     "712_serif.ttf",
        # #     "BPdotsSquareBold.otf",
        # #     "5X5-B___.TTF",
        # #     "origa___.ttf",
        # #     "origap__.ttf",
        # ]
        # for font_name in font_names:
        #     print("font_name: {}".format(font_name))
        #     pix2 = np.zeros((700, 1600, 3), dtype=np.uint8)
        #     img2 = Image.fromarray(pix2)
        #     d = ImageDraw.Draw(img2)

        #     y_pos = 5
        #     # for y in range(16, 65, 16):
        #     # for y in range(4, 65, 1):
        #     for y in range(8, 65, 8):
        #         fnt = ImageFont.truetype('../fonts/{}'.format(font_name), y)
        #         d.text((8, y_pos), "size: {}, ".format(y)+string.ascii_letters+"0123456789-_?!=()|&", font=fnt, fill=(255, 255, 255))

        #         y_pos += y+1

        #     pix2_new = np.array(img2)

        #     img2.save(path_test_font_images+"test_font_{}.png".format(font_name))

        #     pix2_remove_not_white = np.array(img2)
        #     idx_not_white = np.all(pix2_remove_not_white != 255, axis=2)
        #     pix2_remove_not_white[idx_not_white] = 0

        #     Image.fromarray(pix2_remove_not_white).save(path_test_font_images+"test_font_{}_remove_not_white.png".format(font_name))

        #     # pix2_complement = np.any(pix2 > 0, axis=2)^np.any(pix2_remove_not_white > 0, axis=2)
        #     pix2_complement = pix2_new+0
        #     idx_complement = ~(np.any(pix2_complement < 255, axis=2))
        #     pix2_complement[idx_complement] = 0
        #     idx = np.any(pix2_complement > 0, axis=2)
        #     pix2_complement[idx] = 255


        #     globals()["idx_complement"] = idx_complement

        #     Image.fromarray(pix2_complement).save(path_test_font_images+"test_font_{}_complement.png".format(font_name))

        # sys.exit(-5)

        # with_frame = False
        with_frame = True
        bit_neighbor_manipulation = BitNeighborManipulation(ft=1, with_frame=with_frame, lambda_str_funcs_lst=function_str_lst)

        # so long there are white pixels, repeat the elimination_process!
        it_max = 2
        it = 1
        pix_bw_prev = pix_bw.copy()
        pixs = [pix_bw.copy()]
        # repeat anything until it is complete blank / black / 0
        while np.sum(pix_bw == 1) > 0 and it < it_max:
            # print("it: {}".format(it))
            
            # TODO: need to be fixed!
            pix_bw = bit_neighbor_manipulation.apply_neighbor_logic_1_bit(pix_bw)
            # pix_bw = apply_neighbour_logic(pix_bw)

            # check if pix_bw is equal to one of previous one!
            is_prev_equal = False
            for pix in pixs:
                if np.all(pix_bw==pix):
                    print("Previous one was equal to pix_bw!!!")
                    is_prev_equal = True
                    break

            if is_prev_equal:
                break

            # Image.fromarray(pix_bw*255).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, it))
            it += 1
            
            if np.any(pix_bw!=pix_bw_prev) == False:
                break

            pixs.append(pix_bw_prev)
            pix_bw_prev = pix_bw.copy()

        print("len(pixs): {}".format(len(pixs)))

        # if np.sum(pix_bw==1) > 0:
        if np.sum(pix_bw==1) > height:
            print("Is not completly black image!")
            it = it_max

        if it < it_max:
            pixs.append(pix_bw)
            
            print("It worked!")
            # create the images and the gif too!
            suffix, path_pictures = generate_folder()

            with open(path_pictures+"lambdas.txt", "w") as fout:
                for line in function_str_lst:
                    fout.write(line+"\n")

            for i, pix in enumerate(pixs):
                pix_1 = np.dstack((pix, pix, pix)).astype(np.uint8)*255
                Image.fromarray(np.hstack((pix2_1, pix_1))).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, i))

            print("Create the gif image!")
            os.system("convert -delay 5 -loop 0 ./{}/*.png ./{}/myimage.gif".format(path_pictures, path_pictures))

            path_gifs = "images/animations/"
            if not os.path.exists(path_gifs):
                os.makedirs(path_gifs)

            # TODO: maybe also create a image map of each individual lambda funcstions, how the calculation was done!
            # TODO: add the function str in each picture too!!
            shutil.copy("{}/myimage.gif".format(path_pictures), path_gifs+"1bit_{}.gif".format(suffix))

            it_amount += 1
        else:
            print("Mhhh, maybe too many iterations?!")



def create_1_byte_neighbour_pictures(height, width):
    path_pictures = "images/changing_bw_1_byte_{}_{}/".format(height, width)
    if not os.path.exists(path_pictures):
        os.system("rm -rf {}".format(path_pictures))
    if not os.path.exists(path_pictures):
        os.makedirs(path_pictures)

    get_pix_bw = lambda: np.random.randint(0, 2, (height, width), dtype=np.uint8)
    pix_bws = [get_pix_bw() for _ in range(0, 8)]

    def combine_1_bit_neighbours(pix_bws):
        pix = np.zeros(pix_bws[0].shape, dtype=np.uint8)
        for i, pix_bw in enumerate(pix_bws):
            pix += pix_bw<<i
        return pix

    pix_combine = combine_1_bit_neighbours(pix_bws)
    Image.fromarray(pix_combine).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, 0))

    # so long there are white pixels, repeat the elimination_process!
    it = 1
    # repeat anything until it is complete blank / black / 0
    while np.sum([np.sum(pix_bw == 1) for pix_bw in pix_bws]) > 0:
        print("it: {}".format(it))
        
        for i in range(0, 8):
            # TODO: need to be fixed!
            pix_bws[i] = apply_neighbour_logic(pix_bws[i])

        pix_combine = combine_1_bit_neighbours(pix_bws)
        Image.fromarray(pix_combine).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, it))
        it += 1


def create_3_byte_neighbour_pictures(img_type,
    height=None, width=None, same_image=None, with_frame=None,
    image_path=None, max_iterations=-1, path_dir=None,
    resize_params=None, ft=2, num_copies_first_image=3,
    amount_combines=1, gif_delay=5, fps_movie=20, folder_suffix="",
    height_resize=None, width_resize=None):
    prev_folder = os.getcwd()

    get_pix_bws_from_pix_img = lambda pix_img: [(pix_c>>j)&0x1 for pix_c in [pix_img[:, :, i] for i in range(0, 3)] for j in range(0, 8)]
    
    path_suffix = ("" if folder_suffix == "" else "_"+folder_suffix)

    if img_type == "picture":
        if image_path == None:
            sys.exit(-1)

        if not os.path.exists(image_path):
            print("Path to image '{}' does not exists!".format(image_path))
            return -1

        img = Image.open(image_path)
        pix_img = np.array(img)
        height, width = pix_img.shape[:2]

        path_pictures = "images/changing_image_{}_{}{}/".format(height, width, path_suffix)
        
    elif img_type == "random":
        if height == None or \
           width == None or \
           same_image == None or \
           with_frame == None:
            system.exit(-1)

        path_pictures = "images/changing_bw_3_byte_{}_{}{}/".format(height, width, path_suffix)
    
        orig_file_path = "images/orig_image_{}_{}.png".format(height, width)
        if same_image and os.path.exists(orig_file_path):
            pix_img = np.array(Image.open(orig_file_path))
        else:
            pix_img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            Image.fromarray(pix_img).save(orig_file_path)

    pix_bws = get_pix_bws_from_pix_img(pix_img)
   
    if os.path.exists(path_pictures):
        os.system("rm -rf {}*".format(path_pictures))
    else:
        os.makedirs(path_pictures)
    
    path_animations = "images/animations/"
    path_movies = "images/movies/"
    if not os.path.exists(path_animations):
        os.makedirs(path_animations)
    if not os.path.exists(path_movies):
        os.makedirs(path_movies)
    

    # TODO: use this function for combining a RGB image! also modify it for using 8 bits and 1 bits images too!
    # or maybe make separate functions for the other functions!
    def combine_1_byte_neighbours(pix_bws):
        def combine_1_bit_neighbours(pix_bws):
            pix = np.zeros(pix_bws[0].shape, dtype=np.uint8)
            for i, pix_bw in enumerate(pix_bws):
                pix += pix_bw<<i
            return pix

        pix_bw_channels = [combine_1_bit_neighbours(pix_bws[8*i:8*(i+1)]) for i in range(0, 3)]
        pix = np.zeros(pix_bw_channels[0].shape+(3, ), dtype=np.uint8)
        for i, pix_bw_c in enumerate(pix_bw_channels):
            pix[:, :, i] = pix_bw_c
        return pix

    pix_combine = combine_1_byte_neighbours(pix_bws)
    pix_combines = [pix_combine]

    bit_neighbor_manipulation = BitNeighborManipulation(ft=ft, with_frame=with_frame, path_dir=path_dir)

    # so long there are white pixels, repeat the elimination_process!
    # TODO: or add an termination point too!
    it = 1
    # repeat anything until it is complete blank / black / 0
    while np.sum([np.sum(pix_bw == 1) for pix_bw in pix_bws]) > 0 and (it < max_iterations if max_iterations > 0 else True):
        if it%10 == 0:
            print("it: {}".format(it))
        
        pix_bws = bit_neighbor_manipulation.apply_neighbor_logic(pix_bws)

        pix_combine = combine_1_byte_neighbours(pix_bws)
        pix_combines.append(pix_combine)
        it += 1

    # now take each image and interpolate between each image e.g. 10 samples
    def get_pix_between(pix_1, pix_2, alpha=0.5):
        return (pix_1.astype(np.float)*alpha+pix_2.astype(np.float)*(1.-alpha)).astype(np.uint8)
    
    arr_pix_combines = np.array(pix_combines)
    path_template = path_pictures+"rnd_{}_{}_bw_i_{{:03}}_{{:02}}.png".format(height, width)
    for i, (pix_1, pix_2) in enumerate(zip(pix_combines[:-1], pix_combines[1:])):
        Image.fromarray(pix_1).save(path_template.format(i, 0))
        for j in range(1, amount_combines):
            print("i: {}, j: {}".format(i, j))
            Image.fromarray(get_pix_between(pix_1, pix_2, float(amount_combines-j)/amount_combines)).save(path_template.format(i, j))

    Image.fromarray(arr_pix_combines[-1]).save(path_template.format(arr_pix_combines.shape[0]-1, 0))

    os.chdir("./{}".format(path_pictures))

    if height_resize != None and width_resize != None:
     if img_type == "random":
        for root_dir, dirs, files in os.walk("."):
            if not root_dir == ".":
                continue

            for file_name in files:
                if not ".png" in file_name or file_name == "orig_image.png":
                    print("continue: file_name: {}".format(file_name))
                    continue
                print("Resize, convert and reduce quality for file: '{}'".format(file_name))
                os.system("convert {} -filter Point -resize {}x{} +antialias {}".format(file_name, height_resize, width_resize, file_name))
    
    for root_dir, dirs, files in os.walk("."):
        if not root_dir == ".":
            continue

        arr = np.sort(np.array(files))
        file_num = 0
        for file_name in arr:
            if not ".png" in file_name or file_name == "orig_image.png":
                continue

            if file_num == 0:
                for _ in range(0, num_copies_first_image):
                    os.system("cp {} pic_{:04d}.png".format(file_name, file_num))
                    file_num += 1
            os.system("mv {} pic_{:04d}.png".format(file_name, file_num))
            file_num += 1

    random_64_bit_num = utils_all.get_random_str_base_64(4)
    suffix_temp = "_{}_{{}}_{{}}_{}".format(img_type, random_64_bit_num)
    suffix = suffix_temp.format(height, width)

    # suffix = "_{}_{}_{}_{}_{}".format(img_type, height, width, (lambda x: "-".join(list(map(str, x.bit_operators_idx[:x.max_bit_operators]))))(bit_neighbor_manipulation), random_64_bit_num)
    print("Create an animation (gif) with png's and suffix '{}'!".format(suffix))
    os.system("convert -delay {} -loop 0 *.png ../../{}animated{}.gif".format(gif_delay, path_animations, suffix))
    print("Create an animation (mp4) with png's and suffix '{}'!".format(suffix))
    os.system("ffmpeg -r {} -i pic_%04d.png -vcodec mpeg4 -y ../../{}movie{}.mp4".format(fps_movie, path_movies, suffix))

    os.chdir(prev_folder)
    if resize_params != None:
        os.chdir(path_animations)

        new_height = height-resize_params[0]-resize_params[1]
        new_width = width-resize_params[2]-resize_params[3]
        
        orig_animated_file_name = "animated{}.gif".format(suffix_temp.format(height, width))
        modif_animated_file_name = "animated{}.gif".format(suffix_temp.format(new_height, new_width))
        print("Now crop the image! (only when needed!)")
        os.system("convert {} -coalesce -repage 0x0 -crop {}x{}+{}+{} +repage {}".format(
            orig_animated_file_name,
            # new_width, new_height, resize_params[2], resize_params[0],
            new_width, new_height, resize_params[3], resize_params[1],
            modif_animated_file_name))


def get_special_functions_str_lst(name):
    functions_str_lst_dict = {
        'conway_game_of_life': [
          ( 'def a():\n'+
            '    x = u+d+r+l+ur+ul+dr\n'+
            '    t1 = np.logical_or.reduce((x==2, x==4, x==5))\n'+
            '    p1 = np.logical_and.reduce((p==1, t1))\n'+
            '    p2 = np.logical_and.reduce((p==0, x==3))\n'+
            '    return np.logical_or.reduce((p1, p2)).astype(np.uint8)' ),
        ]
    }

    if not name in functions_str_lst_dict:
        print("Name '{name}' in dict of functions not found!".format(name=name))
        return []

    return functions_str_lst_dict[name]


def print_variables_content(variables):
    print(" - ft: {}".format(variables.ft))
    print(" - file_name_dm: {}".format(variables.file_name_dm))
    print(" - file_name_txt: {}".format(variables.file_name_txt))
    print(" - min_or: {}".format(variables.min_or))
    print(" - max_or: {}".format(variables.max_or))
    print(" - min_and: {}".format(variables.min_and))
    print(" - max_and: {}".format(variables.max_and))
    print(" - min_n: {}".format(variables.min_n))
    print(" - max_n: {}".format(variables.max_n))

    print(" - height: {}".format(variables.height))
    print(" - width: {}".format(variables.width))
    print(" - next_folder: {}".format(variables.next_folder))
    print(" - with_frame: {}".format(variables.with_frame))
    print(" - width_append_frame: {}".format(variables.width_append_frame))
    print(" - return_pix_array: {}".format(variables.return_pix_array))
    print(" - save_pictures: {}".format(variables.save_pictures))
    print(" - functions_str_lst: {}".format(variables.functions_str_lst))
    print(" - with_resize_image: {}".format(variables.with_resize_image))
    print(" - resize_factor: {}".format(variables.resize_factor))
    print(" - func_by_name: {}".format(variables.func_by_name))
    print(" - lambdas_in_picture: {}".format(variables.lambdas_in_picture))
    print(" - max_it: {}".format(variables.max_it))
    print(" - bits: {}".format(variables.bits))
    print(" - temp_path_lambda_file: {}".format(variables.temp_path_lambda_file))
    print(" - image_by_str: {}".format(variables.image_by_str))


def get_default_variables():
    variables = DotMap()

    variables.file_name_dm = "dm.pkl.gz"
    variables.file_name_txt = "lambdas.txt"
    variables.min_or = 4
    variables.max_or = 4
    variables.min_and = 3
    variables.max_and = 3
    variables.min_n = 2
    variables.max_n = 2

    variables.height = 64
    variables.width = variables.height

    variables.ft = 1

    dt = datetime.datetime.now()
    variables.next_folder = "{year:04}_{month:02}_{day:02}".format(year=dt.year, month=dt.month, day=dt.day)

    variables.with_frame = True
    variables.width_append_frame = 5

    variables.return_pix_array = True
    variables.save_pictures = True

    variables.functions_str_lst = []

    variables.with_resize_image = False
    variables.resize_factor = 3
    variables.bits = 1
    variables.func_by_name = ''
    variables.lambdas_in_picture = True
    variables.max_it = 100

    variables.height_resize = variables.height * variables.resize_factor
    variables.width_resize = variables.width * variables.resize_factor

    variables.temp_path_lambda_file = ''
    variables.image_by_str = ''

    return variables


def parse_argv_to_variables(argv, variables):
    using_vars_type = {
        'ft': int,
        'file_name_dm': str,
        'file_name_txt': str,
        'min_and': int,
        'max_and': int,
        'min_or': int,
        'max_or': int,
        'min_n': int,
        'max_n': int,

        'height': int,
        'width': int,
        'next_folder': str,
        'with_frame': bool,
        'width_append_frame': int,
        'return_pix_array': bool,
        'save_pictures': bool,
        'functions_str_lst': list,
        'with_resize_image': bool,
        'resize_factor': int,
        'func_by_name': str,
        'lambdas_in_picture': bool,
        'max_it': int,
        'bits': int,
        'temp_path_lambda_file': str,
        'image_by_str': str,
    }

    value_str = ""
    if len(argv) > 1:
        value_str = ",".join(argv[1:]).lstrip(",").rstrip(",")
    print("value_str: {}".format(value_str))
    value_str_split = value_str.split(",")
    value_str_split = list(filter(lambda x: x.count("=") == 1, value_str_split))
    var_val_lst = list(map(lambda x: x.split("="), value_str_split))
    print("var_val_lst:\n{}".format(var_val_lst))

    loc_dir = {'variables': variables}

    # Do the check and convert the variable for the given input!
    for var, val in var_val_lst:
        if var in using_vars_type:
            try:
                type_var = using_vars_type[var]
                val_converted = type_var(val)
                if type_var == int:
                    exec("variables.{var} = {val}".format(var=var, val=val_converted), {}, loc_dir)
                elif type_var == str:
                    if var == 'func_by_name':
                        functions_str_lst = get_special_functions_str_lst(val)
                    else:
                        exec("variables.{var} = '{val}'".format(var=var, val=val_converted), {}, loc_dir)
                elif type_var == bool:
                    if val == "0" or val == "False" or val == "false":
                        exec('variables.{var} = False'.format(var=var))
                    elif val == "1" or val == "True" or val == "true":
                        exec('variables.{var} = True'.format(var=var))
                    else:
                        raise
                elif type_var == list:
                    pass
                print("var: {}, val: {}, type_var: {}, val_converted: {}".format(
                    var, val, type_var, val_converted))
            except:
                print("For var '{var}' could not convert to type '{type_var}' of val '{val}'!".format(
                    var=var, type_var=type_var, val=val))


def get_dm_params_lambda(variables):
    dm_params_lambda = DotMap()

    dm_params_lambda.ft = variables.ft
    dm_params_lambda.path_dir = None
    dm_params_lambda.save_data = False
    dm_params_lambda.file_name_dm = variables.file_name_dm
    dm_params_lambda.file_name_txt = variables.file_name_txt
    dm_params_lambda.min_or = variables.min_or
    dm_params_lambda.max_or = variables.max_or
    dm_params_lambda.min_and = variables.min_and
    dm_params_lambda.max_and = variables.max_and
    dm_params_lambda.min_n = variables.min_n
    dm_params_lambda.max_n = variables.max_n

    return dm_params_lambda


def get_dm_params(variables):
    dm_params = DotMap()

    dm_params.height = variables.height
    dm_params.width = variables.width
    dm_params.ft = variables.ft
    dm_params.next_folder = variables.next_folder
    dm_params.suffix = variables.suffix
    dm_params.with_frame = variables.with_frame
    dm_params.return_pix_array = variables.return_pix_array
    dm_params.save_pictures = variables.save_pictures
    dm_params.functions_str_lst = variables.functions_str_lst
    dm_params.width_append_frame = variables.width_append_frame
    dm_params.with_resize_image = variables.with_resize_image
    dm_params.resize_factor = variables.resize_factor
    dm_params.lambdas_in_picture = variables.lambdas_in_picture
    dm_params.max_it = variables.max_it
    dm_params.bits = variables.bits
    dm_params.temp_path_lambda_file = variables.temp_path_lambda_file
    dm_params.image_by_str = variables.image_by_str

    return dm_params


def main(argv):
    print("Now in 'main' of 'approx_random_images.py'")

    print("argv: {}".format(argv))

    path_dir = None
    save_data = False

    variables = get_default_variables()

    variables.functions_str_lst = [
        # "lambda: u",
        # "lambda: ul",
        # "lambda: l",
        # "lambda: dl",
        # "lambda: d",
        # "lambda: dr",
        # "lambda: r",
        # "lambda: ur",

        # 'lambda: (u&ul|ur&r|d&dl|dr&r)&p|(i(l)&ul|dr&i(d))&i(p)', #|r&l&i(p)',

        # 'lambda: u&dr&p|ur&l&i(p)',
        # 'lambda: u&dr&p|ur&l&i(p)|u&d&p|r&l&i(p)',

        # "lambda: u&d&p|r&d&i(p)|p&ur&dl",
        # "lambda: u&d&r&p|r&d&i(p)|p&ur&dl|ul&ur&i(p)",
        # "lambda: u&d&l&p|r&d&i(p)|p&ur&dl&l|l&i(r)&p",
    ]

    print("locals():\n{}".format(locals()))

    print("Values for variables before input:")
    print_variables_content(variables)

    parse_argv_to_variables(argv, variables)

    print("Values for variables after input:")
    print_variables_content(variables)

    variables.suffix = "h{}_w{}_{}_{}".format(
        variables.height,
        variables.width,
        utils_all.get_date_time_str_full_short(),
        utils_all.get_random_str_base_16(4)
    )

    dm_params_lambda = get_dm_params_lambda(variables)
    dm_params = get_dm_params(variables)

    # print("FINISHED EARLIER!!!")
    # sys.exit(-1)

    print("dm_params.functions_str_lst: {}".format(dm_params.functions_str_lst))
    
    # example calling function:
    # TODO: 
    # ../venv/bin/python3.6 approx_random_images.py lambdas_in_picture=False,with_resize_image=False,bits=1,min_or=2,max_or=2,width=200,height=200,temp_path_lambda_file=lambdas.txt
    # sys.exit(-1)
    
    returns = create_bits_neighbour_pictures(dm_params, dm_params_lambda)
    if len(returns) == 3:
        path_pictures, dm_params, dm_params_lambda = returns
    elif len(returns) == 4:
        pixs, pixs_combined, dm_params, dm_params_lambda = returns
        globals()['pixs'] = pixs
        globals()['pixs_combined'] = pixs_combined
        globals()['dm_params'] = dm_params
        globals()['dm_params_lambda'] = dm_params_lambda
    # path_pictures = create_bits_neighbour_pictures(dm_params)
    print("len(pixs): {}".format(len(pixs)))
    print("dm_params.path_dir:\n{}".format(dm_params.path_dir))


def test_nr_1():
    variables = get_default_variables()

    print("Before:")
    print("variables.height: {}".format(variables.height))
    print("variables.width: {}".format(variables.width))
    variables.lambdas_in_picture=False
    variables.with_resize_image=False
    variables.bits=1
    variables.min_or=2
    variables.max_or=2
    variables.width=200
    variables.height=200
    variables.temp_path_lambda_file='lambdas.txt'
    variables.save_pictures=False
    variables.suffix = ' '

    print("After:")
    print("variables.height: {}".format(variables.height))
    print("variables.width: {}".format(variables.width))

    dm_params_lambda = get_dm_params_lambda(variables)
    dm_params = get_dm_params(variables)

    returns = create_bits_neighbour_pictures(dm_params, dm_params_lambda)
    pixs, pixs_combined, dm_params, dm_params_lambda = returns
    globals()['pixs'] = pixs
    globals()['pixs_combined'] = pixs_combined
    globals()['dm_params'] = dm_params
    globals()['dm_params_lambda'] = dm_params_lambda


if __name__ == "__main__":
    # TODO: add some of create_3_byte_neighbour_pictures params to create_bits_neighbour_pictures function!
    # TODO: add some more, lets see
    
    # argv_own = ['approx_random_images.py',
    #             'lambdas_in_picture=False,with_resize_image=False,bits=24,min_or=2,max_or=2,width=450,height=300,temp_path_lambda_file=lambdas.txt,image_by_str=234']
    # approx_random_images.py lambdas_in_picture=False,with_resize_image=False,bits=24,min_or=2,max_or=2,width=450,height=300,temp_path_lambda_file=lambdas.txt,image_by_str=234

    test_nr_1()
    sys.exit(-12345)

    main(sys.argv)
    # main(argv_own)

    # paths_pictures = []
    # for _ in range(0, 10):
    #     path_pictures = create_bits_neighbour_pictures(height, width, next_folder=next_folder)
    #     paths_pictures.append(path_pictures)
    # print("paths_pictures:\n{}".format(paths_pictures))


    # combine_images_from_folders(paths_pictures)
    sys.exit(0)

    # create_1_byte_neighbour_pictures(height, width)
    # create_3_byte_neighbour_pictures("random", (height, width, False))

    # create_3_byte_neighbour_pictures("random", height=height, width=width, same_image=False, with_frame=True)
    # TODO: make the system call multithreaded!

    # create_from_image_neighbour_pictures("images/fall-autumn-red-season.jpg")
    # ## convert fall-autumn-red-season.jpg -resize 320x213 fall-autumn-red-season_resized.jpg
    
    max_iterations = 51
    resize_params = None
    ft = 6
    num_copies_first_image=4
    amount_combines=2
    gif_delay=5
    fps_movie=20

    # with open("lambda_functions/resize_params_2.pkl", "rb") as fout:
    #     dm = dill.load(fout)
    # resize_params = dm.resize_params
    # max_iterations = dm.max_iterations
    # print("resize_params: {}".format(resize_params))
    # print("max_iterations: {}".format(max_iterations))

    folder_suffix = ""
    argv = sys.argv
    if len(argv) > 1:
        folder_suffix = argv[1]

    create_3_byte_neighbour_pictures("random",
                                     height=height,
                                     width=width,
                                     same_image=True,
                                     height_resize=height_resize,
                                     width_resize=width_resize,
    
    # create_3_byte_neighbour_pictures("picture",
    #                                  image_path="images/fall-autumn-red-season_resized.jpg",
    #                                  resize_params=resize_params,

                                     with_frame=True,
                                     path_dir="lambda_functions/lambdas_5.txt",
                                     max_iterations=max_iterations,
                                     ft=ft,
                                     num_copies_first_image=num_copies_first_image,
                                     amount_combines=amount_combines,
                                     gif_delay=gif_delay,
                                     fps_movie=fps_movie,
                                     folder_suffix=folder_suffix)
