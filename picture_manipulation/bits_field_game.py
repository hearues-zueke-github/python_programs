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

from copy import deepcopy

import numpy as np
from array2gif import write_gif

from dotmap import DotMap
from PIL import Image, ImageFont, ImageDraw

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

sys.path.append(PATH_ROOT_DIR+"/..")
import utils_all
import utils_serialization

KWU8 = {'dtype': np.uint8}

def write_dm_obj_txt(dm):
    print("Now in def 'write_dm_obj_txt'")

    print("dm.functions_str_lst: {}".format(dm.functions_str_lst))
    assert dm.path_dir
    assert isinstance(dm.save_data, bool)
    assert dm.functions_str_lst
    assert dm.file_name_dm_params_lambda
    assert dm.file_name_txt

    path_dir = dm.path_dir
    save_data = dm.save_data
    functions_str_lst = dm.functions_str_lst
    file_name_dm_params_lambda = dm.file_name_dm_params_lambda
    file_name_txt = dm.file_name_txt

    print("file_name_dm_params_lambda: {}".format(file_name_dm_params_lambda))
    print("file_name_txt: {}".format(file_name_txt))

    if path_dir == None:
        path_dir = "./"
    if path_dir[-1] != "/":
        path_dir += "/"

    if save_data:
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        with gzip.open(path_dir+file_name_dm_params_lambda, "wb") as fout:
            dill.dump(dm, fout)

        with open(path_dir+file_name_txt, "w") as fout:
            for line in functions_str_lst:
                fout.write("{}\n".format(line))


def get_params_arr(ft):
    # ft...frame thickness
    params_arr = np.empty((ft*2+1, ft*2+1), dtype=np.object)

    params_arr[ft, ft] = "p"
    
    for j in range(1, ft+1):
        params_arr[ft-j, ft] = "u"*j
        params_arr[ft+j, ft] = "d"*j
        params_arr[ft, ft-j] = "l"*j
        params_arr[ft, ft+j] = "r"*j
        for i in range(1, ft+1):
            params_arr[ft-j, ft-i] = "u"*j+"l"*i
            params_arr[ft-j, ft+i] = "u"*j+"r"*i
            params_arr[ft+j, ft-i] = "d"*j+"l"*i
            params_arr[ft+j, ft+i] = "d"*j+"r"*i

    return params_arr


def create_lambda_functions_with_matrices(dm_params):
    print("Now in def 'create_lambda_functions_with_matrices'")

    assert not ( isinstance(dm_params, DotMap) and
                 isinstance(dm_params, list) and
                 isinstance(dm_params, dict) )

    if isinstance(dm_params, list):
        dm_params = dict(dm_params)
    if isinstance(dm_params, dict):
        dm_params = DotMap(dm_params)

    assert dm_params.ft
    assert isinstance(dm_params.save_data, bool)
    assert isinstance(dm_params.path_dir, str) or dm_params.path_dir is None
    assert isinstance(dm_params.file_name_dm_params_lambda, str) or dm_params.file_name_dm_params_lambda is None
    assert isinstance(dm_params.file_name_txt, str) or dm_params.file_name_txt is None
    assert isinstance(dm_params.min_or, int)
    assert isinstance(dm_params.max_or, int)
    assert isinstance(dm_params.min_and, int)
    assert isinstance(dm_params.max_and, int)
    assert isinstance(dm_params.min_n, int)
    assert isinstance(dm_params.max_n, int)

    ft = dm_params.ft
    # path_dir = dm_params.path_dir
    # save_data = dm_params.save_data
    # file_name_dm_params_lambda = dm_params.file_name_dm_params_lambda
    # file_name_txt = dm_params.file_name_txt
    min_or = dm_params.min_or
    max_or = dm_params.max_or
    min_and = dm_params.min_and
    max_and = dm_params.max_and
    min_n = dm_params.min_n
    max_n = dm_params.max_n  
    
    params_arr = get_params_arr(ft)

    params_1 = params_arr.reshape((-1))
    params_0 = np.array(["i({})".format(param) for param in params_1])

    params = np.vstack((params_1, params_0)).T

    def get_random_conjuctions(params, min_n=1, max_n=3):
        # First get used variables for the conjuction
        key = np.random.choice(np.arange(min_n, max_n+1))
        group_num = np.zeros((params.shape[0], ), dtype=np.uint8)
        group_num[:key] = 1
        idx_choosen_param = np.random.permutation(group_num)

        # Now invert some of the params by random
        idx_inv_param = np.random.randint(0, 2, params.shape[0])
        idx_arr_inv_param = np.zeros(params.shape, dtype=np.int)
        idx_arr_inv_param[np.arange(0, params.shape[0]), idx_inv_param] = 1
        choosen_inv_params = params[idx_arr_inv_param==1]

        and_params = choosen_inv_params[idx_choosen_param==1]
        and_str = "&".join(and_params)

        return and_str, idx_choosen_param, idx_inv_param


    def get_random_disjunctions(params, min_and=1, max_and=4, min_n=1, max_n=3):
        amount_and = np.random.randint(min_and, max_and+1)
        and_values = [get_random_conjuctions(params=params, min_n=min_n, max_n=max_n) for _ in range(0, amount_and)]

        and_lst, idx_choosen_params, idx_inv_params = list(zip(*and_values))

        or_str = "lambda: "+"|".join(and_lst)

        return or_str, np.array(idx_choosen_params), np.array(idx_inv_params)


    def get_random_lambdas(params, min_or=1, max_or=8, min_and=1, max_and=4, min_n=1, max_n=3):
        amount_or = np.random.randint(min_or, max_or+1)
        or_lst = [get_random_disjunctions(params=params, min_and=min_and, max_and=max_and, min_n=min_n, max_n=max_n) for _ in range(0, amount_or)]

        return or_lst


    # short description:
    # min/max values ... where the random values can be (including both)
    # _or ... amount for lambdas
    # _and ... amount for disjunctions
    # _n ... amount for conjuctions
    function_str_values = get_random_lambdas(params, min_or=min_or, max_or=max_or, min_and=min_and, max_and=max_and, min_n=min_n, max_n=max_n)
    functions_str_lst, idx_choosen_params_lst, idx_inv_params_lst = list(zip(*function_str_values))
    idx_choosen_params_lst = np.array(idx_choosen_params_lst)
    idx_inv_params_lst = np.array(idx_inv_params_lst)

    
    dm_params.functions_str_lst = functions_str_lst
    print("functions_str_lst: {}".format(functions_str_lst))
    dm_params.idx_choosen_params_lst = idx_choosen_params_lst
    dm_params.idx_inv_params_lst = idx_inv_params_lst

    return dm_params


class BitFieldBWConverter(Exception):
    possible_bits = [1, 3, 8, 24]

    def __init__(self, bits):
        assert isinstance(bits, int)
        assert bits in self.possible_bits
        
        if bits==1:
            self._convert_bws_to_pix = self._convert_1_bit_field_to_pix
        elif bits==3:
            self._convert_bws_to_pix = self._convert_3_bit_field_to_pix
        elif bits==8:
            self._convert_bws_to_pix = self._convert_8_bit_field_to_pix
        elif bits==24:
            self._convert_bws_to_pix = self._convert_24_bit_field_to_pix


    def convert_bws_to_pix(self, bws):
        return self._convert_bws_to_pix(bws)


    def _convert_1_bit_field_to_pix(self, bws):
        assert isinstance(bws, np.ndarray)
        assert len(bws) == 1

        pix_bw = bws[0]
        pix = np.dstack((pix_bw, pix_bw, pix_bw)).astype(np.uint8)*255
        return pix


    def _convert_3_bit_field_to_pix(self, bws):
        assert isinstance(bws, np.ndarray)
        assert len(bws) == 3

        pix_bw_r = np.zeros(bws[0].shape, **KWU8)
        pix_bw_r += bws[0]*255
        pix_bw_g = np.zeros(bws[0].shape, **KWU8)
        pix_bw_g += bws[1]*255
        pix_bw_b = np.zeros(bws[0].shape, **KWU8)
        pix_bw_b += bws[2]*255
        pix = np.dstack((pix_bw_r, pix_bw_g, pix_bw_b)).astype(np.uint8)
        return pix


    def _convert_8_bit_field_to_pix(self, bws):
        assert isinstance(bws, np.ndarray)
        assert len(bws) == 8

        pix_bw = np.zeros(bws[0].shape, **KWU8)
        for i, p in zip(range(7, -1, -1), bws):
            pix_bw += p<<i
        pix = np.dstack((pix_bw, pix_bw, pix_bw)).astype(np.uint8)
        return pix


    def _convert_24_bit_field_to_pix(self, bws):
        assert isinstance(bws, np.ndarray)
        assert len(bws) == 24

        pix_bw_r = np.zeros(bws[0].shape, **KWU8)
        for i, p in zip(range(7, -1, -1), bws[:8]):
            pix_bw_r += p<<i
        pix_bw_g = np.zeros(bws[0].shape, **KWU8)
        for i, p in zip(range(7, -1, -1), bws[8:16]):
            pix_bw_g += p<<i
        pix_bw_b = np.zeros(bws[0].shape, **KWU8)
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

            zero_row = np.zeros((width, ), **KWU8)
            zero_col = np.zeros((height, 1), **KWU8)

            move_arr_u = lambda pix_bw: np.vstack((pix_bw[1:], zero_row))
            move_arr_d = lambda pix_bw: np.vstack((zero_row, pix_bw[:-1]))
            move_arr_l = lambda pix_bw: np.hstack((pix_bw[:, 1:], zero_col))
            move_arr_r = lambda pix_bw: np.hstack((zero_col, pix_bw[:, :-1]))

            pixs = np.zeros((ft*2+1, ft*2+1, height, width), **KWU8)
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

        return np.array(pix_bws_new)


def create_bits_neighbour_pictures(dm_params, dm_params_lambda):
    print("Now in def 'create_bits_neighbour_pictures'")

    assert not ( isinstance(dm_params, DotMap) and
                 isinstance(dm_params, list) and
                 isinstance(dm_params, dict) )

    if isinstance(dm_params, list):
        dm_params = dict(dm_params)
    if isinstance(dm_params, dict):
        dm_params = DotMap(dm_params)

    assert dm_params.path_dir
    assert dm_params.file_name_dm_params_lambda
    assert dm_params.file_name_txt

    assert isinstance(dm_params.height, int)
    assert isinstance(dm_params.width, int)
    assert isinstance(dm_params.ft, int)
    assert isinstance(dm_params.next_folder, str)
    assert isinstance(dm_params.with_frame, bool)
    assert isinstance(dm_params.save_data, bool)
    assert isinstance(dm_params.save_pictures, bool)
    assert isinstance(dm_params.save_gif, bool)
    assert isinstance(dm_params.functions_str_lst, list)
    assert isinstance(dm_params.width_append_frame, int)
    assert isinstance(dm_params.lambdas_in_picture, bool)
    assert isinstance(dm_params.max_it, int)
    assert isinstance(dm_params.with_resize_image, bool)
    assert isinstance(dm_params.resize_factor, int)
    assert isinstance(dm_params.bits, int)
    assert not (len(dm_params.temp_path_lambda_file) > 0 and len(dm_params.functions_str_lst) > 0)
    assert isinstance(dm_params.suffix, str)
    assert isinstance(dm_params.image_by_str, str)
    assert isinstance(dm_params.lambda_by_str, str)
    assert isinstance(dm_params.random_folder_name, bool)

    path_dir = dm_params.path_dir
    file_name_dm_params_lambda = dm_params.file_name_dm_params_lambda
    file_name_txt = dm_params.file_name_txt

    height = dm_params.height
    width = dm_params.width
    ft = dm_params.ft
    next_folder = dm_params.next_folder
    with_frame = dm_params.with_frame
    save_data = dm_params.save_data
    save_pictures = dm_params.save_pictures
    save_gif = dm_params.save_gif
    functions_str_lst = dm_params.functions_str_lst
    width_append_frame = dm_params.width_append_frame
    lambdas_in_picture = dm_params.lambdas_in_picture
    max_it = dm_params.max_it
    with_resize_image = dm_params.with_resize_image
    resize_factor = dm_params.resize_factor
    bits = dm_params.bits
    temp_path_lambda_file = dm_params.temp_path_lambda_file
    suffix = dm_params.suffix
    image_by_str = dm_params.image_by_str
    lambda_by_str = dm_params.lambda_by_str
    random_folder_name = dm_params.random_folder_name

    # assert not(suffix=='' or image_by_str)

    np.random.seed()

    if isinstance(next_folder, str) and (len(next_folder) > 1 and next_folder[:-1] != "/" or len(next_folder) == 0):
        next_folder += '/'

    if len(suffix) > 0 and suffix[0] != '_':
        suffix = '_'+suffix

    if len(image_by_str)>0:
        suffix += '_imgstr_'+image_by_str

    if len(lambda_by_str)>0:
        suffix += '_lmbdstr_'+lambda_by_str

    print("save_pictures: {}".format(save_pictures))
    # if save_pictures:
    font_name = "712_serif.ttf"
    font_size = 16
    fnt = ImageFont.truetype('../fonts/{}'.format(font_name), font_size)

    char_width, char_height = fnt.getsize("a")
    print("char_width of 'a': {}".format(char_width))
    print("char_height of 'a': {}".format(char_height))

    if save_pictures or save_gif or save_data:
        if random_folder_name:
            folder_name = 'rand_'+utils_all.get_random_str_base_16(16)
        else:
            folder_name = "changing_bw_{bits}_bits{suffix}/".format(bits=bits, suffix=suffix)

        path_pictures = PATH_ROOT_DIR+"images/{next_folder}{folder_name}/".format(next_folder=next_folder, folder_name=folder_name)
        print("path_pictures: {}".format(path_pictures))

        # dm_params.path_pictures = path_pictures

        if os.path.exists(path_pictures):
            os.system("rm -rf {}".format(path_pictures))
        if not os.path.exists(path_pictures):
            os.makedirs(path_pictures)

        print("path_pictures:\n{}".format(path_pictures))

        dm_params.path_dir = path_pictures
        dm_params.folder_name = folder_name
        dm_params_lambda.path_dir = path_pictures
        dm_params_lambda.save_data = save_data
    else:
        dm_params.path_dir = None
        dm_params.folder_name = None
        dm_params_lambda.path_dir = None
        dm_params_lambda.save_data = False

    if len(functions_str_lst)>0:
        dm_params_lambda.used_method = "own_defined_lambdas"
        dm_params_lambda.functions_str_lst = functions_str_lst
    elif len(lambda_by_str)>0:
        if len(lambda_by_str)<4 or '.txt'!=lambda_by_str[-4:]:
            lambda_by_str += '.txt'
        # TODO: first check, if file exists, otherwise create random lambdas!
        path_lambda_functions_dir = PATH_ROOT_DIR+'lambda_functions/'
        if not os.path.exists(path_lambda_functions_dir):
            os.makedirs(path_lambda_functions_dir)
        path_lambda_functions_file = path_lambda_functions_dir+lambda_by_str
        if not os.path.exists(path_lambda_functions_file):
            dm_params_lambda_2 = create_lambda_functions_with_matrices(deepcopy(dm_params_lambda))
            functions_str_lst = dm_params_lambda_2.functions_str_lst
            with open(path_lambda_functions_file, 'w') as f:
                f.write('\n'.join(functions_str_lst))
        with open(path_lambda_functions_file, 'r') as f:
            # functions_str_lst = f.readlines()
            functions_str = f.read()
            functions_str_lst = functions_str.split('\n')

        dm_params_lambda.functions_str_lst = functions_str_lst
        dm_params_lambda.used_method = 'create_new_random_lambdas_by_name'
    elif len(temp_path_lambda_file) > 0:
        assert os.path.exists(temp_path_lambda_file)

        bnm = BitNeighborManipulation(path_dir=temp_path_lambda_file)
        dm_params_lambda.functions_str_lst =  bnm.lambdas_str
        dm_params_lambda.used_method = "from_temp_path_file"
        print("dm_params_lambda.functions_str_lst:\n{}".format(dm_params_lambda.functions_str_lst))
        print("len(dm_params_lambda.functions_str_lst): {}".format(len(dm_params_lambda.functions_str_lst)))
    else:
        dm_params_lambda = create_lambda_functions_with_matrices(dm_params_lambda)
        dm_params_lambda.used_method = 'create_new_random_lambdas'
        # dm_params_lambda.functions_str_lst = functions_str_lst

    # print("len(dm_params_lambda.functions_str_lst): {}".format(len(dm_params_lambda.functions_str_lst)))
    # sys.exit(-2)

    if dm_params_lambda.save_data:
        write_dm_obj_txt(dm_params_lambda)

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

        length = len(line)
        if max_chars < length:
            max_chars = length

    pix2 = np.zeros((len(lines_print)*(max_char_height+2)+2, max_width+2, 3), **KWU8)
    pix2 += 0x40
    img2 = Image.fromarray(pix2)
    d = ImageDraw.Draw(img2)

    for i, line in enumerate(lines_print):
        d.text((1, 1+i*(max_char_height+2)), line, font=fnt, fill=(255, 255, 255))

    if save_data:
        img2.save(path_pictures+"lambdas.png")

    # save the lambda functions into as a image!
    pix2_1 = np.array(img2)

    if pix2_1.shape[0] < height:
        diff = height-pix2_1.shape[0]
        h1 = diff//2
        h2 = h1+diff%2
        pix2_1 = np.vstack((
            np.zeros((h1, pix2_1.shape[1], 3), **KWU8),
            pix2_1,
            np.zeros((h2, pix2_1.shape[1], 3), **KWU8)
        ))

    print("pix2_1.shape: {}".format(pix2_1.shape))
    
    print("bits: {}".format(bits))
    bws_to_pix_converter = BitFieldBWConverter(bits=bits)
    convert_bws_to_pix = bws_to_pix_converter.convert_bws_to_pix

    if len(image_by_str) > 0:
        dir_random_images = PATH_ROOT_DIR+"images/random_images_by_name/"
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
                pix = np.vstack((pix, np.random.randint(0, 256, (height-pix.shape[0], pix.shape[1], 3), **KWU8)))
                is_pix_changed = True
            if pix.shape[1] < width:
                pix = np.hstack((pix, np.random.randint(0, 256, (pix.shape[0], width-pix.shape[1], 3), **KWU8)))
                is_pix_changed = True
                
            if is_pix_changed:
                img = Image.fromarray(pix)
                img.save(path_rnd_img)
            
            if pix.shape[0] > height:
                pix = pix[:height]
            if pix.shape[1] > width:
                pix = pix[:, :width]
        else:
            pix = np.random.randint(0, 256, (height, width, 3), **KWU8)
            img = Image.fromarray(pix)
            img.save(path_rnd_img)

        # make a function for this! (or not xD)
        pix_bws = []
        for c in range(0, 3):
            channel = pix[:, :, c]
            for i in range(7, -1, -1):
                pix_bws.append((channel>>i)&0x1)
        pix_bws = np.array(pix_bws[:bits])

        dm_params.path_rnd_img = path_rnd_img
    else:
        # pix_bws = [np.random.randint(0, 2, (height, width), **KWU8) for _ in range(0, 24)]
        pix_bws = np.array([np.random.randint(0, 2, (height, width), **KWU8) for _ in range(0, bits)])
    
    dm_params.first_pix_bws = deepcopy(pix_bws)
    pix_1 = convert_bws_to_pix(pix_bws)
    
    color_frame_image = np.array([0x40, 0x20, 0xFF], **KWU8)

    def add_left_right_frame(pix, color, width):
        field = np.zeros((pix.shape[0], width, 3), **KWU8)+color
        return np.hstack((
            field.copy(),
            pix,
            field
        ))

    if pix_1.shape[0] < pix2_1.shape[0]:
        diff = pix2_1.shape[0]-pix_1.shape[0]
        h1 = diff//2
        h2 = h1+diff%2
        pix1_1 = np.vstack((
            np.zeros((h1, pix_1.shape[1], 3), **KWU8)+color_frame_image,
            pix_1,
            np.zeros((h2, pix_1.shape[1], 3), **KWU8)+color_frame_image,
        ))
    else:
        pix1_1 = pix_1

    pix1_2 = add_left_right_frame(pix1_1, color_frame_image, width_append_frame)

    pix_combined = np.hstack((pix2_1, pix1_2))

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
        
    # print("TEST!!!")
    # print("dm_params_lambda.functions_str_lst: {}".format(dm_params_lambda.functions_str_lst))
    # print("TEST!!!")
    if save_data:
        if len(temp_path_lambda_file) > 0:
            # copy the temp_path_lambda_file as path_pictures+'lambdas.txt' file!
            shutil.copy(temp_path_lambda_file, path_pictures+'lambdas.txt')
        path_lambda_file = path_pictures+"lambdas.txt"
        assert os.path.exists(path_lambda_file)
        bit_neighbor_manipulation = BitNeighborManipulation(ft=ft, with_frame=with_frame, path_dir=path_lambda_file)
    else:
        bit_neighbor_manipulation = BitNeighborManipulation(ft=ft, with_frame=with_frame, lambda_str_funcs_lst=dm_params_lambda.functions_str_lst)

    bit_neighbor_manipulation.bws_to_pix_converter = bws_to_pix_converter
    apply_neighbor_logic = bit_neighbor_manipulation.apply_neighbor_logic

    it = 1
    while it < max_it:
        print("it: {}".format(it))
        
        pix_bws = apply_neighbor_logic(pix_bws)
        # pix_bws = bit_neighbor_manipulation.apply_neighbor_logic(pix_bws)
        pix_1 = convert_bws_to_pix(pix_bws)

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
                np.zeros((h1, pix_1.shape[1], 3), **KWU8)+color_frame_image,
                pix_1,
                np.zeros((h2, pix_1.shape[1], 3), **KWU8)+color_frame_image,
            ))
        else:
            pix1_1 = pix_1
        
        pix1_2 = add_left_right_frame(pix1_1, color_frame_image, width_append_frame)
        pix_combined = np.hstack((pix2_1, pix1_2))

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
        
    if save_gif:
        print("new path_pictures: {path_pictures}".format(path_pictures=path_pictures))

        img, *imgs = [Image.fromarray(pix) for pix in pixs]
        img.save(
            fp="{path_pictures}myimage.gif".format(path_pictures=path_pictures),
            format='GIF', append_images=imgs, save_all=True, duration=15, loop=0
        )
        
        path_gifs = PATH_ROOT_DIR+"images/{next_folder}animations/".format(next_folder=next_folder)
        if not os.path.exists(path_gifs):
            os.makedirs(path_gifs)

        if random_folder_name:
            command2 = "cp {path_pictures}myimage.gif {path_gifs}{folder_name}.gif".format(
                path_pictures=path_pictures, path_gifs=path_gifs, folder_name=folder_name)
        else:
            command2 = "cp {path_pictures}myimage.gif {path_gifs}{bits}_bits{suffix}.gif".format(
                bits=bits, path_pictures=path_pictures, path_gifs=path_gifs, suffix=suffix)
        print("command2: {}".format(command2))
        os.system(command2)

    print("\nPrinting the lambda functions (lines)!")
    for line in lines_print:
        print("{}".format(line))

    dm_params.it = it # save the last it iterator
    dm_params.pix_bws = pix_bws

    # save bit_neighbor_manipulation, pixs and pixs_combined as .pkl.gz objects tooo! (if save_pictures == True)
    if save_data:
        with gzip.open(path_pictures+'bit_neighbor_manipulation.pkl.gz', 'wb') as fout:
            dill.dump(bit_neighbor_manipulation, fout)

        with gzip.open(path_pictures+'dm_params.pkl.gz', 'wb') as fout:
            dill.dump(dm_params, fout)

        with gzip.open(path_pictures+'pixs.pkl.gz', 'wb') as fout:
            dill.dump(np.array(pixs), fout)

        # with gzip.open(path_pictures+'pixs_combined.pkl.gz', 'wb') as fout:
        #     dill.dump(pixs_combined, fout)

    objs = DotMap()
    objs.pixs = pixs
    objs.pixs_combined = pixs_combined

    return objs, dm_params, dm_params_lambda


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
    print(" - file_name_dm_params_lambda: {}".format(variables.file_name_dm_params_lambda))
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

    variables.save_data = True
    variables.path_dir = 'images/'
    variables.file_name_dm_params_lambda = "dm_params_lambda.pkl.gz"
    variables.file_name_txt = "lambdas.txt"
    variables.min_or = 4
    variables.max_or = 4
    variables.min_and = 3
    variables.max_and = 3
    variables.min_n = 2
    variables.max_n = 2

    variables.height = 64
    variables.width = variables.height+20

    variables.ft = 1

    dt = datetime.datetime.now()
    variables.next_folder = "{year:04}_{month:02}_{day:02}".format(year=dt.year, month=dt.month, day=dt.day)

    variables.with_frame = True
    variables.width_append_frame = 5

    variables.save_pictures = True
    variables.save_gif = True

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
    variables.lambda_by_str = ''
    variables.random_folder_name = True

    return variables


def parse_argv_to_variables(argv, variables):
    using_vars_type = {
        'ft': int,
        'file_name_dm_params_lambda': str,
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
    dm_params_lambda.path_dir = variables.path_dir
    dm_params_lambda.save_data = variables.save_data
    dm_params_lambda.file_name_dm_params_lambda = variables.file_name_dm_params_lambda
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
    dm_params.save_data = variables.save_data
    dm_params.save_pictures = variables.save_pictures
    dm_params.save_gif = variables.save_gif
    dm_params.functions_str_lst = variables.functions_str_lst
    dm_params.width_append_frame = variables.width_append_frame
    dm_params.with_resize_image = variables.with_resize_image
    dm_params.resize_factor = variables.resize_factor
    dm_params.lambdas_in_picture = variables.lambdas_in_picture
    dm_params.max_it = variables.max_it
    dm_params.bits = variables.bits
    dm_params.temp_path_lambda_file = variables.temp_path_lambda_file
    dm_params.image_by_str = variables.image_by_str
    dm_params.lambda_by_str = variables.lambda_by_str
    dm_params.random_folder_name = variables.random_folder_name

    dm_params.path_dir = variables.path_dir
    dm_params.file_name_dm_params_lambda = variables.file_name_dm_params_lambda
    dm_params.file_name_txt = variables.file_name_txt

    return dm_params


def test_nr_1():
    variables = get_default_variables()

    variables.lambdas_in_picture = False
    variables.with_resize_image = False
    variables.bits = 24
    variables.min_or = 2
    variables.max_or = 2
    variables.width = 100
    variables.height = 100
    variables.temp_path_lambda_file = ''
    # variables.temp_path_lambda_file = 'lambdas.txt'
    variables.save_data = False
    variables.save_pictures = False
    variables.save_gif = False

    variables.suffix = ''
    variables.image_by_str = '0'
    variables.lambda_by_str = '0'
    variables.random_folder_name = True
    
    variables.max_it = 300

    fields = [
        (1, ''),
        (3, ''),
        (8, ''),
        (24, ''),
        # (1, ''),
        # (3, ''),
        # (8, ''),
        # (24, ''),
        # (1, '1'),
        # (3, '1'),
        # (8, '1'),
        # (24, '1'),
        # (1, '2'),
        # (3, '2'),
        # (8, '2'),
        # (24, '2'),
    ]

    lst_pixs = []

    for bits, image_by_str in fields:
        print("bits: {}, image_by_str: '{}'".format(bits, image_by_str))

        variables_copy = deepcopy(variables)
        variables_copy.bits = bits
        variables_copy.image_by_str = image_by_str

        dm_params_lambda = get_dm_params_lambda(variables_copy)
        dm_params = get_dm_params(variables_copy)

        objs, dm_params, dm_params_lambda = create_bits_neighbour_pictures(dm_params, dm_params_lambda)
        pixs = objs.pixs

        lst_pixs.append(pixs)

    globals()['lst_pixs'] = lst_pixs

        # returns = create_bits_neighbour_pictures(dm_params, dm_params_lambda)
    # dm_params, dm_params_lambda = returns
    # globals()['dm_params'] = dm_params
    # globals()['dm_params_lambda'] = dm_params_lambda

    # objs, dm_params, dm_params_lambda = returns
    # pixs, pixs_combined, dm_params, dm_params_lambda = returns
    # globals()['pixs'] = pixs
    # globals()['pixs_combined'] = pixs_combined
    # globals()['objs'] = objs
    # globals()['dm_params'] = dm_params
    # globals()['dm_params_lambda'] = dm_params_lambda

    # TODO: try applying this function to get images for the window display!


if __name__ == "__main__":
    test_nr_1()
