#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import inspect
import os
import pdb
import shutil
import string
import sys

import numpy as np

from dotmap import DotMap
from PIL import Image, ImageFont, ImageDraw

import create_lambda_functions

path_dir_root = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

sys.path.append(path_dir_root+"/..")
import utils_all

class BitNeighborManipulation(Exception):
    def __init__(self, ft=2, with_frame=True, path_lambda_functions=None, lambda_str_funcs_lst=None):
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
        self.bit_operations = self._generate_lambda_functions(path_lambda_functions, lambda_str_funcs_lst) # list of lambdas
        self.it1 = 0 # for the iterator variable (1st)
        self.it2 = 0 # for the iterator variable (2nd)


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


    def _generate_lambda_functions(self, path_lambda_functions, lambda_str_funcs_lst):
        if path_lambda_functions != None:
            if not os.path.exists(path_lambda_functions):
                print("File path '{}' does not exists!".format(path_lambda_functions))
                print("Will use default lambda functions then instead!")
                sys.exit(-1)

            with open(path_lambda_functions, "r") as fin:
                # lines = fin.readlines()
                lines = list(filter(lambda x: len(x) > 0, fin.read().splitlines()))
        elif lambda_str_funcs_lst != None:
            lines = lambda_str_funcs_lst
        else:
            print("ERROR! No lambda functions can be found!")
            sys.exit(-2)

        # TODO: check every single line, if it is matching with the variable convention!
        # TODO: add a security function, where each line will be checked up
        
        def inv(l):
            return (l+1)%2
        globals()["inv"] = inv

        def return_function(def_func_str):
            local = {}
            # local = {"inv": inv}
            exec(def_func_str, globals(), local)
            # print("local: {}".format(local))
            # sys.exit(-4)
            return local["a"]

        # first find 'def' functions and split it up!
        lambdas = []
        def_func = ""
        is_prev_def = False
        for line in lines:
            # print("111: line: {}".format(line))
            if "def " in line or (len(line) >= 6 and "lambda" != line[:6]):
                if is_prev_def == False:
                    is_prev_def = True
                    def_func = ""
                def_func += line+"\n"
            else:
                if is_prev_def == True:
                    is_prev_def = False
                    lambdas.append(return_function(def_func))
                    # lambdas.append(inspect.getsource(def_func))
                    # lambdas.append(exec(def_func))
                lambdas.append(eval(line))
        if is_prev_def == True:
            # print("def_func:\n{}".format(def_func))
            lambdas.append(return_function(def_func))
            # lambdas.append(inspect.getsource(def_func))
            # lambdas.append(exec(def_func))

        # lambdas = [eval(line) for line in lines]
        # self.max_bit_operators = 5
        self.max_bit_operators = len(lambdas)
        # pdb.set_trace()

        return lambdas

        # TODO: make this as the default lambdas list!
        return [
        lambda:
        p_ulll,
        lambda:
        p_ul,
        lambda:
        ((p_ull&p_ddr)&(p==0))|
        ((p_ul&p_ddrr&p_dl)&(p==1)),
        lambda: \
        ((p_uu&p_dd&p_rr&p_ll|p_u&p_d&p_l&p_r)&(p==0))|
        ((p_uull&p_uurr&p_ddll&p_ddrr|p_ul&p_ur&p_dl&p_dr)&(p==1)),
        lambda: \
        ((p_u&p_r|p_ur&p_dr)&(p==0))|
        ((p_d&p_l|p_ul&p_dl)&(p==1)),
        lambda: \
        ((p_u&p_l)&(p==0))^
        ((p_d&p_r)&(p==1)),
        lambda: \
        ((p_u&p_l|p_d&p_r)&(p==1))^
        ((p_u&p_d|p_l&p_r)&(p==0))^
        ((p_u&p_d&p_l&p_r)),
        lambda: \
        ((p_u&p_l|p_d&p_r)&(p==0))^
        ((p_u&p_d|p_l&p_r)&(p==1))^
        ((p_u&p_d&p_l&p_r)),
        lambda: \
        ((p_u&p_r|p_d&p_l)&(p==1))^
        ((p_u&p_l|p_l&p_d)&(p==0))^
        ((p_u&p_d&p_l&p_r)),
        lambda: \
        ((p_u&p_r|p_d&p_l)&(p==0))^
        ((p_u&p_l|p_r&p_d)&(p==1))^
        ((p_u&p_d&p_l&p_r)),
        lambda: \
        ((p_u&p_r&p_d|p_l)&(p==1))^
        ((p_u&p_r|p_l&p_d)&(p==0))^
        ((p_u&p_d&p_l&p_r)),
        lambda: \
        ((p_u&p_r&p_d|p_l)&(p==1))^
        ((p_u&p_r|p_l&p_d)&(p==0))^
        ((p_u&p_d&p_l&p_r))
        ]


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
        exec("globals()['p'] = pixs[ft, ft]")
        for y in range(0, ft*2+1):
            for x in range(0, ft*2+1):
                if y == ft and x == ft:
                    continue
                # Removed "p_" at the beginning of each variable name!
                var_name = ("u"*(ft-y) if y < ft else "d"*(y-ft))+("l"*(ft-x) if x < ft else "r"*(x-ft))
                # variables.append(var_name)
                exec("globals()['{}'] = pixs[{}, {}]".format(var_name, y, x))

        # idxs_lambda = (self.it1+self.it2)%self.max_bit_operators
        # idxs_lambda = (self.it1)%self.max_bit_operators
        # pix_bw1 = self.bit_operations[idxs_lambda]()
        # pix_bw1 = self.bit_operations[self.bit_operators_idx[(self.it1+self.it2)%self.max_bit_operators]]()
        pix_bw1 = self.bit_operations[(self.it1+self.it2)%self.max_bit_operators]()
        # pix_bw2 = self.bit_operations[(self.it1+1)%self.max_bit_operators]()
        # pix_bw3 = self.bit_operations[self.bit_operators_idx[(self.it1+2)%self.max_bit_operators]]()

        # assert np.sum(pix_prev_1 != pix_prev_2) == 0
        pix_bw = pix_bw1
        # pix_bw = pix_bw1^pix_bw2
        # pix_bw = pix_bw1^pix_bw2^pix_bw3
        # pix_bw = self.bit_operations[self.bit_operators_idx[(self.it1+self.it2)%self.max_bit_operators]]()
        # self.it1 += 1
        self.it2 += 1
        
        if self.remove_frame != None:
            return self.remove_frame(pix_bw)
        return pix_bw


    def apply_neighbor_logic(self, pix_bws):
        pix_bws_new = []

        self.it2 = 0
        for i, pix_bw in enumerate(pix_bws):
           pix_bws_new.append(self.apply_neighbor_logic_1_bit(pix_bw))
        self.it1 += 1

        return pix_bws_new


def create_1_bit_neighbour_pictures(height, width, next_folder="", return_pix_array=False, save_pictures=True):
    if save_pictures:
        suffix = "{}_{}_{}_{}".format(
            height,
            width,
            utils_all.get_date_time_str_full(),
            utils_all.get_random_str_base_16(4)
        )

        font_name = "712_serif.ttf"
        font_size = 16
        fnt = ImageFont.truetype('../fonts/{}'.format(font_name), font_size)

        char_width, char_height = fnt.getsize("a")
        print("char_width: {}".format(char_width))
        print("char_height: {}".format(char_height))

        print("path_dir_root: {}".format(path_dir_root))

        path_pictures = path_dir_root+"images/{}changing_bw_1_bit_{}/".format(next_folder, suffix)
        # path_pictures = path_pictures.replace("/", "\\")
        print("path_pictures: {}".format(path_pictures))

        # sys.exit(-1)

        if os.path.exists(path_pictures):
            os.system("rm -rf {}".format(path_pictures))
        if not os.path.exists(path_pictures):
            os.makedirs(path_pictures)

    # sys.exit(-2)
        print("path_pictures:\n{}".format(path_pictures))

        dm = create_lambda_functions.create_lambda_functions_with_matrices(path_lambda_functions=path_pictures, save_data=save_pictures)
    
        text_sizes = []
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

            text_sizes.append(size)
            length = len(line)
            if max_chars < length:
                max_chars = length

        for i, text_size in enumerate(text_sizes):
            print("i: {}, text_size: {}".format(i, text_size))

        print("max_chars: {}".format(max_chars))

        pix2 = np.zeros((len(lines_print)*(max_char_height+2)+2, max_width+2, 3), dtype=np.uint8)
        pix2 += 0x40
        img2 = Image.fromarray(pix2)
        d = ImageDraw.Draw(img2)

        for i, line in enumerate(lines_print):
            d.text((1, 1+i*(max_char_height+2)), line, font=fnt, fill=(255, 255, 255))
            # d.text((1, 1+i*(font_size)), line, font=fnt, fill=(255, 255, 255))

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

        # img2 = Image.fromarray(pix2_1)

        # img2.show()
        # return
        print("pix2_1.shape: {}".format(pix2_1.shape))
    else:
        dm = create_lambda_functions.create_lambda_functions_with_matrices(path_lambda_functions=None, save_data=save_pictures)

    # function_str_lst = create_lambda_functions.conway_game_of_life_functions(path_pictures)
    # function_str_lst = create_lambda_functions.simplest_lambda_functions(path_pictures)
    # function_str_lst = create_lambda_functions.simple_random_lambda_creation(path_lambda_functions_file=path_pictures)

    function_str_lst = dm.function_str_lst

    lines_print = []
    for i, line in enumerate(function_str_lst):
        lines_print.append("i: {}, {}".format(i, line))

    # char_sizes = []
    # for c in list(string.ascii_letters+"0123456789-_#'+*/()[]{}?!"):
    #     size = fnt.getsize(c)
    #     print("c: {}, fnt.getsize(c): {}".format(c, size))
    #     char_sizes.append(size)

    # print("function_str_lst:\n\n{}".format("\n".join(function_str_lst)))
    # sys.exit(-12342)
    
    pix_bw = np.random.randint(0, 2, (height, width), dtype=np.uint8)
    # print("type(pix_bw): {}".format(type(pix_bw)))
    pix_1 = np.dstack((pix_bw, pix_bw, pix_bw)).astype(np.uint8)*255
    
    color_frame_image = np.array([0x40, 0x80, 0xFF], dtype=np.uint8)

    def add_left_right_frame(pix, color, width):
        field = np.zeros((pix.shape[0], width, 3), dtype=np.uint8)+color
        return np.hstack((
            field.copy(),
            pix,
            field
        ))


    # if pix_1.shape[0] < pix2_1.shape[0]:
    #     diff = pix2_1.shape[0]-pix_1.shape[0]
    #     h1 = diff//2
    #     h2 = h1+diff%2
    #     pix_1 = np.vstack((
    #         np.zeros((h1, pix_1.shape[1], 3), dtype=np.uint8)+color_frame_image,
    #         pix_1,
    #         np.zeros((h2, pix_1.shape[1], 3), dtype=np.uint8)+color_frame_image,
    #     ))

    # width_frame = 10
    # pix_1 = add_left_right_frame(pix_1, color_frame_image, width_frame)

    # Image.fromarray(pix_1).show()
    # Image.fromarray(pix2_1).show()

    pix_complete = pix_1

    # pix_complete = np.hstack((pix2_1, pix_1))
    if save_pictures:
        Image.fromarray(pix_complete).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, 0))
        
    pixs = [pix_complete]
    # Image.fromarray(pix_bw*255).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, 0))

    # with_frame = False
    with_frame = True
    
    if save_pictures:
        bit_neighbor_manipulation = BitNeighborManipulation(ft=1, with_frame=with_frame, path_lambda_functions=path_pictures+"lambdas.txt")
    else:
        bit_neighbor_manipulation = BitNeighborManipulation(ft=1, with_frame=with_frame, lambda_str_funcs_lst=dm.function_str_lst)

    # so long there are white pixels, repeat the elimination_process!
    it_max = 300
    # it_max = height
    it = 1
    # pix_bw_prev = pix_bw.copy()
    # pixs = [pix_bw.copy()]
    # repeat anything until it is complete blank / black / 0
    while it < it_max:
    # while np.sum(pix_bw == 1) > 0 and it < it_max:
        print("it: {}".format(it))
        
        # TODO: need to be fixed!
        pix_bw = bit_neighbor_manipulation.apply_neighbor_logic_1_bit(pix_bw)
        # print("type(pix_bw): {}".format(type(pix_bw)))
        # pix_bw = apply_neighbour_logic(pix_bw)

        pix_1 = np.dstack((pix_bw, pix_bw, pix_bw)).astype(np.uint8)*255

        # if pix_1.shape[0] < pix2_1.shape[0]:
        #     diff = pix2_1.shape[0]-pix_1.shape[0]
        #     h1 = diff//2
        #     h2 = h1+diff%2
        #     pix_1 = np.vstack((
        #         np.zeros((h1, pix_1.shape[1], 3), dtype=np.uint8)+color_frame_image,
        #         pix_1,
        #         np.zeros((h2, pix_1.shape[1], 3), dtype=np.uint8)+color_frame_image,
        #     ))
        
        # pix_1 = add_left_right_frame(pix_1, color_frame_image, width_frame)
        pix_complete = pix_1

        is_already_available = False
        for pix in pixs:
            if np.all(pix==pix_complete):
                is_already_available = True
                break

        if is_already_available:
            break
        pixs.append(pix_complete)

        # pix_complete = np.hstack((pix2_1, pix_1))
        if save_pictures:
            Image.fromarray(pix_complete).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, it))
        # Image.fromarray(pix_bw*255).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, it))
        it += 1
        
        # if np.any(pix_bw!=pix_bw_prev) == False:
        #     break
        # pix_bw_prev = pix_bw.copy()

    if save_pictures:
        path_pictures = path_pictures[11:]
        print("new path_pictures: {}".format(path_pictures))
        command = "convert -delay 10 -loop 0 {}*.png {}myimage.gif".format(path_pictures, path_pictures)
        # command = command.replace("/", "\\\\")
        print("command:\n{}".format(command))
        os.system(command)
        # os.system("convert -delay 5 -loop 0 ./{}/*.png ./{}/myimage.gif".format(path_pictures, path_pictures))
        
        path_gifs = path_dir_root[11:]+"images/animations/{}".format(next_folder)
        # path_gifs = path_gifs.replace("/", "\\\\")
        if not os.path.exists(path_gifs):
            os.makedirs(path_gifs)

        # shutil.copy("{}myimage.gif".format(path_pictures), "{}1bit_{}.gif".format(path_gifs, suffix))
        command2 = "cp {}myimage.gif".format(path_pictures)+" {}1bit_{}.gif".format(path_gifs, suffix)
        print("command2: {}".format(command2))
        os.system(command2)

    for line in lines_print:
        print("{}".format(line))


    if return_pix_array:
        return pixs, dm

    return path_pictures


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
    image_path=None, max_iterations=-1, path_lambda_functions=None,
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

    bit_neighbor_manipulation = BitNeighborManipulation(ft=ft, with_frame=with_frame, path_lambda_functions=path_lambda_functions)

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


if __name__ == "__main__":
    height = 64
    # height = 128
    # height = 256
    # height = 256
    # height = 512
    width = height

    height_resize = height*3
    width_resize = width*3

    # create_1_bit_neighbour_pictures_only_good_ones(height, width, amount=100)
    # sys.exit(0)
    
    next_folder="{}_{}/".format(
        utils_all.get_date_time_str_full(),
        utils_all.get_random_str_base_16(4)
    )

    path_pictures = create_1_bit_neighbour_pictures(height, width, next_folder="")
    print("path_pictures:\n{}".format(path_pictures))

    # paths_pictures = []
    # for _ in range(0, 10):
    #     path_pictures = create_1_bit_neighbour_pictures(height, width, next_folder=next_folder)
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
    ft = 3
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
                                     path_lambda_functions="lambda_functions/lambdas_5.txt",
                                     max_iterations=max_iterations,
                                     ft=ft,
                                     num_copies_first_image=num_copies_first_image,
                                     amount_combines=amount_combines,
                                     gif_delay=gif_delay,
                                     fps_movie=fps_movie,
                                     folder_suffix=folder_suffix)
