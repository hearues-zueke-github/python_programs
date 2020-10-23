#! /usr/bin/python3

import os
import sys

import multiprocessing as mp
import numpy as np

from PIL import Image
from copy import deepcopy

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

class HashIterator(Exception):
    def __init__(self, l_val, mod_num_bits=64):
    # def __init__(self, I, x, y, z, w):
        self.mod_num_bits = mod_num_bits
        self.mod_num = 2**mod_num_bits

        self.l_val = deepcopy(l_val)

        # self.I = I
        # self.x = x
        # self.y = y
        # self.z = z
        # self.w = w

        self.l_l_str_val = [["{:064b}".format(v)] for v in self.l_val]

        # self.l_str_I = []
        # self.l_str_x = []
        # self.l_str_y = []
        # self.l_str_z = []
        # self.l_str_w = []

        # self.l_str_I.append("{:064b}".format(I))
        # self.l_str_x.append("{:064b}".format(x))
        # self.l_str_y.append("{:064b}".format(y))
        # self.l_str_z.append("{:064b}".format(z))
        # self.l_str_w.append("{:064b}".format(w))


    def do_next_hash_step(self):
        # self.I += 1
        # self.x += self.I + (self.x >> 1)
        # self.y = (self.x ^ self.y) + (self.x >> 1)
        # self.z = self.x * self.I + (self.z << 1) + (self.I ^ self.z)
        # self.w = (self.z << 1) ^ (self.y >> 1)

        self.l_val[0] += 1
        self.l_val[1] += self.l_val[0] + (self.l_val[1] >> 1)
        self.l_val[2] = (self.l_val[1] ^ self.l_val[2]) + (self.l_val[1] >> 1)
        self.l_val[3] = self.l_val[1] * self.l_val[0] + (self.l_val[3] << 1) + (self.l_val[0] ^ self.l_val[3])
        self.l_val[4] = (self.l_val[3] << 1) ^ (self.l_val[2] >> 1) + self.l_val[4]

        for i in range(0, len(self.l_val)):
            self.l_val[i] = self.l_val[i] % self.mod_num
            self.l_l_str_val[i].append("{:064b}".format(self.l_val[i]))

        # self.I = self.I % self.mod_num
        # self.x = self.x % self.mod_num
        # self.y = self.y % self.mod_num
        # self.z = self.z % self.mod_num
        # self.w = self.w % self.mod_num

        # self.l_str_I.append("{:064b}".format(self.I))
        # self.l_str_x.append("{:064b}".format(self.x))
        # self.l_str_y.append("{:064b}".format(self.y))
        # self.l_str_z.append("{:064b}".format(self.z))
        # self.l_str_w.append("{:064b}".format(self.w))


if __name__ == '__main__':
    l_var_name = ['I', 'x', 'y', 'z', 'w']
    l_folder_path = [TEMP_DIR+'combined_hashed_images_64_bits_{}/'.format(var_name) for var_name in l_var_name]

    for path in l_folder_path:
        if not os.path.exists(path):
            os.makedirs(path)

    AMOUNT_DIFFS = 30

    # folder_dir_diff_00 = TEMP_DIR+'combined_hashed_images_64_bits_diff_00/'
    # if not os.path.exists(folder_dir_diff_00):
    #     os.makedirs(folder_dir_diff_00)

    # folder_dir_diff_01 = TEMP_DIR+'combined_hashed_images_64_bits_diff_01/'
    # if not os.path.exists(folder_dir_diff_01):
    #     os.makedirs(folder_dir_diff_01)

    # folder_dir_diff_02 = TEMP_DIR+'combined_hashed_images_64_bits_diff_02/'
    # if not os.path.exists(folder_dir_diff_02):
    #     os.makedirs(folder_dir_diff_02)

    # folder_dir_diff_03 = TEMP_DIR+'combined_hashed_images_64_bits_diff_03/'
    # if not os.path.exists(folder_dir_diff_03):
    #     os.makedirs(folder_dir_diff_03)

    # folder_dir_diff_04 = TEMP_DIR+'combined_hashed_images_64_bits_diff_04/'
    # if not os.path.exists(folder_dir_diff_04):
    #     os.makedirs(folder_dir_diff_04)

    # folder_dir_combined = TEMP_DIR+'combined_hashed_images_64_bits_combined/'
    # if not os.path.exists(folder_dir_combined):
    #     os.makedirs(folder_dir_combined)

    cpu_count = mp.cpu_count()
    print("cpu_count: {}".format(cpu_count))

    l_all_i_num = list(range(-1, 64))
    l_all_i_num = [((1<<i_num) if i_num > 0 else 0) for i_num in l_all_i_num]

    amount_nums = len(l_all_i_num)
    l_ranges = [0]+[int(amount_nums/cpu_count*i) for i in range(1, cpu_count)]+[amount_nums]
    l_l_i_num = [l_all_i_num[i1:i2] for i1, i2 in zip(l_ranges[:-1], l_ranges[1:])]
    print("l_ranges: {}".format(l_ranges))
    print("l_l_i_num: {}".format(l_l_i_num))

    def create_simple_hashes(l_i_num):
        for i_num in l_i_num:
            print("i_num: {}".format(i_num)) 
            l_init_vals = [
                i_num,
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
            ]

            hi = HashIterator(l_val=l_init_vals)
            for _ in range(0, 3000):
                hi.do_next_hash_step()
            
            arr_colors_bw = np.array([
                [0x00, 0x00, 0x00],
                [0xFF, 0xFF, 0xFF],
            ], dtype=np.uint8)

            l_arr_prev = [
                np.array(list(map(lambda x: list(map(int, list(x))), l)), dtype=np.uint8)
                for l in hi.l_l_str_val
            ]
            l_l_arr = [l_arr_prev]
            for _ in range(0, AMOUNT_DIFFS):
                l_arr_next = [(arr[:-1] != arr[1:]).astype(np.uint8) for arr in l_arr_prev]
                l_l_arr.append(l_arr_next)
                l_arr_prev = l_arr_next

            l_l_arr_T = list(zip(*l_l_arr))

            height = l_l_arr[0][0].shape[0]
            arr_vertical_separate = np.zeros((height, 1, 3), dtype=np.uint8) + np.array((0xFF, 0x00, 0x00), dtype=np.uint8)

            for i, l_arr_T in enumerate(l_l_arr_T, 0):
                l_combined = [arr_colors_bw[l_arr_T[0]]]
                for arr in l_arr_T[1:]:
                    pix = arr_colors_bw[arr]
                    pix_diff = np.vstack((pix, np.zeros((height-pix.shape[0], pix.shape[1], 3), dtype=np.uint8) + np.array((0x80, 0x80, 0x80), dtype=np.uint8)))
                    l_combined.append(arr_vertical_separate)
                    l_combined.append(pix_diff)
                pix_combined = np.hstack(l_combined)

                img_combined = Image.fromarray(pix_combined)
                img_combined.save(l_folder_path[i]+"0x{:016X}.png".format(l_init_vals[0]))


            globals()['l_l_arr'] = l_l_arr
            return

            # l_al_arr_00 = [
            #     np.array(list(map(lambda x: list(map(int, list(x))),hi.l_str_I)), dtype=np.uint8),
            #     np.array(list(map(lambda x: list(map(int, list(x))),hi.l_str_x)), dtype=np.uint8),
            #     np.array(list(map(lambda x: list(map(int, list(x))),hi.l_str_y)), dtype=np.uint8),
            #     np.array(list(map(lambda x: list(map(int, list(x))),hi.l_str_z)), dtype=np.uint8),
            #     np.array(list(map(lambda x: list(map(int, list(x))),hi.l_str_w)), dtype=np.uint8),
            # ]
            # pix_red_col = np.zeros((l_arr_00[0].shape[0], 1, 3), dtype=np.uint8) + np.array([0xFF, 0x00, 0x00], dtype=np.uint8)
            # l = [arr_colors_bw[l_arr_00[0]]]
            # for arr in l_arr_00[1:]:
            #     l.append(pix_red_col)
            #     l.append(arr_colors_bw[arr])
            # pix_00 = np.hstack(l)
            # img_00 = Image.fromarray(pix_00)
            # img_00.save(folder_dir_diff_00+"0x{:016X}.png".format(l_init_vals[0]))

            # l_arr_01 = [(arr[:-1] != arr[1:]).astype(np.uint8) for arr in l_arr_00]
            # pix_red_col = np.zeros((l_arr_01[0].shape[0], 1, 3), dtype=np.uint8) + np.array([0xFF, 0x00, 0x00], dtype=np.uint8)
            # l = [arr_colors_bw[l_arr_01[0]]]
            # for arr in l_arr_01[1:]:
            #     l.append(pix_red_col)
            #     l.append(arr_colors_bw[arr])
            # pix_01 = np.hstack(l)
            # img_01 = Image.fromarray(pix_01)
            # img_01.save(folder_dir_diff_01+"0x{:016X}.png".format(l_init_vals[0]))

            # l_arr_02 = [(arr[:-1] != arr[1:]).astype(np.uint8) for arr in l_arr_01]
            # pix_red_col = np.zeros((l_arr_02[0].shape[0], 1, 3), dtype=np.uint8) + np.array([0xFF, 0x00, 0x00], dtype=np.uint8)
            # l = [arr_colors_bw[l_arr_02[0]]]
            # for arr in l_arr_02[1:]:
            #     l.append(pix_red_col)
            #     l.append(arr_colors_bw[arr])
            # pix_02 = np.hstack(l)
            # img_02 = Image.fromarray(pix_02)
            # img_02.save(folder_dir_diff_02+"0x{:016X}.png".format(l_init_vals[0]))

            # l_arr_03 = [(arr[:-1] != arr[1:]).astype(np.uint8) for arr in l_arr_02]
            # pix_red_col = np.zeros((l_arr_03[0].shape[0], 1, 3), dtype=np.uint8) + np.array([0xFF, 0x00, 0x00], dtype=np.uint8)
            # l = [arr_colors_bw[l_arr_03[0]]]
            # for arr in l_arr_03[1:]:
            #     l.append(pix_red_col)
            #     l.append(arr_colors_bw[arr])
            # pix_03 = np.hstack(l)
            # img_03 = Image.fromarray(pix_03)
            # img_03.save(folder_dir_diff_03+"0x{:016X}.png".format(l_init_vals[0]))

            # l_arr_04 = [(arr[:-1] != arr[1:]).astype(np.uint8) for arr in l_arr_03]
            # pix_red_col = np.zeros((l_arr_04[0].shape[0], 1, 3), dtype=np.uint8) + np.array([0xFF, 0x00, 0x00], dtype=np.uint8)
            # l = [arr_colors_bw[l_arr_04[0]]]
            # for arr in l_arr_04[1:]:
            #     l.append(pix_red_col)
            #     l.append(arr_colors_bw[arr])
            # pix_04 = np.hstack(l)
            # img_04 = Image.fromarray(pix_04)
            # img_04.save(folder_dir_diff_04+"0x{:016X}.png".format(l_init_vals[0]))
            

            # pix_01_ext = np.vstack((pix_01, np.zeros((pix_00.shape[0]-pix_01.shape[0], pix_01.shape[1], 3), dtype=np.uint8) + np.array((0x80, 0x80, 0x80), dtype=np.uint8)))
            # pix_02_ext = np.vstack((pix_02, np.zeros((pix_00.shape[0]-pix_02.shape[0], pix_02.shape[1], 3), dtype=np.uint8) + np.array((0x80, 0x80, 0x80), dtype=np.uint8)))
            # pix_03_ext = np.vstack((pix_03, np.zeros((pix_00.shape[0]-pix_03.shape[0], pix_03.shape[1], 3), dtype=np.uint8) + np.array((0x80, 0x80, 0x80), dtype=np.uint8)))
            # pix_04_ext = np.vstack((pix_04, np.zeros((pix_00.shape[0]-pix_04.shape[0], pix_04.shape[1], 3), dtype=np.uint8) + np.array((0x80, 0x80, 0x80), dtype=np.uint8)))

            # pix_blue_col = np.zeros((pix_00.shape[0], 5, 3), dtype=np.uint8) + np.array((0x00, 0x00, 0xFF), dtype=np.uint8)

            # pix_combind = np.hstack((
            #     pix_00,
            #     pix_blue_col, pix_01_ext,
            #     pix_blue_col, pix_02_ext,
            #     pix_blue_col, pix_03_ext,
            #     pix_blue_col, pix_04_ext,
            # ))
            # img_combined = Image.fromarray(pix_combind)
            # img_combined.save(folder_dir_combined+"0x{:016X}.png".format(l_init_vals[0]))


    create_simple_hashes(l_l_i_num[0])

    # l_proc = [mp.Process(target=create_simple_hashes, args=(l_i_num, )) for l_i_num in l_l_i_num]
    # for proc in l_proc: proc.start()
    # for proc in l_proc: proc.join()
