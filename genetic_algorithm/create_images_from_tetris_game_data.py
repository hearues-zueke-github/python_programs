#! /usr/bin/python3

# -*- coding: utf-8 -*-

import datetime
import os
import sys
import subprocess
import shutil

import numpy as np

from PIL import Image

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

from utils_tetris import parse_tetris_game_data

import platform
print("platform.system(): {}".format(platform.system()))

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

if __name__ == "__main__":
    f = tempfile.NamedTemporaryFile(prefix="Test", suffix=".txt")
    TEMP_DIR = f.name[:f.name.rfind("/")]+"/"

    suffix = sys.argv[1]
    # file_name = sys.argv[1]
    # suffix = sys.argv[1]

    TEMP_DIR_PICS = TEMP_DIR+f"tetris_pictures_{suffix}/"
    if not os.path.exists(TEMP_DIR_PICS):
        os.makedirs(TEMP_DIR_PICS)
    else:
        shutil.rmtree(TEMP_DIR_PICS)
        os.makedirs(TEMP_DIR_PICS)

    print("TEMP_DIR_PICS: {}".format(TEMP_DIR_PICS))

    # file_name = f'tetris_game_data/{file_name}.ttrsfields'
    file_name = f'tetris_game_data/data_fields_{suffix}.ttrsfields'
    resize = 54
    
    DATA_FILE_PATH = PATH_ROOT_DIR+file_name

    l_colors = [
        [0x00, 0x00, 0x00],
        [0xFF, 0x00, 0x00],
        [0x00, 0xFF, 0x00],
        [0x00, 0x00, 0xFF],
        [0xFF, 0xFF, 0x00],
        [0xFF, 0x00, 0xFF],
        [0x00, 0xFF, 0xFF],
        [0x80, 0x80, 0xFF],
        [0x80, 0x80, 0x80],
    ]
    l_colors[-1:-1] = np.random.randint(64, 256, (40, 3), dtype=np.uint8).tolist()
    arr_colors = np.array(l_colors, dtype=np.uint8)


    d_data = parse_tetris_game_data(DATA_FILE_PATH)
    rows = d_data['rows']
    cols = d_data['cols']
    arr_fields = d_data['arr_fields']

    amount_pieces = d_data['amount_pieces']
    l_amount_group_pieces = d_data['l_amount_group_pieces']
    l_group_pieces = d_data['l_group_pieces']
    arr_pcs_idx_pos = d_data['arr_pcs_idx_pos']
    l_pcs_idx_l_index_to_group_idx_pos = d_data['l_pcs_idx_l_index_to_group_idx_pos']

    i = 0
    l_pcs_idx_l_group_pieces = []
    for pcs_idx in range(0, amount_pieces):
        l_group_pieces_part = []
        for _ in range(0, l_amount_group_pieces[pcs_idx]):
            l_group_pieces_part.append(l_group_pieces[i])
            i += 1
        l_pcs_idx_l_group_pieces.append(l_group_pieces_part)

    print("arr_fields.shape: {}".format(arr_fields.shape))
    print("arr_pcs_idx_pos.shape: {}".format(arr_pcs_idx_pos.shape))

    cells_amount_max = max([len(l)//2 for l in l_group_pieces])
    frame_w = cells_amount_max
    frame_h = (cells_amount_max+1)//2

    # check, if the l_pcs_idx_l_group_pieces is OK or not!

    for l_group_pieces_part in l_pcs_idx_l_group_pieces:
        # do for one piece 3 rotations and check the length of the set with the list!
        s_group_pieces_part = set([tuple(sorted(map(tuple, np.array(l).reshape((-1, 2)).tolist()))) for l in l_group_pieces_part])
        
        l_piece_pos_orig = l_group_pieces_part[0]
        l_piece_pos = sorted([tuple(l) for l in np.array(l_piece_pos_orig).reshape((-1, 2)).tolist()])
        l_t_piece_pos = [tuple(l_piece_pos)]
        for _ in range(0, 3):
            # transpose pos
            l_piece_pos_T = [(x, y) for y, x in l_piece_pos]
            # mirror x
            max_x = 0
            for _, x in l_piece_pos_T:
                if max_x < x:
                    max_x = x

            l_piece_pos_T_mirror = sorted([(y, max_x - x) for y, x in l_piece_pos_T])

            l_piece_pos = l_piece_pos_T_mirror
            l_t_piece_pos.append(tuple(l_piece_pos))

        s_t_piece_pos = set(l_t_piece_pos)
        assert s_group_pieces_part==s_t_piece_pos


    # find the same pieces, which are mirrored
    # also find the pieces, which will fit into the box with the sizes (h, w) = (frame_h, frame_w)

    d_pcs_idx_to_l_group = {pcs_idx: [tuple(map(tuple, sorted(np.array(l).reshape((-1, 2)).tolist()))) for l in l_group_pieces] for pcs_idx, l_group_pieces in enumerate(l_pcs_idx_l_group_pieces, 0)}

    l_pcs_idx_l_possible_pieces = []
    # find the piece_pos for the pcs_idx, where the piece can be shown in the frame with the size (frame_h, frame_w)!
    for l_group_pieces_part in l_pcs_idx_l_group_pieces:
        l_possible_pieces = []
        for l_piece_pos in l_group_pieces_part:
            arr = np.array(l_piece_pos).reshape((-1, 2))
            max_y, max_x = np.max(arr, axis=0)
            if max_y >= frame_h:
                continue

            l_possible_pieces.append(tuple(sorted(map(tuple, arr.tolist()))))
        l_pcs_idx_l_possible_pieces.append(sorted(set(l_possible_pieces)))

    l_pcs_idx_one_piece_pos = [l[0] for l in l_pcs_idx_l_possible_pieces]
    d_pcs_idx_to_one_piece_pos = {pcs_idx: one_piece_pos for pcs_idx, one_piece_pos in enumerate(l_pcs_idx_one_piece_pos, 0)}

    # sys.exit(0)

    # # TODO: need to be added later!
    # using_pieces = d_data['using_pieces']
    using_pieces = 4


    arr_1_col = np.zeros((rows, 1), dtype=np.uint8)+len(arr_colors)-1
    # arr_2_col = np.zeros((rows, 1), dtype=np.uint8)+len(arr_colors)-1
    arr_2_col = np.zeros((rows, 1+frame_w+1), dtype=np.uint8)+len(arr_colors)-1
    arr_1_row = np.zeros((1, cols+arr_1_col.shape[1]+arr_2_col.shape[1]), dtype=np.uint8)+len(arr_colors)-1

    start_y = 3
    start_x = cols+2
    l_y_x_show_pos = [(start_y+i*(frame_h+1), start_x) for i in range(0, using_pieces-1)]

    between_frames = 1

    h_total = 1080
    w_total = 1920

    def create_new_image(field_i, arr):
        arr = np.hstack((arr_1_col, arr, arr_2_col))
        arr = np.vstack((arr_1_row, arr, arr_1_row))

        for y, x in l_y_x_show_pos:
            arr[y:y+frame_h, x:x+frame_w] = 0

        for (y_start, x_start), pcs_idx in zip(l_y_x_show_pos, arr_pcs_idx_pos[field_i:field_i+using_pieces-1, 0]):
            for y, x in l_pcs_idx_one_piece_pos[pcs_idx]:
                arr[y_start+y, x_start+x] = pcs_idx+1

        pix = arr_colors[arr]
        img = Image.fromarray(pix)
        width, height = img.size
        img = img.resize((width*resize, height*resize))

        w, h = img.size
        w_n = (h_total * w) //h
        img = img.resize((w_n, h_total), Image.LANCZOS)

        w_left = (w_total-w_n)//2
        w_right = w_total-w_left-w_n

        pix = np.array(img)
        pix_left = np.zeros((h_total, w_left, 3), dtype=np.uint8)
        pix_right = np.zeros((h_total, w_right, 3), dtype=np.uint8)

        pix = np.hstack((pix_left, pix, pix_right))
        img = Image.fromarray(pix)

        return img


    arr_fields_before, arr_fields_after = arr_fields[0::2], arr_fields[1::2]
    i = 0
    for field_i, t_arr in enumerate(zip(arr_fields_before, arr_fields_after[:-using_pieces+1]), 0):
        print("field_i: {}".format(field_i))
        for arr in t_arr:
            img = create_new_image(field_i, arr)
            for _ in range(0, between_frames):
                img.save(TEMP_DIR_PICS+f"field_nr_{i:05}.png")
                i += 1

    # create gif command: convert -delay 5 -loop 0 *.png animatedGIF.gif

    prefix_file_name = "output"
    extension_file_name = "mp4"
    FPS = 60
    COMMAND = f"ffmpeg -framerate 20 -i field_nr_%05d.png -r {FPS}"
    # COMMAND = f"ffmpeg -framerate {FPS} -i field_nr_%05d.png"

    old_file_name = f"{prefix_file_name}.{extension_file_name}"
    FNULL = open(os.devnull, "wb")
    proc = subprocess.Popen(f"cd {TEMP_DIR_PICS} && {COMMAND} {old_file_name}", shell=True, stdout=FNULL)
    # proc = subprocess.Popen("cd {} && convert -delay 5 -loop 0 *.png animatedGIF.gif".format(TEMP_DIR_PICS), shell=True)
    FNULL.close()
    proc.wait()

    DIR_GIF_IMAGES = TEMP_DIR+"tetris_gifs/"
    # DIR_GIF_IMAGES = PATH_ROOT_DIR+"gifs/"
    if not os.path.exists(DIR_GIF_IMAGES):
        os.makedirs(DIR_GIF_IMAGES)

    l_hex = np.array(list('0123456789ABCDEF'))
    def get_random_hex():
        return ''.join(np.random.choice(l_hex, (4, )))


    def get_new_file_name():
        dt_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        hex_str = get_random_hex()
        return f"{extension_file_name}_suf_{suffix}_{dt_str}_{hex_str}.{extension_file_name}"

    new_file_name = get_new_file_name()

    shutil.copyfile(TEMP_DIR_PICS+old_file_name, DIR_GIF_IMAGES+new_file_name)
