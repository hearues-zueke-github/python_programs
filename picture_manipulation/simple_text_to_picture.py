#! /usr/bin/python3.6

import os

import numpy as np

from functools import reduce
from textwrap import wrap

# from PIL import Image
from PIL import Image, ImageDraw, ImageFont

if __name__ == "__main__":

    font_size = 8*1

    font_name = "kongtext.ttf"

    # alphabet_small = "abcdefghijklmnopqrstuvwxyz"
    # alphabet_big = alphabet_small.upper()
    # symbols = ".,:;-_?!=()[]{}\'\"#+-*/\\$%&"
    # symbols_2 = "!?@<>|"
    # lines = [alphabet_small, alphabet_big, symbols, symbols_2]

    # lines_amount = 40
    # with open("simple_text_to_picture.py") as fin:
    #     lines = list(map(lambda x: x.replace("\n", ""), fin.readlines()))
    #     print("len(lines): {}".format(len(lines)))
    # lines = lines[:((lambda x: x if x < lines_amount else lines_amount)(len(lines)))]

    lines = wrap(str(7**1230), 40)

    col_white = (255, 255, 255)
    col_green = [0, 255, 0]
    
    # is_show_lines = True
    is_show_lines = False
    if is_show_lines:
        x_offset = 1
        y_offset = 1
        x_space = font_size+1
        y_space = font_size+1
    else:
        x_offset = 0
        y_offset = 0
        x_space = font_size
        y_space = font_size

    rows = len(lines)
    cols = reduce(lambda a, b: a if a > len(b) else len(b), lines, 0)
    pix = np.zeros((y_offset+y_space*rows, x_offset+x_space*cols, 3), dtype=np.uint8)
    print("rows: {}".format(rows))
    print("cols: {}".format(cols))

    if is_show_lines:
        for i in range(0, rows+1):
            pix[i*(font_size+1)] = col_green
        for i in range(0, cols+1):
            pix[:, i*(font_size+1)] = col_green
    
    img = Image.fromarray(pix)
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(font_name, font_size)
    
    def draw_char(x, y, c, color):
        # draw.text((x, y), c, tuple(np.random.randint(0, 256, (3, )).tolist()), font=font)
        # draw.text((x, y), c, col_white, font=font)
        draw.text((x, y), c, color, font=font)

    for j, line in enumerate(lines):
        # rand_col = (np.random.randint(50, 256),
        #             np.random.randint(50, 256),
        #             np.random.randint(50, 256))
        for i, c in enumerate(line):
            # draw_char(x_offset+x_space*i, y_offset+y_space*j, c, rand_col)
            draw_char(x_offset+x_space*i, y_offset+y_space*j, c, col_white)

    if not os.path.exists("images"):
        os.makedirs("images")
    img.save("images/text_to_picture.png")
