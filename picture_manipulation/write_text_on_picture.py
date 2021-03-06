#! /usr/bin/python3.6

import os
import sys

import numpy as np

# from PIL import Image
from PIL import Image, ImageDraw, ImageFont

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

if __name__ == "__main__":
    print("Hello World!")

    pix = np.zeros((100, 400, 3), dtype=np.uint8)
    img = Image.fromarray(pix)
    draw = ImageDraw.Draw(img)

    # font_name = "joystix monospace.ttf"
    font_name = "kongtext.ttf"
    alphabet_small = "abcdefghijklmnopqrstuvwxytz"
    alphabet_big = alphabet_small.upper()
    col_white = (255, 255, 255)
    # for row in range(0, 6):
    #     y = 10+35*row
    #     x = 20
    #     font_size = 8+row*8
    #     font = ImageFont.truetype(font_name, font_size)
    #     draw.text((x, y), "x: {}, y: {}, font_size: {}, TEst Hello Text!".format(x, y, font_size), col_white, font=font)
    #     draw.text((x, y), "x: {}, y: {}, font_size: {}, TEst Hello Text!".format(x, y, font_size), col_white, font=font)
    def draw_small(x, y):
        draw.text((x, y), "x: {}, y: {}, ".format(x, y)+alphabet_small, col_white, font=font)
    def draw_big(x, y):
        draw.text((x, y), "x: {}, y: {}, ".format(x, y)+alphabet_big, (255, 255, 0), font=font)
    
    font_size = 8
    font = ImageFont.truetype(font_name, font_size)

    draw_small(16, 33)
    draw_big(24, 8)
    # draw_small(20, 50)

    # draw_small(0, 0)
    # draw_big(0, 8)
    # draw_small(8, 16)
    # draw_big(16, 24)

    # img.show()
    if not os.path.exists("images"):
        os.makedirs("images")
    img.save("images/test_font_sizes.png")
