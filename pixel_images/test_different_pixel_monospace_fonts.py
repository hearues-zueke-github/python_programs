#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import string

import numpy as np

from PIL import Image, ImageDraw, ImageFont

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

# def doing_img_example_2():
#     pix = np.zeros((100, 100, 3), dtype=np.uint8)
#     img = Image.fromarray(pix)
#     draw = ImageDraw.Draw(img)

#     font_name = "kongtext.ttf"

#     font_size = 8
#     font = ImageFont.truetype(font_name, font_size)
#     font_8 = ImageFont.truetype(font=font_name, size=font_size)

#     color_black = (0, 255, 80)

#     draw.text(xy=(10, 20), text="AAde", fill=(255, 80, 0), font=font_8)
#     draw.text(xy=(18, 28), text="AAde", fill=(255, 80, 0), font=font_8)
#     draw.text(xy=(26, 36), text="A!de", fill=(255, 80, 0), font=font_8)
#     draw.text(xy=(18, 44), text="AA.?", fill=(255, 80, 0), font=font_8)

#     img = img.resize((img.width*4, img.height*4))
#     img.show()


def test_pixel_font_sizes():
    pix = np.zeros((300, 900, 3), dtype=np.uint8)
    img = Image.fromarray(pix)
    draw = ImageDraw.Draw(img)
    l_font_name = [
        'basis33.ttf',
        'kongtext.ttf',
        'monogram_extended.ttf',
        'monogram.ttf',
        'GnuUnifontFull-Pm9P.ttf',
        'MozartNbp-93Ey.ttf',
    ]

    # font_size = 8
    font_size = 16
    l_font = [ImageFont.truetype(font=font_name, size=font_size) for font_name in l_font_name]

    font_color = (0, 255, 80)

    for i, font in enumerate(l_font, 0):
        text = '({}, {}), {}'.format(l_font_name[i], font_size, string.digits+string.ascii_letters)
        # test = 'font_name: {}, font_size: {}'.format(l_font_name[i], font_size)
        draw.text(xy=(20, 20+font_size*i), text=text, fill=font_color, font=font)
    img = img.resize((img.width, img.height))
    
    # img.show()

    img.save(TEMP_DIR+'font_test_temp.png')


def test_one_font_sizes(font_name):
    pix = np.zeros((1800, 1000, 3), dtype=np.uint8)
    img = Image.fromarray(pix)
    draw = ImageDraw.Draw(img)

    font_size_max = 45
    l_font = [(font_size, ImageFont.truetype(font=font_name, size=font_size)) for font_size in range(1, font_size_max)]

    font_color = (0, 255, 80)
    for i, (font_size, font) in enumerate(l_font, 0):
        text = '({}, {})'.format(font_name, font_size)
        draw.text(xy=(20, 20+font_size_max*i), text=text, fill=font_color, font=font)
    img = img.resize((img.width, img.height))
    
    pix_txt = np.array(img)
    idxs = (pix_txt == font_color) | (pix_txt == (0, 0, 0))
    pix_txt[idxs] = 255
    pix_txt[~idxs] = 0
    img_txt = Image.fromarray(pix_txt)

    img.save(TEMP_DIR+'font_test_temp.png')
    img_txt.save(TEMP_DIR+'font_test_txt_temp.png')


if __name__ == '__main__':
    print("Hello World!")

    # test_pixel_font_sizes()

    # 16, 32
    test_one_font_sizes('basis33.ttf')

    # 8, 16, 24, 32
    # test_one_font_sizes('kongtext.ttf')

    # 16, 32
    # test_one_font_sizes('monogram_extended.ttf')

    # 16, 32
    # test_one_font_sizes('monogram.ttf')

    # # 16, 32
    # test_one_font_sizes('GnuUnifontFull-Pm9P.ttf')
    
    # # 16, 32
    # test_one_font_sizes('MozartNbp-93Ey.ttf')
