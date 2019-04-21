#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import string

import numpy as np

from textwrap import wrap

from PIL import Image, ImageFont, ImageDraw

if __name__ == "__main__":
    font_name = "Graph-35-pix.ttf"
    # font_name = "712_serif.ttf"
    # font_size = 8
    font_size = 8*1

    fnt = ImageFont.truetype('../fonts/{}'.format(font_name), font_size)
    tw, th = fnt.getsize("a")

    chars_per_line = 35
    lines_amount = 4

    height = th*lines_amount+lines_amount+1
    width = tw*chars_per_line

    pix2 = np.zeros((height, width, 3), dtype=np.uint8)
    img2 = Image.fromarray(pix2)
    d = ImageDraw.Draw(img2)

    string_temp = np.array(list(string.ascii_letters+"0123456789-_?!=()|&"))
    idx = np.random.randint(0, string_temp.shape[0], (chars_per_line*lines_amount))
    print("idx: {}".format(idx))
    print("idx.shape: {}".format(idx.shape))
    one_liner = "".join(string_temp[idx])
    print("one_liner: {}".format(one_liner))
    random_text_lines = wrap(one_liner, chars_per_line)

    print("random_text_lines: {}".format(random_text_lines))

    for i, line in enumerate(random_text_lines):
        d.text((1, th*i+1+i), line, font=fnt, fill=(255, 255, 255))

    img2.save("images/test.png")
