#! /usr/bin/python3.6

import os

import numpy as np

# from PIL import Image
from PIL import Image, ImageDraw, ImageFont

if __name__ == "__main__":
    rows = 10
    cols = 17

    pix_field = np.zeros((rows*2+1, cols*2+1, 3), dtype=np.uint8)

    # set start and end pixels
    pix_field[0, 1] = (0, 0, 255)
    pix_field[-1, -2] = (255, 0, 0)

    # fill all in between white pixels
    for y in range(0, rows):
        for x in range(0, cols):
            pix_field[1+y*2, 1+x*2] = (255, 255, 255)

    # now add some gaps in between rows and cols
    rows_gaps = np.random.randint(0, 2, (rows-1, cols))
    cols_gaps = np.random.randint(0, 2, (rows, cols-1))

    pix_field[(lambda x: (x[0]*2+2, x[1]*2+1))(np.where(rows_gaps==1))] = (255, 255, 255)
    pix_field[(lambda x: (x[0]*2+1, x[1]*2+2))(np.where(cols_gaps==1))] = (255, 255, 255)

    img = Image.fromarray(pix_field)
    # img.show()
    if not os.path.exists("images"):
        os.makedirs("images")
    img.save("images/labyrinth.png", "PNG")
