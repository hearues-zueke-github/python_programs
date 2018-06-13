#! /usr/bin/python3.5

import os

import numpy as np

from PIL import Image

with open("/usr/bin/python2.7", "rb") as fin:
    arr = np.fromfile(fin, dtype=np.uint8, count=-1)

print("arr.shape: {}".format(arr.shape))

# image_ratio = (3, 4, 3) # height vs width, plus channelsize
image_ratio = (3, 4) # height vs width, plus channelsize

max_ratios = int(np.sqrt(arr.shape[0] / np.prod(image_ratio)))
print("max_ratios: {}".format(max_ratios))

from os.path import expanduser
home = expanduser("~")
print("home: {}".format(home))

folder_path = home+"/Pictures/bin_to_image/"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

height = max_ratios*image_ratio[0]
width = max_ratios*image_ratio[1]

pix = arr[:np.prod(image_ratio)*max_ratios**2] \
     .reshape((height,
               width))#,
               # image_ratio[2]))

img = Image.fromarray(pix)
img.save(folder_path+"python2_7_as_image.png", "PNG")
