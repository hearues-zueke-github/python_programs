import numpy as np

def get_square_tiles(pix, tw):
    x, y, z = pix.shape
    return pix.transpose(0, 2, 1) \
              .reshape((x//tw, tw*z, y)) \
              .transpose(0, 2, 1) \
              .reshape((x*y//tw//tw, tw, tw, z)) \
              .transpose(0, 2, 1, 3) \
              .reshape((x//tw, y//tw, tw, tw, z)) # this is crazy!
