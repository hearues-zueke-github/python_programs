#! /usr/bin/python3.5

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

mod = 256
n = 3
a = np.random.randint(0, mod, (n, n)).astype(np.int64)
b = np.random.randint(0, mod, (n, n)).astype(np.int64)
c = np.random.randint(0, mod, (n, n)).astype(np.int64)
d = np.random.randint(0, mod, (n, n)).astype(np.int64)

random_pos = tuple(map(tuple, np.random.randint(0, n, (12, 2))))
print("random_pos: {}".format(random_pos))

width = 1000
height = 1000

length = width*height
channel_r = []
channel_g = []
channel_b = []
for i in range(0, length//4):
    d = (a.dot(b)+c^d*2+4) % mod
    channel_r.append(d[random_pos[0]]) # 0, 0])
    channel_g.append(d[random_pos[1]]) # 0, 1])
    channel_b.append(d[random_pos[2]]) # 1, 0])
    c = (a.dot(b)^c+d*2+3) % mod
    channel_r.append(c[random_pos[3]]) # 0, 0])
    channel_g.append(c[random_pos[4]]) # 0, 1])
    channel_b.append(c[random_pos[5]]) # 1, 0])
    b = (a.dot(b)+c^d*2+2) % mod
    channel_r.append(b[random_pos[6]]) # 0, 0])
    channel_g.append(b[random_pos[7]]) # 0, 1])
    channel_b.append(b[random_pos[8]]) # 1, 0])
    a = (a.dot(b)^c+d*2+1) % mod
    channel_r.append(a[random_pos[9]]) # 0, 0])
    channel_g.append(a[random_pos[10]]) # 0, 1])
    channel_b.append(a[random_pos[11]]) # 1, 0])
    a = a.T
    
channel_r = channel_r[::-1]
channel_g = channel_g[::-1]
channel_b = channel_b[::-1]
# print("channel_r:\n{}".format(channel_r))
# print("channel_g:\n{}".format(channel_g))
# print("channel_b:\n{}".format(channel_b))
# print("channel_r: {}".format(channel_r))
print("len(channel_r): {}".format(len(channel_r)))
print("len(channel_g): {}".format(len(channel_g)))
print("len(channel_b): {}".format(len(channel_b)))

# x = np.arange(0, len(channel_r))
# plt.figure()
# plt.plot(x, channel_r, "b.")
# plt.show()

pix = np.zeros((height, width, 3)).astype(np.uint8)
layer_r = np.array(channel_r).astype(np.uint8).reshape((height, width))
layer_g = np.array(channel_g).astype(np.uint8).reshape((height, width))
layer_b = np.array(channel_b).astype(np.uint8).reshape((height, width))
pix[:, :, 0] = layer_r
pix[:, :, 1] = layer_g
pix[:, :, 2] = layer_b
img = Image.fromarray(pix)
img.save("test_random_pixels_n_{}_h_{}_w_{}.png".format(n, height, width))
