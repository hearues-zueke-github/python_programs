#! /usr/bin/python3.5

import numpy as np

import matplotlib.pyplot as plt

from functools import reduce
from PIL import Image, ImageDraw, ImageFont

randint = np.random.randint

def one_more_match(results, max_x, max_y):
    results = np.copy(results)
    x = 0
    y = 0

    results[0, 0] += 1

    while (x < max_x and y < max_y) and (x != max_x-1 or y != max_y-1):
    # while (x < 16 and y < 16) and (x != 15 or y != 15):
        if randint(0, 2) == 0:
            x += 1
        else:
            y += 1

        results[y, x] += 1

    return results

def play_matches(max_matches):
    results = np.zeros((max_y+1, max_x+1)).astype(np.int)

    for _ in range(0, max_matches):
        results = one_more_match(results, max_x, max_y)

    return results

max_matches = 10000

max_x = 40
max_y = max_x

results = play_matches(max_matches)

results_percents = np.zeros(results.shape)
results_percents[0, 0] = 1.
for i in range(1, max_x):
    sum_matches = np.sum(results[:i, i])+np.sum(results[i, :i+1])
    for j in range(0, i):
        results_percents[j, i] = results[j, i]/sum_matches*2
    # for j in range(0, i):
    #     results_percents[i, j] = results[i, j]/sum_matches
    results_percents[i, i] = results[i, i]/sum_matches

results_with_scores = list(map(lambda y: list(map(lambda x: (x, y, results[y, x]), range(0, 17))), range(0, 17)))
results_with_scores = reduce(lambda a, b: a+b, results_with_scores, [])

results_with_scores_sorted = sorted(results_with_scores, key=lambda x: x[::-1])[::-1]

print("max_matches: {}".format(max_matches))
print("results:\n{}".format(results))
print("results_with_scores_sorted: {}".format(results_with_scores_sorted))

def test_pixel_text():
    pix = np.zeros((800, 800, 3)).astype(np.uint8)

    img = Image.fromarray(pix)
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype("sans-serif.ttf", 16)

    font = ImageFont.truetype("Commodore Pixelized v1.2.ttf", 10)

    sample_text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?!-"

    draw.text((0,  0), " "*0+sample_text, (255, 255, 255), font=font)
    draw.text((0, 10), " "*1+sample_text, (255, 255, 255), font=font)
    draw.text((0, 20), " "*2+sample_text, (255, 255, 255), font=font)
    draw.text((0, 30), " "*1+sample_text, (255, 255, 255), font=font)
    draw.text((0, 40), " "*0+sample_text, (255, 255, 255), font=font)

    # for x in range(5, 41):
        # font = ImageFont.truetype("Commodore Pixelized v1.2.ttf", x)
    #     draw.text((0, x*25),str(x)+" Sample Text",(255,255,255),font=font)

    img.save("test_image.png", fomrat="PNG")

max_value = float(np.max(results))
print("max_value: {}".format(max_value))

width = 60
height = width
pix = np.zeros((height*max_y, width*max_x, 3)).astype(np.uint8)

for y in range(0, max_y):
    for x in range(0, max_x):
        pix[height*y:height*(y+1), width*x:width*(x+1)] = np.min((int(results_percents[y, x]*256), 255))
        # pix[height*y:height*(y+1), width*x:width*(x+1)] = np.min((int(results[y, x]/max_value*256), 255))

img = Image.fromarray(pix)
draw = ImageDraw.Draw(img)

font_size = 16
font = ImageFont.truetype("arial.ttf", font_size)
# font = ImageFont.truetype("Commodore Pixelized v1.2.ttf", font_size)

color_1 = (255, 255, 0)
color_2 = (0, 255, 255)

text_x = 60
text_y = 60

for y in range(0, max_y):
    for x in range(0, max_x):
        res = results[y, x]
        text_move_x = int(len(str(res))/2.*font_size*0.5)
        res_percent = results_percents[y, x]
        str_text = "0." if res_percent < 0.0001 else str(res_percent)
        # str_text = str(res_percent)
        draw.text((x*text_x+30-text_move_x, y*text_y+25), str_text[:np.min((5, len(str_text)))], color_1, font=font)

img.save("results_csgo_8_13.png", format="PNG")
