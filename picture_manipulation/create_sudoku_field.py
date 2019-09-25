#! /usr/bin/python3.6

import os

import numpy as np

# from PIL import Image
from PIL import Image, ImageDraw, ImageFont

def doing_img_example_1():
    pix = np.zeros((100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(pix)
    draw = ImageDraw.Draw(img)

    # font_name = "joystix monospace.ttf"
    font_name = "kongtext.ttf"
    alphabet_small = "abcdefghijklmnopqrstuvwxytz"
    alphabet_big = alphabet_small.upper()
    col_white = (255, 255, 255)
    
    font_size = 8
    font = ImageFont.truetype(font_name, font_size)
    font_8 = ImageFont.truetype(font=font_name, size=font_size)

    color_1 = (0, 255, 80)
    x1 = 0; x2 = 99; y = 20
    draw.line(xy=[(x1, y), (x2, y)], fill=color_1)
    x1 = 0; x2 = 99; y = 27
    draw.line(xy=[(x1, y), (x2, y)], fill=color_1)

    y1 = 0; y2 = 99; x = 10
    draw.line(xy=[(x, y1), (x, y2)], fill=color_1)
    y1 = 0; y2 = 99; x = 10+7*1
    draw.line(xy=[(x, y1), (x, y2)], fill=color_1)
    y1 = 0; y2 = 99; x = 10+7*2
    draw.line(xy=[(x, y1), (x, y2)], fill=color_1)
    y1 = 0; y2 = 99; x = 24+15
    draw.line(xy=[(x, y1), (x, y2)], fill=color_1)
    y1 = 0; y2 = 99; x = 39+23
    draw.line(xy=[(x, y1), (x, y2)], fill=color_1)

    draw.text(xy=(10, 20), text="A", fill=(255, 80, 0), font=font_8)
    draw.text(xy=(17, 20), text="B", fill=(255, 80, 0), font=font_8)
    draw.text(xy=(24, 20), text="AB", fill=(255, 80, 0), font=font_8)
    draw.text(xy=(39, 20), text="ABc", fill=(255, 80, 0), font=font_8)

    img = img.resize((img.width*4, img.height*4))

    img.show()


def doing_img_example_2():
    pix = np.zeros((100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(pix)
    draw = ImageDraw.Draw(img)

    # font_name = "joystix monospace.ttf"
    font_name = "kongtext.ttf"

    font_size = 8
    font = ImageFont.truetype(font_name, font_size)
    font_8 = ImageFont.truetype(font=font_name, size=font_size)

    color_1 = (0, 255, 80)

    draw.text(xy=(10, 20), text="AAde", fill=(255, 80, 0), font=font_8)
    draw.text(xy=(18, 28), text="AAde", fill=(255, 80, 0), font=font_8)
    draw.text(xy=(26, 36), text="A!de", fill=(255, 80, 0), font=font_8)
    draw.text(xy=(18, 44), text="AA.?", fill=(255, 80, 0), font=font_8)

    img = img.resize((img.width*4, img.height*4))
    img.show()


def doing_img_example_3():
    pix = np.zeros((150, 150, 3), dtype=np.uint8)
    img = Image.fromarray(pix)
    draw = ImageDraw.Draw(img)

    # font_name = "joystix monospace.ttf"
    font_name = "kongtext.ttf"

    font_size = 8
    font = ImageFont.truetype(font_name, font_size)
    font_8 = ImageFont.truetype(font=font_name, size=font_size)

    color_1 = (0, 255, 80)

    y = 0
    for size in range(8, 33, 8):
        draw.text(xy=(0, y), text="AAde", fill=(255, 80, 0), font=ImageFont.truetype(font=font_name, size=size))
        y += size

    img = img.resize((img.width*4, img.height*4))
    img.show()


def doing_img_example_4():
    frame_width = 5
    inner_frame_width = 3
    most_inner_frame_width = 1
    font_size = 8
    
    fw = frame_width*2+inner_frame_width*2+most_inner_frame_width*2*3+font_size*3*3*3
    fh = frame_width*2+inner_frame_width*2+most_inner_frame_width*2*3+font_size*3*3*3
    pix = np.zeros((fh, fw, 3), dtype=np.uint8)
    img = Image.fromarray(pix)
    draw = ImageDraw.Draw(img)

    # font_name = "joystix monospace.ttf"
    font_name = "kongtext.ttf"

    font = ImageFont.truetype(font_name, font_size)
    font_placed = ImageFont.truetype(font=font_name, size=font_size*3)
    font_guess = ImageFont.truetype(font=font_name, size=font_size)

    color_1 = (0, 0, 0)
    color_2 = (255, 255, 255)
    color_digit_placed = (0, 0, 0)
    color_digit_filled = (0, 0, 255)
    color_digit_guess = (0, 255, 0)

    def draw_frame(draw, xy, width=1, color=(0, 0, 0)):
        x1, y1 = xy[0]
        x2, y2 = xy[1]
        draw.rectangle(xy=[(x1, y1), (x2, y1+width-1)], fill=color)
        draw.rectangle(xy=[(x2-width+1, y1), (x2, y2)], fill=color)
        draw.rectangle(xy=[(x1, y2-width+1), (x2, y2)], fill=color)
        draw.rectangle(xy=[(x1, y1), (x1+width-1, y2)], fill=color)

    draw_frame(draw=draw, xy=[(0, 0), (fw-1, fh-1)], width=frame_width, color=color_1)

    draw.rectangle(xy=[(frame_width, frame_width), (fw-frame_width-1, fh-frame_width-1)], fill=color_2)
    
    draw_frame(draw=draw, xy=[(frame_width+font_size*3*3+most_inner_frame_width*2, 0), (frame_width+(font_size*3*3+most_inner_frame_width*2)*2+inner_frame_width*2-1, fh-1)], color=color_1, width=inner_frame_width)
    draw_frame(draw=draw, xy=[(0, frame_width+font_size*3*3+most_inner_frame_width*2), (fw-1, frame_width+(font_size*3*3+most_inner_frame_width*2)*2+inner_frame_width*2-1)], color=color_1, width=inner_frame_width)
    
    x_width_space = font_size*3*3+most_inner_frame_width*2+inner_frame_width
    y_height_space = font_size*3*3+most_inner_frame_width*2+inner_frame_width
    for i in range(0, 3):
        draw_frame(draw=draw, xy=[(frame_width+font_size*3+x_width_space*i, 0), (frame_width+font_size*3*2+most_inner_frame_width*2-1+x_width_space*i, fh-1)], color=color_1, width=most_inner_frame_width)
        draw_frame(draw=draw, xy=[(0, frame_width+font_size*3+y_height_space*i), (fw-1, frame_width+font_size*3*2+most_inner_frame_width*2-1+y_height_space*i)], color=color_1, width=most_inner_frame_width)

    lst_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    # first, create the map for each cell pixel upper left corner positions!
    dict_pixel_pos_cell = {}

    for j1 in range(0, 3):
        for i1 in range(0, 3):
            for j2 in range(0, 3):
                for i2 in range(0, 3):
                    dict_pixel_pos_cell[(i1*3+i2, j1*3+j2)] = (frame_width+x_width_space*i1+(font_size*3+most_inner_frame_width)*i2, frame_width+y_height_space*j1+(font_size*3+most_inner_frame_width)*j2)

    digits_xy = {
        1: (0, 0), 2: (1, 0), 3: (2, 0),
        4: (0, 1), 5: (1, 1), 6: (2, 1),
        7: (0, 2), 8: (1, 2), 9: (2, 2),
    }

    # for i, (x, y) in enumerate(sorted(dict_pixel_pos_cell.values()), 0):
    #     # draw.rectangle(xy=[(x, y), (x+font_size*3-1, y+font_size*3-1)], fill=lst_colors[(i+x)%len(lst_colors)])
        
    #     for j, (x1, y1) in enumerate(sorted(digits_xy.values()), 0):
    #         xs = x+font_size*x1
    #         ys = y+font_size*y1
    #         draw.rectangle(xy=[(xs, ys), (xs+font_size-1, ys+font_size-1)], fill=lst_colors[(i+j+x1)%len(lst_colors)])

    pos_cell = (3, 6) # x, y
    x1, y1 = dict_pixel_pos_cell[pos_cell]
    used_digits_in_cell = [2, 5, 6, 7]
    
    for d in used_digits_in_cell:
        x, y = digits_xy[d]
        draw.text(xy=(x1+font_size*x, y1+font_size*y), text=str(d), fill=color_digit_guess, font=font_guess)


    pos_cell = (2, 3)
    x1, y1 = dict_pixel_pos_cell[pos_cell]
    d = 2
    draw.text(xy=(x1, y1), text=str(d), fill=color_digit_placed, font=font_placed)

    pos_cell = (4, 7)
    x1, y1 = dict_pixel_pos_cell[pos_cell]
    d = 6
    draw.text(xy=(x1, y1), text=str(d), fill=color_digit_filled, font=font_placed)

    # TODO: create a loop, where all guessed numbers are placed in the correct cell
    # plus also place the placed and filled digits too for each cell!

    img.save('images/test_image.png')
    img = img.resize((img.width*2, img.height*2))
    img.show()


if __name__ == "__main__":
    print("Hello World!")

    if not os.path.exists("images"):
        os.makedirs("images")

    # doing_img_example_1()

    # doing_img_example_2()
    
    # doing_img_example_3()

    doing_img_example_4()
