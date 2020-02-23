#! /usr/bin/python3.6

import os

import numpy as np

from PIL import Image, ImageDraw, ImageFont

def doing_img_example_1():
    pix = np.zeros((100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(pix)
    draw = ImageDraw.Draw(img)

    font_name = "kongtext.ttf"
    alphabet_small = "abcdefghijklmnopqrstuvwxytz"
    alphabet_big = alphabet_small.upper()
    col_white = (255, 255, 255)
    
    font_size = 8
    font = ImageFont.truetype(font_name, font_size)
    font_8 = ImageFont.truetype(font=font_name, size=font_size)

    color_black = (0, 255, 80)
    x1 = 0; x2 = 99; y = 20
    draw.line(xy=[(x1, y), (x2, y)], fill=color_black)
    x1 = 0; x2 = 99; y = 27
    draw.line(xy=[(x1, y), (x2, y)], fill=color_black)

    y1 = 0; y2 = 99; x = 10
    draw.line(xy=[(x, y1), (x, y2)], fill=color_black)
    y1 = 0; y2 = 99; x = 10+7*1
    draw.line(xy=[(x, y1), (x, y2)], fill=color_black)
    y1 = 0; y2 = 99; x = 10+7*2
    draw.line(xy=[(x, y1), (x, y2)], fill=color_black)
    y1 = 0; y2 = 99; x = 24+15
    draw.line(xy=[(x, y1), (x, y2)], fill=color_black)
    y1 = 0; y2 = 99; x = 39+23
    draw.line(xy=[(x, y1), (x, y2)], fill=color_black)

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

    font_name = "kongtext.ttf"

    font_size = 8
    font = ImageFont.truetype(font_name, font_size)
    font_8 = ImageFont.truetype(font=font_name, size=font_size)

    color_black = (0, 255, 80)

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

    font_name = "kongtext.ttf"

    font_size = 8
    font = ImageFont.truetype(font_name, font_size)
    font_8 = ImageFont.truetype(font=font_name, size=font_size)

    color_black = (0, 255, 80)

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

    font_name = "kongtext.ttf"

    font = ImageFont.truetype(font_name, font_size)
    font_placed = ImageFont.truetype(font=font_name, size=font_size*3)
    font_guess = ImageFont.truetype(font=font_name, size=font_size)

    color_black = (0, 0, 0)
    color_white = (255, 255, 255)
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

    draw_frame(draw=draw, xy=[(0, 0), (fw-1, fh-1)], width=frame_width, color=color_black)

    draw.rectangle(xy=[(frame_width, frame_width), (fw-frame_width-1, fh-frame_width-1)], fill=color_white)
    
    draw_frame(draw=draw, xy=[(frame_width+font_size*3*3+most_inner_frame_width*2, 0), (frame_width+(font_size*3*3+most_inner_frame_width*2)*2+inner_frame_width*2-1, fh-1)], color=color_black, width=inner_frame_width)
    draw_frame(draw=draw, xy=[(0, frame_width+font_size*3*3+most_inner_frame_width*2), (fw-1, frame_width+(font_size*3*3+most_inner_frame_width*2)*2+inner_frame_width*2-1)], color=color_black, width=inner_frame_width)
    
    x_width_space = font_size*3*3+most_inner_frame_width*2+inner_frame_width
    y_height_space = font_size*3*3+most_inner_frame_width*2+inner_frame_width
    for i in range(0, 3):
        draw_frame(draw=draw, xy=[(frame_width+font_size*3+x_width_space*i, 0), (frame_width+font_size*3*2+most_inner_frame_width*2-1+x_width_space*i, fh-1)], color=color_black, width=most_inner_frame_width)
        draw_frame(draw=draw, xy=[(0, frame_width+font_size*3+y_height_space*i), (fw-1, frame_width+font_size*3*2+most_inner_frame_width*2-1+y_height_space*i)], color=color_black, width=most_inner_frame_width)

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

    field = np.zeros((9, 9), dtype=np.int)
    # field[:] = [
    #     [3, 0, 8, 6, 0, 0, 0, 5, 0],
    #     [0, 0, 0, 0, 0, 7, 8, 0, 0],
    #     [0, 0, 5, 0, 0, 0, 0, 4, 2],
    #     [0, 0, 0, 0, 1, 6, 0, 0, 0],
    #     [0, 5, 0, 0, 0, 3, 0, 0, 7],
    #     [4, 2, 0, 0, 9, 0, 5, 0, 6],
    #     [0, 0, 0, 2, 0, 9, 0, 0, 0],
    #     [0, 7, 0, 0, 0, 0, 0, 0, 0],
    #     [8, 9, 0, 0, 0, 0, 4, 0, 0],
    # ]

    field[:] = [
        [8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 6, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 9, 0, 2, 0, 0],
        [0, 5, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 4, 5, 7, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 3, 0],
        [0, 0, 1, 0, 0, 0, 0, 6, 8],
        [0, 0, 8, 5, 0, 0, 0, 1, 0],
        [0, 9, 0, 0, 0, 0, 4, 0, 0],
    ]

    # pos_cell = (3, 6) # x, y
    # x1, y1 = dict_pixel_pos_cell[pos_cell]
    # used_digits_in_cell = [2, 5, 6, 7]
    
    # for d in used_digits_in_cell:
    #     x, y = digits_xy[d]
    #     draw.text(xy=(x1+font_size*x, y1+font_size*y), text=str(d), fill=color_digit_guess, font=font_guess)


    for y in range(0, 9):
        for x in range(0, 9):
            d = field[y, x]
            if d==0:
                continue
            pos_cell = (x, y)
            x1, y1 = dict_pixel_pos_cell[pos_cell]
            draw.text(xy=(x1, y1), text=str(d), fill=color_digit_placed, font=font_placed)

    # mappings for each row, col and 3x3 field!
    dict_row = {i: field[i, :] for i in range(0, 9)}
    dict_col = {i: field[:, i] for i in range(0, 9)}
    dict_3x3_field = {(j, i): field[3*(j+0):3*(j+1), 3*(i+0):3*(i+1)] for j in range(0, 3) for i in range(0, 3)}

    def get_possible_digits_not_finished_cells():
        s_all_digits = set(range(1, 9+1))
        d_possible_digits_in_cell = {}
        # get all possible numbers for each cell!
        not_finished_cells = []
        for y in range(0, 9):
            for x in range(0, 9):
                d = field[y, x]
                if d!=0:
                    d_possible_digits_in_cell[(y, x)] = set()
                    continue
                print("x: {}, y: {}".format(x, y))
                # continue
                # get all_available numbers of the field!
                r = dict_row[y]
                c = dict_col[x]
                f = dict_3x3_field[(y//3, x//3)]

                l = r[r!=0].flatten().tolist()+c[c!=0].flatten().tolist()+f[f!=0].flatten().tolist()
                available_digits = s_all_digits-set(l)
                # available_digits = list(s_all_digits-set(l))
                d_possible_digits_in_cell[(y, x)] = available_digits

                # print("r:\n{}".format(r))
                # print("c:\n{}".format(c))
                # print("f:\n{}".format(f))
                # print("l: {}".format(l))
                # print("available_digits: {}".format(available_digits))
                # break
            # break

                not_finished_cells.append((y, x))
        return d_possible_digits_in_cell, not_finished_cells

    d_possible_digits_in_cell, not_finished_cells = get_possible_digits_not_finished_cells()

    for y, x in not_finished_cells:
        pos_cell = (x, y)
        x1, y1 = dict_pixel_pos_cell[pos_cell]
        draw.rectangle(xy=[(x1, y1), (x1+font_size*3-1, y1+font_size*3-1)], fill=color_white)

        available_digits = d_possible_digits_in_cell[(y, x)]
        for d in available_digits:
            x2, y2 = digits_xy[d]
            draw.text(xy=(x1+font_size*x2, y1+font_size*y2), text=str(d), fill=color_digit_guess, font=font_guess)

    # img2 = img.resize((img.width*2, img.height*2))
    # img2.show()
    img.save('images/sudoku_solver_iter_{:02}.png'.format(0))


    # create a mapping, where every cells are in a list contained, where
    # they are going together!
    dict_lst_row = {j: [(j, i) for i in range(0, 9)] for j in range(0, 9)}
    dict_lst_col = {i: [(j, i) for j in range(0, 9)] for i in range(0, 9)}
    dict_lst_3x3_field = {(j1, i1): [(j1*3+j2, i1*3+i2) for j2 in range(0, 3) for i2 in range(0, 3)] for j1 in range(0, 3) for i1 in range(0, 3)}

    for iterations in range(1, 9):
        finished_cells = []
        for y, x in not_finished_cells:
            print("y: {}, x: {}".format(y, x))
            s = d_possible_digits_in_cell[(y, x)]
            if len(s)==1:
                finished_cells.append((y, x))
                field[y, x] = list(s)[0]
                continue

            l1_row = list(dict_lst_row[y])
            l1_row.remove((y, x))
            s_row = set(d_possible_digits_in_cell[(y, x)])
            for p in l1_row:
                s_row -= d_possible_digits_in_cell[p]

            l1_col = list(dict_lst_col[x])
            l1_col.remove((y, x))
            s_col = set(d_possible_digits_in_cell[(y, x)])
            for p in l1_col:
                s_col -= d_possible_digits_in_cell[p]

            l1_3x3_field = list(dict_lst_3x3_field[(y//3, x//3)])
            l1_3x3_field.remove((y, x))
            s_3x3_field = set(d_possible_digits_in_cell[(y, x)])
            for p in l1_3x3_field:
                s_3x3_field -= d_possible_digits_in_cell[p]

            print("- s_row: {}".format(s_row))
            print("- s_col: {}".format(s_col))
            print("- s_3x3_field: {}".format(s_3x3_field))

            if len(s_row)==1:
                finished_cells.append((y, x))
                field[y, x] = list(s_row)[0]
            elif len(s_col)==1:
                finished_cells.append((y, x))
                field[y, x] = list(s_col)[0]
            elif len(s_3x3_field)==1:
                finished_cells.append((y, x))
                field[y, x] = list(s_3x3_field)[0]

        for pos in finished_cells:
            not_finished_cells.remove(pos)

        for y, x in finished_cells:
            d = field[y, x]
            if d==0:
                continue
            pos_cell = (x, y)
            x1, y1 = dict_pixel_pos_cell[pos_cell]
            draw.rectangle(xy=[(x1, y1), (x1+font_size*3-1, y1+font_size*3-1)], fill=color_white)
            draw.text(xy=(x1, y1), text=str(d), fill=color_digit_filled, font=font_placed)

        d_possible_digits_in_cell, not_finished_cells = get_possible_digits_not_finished_cells()

        for y, x in not_finished_cells:
            pos_cell = (x, y)
            x1, y1 = dict_pixel_pos_cell[pos_cell]
            draw.rectangle(xy=[(x1, y1), (x1+font_size*3-1, y1+font_size*3-1)], fill=color_white)

            available_digits = d_possible_digits_in_cell[(y, x)]
            for d in available_digits:
                x2, y2 = digits_xy[d]
                draw.text(xy=(x1+font_size*x2, y1+font_size*y2), text=str(d), fill=color_digit_guess, font=font_guess)

        globals()['d_possible_digits_in_cell'] = d_possible_digits_in_cell
        globals()['not_finished_cells'] = not_finished_cells
        img.save('images/sudoku_solver_iter_{:02}.png'.format(iterations))

        # img2 = img.resize((img.width*2, img.height*2))
        # img2.show()
    
    # TODO: create a loop, where all guessed numbers are placed in the correct cell
    # plus also place the placed and filled digits too for each cell!

    # img.save('images/sudoku_solver_iter_{:02}.png'.format(iterations+1))


if __name__ == "__main__":
    print("Hello World!")

    if not os.path.exists("images"):
        os.makedirs("images")

    # doing_img_example_1()
    # doing_img_example_2()
    # doing_img_example_3()
    doing_img_example_4()
