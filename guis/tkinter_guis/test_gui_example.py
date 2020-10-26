#! /usr/bin/python3

import os
import sys

import numpy as np
import tkinter as tk

from PIL import Image, ImageTk

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

class App(tk.Frame):
    def __init__(self, parent, x_amount, y_amount, box_width, seed=0):
        print("parent.winfo_width(): {}".format(parent.winfo_width()))
        print("parent.winfo_height(): {}".format(parent.winfo_height()))
        # position...(y, x)

        self.seed = seed
        
        self.d_coordinate_to_move = {
            'N': 'U', 'S': 'D', 'W': 'L', 'E': 'R'
        }

        assert box_width%10==0
        self.box_width = box_width
        self.box_height = self.box_width
        self.x_amount = x_amount # 400//self.box_width
        self.y_amount = y_amount # 400//self.box_height

        self.modulo_number = x_amount*y_amount

        self.color_idx = 0
        self.colors = [
            '#FF0000', '#00FF00', '#0000FF',
            '#FF8000', '#80FF00', '#8000FF'
        ]
        self.current_color = self.colors[self.color_idx]

        # self.tiles_folder = PATH_ROOT_DIR+'tiles/'
        self.tiles_folder = PATH_ROOT_DIR+'tiles_snake/'

        self.img_arrow_up = Image.open(self.tiles_folder+'arrow_up_10x10_thick.png')
        self.img_line_ver_end_up = Image.open(self.tiles_folder+'line_straight_end_up_10x10_thick.png')
        self.img_line_ver = Image.open(self.tiles_folder+'line_straight_vertical_10x10_thick.png')
        self.img_line_dl = Image.open(self.tiles_folder+'line_down_left_10x10_thick.png')
        self.img_food = Image.open(self.tiles_folder+'food.png')

        self.img_arrow_up = self.img_arrow_up.resize((self.box_width, self.box_height))
        self.img_line_ver_end_up = self.img_line_ver_end_up.resize((self.box_width, self.box_height))
        self.img_line_ver = self.img_line_ver.resize((self.box_width, self.box_height))
        self.img_line_dl = self.img_line_dl.resize((self.box_width, self.box_height))      
        self.img_food = self.img_food.resize((self.box_width, self.box_height))      
        

        self.arr_arrow_up = np.array(self.img_arrow_up)
        self.arr_arrow_down = np.flip(self.arr_arrow_up, axis=0).copy()
        self.arr_arrow_right = self.arr_arrow_down.transpose(1, 0, 2).copy()
        self.arr_arrow_left = np.flip(self.arr_arrow_right, axis=1).copy()

        self.arr_line_end_up = np.array(self.img_line_ver_end_up)
        self.arr_line_end_down = np.flip(self.arr_line_end_up, axis=0).copy()
        self.arr_line_end_right = self.arr_line_end_down.transpose(1, 0, 2).copy()
        self.arr_line_end_left = np.flip(self.arr_line_end_right, axis=1).copy()

        self.arr_line_ver = np.array(self.img_line_ver)
        self.arr_line_hor = self.arr_line_ver.transpose(1, 0, 2).copy()

        self.arr_line_dl = np.array(self.img_line_dl)
        self.arr_line_dr = np.flip(self.arr_line_dl, axis=1).copy()
        self.arr_line_ul = np.flip(self.arr_line_dl, axis=0).copy()
        self.arr_line_ur = np.flip(self.arr_line_ul, axis=1).copy()

        assert box_width%10==0
        self.box_width = box_width
        self.box_height = self.box_width
        self.x_amount = x_amount # 400//self.box_width
        self.y_amount = y_amount # 400//self.box_height

        self.imgtk_arrow_up = ImageTk.PhotoImage(Image.fromarray(self.arr_arrow_up))
        self.imgtk_arrow_down = ImageTk.PhotoImage(Image.fromarray(self.arr_arrow_down))
        self.imgtk_arrow_right = ImageTk.PhotoImage(Image.fromarray(self.arr_arrow_right))
        self.imgtk_arrow_left = ImageTk.PhotoImage(Image.fromarray(self.arr_arrow_left))
        
        self.imgtk_line_end_up = ImageTk.PhotoImage(Image.fromarray(self.arr_line_end_up))
        self.imgtk_line_end_down = ImageTk.PhotoImage(Image.fromarray(self.arr_line_end_down))
        self.imgtk_line_end_right = ImageTk.PhotoImage(Image.fromarray(self.arr_line_end_right))
        self.imgtk_line_end_left = ImageTk.PhotoImage(Image.fromarray(self.arr_line_end_left))

        self.imgtk_line_ver = ImageTk.PhotoImage(Image.fromarray(self.arr_line_ver))
        self.imgtk_line_hor = ImageTk.PhotoImage(Image.fromarray(self.arr_line_hor))
        
        self.imgtk_line_dl = ImageTk.PhotoImage(Image.fromarray(self.arr_line_dl))
        self.imgtk_line_dr = ImageTk.PhotoImage(Image.fromarray(self.arr_line_dr))
        self.imgtk_line_ul = ImageTk.PhotoImage(Image.fromarray(self.arr_line_ul))
        self.imgtk_line_ur = ImageTk.PhotoImage(Image.fromarray(self.arr_line_ur))

        self.imgtk_food = ImageTk.PhotoImage(self.img_food)

        tk.Frame.__init__(self, parent)
        # self.canvas_bg = '#00ff00'
        # self.canvas_bg = '#95a4e5'
        self.canvas_bg = '#32469f'
        # self.canvas_bg = '#F0FFFF'
        self._createVariables(parent)
        self.ht_canvas = 5
        self._createCanvas()

        self.parent.bind('s', self.draw_random_rectangles)
        self.parent.bind('S', self.draw_random_rectangles)
        self.parent.bind('n', self.execute_moves)

        self.parent.bind('<Left>', self.move_left)
        self.parent.bind('<Right>', self.move_right)
        self.parent.bind('<Up>', self.move_up)
        self.parent.bind('<Down>', self.move_down)
        
        self.parent.bind('<r>', self.reset_snake)

        self.parent.bind('<space>', self.print_infos)

        self.pixel_1x1_image = tk.PhotoImage(width=1, height=1)

        self.canvas.update()

        x_right_canvas = self.canvas.winfo_x()+self.canvas.winfo_width()+10
        y_down_canvas = self.canvas.winfo_y()+self.canvas.winfo_height()+10
        print("x_right_canvas: {}".format(x_right_canvas))

        self.parent.geometry('{}x{}'.format(x_right_canvas+130, y_down_canvas+10))
        self.parent.update()

        # font = tk.font.Font(family=tk.font.nametofont("TkDefaultFont").cget("family"), size=15)
        self.btn_test = tk.Button(master=self.parent, text='create something!', image=self.pixel_1x1_image, compound=tk.CENTER, font=('helvetica', 8))
        self.btn_test.place(x=x_right_canvas, y=20, width=120, height=30)
        self.btn_test.bind('<Button-1>', self.draw_random_rectangles)

        self.txt_moves = tk.Text(master=self.parent)
        self.txt_moves.place(x=x_right_canvas, y=60, width=120, height=30)

        self.reset_snake(event=None)


    def reset_snake(self, event):
        self.l_positions = [(0, i) for i in range(0, 3)]
        self.l_possible_positions = [(y, x) for y in range(0, self.y_amount) for x in range(0, self.x_amount)]
        for p in self.l_positions:
            self.l_possible_positions.remove(p)
        self.amount_possible_positions = len(self.l_possible_positions)
        self.l_directions = ['E']*(len(self.l_positions)-1)
        self.l_saved_moves = []
        first_pos = self.l_positions[-1]

        self.current_x = first_pos[1]
        self.current_y = first_pos[0]
        self.prev_x = self.current_x
        self.prev_y = self.current_y
        self.current_direction = 'E'
        
        self.last_number = 0
        self.l_food_positions = []

        self.canvas.create_rectangle(
            0, 0, self.ht_canvas+self.x_amount*self.box_width, self.ht_canvas+self.y_amount*self.box_height, fill=self.canvas_bg, width=0, # disabledoutline=tk.DISABLED,
            # x0=0, y0=0, x1=self.x_amount*self.box_width, y1=self.y_amount*self.box_height, fill=self.canvas_bg, width=0, # disabledoutline=tk.DISABLED,
        )

        self.canvas.create_image(self.ht_canvas+self.box_width*0, self.ht_canvas+0, image=self.imgtk_line_end_right, anchor=tk.NW)
        for i in range(1, len(self.l_positions)-1):
            self.canvas.create_image(self.ht_canvas+self.box_width*i, self.ht_canvas+0, image=self.imgtk_line_hor, anchor=tk.NW)
        self.canvas.create_image(self.ht_canvas+self.box_width*(len(self.l_positions)-1), self.ht_canvas+0, image=self.imgtk_arrow_right, anchor=tk.NW)

        print('')
        self.set_pseudo_random_food_position()


    def execute_moves(self, event):
        txt_str = self.txt_moves.get('1.0', 'end-1c')
        txt_str = txt_str.upper()
        move_nr = 1
        for i, c in enumerate(txt_str, 0):
            if not c in ['U', 'D', 'L', 'R']:
                continue
            if c=='U':
                ret_val = self.move_up(event=None)
            elif c=='D':
                ret_val = self.move_down(event=None)
            elif c=='L':
                ret_val = self.move_left(event=None)
            elif c=='R':
                ret_val = self.move_right(event=None)
            
            if ret_val==True:
                print("move_nr: {}, char_pos: {}, direction: {}".format(move_nr, i, c))
                move_nr += 1


    def set_pseudo_random_food_position(self):
        # need entropy for the RNG: inputs, positions, seed etc.
        if len(self.l_positions)>=self.x_amount*self.y_amount:
            self.food_x = -1
            self.food_y = -1
            print('\nNo more moves possible!!!!!!')
            print("self.l_saved_moves: {}".format(''.join(self.l_saved_moves)))
            arr = np.array(self.l_saved_moves)
            direction_changes = np.sum(arr[1:]!=arr[:-1])
            print("direction_changes: {}".format(direction_changes))
            print("len(self.l_saved_moves): {}".format(len(self.l_saved_moves)))
            return
        # add all positions of the current snake to the number!
        for y, x in self.l_positions:
            self.last_number += x*self.y_amount+y
        for y, x in self.l_food_positions:
            self.last_number += y*self.x_amount+x
        # add also the last moves used before getting the food!
        d_move_value = {'U': 1, 'D': self.y_amount, 'L': 1, 'R': self.x_amount}
        for i, m in enumerate(self.l_saved_moves[-30:] if len(self.l_saved_moves)>=30 else self.l_saved_moves, 0):
            self.last_number += i+d_move_value[m]
        self.last_number += 1
        self.last_number = self.last_number%self.amount_possible_positions
        
        # self.last_number = self.last_number%self.modulo_number
        # food_y = self.last_number//self.x_amount
        # food_x = self.last_number%self.x_amount
        # while True:
        #     food_x += 1
        #     if food_x>=self.x_amount:
        #         food_x = 0
        #         food_y += 1
        #         if food_y>=self.y_amount:
        #             food_y = 0

        #     food_pos = (food_y, food_x)

        #     sys.stdout.write('{}, '.format(food_pos))
        #     sys.stdout.flush()

        #     if not food_pos in self.l_positions:
        #         break

        food_pos = self.l_possible_positions[self.last_number]
        food_y, food_x = food_pos
        # food_pos = self.l_possible_positions.pop(self.last_number)
        # self.amount_possible_positions -= 1

        self.l_food_positions.append(food_pos)
        self.food_x = food_x
        self.food_y = food_y
        self.canvas.create_image(self.ht_canvas+self.box_width*self.food_x, self.ht_canvas+self.box_height*self.food_y, image=self.imgtk_food, anchor=tk.NW)


    def print_infos(self, event):
        print("self.l_positions: {}".format(self.l_positions))
        print("self.l_possible_positions: {}".format(self.l_possible_positions))
        print("self.l_directions: {}".format(self.l_directions))
        print("self.l_saved_moves: {}".format(''.join(self.l_saved_moves)))


    def do_the_move(self, new_direction):
        self.l_saved_moves.append(self.d_coordinate_to_move[new_direction])

        if new_direction=='W':
            new_arrow_img = self.imgtk_arrow_left
            if self.current_direction=='S':
                new_prev_tile = self.imgtk_line_ul
            elif self.current_direction=='N':
                new_prev_tile = self.imgtk_line_dl
            else:
                new_prev_tile = self.imgtk_line_hor
        elif new_direction=='E':
            new_arrow_img = self.imgtk_arrow_right
            if self.current_direction=='S':
                new_prev_tile = self.imgtk_line_ur
            elif self.current_direction=='N':
                new_prev_tile = self.imgtk_line_dr
            else:
                new_prev_tile = self.imgtk_line_hor
        elif new_direction=='N':
            new_arrow_img = self.imgtk_arrow_up
            if self.current_direction=='W':
                new_prev_tile = self.imgtk_line_ur
            elif self.current_direction=='E':
                new_prev_tile = self.imgtk_line_ul
            else:
                new_prev_tile = self.imgtk_line_ver
        elif new_direction=='S':
            new_arrow_img = self.imgtk_arrow_down
            if self.current_direction=='W':
                new_prev_tile = self.imgtk_line_dr
            elif self.current_direction=='E':
                new_prev_tile = self.imgtk_line_dl
            else:
                new_prev_tile = self.imgtk_line_ver

        self.draw_blank_previuos(x=self.prev_x, y=self.prev_y)
        self.canvas.create_image(self.ht_canvas+self.prev_x*self.box_width, self.ht_canvas+self.prev_y*self.box_height, image=new_prev_tile, anchor=tk.NW)

        if self.current_x==self.food_x and self.current_y==self.food_y:
            self.draw_blank_previuos(x=self.food_x, y=self.food_y)
            food_pos = (self.food_y, self.food_x)
            self.l_possible_positions.remove(food_pos)
            self.amount_possible_positions -= 1

            self.l_positions.append(food_pos)
            self.set_pseudo_random_food_position()
        else:
            last_pos = self.l_positions.pop(0)
            last_y, last_x = last_pos
            self.l_possible_positions.append(last_pos)
            self.draw_blank_previuos(x=last_x, y=last_y)
            new_pos = (self.current_y, self.current_x)
            self.l_possible_positions.remove(new_pos)
            self.l_positions.append(new_pos)

            self.l_directions.pop(0)

            last_direction = self.l_directions[0]
            last2_y, last2_x = self.l_positions[0]
            self.draw_blank_previuos(x=last2_x, y=last2_y)
            if last_direction=='W':
                self.canvas.create_image(self.ht_canvas+last2_x*self.box_width, self.ht_canvas+last2_y*self.box_height, image=self.imgtk_line_end_left, anchor=tk.NW)
            elif last_direction=='E':
                self.canvas.create_image(self.ht_canvas+last2_x*self.box_width, self.ht_canvas+last2_y*self.box_height, image=self.imgtk_line_end_right, anchor=tk.NW)
            elif last_direction=='N':
                self.canvas.create_image(self.ht_canvas+last2_x*self.box_width, self.ht_canvas+last2_y*self.box_height, image=self.imgtk_line_end_up, anchor=tk.NW)
            elif last_direction=='S':
                self.canvas.create_image(self.ht_canvas+last2_x*self.box_width, self.ht_canvas+last2_y*self.box_height, image=self.imgtk_line_end_down, anchor=tk.NW)

        self.prev_x = self.current_x
        self.prev_y = self.current_y
        self.current_direction = new_direction
        self.l_directions.append(self.current_direction)
        self.canvas.create_image(self.ht_canvas+self.current_x*self.box_width, self.ht_canvas+self.current_y*self.box_height, image=new_arrow_img, anchor=tk.NW)
        

    def move_left(self, event):
        if self.current_direction=='E':
            return

        if self.current_x<=0:
            return
        
        new_x = self.current_x-1
        new_y = self.current_y

        new_pos = (new_y, new_x)
        if new_pos in self.l_positions and new_pos!=self.l_positions[0]:
            return

        self.current_x -= 1
        self.do_the_move(new_direction='W')
        return True


    def move_right(self, event):
        if self.current_direction=='W':
            return

        if self.current_x>=self.x_amount-1:
            return
        
        new_x = self.current_x+1
        new_y = self.current_y

        new_pos = (new_y, new_x)
        if new_pos in self.l_positions and new_pos!=self.l_positions[0]:
            return

        self.current_x += 1
        self.do_the_move(new_direction='E')
        return True


    def move_up(self, event):
        if self.current_direction=='S':
            return

        if self.current_y<=0:
            return
        
        new_x = self.current_x
        new_y = self.current_y-1

        new_pos = (new_y, new_x)
        if new_pos in self.l_positions and new_pos!=self.l_positions[0]:
            return

        self.current_y -= 1
        self.do_the_move(new_direction='N')
        return True


    def move_down(self, event):
        if self.current_direction=='N':
            return

        if self.current_y>=self.y_amount-1:
            return
        
        new_x = self.current_x
        new_y = self.current_y+1

        new_pos = (new_y, new_x)
        if new_pos in self.l_positions and new_pos!=self.l_positions[0]:
            return

        self.current_y += 1
        self.do_the_move(new_direction='S')
        return True
        

    def draw_current_box_in_canvas(self):
        current_color = self.colors[self.color_idx]
        # self.current_color = self.colors[self.color_idx]

        x = self.current_x
        y = self.current_y
        x1 = self.box_width*x
        y1 = self.box_height*y
        x2 = x1+self.box_width
        y2 = y1+self.box_height
        # colors = ['#00FF00', '#808000', '#00FF80']
        self.rectid = self.canvas.create_rectangle(
            x1, y1, x2, y2, fill=current_color, width=0, # disabledoutline=tk.DISABLED,
        )


    def draw_blank_previuos(self, x, y):
        # x = self.prev_x
        # y = self.prev_y
        x1 = self.ht_canvas+self.box_width*x
        y1 = self.ht_canvas+self.box_height*y
        x2 = x1+self.box_width
        y2 = y1+self.box_height
        self.rectid = self.canvas.create_rectangle(
            x1, y1, x2, y2, fill=self.canvas_bg, width=0,
        )


    def _createVariables(self, parent):
        self.parent = parent
        self.rectx0 = 0
        self.recty0 = 0
        self.rectx1 = 0
        self.recty1 = 0
        self.rectid = None


    def _createCanvas(self):
        self.canvas = tk.Canvas(self.parent, width=self.x_amount*self.box_width, height=self.y_amount*self.box_height,
            bg=self.canvas_bg, # image=self.pixel_1x1_image
            bd=0, highlightthickness=self.ht_canvas, relief='groove', highlightbackground='#000000'
        )
        self.canvas.place(x=10, y=10)
        def set_focus_on_canvas(event):
            self.canvas.focus_set()            
        self.canvas.bind('<Button-1>', set_focus_on_canvas)
        # self.canvas.grid(row=0, column=0, sticky='nsew')


    def onKeyDown(e):
        # The obvious information
        c = e.keysym
        s = e.state

        # Manual way to get the modifiers
        ctrl  = (s & 0x4) != 0
        alt   = (s & 0x8) != 0 or (s & 0x80) != 0
        shift = (s & 0x1) != 0

        # Merge it into an output
        # if alt:
        #     c = 'alt+' + c
        if shift:
            c = 'shift+' + c
        if ctrl:
            c = 'ctrl+' + c
        # print(c)

        return c


    def draw_random_rectangles(self, event):
        c = App.onKeyDown(event)
        print("c: {}".format(c))
        width = 20
        height = 20
        rounds = 1
        
        if 'shift' in c:
            rounds = 100

        for i in range(0, rounds):
            x = np.random.randint(0, self.x_amount)
            y = np.random.randint(0, self.y_amount)
            x1 = self.box_width*x
            y1 = self.box_height*y
            x2 = x1+self.box_width
            y2 = y1+self.box_height
            colors = ['#00FF00', '#808000', '#00FF80']
            self.rectid = self.canvas.create_rectangle(
                x1, y1, x2, y2, fill=colors[np.random.randint(0, 3)], width=0, # disabledoutline=tk.DISABLED,
            )


if __name__ == "__main__":
    root = tk.Tk()
    # root.geometry("620x500")
    root.update()
    app = App(root, y_amount=12, x_amount=12, box_width=20)
    # app = App(root, y_amount=12, x_amount=12, box_width=50)
    root.mainloop()
