#! /usr/bin/python3

import numpy as np
import tkinter as tk

from PIL import Image, ImageTk

class App(tk.Frame):
    def __init__( self, parent):
        # position...(y, x)
        self.l_positions = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        
        first_pos = self.l_positions[-1]

        self.current_x = first_pos[1]
        self.current_y = first_pos[0]
        self.prev_x = self.current_x
        self.prev_y = self.current_y
        # self.prev_direction = 'E'
        self.current_direction = 'E'

        self.box_width = 50
        self.box_height = 50
        self.x_amount = 400//self.box_width
        self.y_amount = 400//self.box_height


        while True:
            food_x = np.random.randint(0, self.x_amount)
            food_y = np.random.randint(0, self.y_amount)
            food_pos = (food_y, food_x)

            if not food_pos in self.l_positions:
                break
        self.food_x = food_x
        self.food_y = food_y


        self.color_idx = 0
        self.colors = [
            '#FF0000', '#00FF00', '#0000FF',
            '#FF8000', '#80FF00', '#8000FF'
        ]
        self.current_color = self.colors[self.color_idx]


        self.img_arrow_up = Image.open('arrow_up_10x10.png')
        self.img_line_ver = Image.open('line_streight_vertical_10x10.png')
        self.img_line_dl = Image.open('line_down_left_10x10.png')
        self.img_food = Image.open('food.png')

        self.img_arrow_up = self.img_arrow_up.resize((self.box_width, self.box_height))
        self.img_line_ver = self.img_line_ver.resize((self.box_width, self.box_height))
        self.img_line_dl = self.img_line_dl.resize((self.box_width, self.box_height))      
        self.img_food = self.img_food.resize((self.box_width, self.box_height))      
        

        self.arr_arrow_up = np.array(self.img_arrow_up)
        self.arr_arrow_down = np.flip(self.arr_arrow_up, axis=0).copy()
        self.arr_arrow_right = self.arr_arrow_down.transpose(1, 0, 2).copy()
        self.arr_arrow_left = np.flip(self.arr_arrow_right, axis=1).copy()

        self.arr_line_ver = np.array(self.img_line_ver)
        self.arr_line_hor = self.arr_line_ver.transpose(1, 0, 2).copy()

        self.arr_line_dl = np.array(self.img_line_dl)
        self.arr_line_dr = np.flip(self.arr_line_dl, axis=1).copy()
        self.arr_line_ul = np.flip(self.arr_line_dl, axis=0).copy()
        self.arr_line_ur = np.flip(self.arr_line_ul, axis=1).copy()


        self.imgtk_arrow_up = ImageTk.PhotoImage(Image.fromarray(self.arr_arrow_up))
        self.imgtk_arrow_down = ImageTk.PhotoImage(Image.fromarray(self.arr_arrow_down))
        self.imgtk_arrow_right = ImageTk.PhotoImage(Image.fromarray(self.arr_arrow_right))
        self.imgtk_arrow_left = ImageTk.PhotoImage(Image.fromarray(self.arr_arrow_left))
        
        self.imgtk_line_ver = ImageTk.PhotoImage(Image.fromarray(self.arr_line_ver))
        self.imgtk_line_hor = ImageTk.PhotoImage(Image.fromarray(self.arr_line_hor))
        
        self.imgtk_line_dl = ImageTk.PhotoImage(Image.fromarray(self.arr_line_dl))
        self.imgtk_line_dr = ImageTk.PhotoImage(Image.fromarray(self.arr_line_dr))
        self.imgtk_line_ul = ImageTk.PhotoImage(Image.fromarray(self.arr_line_ul))
        self.imgtk_line_ur = ImageTk.PhotoImage(Image.fromarray(self.arr_line_ur))

        self.imgtk_food = ImageTk.PhotoImage(self.img_food)

        tk.Frame.__init__(self, parent)
        self.canvas_bg = '#F0FFFF'
        self._createVariables(parent)
        self._createCanvas()

        self.btn_test = tk.Button(master=self.parent, text='create something!')
        self.btn_test.place(x=510, y=50)
        self.btn_test.bind('<Button-1>', self.draw_random_rectangles)

        self.parent.bind('s', self.draw_random_rectangles)
        self.parent.bind('S', self.draw_random_rectangles)

        self.parent.bind('<Left>', self.move_left)
        self.parent.bind('<Right>', self.move_right)
        self.parent.bind('<Up>', self.move_up)
        self.parent.bind('<Down>', self.move_down)

        self.parent.bind('<space>', self.print_infos)

        for i in range(0, len(self.l_positions)-1):
            self.canvas.create_image(self.box_width*i, 0, image=self.imgtk_line_hor, anchor=tk.NW)
        self.canvas.create_image(self.box_width*(len(self.l_positions)-1), 0, image=self.imgtk_arrow_right, anchor=tk.NW)

        self.canvas.create_image(self.box_width*self.food_x, self.box_height*self.food_y, image=self.imgtk_food, anchor=tk.NW)


    def print_infos(self, event):
        print("self.l_positions: {}".format(self.l_positions))


    def move_left(self, event):
        if self.current_direction=='E':
            return

        if self.current_x<=0:
            return
        
        new_x = self.current_x-1
        new_y = self.current_y

        new_pos = (new_y, new_x)
        if new_pos in self.l_positions:
            return

        self.current_x -= 1

        self.draw_blank_previuos(x=self.prev_x, y=self.prev_y)
        if self.current_direction=='S':
            self.canvas.create_image(self.prev_x*self.box_width, self.prev_y*self.box_height, image=self.imgtk_line_ul, anchor=tk.NW)
        elif self.current_direction=='N':
            self.canvas.create_image(self.prev_x*self.box_width, self.prev_y*self.box_height, image=self.imgtk_line_dl, anchor=tk.NW)
        else:
            self.canvas.create_image(self.prev_x*self.box_width, self.prev_y*self.box_height, image=self.imgtk_line_hor, anchor=tk.NW)

        if self.current_x==self.food_x and self.current_y==self.food_y:
            self.draw_blank_previuos(x=self.food_x, y=self.food_y)
            self.l_positions.append((self.food_y, self.food_x))
        else:
            last_y, last_x = self.l_positions.pop(0)
            self.draw_blank_previuos(x=last_x, y=last_y)
            self.l_positions.append((self.current_y, self.current_x))

        self.prev_x = self.current_x
        self.prev_y = self.current_y
        self.current_direction = 'W'
        self.canvas.create_image(self.current_x*self.box_width, self.current_y*self.box_height, image=self.imgtk_arrow_left, anchor=tk.NW)


    def move_right(self, event):
        if self.current_direction=='W':
            return

        if self.current_x>=self.x_amount-1:
            return
        
        new_x = self.current_x+1
        new_y = self.current_y

        new_pos = (new_y, new_x)
        if new_pos in self.l_positions:
            return

        self.current_x += 1

        self.draw_blank_previuos(x=self.prev_x, y=self.prev_y)
        if self.current_direction=='S':
            self.canvas.create_image(self.prev_x*self.box_width, self.prev_y*self.box_height, image=self.imgtk_line_ur, anchor=tk.NW)
        elif self.current_direction=='N':
            self.canvas.create_image(self.prev_x*self.box_width, self.prev_y*self.box_height, image=self.imgtk_line_dr, anchor=tk.NW)
        else:
            self.canvas.create_image(self.prev_x*self.box_width, self.prev_y*self.box_height, image=self.imgtk_line_hor, anchor=tk.NW)

        last_y, last_x = self.l_positions.pop(0)
        self.draw_blank_previuos(x=last_x, y=last_y)
        self.l_positions.append((self.current_y, self.current_x))

        self.prev_x = self.current_x
        self.prev_y = self.current_y
        self.current_direction = 'E'
        self.canvas.create_image(self.current_x*self.box_width, self.current_y*self.box_height, image=self.imgtk_arrow_right, anchor=tk.NW)


    def move_up(self, event):
        if self.current_direction=='S':
            return

        if self.current_y<=0:
            return
        
        new_x = self.current_x
        new_y = self.current_y-1

        new_pos = (new_y, new_x)
        if new_pos in self.l_positions:
            return

        self.current_y -= 1

        self.draw_blank_previuos(x=self.prev_x, y=self.prev_y)
        if self.current_direction=='W':
            self.canvas.create_image(self.prev_x*self.box_width, self.prev_y*self.box_height, image=self.imgtk_line_ur, anchor=tk.NW)
        elif self.current_direction=='E':
            self.canvas.create_image(self.prev_x*self.box_width, self.prev_y*self.box_height, image=self.imgtk_line_ul, anchor=tk.NW)
        else:
            self.canvas.create_image(self.prev_x*self.box_width, self.prev_y*self.box_height, image=self.imgtk_line_ver, anchor=tk.NW)

        last_y, last_x = self.l_positions.pop(0)
        self.draw_blank_previuos(x=last_x, y=last_y)
        self.l_positions.append((self.current_y, self.current_x))
        
        self.prev_x = self.current_x
        self.prev_y = self.current_y
        self.current_direction = 'N'
        self.canvas.create_image(self.current_x*self.box_width, self.current_y*self.box_height, image=self.imgtk_arrow_up, anchor=tk.NW)


    def move_down(self, event):
        if self.current_direction=='N':
            return

        if self.current_y>=self.y_amount-1:
            return
        
        new_x = self.current_x
        new_y = self.current_y+1

        new_pos = (new_y, new_x)
        if new_pos in self.l_positions:
            return

        self.current_y += 1

        self.draw_blank_previuos(x=self.prev_x, y=self.prev_y)
        if self.current_direction=='W':
            self.canvas.create_image(self.prev_x*self.box_width, self.prev_y*self.box_height, image=self.imgtk_line_dr, anchor=tk.NW)
        elif self.current_direction=='E':
            self.canvas.create_image(self.prev_x*self.box_width, self.prev_y*self.box_height, image=self.imgtk_line_dl, anchor=tk.NW)
        else:
            self.canvas.create_image(self.prev_x*self.box_width, self.prev_y*self.box_height, image=self.imgtk_line_ver, anchor=tk.NW)

        last_y, last_x = self.l_positions.pop(0)
        self.draw_blank_previuos(x=last_x, y=last_y)
        self.l_positions.append((self.current_y, self.current_x))
        
        self.prev_x = self.current_x
        self.prev_y = self.current_y
        self.current_direction = 'S'
        self.canvas.create_image(self.current_x*self.box_width, self.current_y*self.box_height, image=self.imgtk_arrow_down, anchor=tk.NW)
        

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
        x1 = self.box_width*x
        y1 = self.box_height*y
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
        # self.pixel_1x1_image = tk.PhotoImage(width=1, height=1)
        self.canvas = tk.Canvas(self.parent, width=400, height=400,
            bg=self.canvas_bg, # image=self.pixel_1x1_image
            bd=0, highlightthickness=0, relief='ridge',
            # borderwidth=0, highlightthickness=0
        )
        self.canvas.place(x=0, y=0)
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
            x = np.random.randint(0, 500//width)
            y = np.random.randint(0, 500//height)
            x1 = 20*x
            y1 = 20*y
            x2 = x1+20
            y2 = y1+20
            colors = ['#00FF00', '#808000', '#00FF80']
            self.rectid = self.canvas.create_rectangle(
                x1, y1, x2, y2, fill=colors[np.random.randint(0, 3)], width=0, # disabledoutline=tk.DISABLED,
            )


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry( "620x500" )
    app = App(root)
    root.mainloop()
