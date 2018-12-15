#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np

from PIL import Image, ImageTk

# Simple enough, just import everything from tkinter.
# from tkinter import *
import tkinter as tk


# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class Window(tk.Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        
        # parameters that you want to send through the Frame class. 
        tk.Frame.__init__(self, master)

        #reference to the master widget, which is the tk window                 
        self.master = master

        #with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

        self.frames = [ImageTk.PhotoImage(
            Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)+0x10*i),
            format = 'gif -index %i' %(i))
            for i in range(0, 16)
        ]
        self.frames2 = [ImageTk.PhotoImage(
            Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)+0x20*i),
            format = 'gif -index %i' %(i))
            for i in range(0, 8)
        ]

        self.increment_idx = 0
        self.decrement_idx = 0
        # self.frames = [ImageTk.PhotoImage(
        #     Image.fromarray(np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)),
        #     format = 'gif -index %i' %(i))
        #     for i in range(100)
        # ]


    def update(self, ind):
        frame = self.frames[ind]
        ind += 1

        if self.increment_idx:
            ind += self.increment_idx
            self.increment_idx = 0
        # if ind >= len(self.frames):
        #     ind = 0

        if self.decrement_idx:
            ind -= self.decrement_idx
            self.decrement_idx = 0
        # if ind < 0:
        #     ind = len(self.frames)-1
        ind = ind % len(self.frames)
        print("ind: {}".format(ind))

        self.labl.configure(image=frame)
        self.after(100, self.update, ind)

    def update2(self, ind):
        frame = self.frames2[ind]
        ind += 1
        if ind >= len(self.frames2):
            ind = 0
        print("ind: {}".format(ind))

        self.labl2.configure(image=frame)
        self.after(350, self.update2, ind)


    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget      
        self.master.title("GUI")

        # allowing the widget to take the full space of the root window
        self.pack(fill=tk.BOTH, expand=1)

        # creating a menu instance
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)

        # create the file object)
        file = tk.Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label="Exit", command=self.client_exit_menu_btn)

        #added "file" to our menu
        menu.add_cascade(label="File", menu=file)

        # create the file object)
        edit = tk.Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        edit.add_command(label="Undo")

        #added "file" to our menu
        menu.add_cascade(label="Edit", menu=edit)


        self.quitButton = tk.Button(self, text="Create Image",command=self.create_new_image)
        self.quitButton.place(x=0, y=0)


        self.pix = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        self.img = Image.fromarray(self.pix)
        self.imgtk = ImageTk.PhotoImage(self.img)

        self.labl = tk.Label(self, text="TEST!", bg="#4080FF", image=self.imgtk, borderwidth=4)
        self.labl.place(x=30, y=40)

        self.labl2 = tk.Label(self, text="TEST2!", bg="#FF8040", image=self.imgtk, borderwidth=4)
        self.labl2.place(x=30+128+4+4, y=40)

        self.labls = [tk.Label(self, text="jup!", bg="#208090", width=50, height=30, borderwidth=0, highlightbackground="#00FF00")]
        self.labls[0].place(x=30, y=40+128+10)
        # self.labls[0].maxsize((50, 30))

        self.labl2.bind("<Button-1>", self.print_text2)
        self.labl2.bind("<Button-2>", self.print_text3)

        # self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.label_frame = tk.LabelFrame(self, text="My first label frame!", width=200, height=100)
        self.label_frame.place(x=250, y=20)
        # self.label_frame.geometry("200x50")

        # self.label_frame.grid_rowconfigure(0, weight=1, uniform="x")
        # self.label_frame.grid_columnconfigure(0, weight=5, uniform="x")
        # self.label_frame.grid_columnconfigure(1, weight=1, uniform="x")
        # self.label_frame.grid_propagate(False)
        self.label_frame.pack_propagate(False)

        self.txt_box = tk.Text(self.label_frame)
        self.txt_box.pack(expand=True, fill='both')

        self.scrollbar = tk.Scrollbar(self.txt_box, command=self.txt_box.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_box.config(yscrollcommand=self.scrollbar.set)


        self.btnFrame = tk.Frame(self, width = 100, height=80)
        self.btnFrame.place(x=380, y=120)
        self.btnFrame.grid_propagate(False)
        self.btnFrame.columnconfigure(0, weight=1)
        self.btnFrame.rowconfigure(0, weight=1)
        self.btnFrame.rowconfigure(1, weight=1)

        self.btnWrite = tk.Button(self.btnFrame, text="Write to text box") # , command=self.write_to_text_box)
        self.btnWrite.bind("<Button-1>", self.write_to_text_box)
        self.btnWrite.grid(row=0, column=0, sticky="news")

        self.btnDelete = tk.Button(self.btnFrame, text="Delete from\ntext box") # , command=self.write_to_text_box)
        self.btnDelete.bind("<Button-1>", self.delete_from_text_box)
        self.btnDelete.grid(row=1, column=0, sticky="news")


    def write_to_text_box(self, event):
        print("event: {}".format(event))
        self.txt_box.insert(tk.END, "TEST! {:03}\n".format(np.random.randint(0, 1000)))
        self.txt_box.see(tk.END)


    def delete_from_text_box(self, event):
        print("event: {}".format(event))
        self.txt_box.delete('1.0', tk.END)

    
    def print_text2(self, event):
        self.increment_idx += 1
        print("Hello button-1!!! It is a label!")
        print("event: {}".format(event))

    def print_text3(self, event):
        self.decrement_idx += 1
        print("Hello button-2!!! It is a label!!!!!")
        print("event: {}".format(event))

    def client_exit_menu_btn(self):
        print("Pressed the menu btn button!")
        exit()


    # def client_exit_btn(self):
    def create_new_image(self):
        print("Pressed the normal btn button!")
        self.pix = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        self.img = Image.fromarray(self.pix)
        self.imgtk = ImageTk.PhotoImage(self.img)

        self.labl.configure(image=self.imgtk)
        self.labl.image = self.imgtk
        print("self.labl.image: {}".format(self.labl.image))

        
# root window created. Here, that would be the only window, but
# you can later have windows within windows.
root = tk.Tk()

root.geometry("600x450")

#creation of an instance
app = Window(root)

root.after(0, app.update, 0)
root.after(0, app.update2, 0)

#mainloop 
root.mainloop()
