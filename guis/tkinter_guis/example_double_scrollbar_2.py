#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk

# Top-level frame
root = tk.Tk()
root.title( "Double scrollbar with tkinter" )
root.minsize(width = 600, height = 600)
frame = ttk.Frame(root, relief="sunken")

# Canvas creation with double scrollbar
hscrollbar = ttk.Scrollbar(frame, orient = tk.HORIZONTAL)
vscrollbar = ttk.Scrollbar(frame, orient = tk.VERTICAL)
sizegrip = ttk.Sizegrip(frame)
canvas = tk.Canvas(frame, bd=0, highlightthickness=0, yscrollcommand = vscrollbar.set, xscrollcommand = hscrollbar.set)
vscrollbar.config(command = canvas.yview)
hscrollbar.config(command = canvas.xview)


# Add controls here
subframe = ttk.Frame(canvas)

#Packing everything
subframe.pack(padx   = 15, pady   = 15, fill = tk.BOTH, expand = tk.TRUE)
hscrollbar.pack( fill=tk.X, side=tk.BOTTOM, expand=tk.FALSE)
vscrollbar.pack( fill=tk.Y, side=tk.RIGHT, expand=tk.FALSE)
sizegrip.pack(in_ = hscrollbar, side = tk.BOTTOM, anchor = "se")
canvas.pack(side = tk.LEFT, padx  = 5, pady   = 5, fill = tk.BOTH, expand= tk.TRUE)
frame.pack( padx   = 5, pady   = 5, expand = True, fill = tk.BOTH)


canvas.create_window(0,0, window = subframe)
root.update_idletasks() # update geometry
canvas.config(scrollregion = canvas.bbox("all"))
canvas.xview_moveto(0) 
canvas.yview_moveto(0)


# launch the GUI
root.mainloop()
