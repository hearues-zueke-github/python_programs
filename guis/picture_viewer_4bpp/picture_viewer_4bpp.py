#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import os
import sys
import traceback

import numpy as np

from datetime import datetime

from PIL import Image
from PIL.ImageQt import ImageQt

from PyQt4 import Qt, QtCore, QtGui, QtTest, uic

import utils_4bpp

form_class = uic.loadUiType("picture_viewer_4bpp.ui")[0]

class MyWindowClass(QtGui.QMainWindow, form_class):

    def focusOutEvent(self,event):
        self.labelPALRow.update()
        print("Outside of window!")

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.WindowDeactivate:
            # self.setTopLevelWindow()
            # self.dialog.close()
            print("Out of window!")

            return True

        return False

    def focusChanged(self, event):
        print("Focus changed!")

    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        # TODO: make more functions!
        # TODO: make more gui elements!
        # TODO: add mouse event on picture!
        # TODO: add outline animation on selected tile/tiles
        self.seperator_line_width = 0
        self.seperator_color = (0xFF, 0xC0, 0x40)
        
        self.file_path = "/home/doublepmcl/git/python_programs/picture_manipulation/images/autumn-colorful-colourful-33109.jpg"
        self.file_path_PAL = "/home/doublepmcl/Documents/own_custom_kaizo_tricks/level_3.pal"
        self.file_path_4bpp = "/home/doublepmcl/Documents/own_custom_kaizo_tricks/Graphics/GFX0D.bin"

        self.lineChoosenFile.setText(self.file_path)
        self.lineChoosenFile4bpp.setText(self.file_path_4bpp)

        def event(self, event):
            if (event.type()==QEvent.KeyPress) and (event.key()==Qt.Key_Tab or
                event.key()==Qt.Key_Left):
                self.emit(Qt.SIGNAL("tabPressed"))
                return True

            return Qt.QLineEdit.event(self, event)

        self.lineChoosenFile.event = event.__get__(self.lineChoosenFile, Qt.QLineEdit)

        self.actionExit.triggered.connect(self.actionExit_triggered)
        self.btnChooseFile.clicked.connect(self.btnChooseFile_triggered)
        self.btnLoadPicture.clicked.connect(self.btnLoadPicture_triggered)

        self.lineChoosenFile.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btnLoadPicture.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btnChooseFile.setFocusPolicy(QtCore.Qt.NoFocus)

        self.btnLoadPicture4bpp.clicked.connect(self.btnLoadPicture4bpp_triggered)
        self.btnChooseFile4bpp.clicked.connect(self.btnChooseFile4bpp_triggered)
        self.btnLoadPAL.clicked.connect(self.btnLoadPAL_triggered)
        self.btnChooseFilePAL.clicked.connect(self.btnChooseFilePAL_triggered)

        self.lineChoosenFile4bpp.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btnLoadPicture4bpp.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btnChooseFile4bpp.setFocusPolicy(QtCore.Qt.NoFocus)

        self.btnChooseFilePAL.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btnLoadPAL.setFocusPolicy(QtCore.Qt.NoFocus)

        self.labelShowOutput.setFocusPolicy(QtCore.Qt.NoFocus)

        self.connect(self.lineChoosenFile, Qt.SIGNAL("tabPressed"),
                     self.update)
  

    def focusInEvent(self, event):
        print('Got focus')

    def focusOutEvent(self, event):
        print('Lost focus')

    def update(self):
        print("Tab pressed!")

    def get_datetime_str(self):
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def btnChooseFile_triggered(self):
        self.file_path = str(Qt.QFileDialog.getOpenFileName())
        self.lineChoosenFile.setText(self.file_path)

    def btnLoadPicture_triggered(self):
        file_path = self.file_path
        print("file_path: {}".format(file_path))
        if not os.path.exists(file_path):
            print("No path defined!")
            return

        print("Passed!")

        try:
            self.img = Image.open(file_path)

            self.pix = np.array(self.img)
            self.h = self.labelPicture.geometry().height()
            self.w = self.labelPicture.geometry().width()
            self.offset_x = 0
            self.offset_y = 0

            self.img_2 = Image.fromarray(self.pix[self.offset_y:self.offset_y+self.h, self.offset_x:self.offset_x+self.w])
            self.pixmap = Qt.QPixmap.fromImage(ImageQt(self.img_2))
            self.labelPicture.setPixmap(self.pixmap)
            print("Image loaded!")
        except Exception as e:
            traceback.print_exc()
            # e.stack_trace()
            print("Not an Image!")


    def btnChooseFilePAL_triggered(self):
        self.file_path_PAL = str(Qt.QFileDialog.getOpenFileName())
        self.labelShowOutput.setText("Choosen the path for the PAL!\nfile_path_PAL: {}".format(self.file_path_PAL))
        print("self.file_path_PAL: {}".format(self.file_path_PAL))

    def btnLoadPAL_triggered(self):
        self.labelShowOutput.setText("Load an image for PAL!")
        with open(self.file_path_PAL, "rb") as fin:
            pal_bin = np.fromfile(fin, dtype=np.uint8)
        self.pix_pal = pal_bin.reshape((16, 16, 3))
        self.pix_pal[:, 0] = (0x80, )*3

        print("self.pix_pal[0x8]: {}".format(self.pix_pal[0x8]))
        self.pix_pal_choosen = self.pix_pal[0x8]
        pal_row_extend = np.zeros((16, 16*16, 3), dtype=np.uint8)+self.pix_pal_choosen.reshape((16, 1, 3))
        self.pix_pal_row = pal_row_extend.reshape((-1, 16, 3)).transpose(1, 0, 2)
        # self.pix_pal_row = np.zeros((16, 16*16, 3), dtype=np.uint8)+0x40
        # self.pix_pal_row = np.random.randint(0, 256, (16, 16*16, 3), dtype=np.uint8)

        self.img_pal_row = Image.fromarray(self.pix_pal_row)
        self.pixmap_pal_row = Qt.QPixmap.fromImage(ImageQt(self.img_pal_row))
        self.labelPALRow.setPixmap(self.pixmap_pal_row)

    def btnChooseFile4bpp_triggered(self):
        self.file_path_4bpp = str(Qt.QFileDialog.getOpenFileName())
        self.lineChoosenFile4bpp.setText(self.file_path_4bpp)
        self.labelShowOutput.setText("Choosen the path for the image!\nfile_path_4bpp: {}".format(self.file_path_4bpp))
        print("self.file_path_4bpp: {}".format(self.file_path_4bpp))

    def btnLoadPicture4bpp_triggered(self):
        self.labelShowOutput.setText("Load an image!")

        file_path_4bpp = self.file_path_4bpp
        # print("file_path_4bpp: {}".format(file_path_4bpp))
        self.labelShowOutput.setText(str(self.labelShowOutput.text())+
            "\nfile_path_4bpp: {}".format(file_path_4bpp))
        
        if not os.path.exists(file_path_4bpp):
            print("No path defined!")
            return

        print("Passed!")

        try:
            with open(file_path_4bpp, "rb") as fin:
                self.pix_bin = np.fromfile(fin, dtype=np.uint8)
            if self.pix_bin.shape[0] % 0x40 != 0:
                self.labelShowOutput.setText(str(self.labelShowOutput.text())+
                    "\nself.pix_bin is not % 0x40!!! self.pix_bin.shape[0]: 0x{:0X}".format(self.pix_bin.shape[0]))
        
                raise Exception

            self.pix_idx = utils_4bpp.bin_to_bpp4_idx(self.pix_bin)
            self.idx_orig_size = self.pix_idx.shape[0]

            if self.pix_idx.shape[0] % 0x10 != 0:
                self.pix_idx = np.vstack((self.pix_idx, np.zeros((16-self.pix_idx.shape[0]%16, 8, 8), dtype=np.uint8)))

            self.pix_idx_row = (self.pix_idx
                .transpose(0, 2, 1)
                .reshape((-1, 128, 16))
                .transpose(0, 2, 1)
                .reshape((-1, 128)))

            self.pix_4bpp_rows = self.pix_pal_choosen[self.pix_idx_row]

            # add black lines between the 8x8 tiles!
            # set the width of the seperator line too!
            
            height, width, depth = self.pix_4bpp_rows.shape

            rows = height // 8
            cols = width // 8

            lw = self.seperator_line_width
            pix_4bpp_rows_new = np.zeros((rows*8+(rows+1)*lw,
                                          cols*8+(cols+1)*lw, depth), dtype=np.uint8)
            yh = 8+lw
            xw = 8+lw
            for y in range(0, rows):
                for x in range(0, cols):
                    j = lw+yh*y
                    i = lw+xw*x
                    pix_4bpp_rows_new[j:j+8, i:i+8] = self.pix_4bpp_rows[8*y:8*(y+1), 8*x:8*(x+1)]
            
            if lw > 0:
                for y in range(0, rows+1):
                    pix_4bpp_rows_new[yh*y:yh*y+lw] = self.seperator_color
                for x in range(0, cols+1):
                    pix_4bpp_rows_new[:, xw*x:xw*x+lw] = self.seperator_color

            self.pix_4bpp_rows = pix_4bpp_rows_new

            self.img_2 = Image.fromarray(self.pix_4bpp_rows)
            self.pixmap = Qt.QPixmap.fromImage(ImageQt(self.img_2.resize((self.img_2.width*2, self.img_2.height*2))))
            self.labelPictureTiles.setPixmap(self.pixmap)
            print("Image loaded!")
        except Exception as e:
            traceback.print_exc()
            # e.stack_trace()
            print("Not an Image!")

    def actionExit_triggered(self):
        QtCore.QCoreApplication.instance().quit()

    def keyPressEvent(self, event):
        key = event.key()
        print(key)
        
        should_update_picture = False
        increment = 30
        if key == QtCore.Qt.Key_Up:
            print('Up Arrow Pressed')
            pix_h = self.pix.shape[0]
            if self.offset_y+1+self.h < pix_h:
                self.offset_y += increment
                if self.offset_y >= pix_h-self.h:
                    self.offset_y = pix_h-self.h-1
                should_update_picture = True
        elif key == QtCore.Qt.Key_Down:
            print('Down Arrow Pressed')
            if self.offset_y > 0:
                self.offset_y -= increment
                if self.offset_y < 0:
                    self.offset_y = 0
                should_update_picture = True
        elif key == QtCore.Qt.Key_Left:
            print('Left Arrow Pressed')
            pix_w = self.pix.shape[1]
            if self.offset_x+1+self.w < pix_w:
                self.offset_x += increment
                if self.offset_x >= pix_w-self.w:
                    self.offset_x = pix_w-self.w-1
                should_update_picture = True
        elif key == QtCore.Qt.Key_Right:
            print('Right Arrow Pressed')
            if self.offset_x > 0:
                self.offset_x -= increment
                if self.offset_x < 0:
                    self.offset_x = 0
                should_update_picture = True
        elif key == QtCore.Qt.Key_Tab:
            print('Tab Pressed!!!')

        if should_update_picture:
            self.img_2 = Image.fromarray(self.pix[self.offset_y:self.offset_y+self.h, self.offset_x:self.offset_x+self.w])
            self.pixmap = Qt.QPixmap.fromImage(ImageQt(self.img_2))
            self.labelPicture.setPixmap(self.pixmap)

def changedFocusSlot(old, now):
    if (now==None and QApplication.activeWindow()!=None):
        print "set focus to the active window"
        QApplication.activeWindow().setFocus()

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    QtCore.QObject.connect(app, Qt.SIGNAL("focusChanged(QWidget *, QWidget *)"), changedFocusSlot)
    
    myWindow = MyWindowClass(None)
    myWindow.show()
    
    app.exec_()
