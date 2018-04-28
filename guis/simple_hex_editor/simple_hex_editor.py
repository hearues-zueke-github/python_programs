#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

from datetime import datetime

from PIL import Image
from PyQt4 import Qt, QtCore, QtGui, QtTest, uic

# form_class_dialog = uic.loadUiType("info_dialog.ui")[0]
form_class = uic.loadUiType("main.ui")[0]

class MyWindowClass(QtGui.QMainWindow, form_class):

    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.actionExit.triggered.connect(self.actionExit_triggered)
        # self.actionInfo.triggered.connect(self.action_Info_triggered)

    def get_datetime_str(self):
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def actionExit_triggered(self):
        QtCore.QCoreApplication.instance().quit()


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myWindow = MyWindowClass(None)
    myWindow.show()
    app.exec_()
