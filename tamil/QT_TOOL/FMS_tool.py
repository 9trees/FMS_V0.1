from PyQt5 import QtCore, QtGui, uic, QtWidgets
import sys
import cv2
import numpy as np
import threading
import time
from multiprocessing import Queue
import easygui
import os
from matplotlib import pyplot as plt
import glob

running = False
capture_thread = None
form_class = uic.loadUiType("fms.ui")[0]
q = Queue()
HSV = False
laplacian = False
sobel = False
sobel_x = False
sobel_y = False
source = False
contrast = False
image_f = False
dft = False


def grab(file_name):
    global running, image_f
    image_f = cv2.imread(file_name)
    running = True


class OwnImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()


class MyWindowClass(QtWidgets.QMainWindow, form_class):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.startImage.clicked.connect(self.change_image)
        self.imgs_list.itemClicked.connect(self.list_clicked)
        self.HSV.stateChanged.connect(self.update_setting)
        self.H1.valueChanged.connect(self.update_setting)
        self.H2.valueChanged.connect(self.update_setting)
        self.S1.valueChanged.connect(self.update_setting)
        self.S2.valueChanged.connect(self.update_setting)
        self.V1.valueChanged.connect(self.update_setting)
        self.V2.valueChanged.connect(self.update_setting)

        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget(self.ImgWidget)

        self.update_frame()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)
        self.close_button.clicked.connect(self.close_window)
        self.save.clicked.connect(self.save_image)


    def close_window(self):
        self.close()

    def change_image(self):
        # global image_f, running
        file_name = easygui.diropenbox()
        self.startImage.setText(file_name)
        # image_f = cv2.imread(file_name)
        self.list_paths(file_name)
        # running = True

    def update_setting(self):
        global running
        running = True

    def update_frame(self):
        global running, image, dft, img
        if running:
            img = image_f.copy()
            img_height, img_width, img_colors = img.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1

            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = bpc * width

            if self.HSV.isChecked():
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, (self.H1.value(), self.S1.value(), self.V1.value()),
                                   (self.H2.value(), self.S2.value(), self.V2.value()))

                imask = mask > 0
                green = np.zeros_like(img, np.uint8)
                green[imask] = img[imask]
                img = green

            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)

            self.ImgWidget.setImage(image)
            running = False

    def save_image(self):
        global running
        cv2.imwrite('hj.jpg', img)
        running = True

    def closeEvent(self, event):
        global running
        running = False

    def list_paths(self,file_name):
        image_paths = glob.glob(file_name + '/*.JPG') + glob.glob(file_name + '/*.png')
        for i in image_paths:
            self.imgs_list.addItem(i)

        # self.itemClicked.connect(self.list_clicked)

    def list_clicked(self, item):
        global image_f, running
        image_f = cv2.imread(item.text())
        running = True

app = QtWidgets.QApplication(sys.argv)
w = MyWindowClass(None)
w.setWindowTitle('HSV Slicing Tool')
w.show()
app.exec_()
