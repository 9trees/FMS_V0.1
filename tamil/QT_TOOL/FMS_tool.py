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
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import copy
import cv2
import detect_paper
import get_roi
from skimage.io import imread
import copy
import hsv_slicer
import length_width as lw
import Get_the_color as gc
from pathlib import Path
import pandas as pd
import data_annotation
import PyQt5

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
            painter = QPainter(self)
            pixmap = QPixmap("myPic.png")
            painter.drawPixmap(self.rect(), pixmap)
            pen = QPen(Qt.red, 3)
            painter.setPen(pen)
            # painter.drawLine(10, 10, self.rect().width() - 10, 10)
            # painter.drawRect(10, 10, 20, 20)
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
        self.select_roi.clicked.connect(self.draw_roi)

        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget(self.ImgWidget)

        self.update_frame()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)
        self.close_button.clicked.connect(self.close_window)
        self.process_roi.clicked.connect(self.find_len)
        self.clr_consle.clicked.connect(self.console_clear)
        self.save_annotation_bt.clicked.connect(self.save_json)

        # initiating the labels from the .txt file
        with open('lables.txt', encoding='utf-8') as f:
            labels = f.read()
        labels = labels.split('\n')[:-1]
        for i in labels:
            self.labels_box.addItem(i)


    def close_window(self):
        self.close()

    def change_image(self):
        global file_name
        file_name = easygui.diropenbox()
        self.startImage.setText(file_name)
        self.list_paths(file_name)
        # folder_name = Path(file_name).name
        # parent_name = str(Path(file_name).parent)
        if not os.path.exists(file_name + '/' + 'output'):
            os.mkdir(file_name + '/' + 'output')

        if not os.path.exists(file_name + '/' + 'annotation'):
            os.mkdir(file_name + '/' + 'annotation')

    def update_setting(self):
        global running
        running = True

    def update_frame(self):
        global running, dft, animate,rois
        if running:
            if rois:
                img = box_draw
                rois = False
            elif animate:
                img = animate_img
                animate = False
            else:
                img = image_f.copy()

            img_height, img_width, img_colors = img.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1

            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            height, width, bpc = img.shape
            bpl = bpc * width

            if self.HSV.isChecked():
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                mask = cv2.inRange(hsv, (self.H1.value(), self.S1.value(), self.V1.value()),
                                   (self.H2.value(), self.S2.value(), self.V2.value()))

                imask = mask > 0
                green = np.zeros_like(img, np.uint8)
                green[imask] = img[imask]
                img = green

            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)

            self.ImgWidget.setImage(image)
            running = False

    def closeEvent(self, event):
        global running
        running = False

    def list_paths(self, file_name):
        image_paths = glob.glob(file_name + '/*.JPG') + glob.glob(file_name + '/*.png') + glob.glob(
            file_name + '/*.jpg') + glob.glob(file_name + '/*.jpeg')
        for i in image_paths:
            self.imgs_list.addItem(i)

        # self.itemClicked.connect(self.list_clicked)

    def console_clear(self):
        self.imgs_list_3.clear()

    def list_clicked(self, item):
        global image_f, running, cm_to_pixel, current_img, animate_img, animate, img_name, img_name_ext, current_index,rois
        animate = False
        rois = False
        current_img = item.text()
        current_index = self.imgs_list.indexFromItem(item).row()
        img_name_ext = Path(current_img).name
        img_name = Path(current_img).name.split('.')[0]
        image = imread(current_img)
        image_f = detect_paper.get_a4(image)
        if image_f is None:
            self.imgs_list_3.addItem(img_name_ext + ': !!!Unable to Detect A4!!!')
        animate_img = copy.copy(image_f)
        cm_to_pixel = detect_paper.get_cm_per_pixel(image_f)
        self.imgs_list_3.addItem(img_name_ext + ':Read successfully')
        running = True


    def draw_roi(self):
        global list_off_cords, box_draw, running, green,rois
        hsv = cv2.cvtColor(image_f, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (self.H1.value(), self.S1.value(), self.V1.value()),
                           (self.H2.value(), self.S2.value(), self.V2.value()))

        imask = mask > 0
        green = np.zeros_like(image_f, np.uint8)
        green[imask] = image_f[imask]

        list_off_cords = get_roi.get_roi(cv2.cvtColor(green, cv2.COLOR_RGB2BGR))
        box_draw = copy.copy(green)
        get_roi.draw_rois(box_draw, list_off_cords)
        self.imgs_list_3.addItem(img_name_ext + ':ROI Drawn')
        rois = True
        running = True

    def find_len(self):
        global animate, running, contours

        source_dict = {}
        source_dict.update({img_name: {}})

        contours = []
        count = 1
        for coord in list_off_cords:
            crop_img = green[coord[1]:coord[3], coord[0]:coord[2]]
            length = lw.get_length(crop_img, cm_to_pixel)
            width = lw.get_width(crop_img, cm_to_pixel)
            color = gc.find_color(crop_img)
            cnt = lw.animate(animate_img, crop_img, length, width, color, count, coord)
            contours.append(cnt)
            sample_id = str(count)
            source_dict[img_name].update({sample_id: {
                'length': length,
                'width': width,
                'color': color
            }})
            count += 1
        animate = True
        running = True

        samples = []
        colors = []
        lengths = []
        widths = []
        for i, j in source_dict[img_name].items():
            samples.append(i)
            colors.append(j['color'])
            lengths.append(j['length'])
            widths.append(j['width'])

        source_list = [samples, colors, lengths, widths]

        df = pd.DataFrame([source_list])
        df.to_csv(file_name + '/' + 'output' + '/' + img_name + '.csv', index=False, header=False)

        self.imgs_list_3.addItem(img_name_ext + ':Animated successfully')

        cv2.imwrite(file_name + '/' + 'output' + '/' + img_name + '.jpg',
                    cv2.cvtColor(animate_img, cv2.COLOR_RGB2BGR))

    def save_json(self):

        img_path = file_name + '/' + 'annotation' + '/' + img_name + '.jpg'
        cv2.imwrite(img_path, cv2.cvtColor(image_f, cv2.COLOR_RGB2BGR))
        data_annotation.save_annotation(contours, self.labels_box.currentText(), image_f, img_path, img_name,
                                        file_name + '/' + 'annotation' + '/' + img_name + '.json')

        self.imgs_list_3.addItem(img_name_ext + ':Annotation Saved')
        self.imgs_list_2.addItem(current_img)
        self.imgs_list.takeItem(current_index)



app = QtWidgets.QApplication(sys.argv)
w = MyWindowClass(None)
w.setWindowTitle('HSV Slicing Tool')
w.show()
app.exec_()
