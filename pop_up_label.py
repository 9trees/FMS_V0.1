# importing libraries
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import threading


class Window(QMainWindow):

    def __init__(self):
        super().__init__()

        # setting title
        self.setWindowTitle("Python ")

        # setting geometry
        self.setGeometry(100, 100, 600, 400)

        # calling method
        self.UiComponents()

        # showing all the widgets
        self.show()

    # method for widgets
    def UiComponents(self):
        global button
        # creating a combo box widget
        self.combo_box = QComboBox(self)

        # setting geometry of combo box
        self.combo_box.setGeometry(200, 150, 120, 30)

        # geek list
        with open('lables.txt', encoding='utf-8') as f:
            labels = f.read()

        labels = labels.split('\n')[:-1]

        # adding list of items to combo box
        self.combo_box.addItems(labels)

        # creating push button
        button = QPushButton("Select", self)

        button.setGeometry(400, 150, 120, 30)

        # adding action to the button
        button.pressed.connect(self.action)

    def action(self):
        global lable
        # showing the pop up
        lable = self.combo_box.currentText()
        button.clicked.connect(self.close_window)
        return lable


    def close_window(self):
        self.close()


def pop_up_class():
    # create pyqt5 app
    App = QApplication(sys.argv)
    # create the instance of our Window
    window = Window()
    #start the app
    window.show()
    App.exec_()
    label = window.action()
    App.exit()
    return label

# f = pop_up_class()
# print(f)