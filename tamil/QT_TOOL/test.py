import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(30, 30, 500, 500)

    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = QPixmap("myPic.png")
        painter.drawPixmap(self.rect(), pixmap)
        pen = QPen(Qt.red, 3)
        painter.setPen(pen)
        painter.drawRect(end_x, end_y, end_x, end_y)
        painter.end()

    def mouseMoveEvent(self, event):
        print(event.x(), event.y())

    def mouseReleaseEvent(self, event):
        global end_x, end_y
        print('end', event.x(), event.y())
        end_x, end_y = event.x(), event.y()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec_())

#
# import sys
# from PyQt5 import QtCore, QtGui, QtWidgets, uic
# from PyQt5.QtCore import Qt
#
#
# class MainWindow(QtWidgets.QMainWindow):
#
#     def __init__(self):
#         super().__init__()
#
#         self.label = QtWidgets.QLabel()
#         canvas = QtGui.QPixmap(400, 300)
#         self.label.setPixmap(canvas)
#         self.setCentralWidget(self.label)
#
#         self.last_x, self.last_y = None, None
#
#     def mouseMoveEvent(self, e):
#         if self.last_x is None: # First event.
#             self.last_x = e.x()
#             self.last_y = e.y()
#             return # Ignore the first time.
#
#         painter = QtGui.QPainter(self.label.pixmap())
#         painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
#         painter.end()
#         self.update()
#
#         # Update the origin for next time.
#         self.last_x = e.x()
#         print(self.last_x)
#         self.last_y = e.y()
#
#     def mouseReleaseEvent(self, e):
#         self.last_x = None
#         self.last_y = None
#
#
# app = QtWidgets.QApplication(sys.argv)
# window = MainWindow()
# window.show()
# app.exec_()
