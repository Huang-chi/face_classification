import sys
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QBrush, QPixmap
import os

scanning_face_path = os.path.join(os.path.dirname(__file__),'fw\FaceSwap')

sys.path.append(scanning_face_path)

from scanning_face import show_face_information

global flag
flag = True


class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent= None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x =0
        self.count = 0
        
    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout() 
        self.__layout_fun_button = QtWidgets.QVBoxLayout() 
        self.__layout_data_show = QtWidgets.QVBoxLayout() 
        self.button_open_camera = QtWidgets.QPushButton(u'打开相机')
        self.button_change = QtWidgets.QPushButton(u'切換功能')
        self.button_close = QtWidgets.QPushButton(u'退出') 
        self.button_open_camera.setMinimumHeight(50)
        self.button_change.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)
        
        self.label_show_camera = QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(80,100)  # Camera　frame　size
        
        self.label_show_camera.setFixedSize(1060, 1000)   # Main frame size
        self.label_show_camera.setAutoFillBackground(False)
        
        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addLayout(self.__layout_data_show)
        
        self.__layout_data_show.addWidget(self.label_show_camera)
        
        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_change)
        
        self.__layout_fun_button.addWidget(self.button_close)
        
        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'攝像頭')

    def show_camera(self):
        flag, bgr_image = self.cap.read()

        if flag:
            print("camera start")
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            show = cv2.resize(rgb_image, (1080, 960))
            showImage = QtGui.QImage(show, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def show_view(self):
        flag, bgr_image = self.cap.read()
        
        if flag:
#             rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            print("camera start")
            
            faces_indention = scanning_face(flag,bgr_image)
            rgb_image = faces_indention.show_face_information()
            
            show = cv2.resize(rgb_image, (1080, 960))
            showImage = QtGui.QImage(show, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
        
    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.button_change.clicked.connect(self.button_add_face_emotion)

        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_camera.timeout.connect(self.show_view)
#         self.button_close.clicked.connect(self.close)
#         self.button_test.clicked.connect(self.test_click)

    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"請檢測相機與電腦是否連線正確", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)

            else:
                self.timer_camera.start(30)
                self.button_open_camera.setText(u'123')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()

    
    def button_add_face_emotion(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"請檢測相機與電腦是否連線正確", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)

            else:
                self.timer_camera.start(30)
                self.button_open_camera.setText(u'123')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
        
if __name__ == "__main__":
    App = QApplication(sys.argv)
    ex = Ui_MainWindow()
    ex.show()
    sys.exit(App.exec_())