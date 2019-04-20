#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QBrush, QPixmap
import os

import sys

from statistics import mode

from keras.models import load_model
import numpy as np

from utils.datasets import get_labels

from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import draw_solid_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

from emotion_icon import load_emotion_icon
from emotion_icon import Addemotion
from emotion_icon import Addemotion_word

from DL_args import get_args
from keras.utils.data_utils import get_file
from contextlib import contextmanager
from wide_resnet import WideResNet
import argparse
from pathlib import Path
import dlib


import numpy as np

from PIL import Image,ImageTk
from utils.grad_cam import compile_gradient_function
from utils.grad_cam import compile_saliency_function
from utils.grad_cam import register_gradient
from utils.grad_cam import modify_backprop
from utils.grad_cam import calculate_guided_gradient_CAM
from utils.inference import detect_faces
from utils.inference import apply_offsets

from utils.datasets import get_class_to_arg
from DL_args import get_args


#################
from test_function import get_args
from test_function import draw_label
from test_function import video_capture
from test_function import yield_images
from test_function import yield_images_from_dir
#################

#################
import utils
scanning_face_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fw\FaceSwap')
sys.path.append(scanning_face_path)
print("22222")
print(scanning_face_path)
print("22222")
from ImportIamage import return_image_path
#################



global flag
flag = True
global IsRunning
IsRunning = True


# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'

# hyper-parameters for bounding boxes shape
frame_window = 10
gender_offsets = (30, 60)
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# starting lists for calculating modes
age_window = []

# starting video streaming
#cv2.namedWindow('window_frame')

# video_capture = cv2.VideoCapture(0)


# load in all emotion icon
icon_dict , words_dict = load_emotion_icon()

###########
args = get_args()
depth = args.depth
k = args.width
weight_file = args.weight_file
margin = args.margin
image_dir = args.image_dir
###########

if not weight_file:
    weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
            file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

# for face detection
detector = dlib.get_frontal_face_detector()

# load model and weights
img_size = 64
model = WideResNet(img_size, depth=depth, k=k)()
model.load_weights(weight_file)


# image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()

class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        
        # self.face_recong = face.Recognition()
        self.timer_camera = QtCore.QTimer()

        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x = 0
        self.count = 0
        self.frq = 0
        self.age_position = []

    def set_ui(self):
        
        self.__layout_main = QtWidgets.QVBoxLayout()  # 垂直排版
        self.__layout_fun_button = QtWidgets.QHBoxLayout()  # 水平排版
        self.__layout_data_show = QtWidgets.QHBoxLayout()
        self.__layout_logo_show = QtWidgets.QHBoxLayout()
        
        ##
#         self.lb1 = QtWidgets.QLabel('實現夢想 在中正 ~！',self)
#         self.lb1.resize(300,500)
#         self.lb1.setFrameStyle(QFrame.Panel | QFrame.Sunken)
#         self.lb1.setAlignment(Qt.AlignBottom | Qt.AlignRight)
#         self.lb1.resultLabel.setText("<h2>實現夢想 在中正 ~！</h2>")
        ##
        # Set image on the button  start
        ICON_HEIGHT = 300 
        ICON_WIDTH = 200
        self.button_test = QtWidgets.QPushButton(u'')##  +button
        self.button_test.setIcon(QtGui.QIcon('./img/AR.png'))
        self.button_test.setIconSize(QtCore.QSize(ICON_HEIGHT, ICON_WIDTH))
        
        self.button_open_camera = QtWidgets.QPushButton(u'')
        self.button_open_camera.setIcon(QtGui.QIcon('./img/age_gender_emotion.png'))
        self.button_open_camera.setIconSize(QtCore.QSize(ICON_HEIGHT, ICON_WIDTH))
        
        self.button_close = QtWidgets.QPushButton(u'')
        self.button_close.setIcon(QtGui.QIcon('./img/face_fustion.PNG'))
        self.button_close.setIconSize(QtCore.QSize(ICON_HEIGHT, ICON_WIDTH))
        
        # Set image on the button  end
        
        #Button 的顏色修改
        button_color = [self.button_open_camera, self.button_close, self.button_test] ##  +button
        for i in range(3):
            button_color[i].setStyleSheet("QPushButton{color:black}"
                                          "QPushButton:hover{color:red}"
                                          "QPushButton{background-color:rgb(205,190,112)}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:10px}"
                                          "QPushButton{padding:2px 4px}")

        # set button size
        BUTTON_HEIGHT = 250
        BUTTON_WIDTH = 40
        self.button_open_camera.setMinimumHeight(BUTTON_HEIGHT)
        self.button_open_camera.setMinimumWidth(BUTTON_WIDTH)
        self.button_close.setMinimumHeight(BUTTON_HEIGHT)
        self.button_close.setMinimumWidth(BUTTON_WIDTH)
        self.button_test.setMinimumHeight(BUTTON_HEIGHT) # + button
        self.button_test.setMinimumWidth(BUTTON_WIDTH)
        # move()方法移動視窗在螢幕上的位置到x = 300，y = 300座標。
#         self.move(300,300)
        self.setGeometry(100, 100, 1217, 684)

    
        # 全大運圖片
        pix = QPixmap('./img/17.png')
        self.lb1 = QLabel()
        self.lb1.setFixedSize(300, 300)
        self.lb1.setPixmap(pix)
        
        pix2 = QPixmap('./img/logo.png')
        self.lb2 = QLabel()
        self.lb2.setFixedSize(300, 330)
        self.lb2.setPixmap(pix2)
        
        
        # 資訊顯示
        self.label_show_camera = QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(80, 100)  # Camera　frame　size

        self.label_show_camera.setFixedSize(1060, 1000)   # Main frame size
        self.label_show_camera.setAutoFillBackground(False)

#         self.__layout_fun_button.addWidget(self.label_move)

        # layer main
#         self.__layout_main.addLayout(self.__layout_data_show)
        self.__layout_main.addWidget(self.lb2)
        self.__layout_main.addWidget(self.label_show_camera)
        self.__layout_main.addLayout(self.__layout_data_show)
        self.__layout_main.addLayout(self.__layout_fun_button)
        
        # Layer data show
#         self.__layout_logo_show.addWidget(self.lb2)
#         self.__layout_logo_show.addWidget(self.lb2)
#         self.__layout_logo_show.addWidget(self.lb2)
        
        self.__layout_data_show.addWidget(self.lb1)
        self.__layout_data_show.addWidget(self.lb1)
        self.__layout_data_show.addWidget(self.lb1)
        
        # layer button
        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.button_test) # button
        
        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'攝像頭')

        # 設定背景圖片
        palette1 = QPalette()
        palette1.setBrush(self.backgroundRole(), QBrush(QPixmap('./img/background_3.jpg')))
        self.setPalette(palette1)


    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.button_close.clicked.connect(self.close)
        # self.timer_camera.timeout.connect(self.face_Fusion)
        # self.button_test.clicked.connect(self.face_Fusion)
    
    def test_click(self, flag):
                
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"請檢測相機與電腦是否連線正確", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
            # if msg==QtGui.QMessageBox.Cancel:
            #                     pass
            else:
                self.timer_camera.start(60)
#                 self.button_open_camera.setText(u'關閉相機')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
#             self.button_open_camera.setText(u'開啟相機')


    def button_open_camera_click(self):
        self.button_open_camera.setStyleSheet("background-color:rgb(139,129,76)")
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"請檢測相機與電腦是否連線正確", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
            # if msg==QtGui.QMessageBox.Cancel:
            #                     pass
            else:
                self.timer_camera.start(60)
                self.button_open_camera.setText(u'')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
#             self.button_open_camera.setText(u'開啟相機')

#############
    def face_Fusion(self):
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fw\shape_predictor_68_face_landmarks.dat')
        predictor_path = model_path
        #predictor_path = "../shape_predictor_68_face_landmarks.dat"

        #image_name = "../data/.png"
        #the smaller this value gets the faster the detection will work
        #if it is too small, the user's face might not be detected
        maxImageSizeForDetection = 320

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel("../candide.npz")

        projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

        modelParams = None
        lockedTranslation = False
        drawOverlay = False
        #cv2.namedWindow('window_frame')
        cap = cv2.VideoCapture(cv2.CAP_DSHOW)

        writer = None
        cameraImg = cap.read()[1]


        while IsRunning:
            textureCoords = utils.getFaceTextureCoords(self.textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)
            renderer = FaceRendering.FaceRenderer(cameraImg, self.textureImg, textureCoords, mesh)
            cameraImg = cap.read()[1]
            shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)

            if shapes2D is not None:
                for shape2D in shapes2D:
                    #3D model parameter initialization
                    modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

                    #3D model parameter optimization
                    modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)

                    #rendering the model to an image
                    shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
                    renderedImg = renderer.render(shape3D)

                    #blending of the rendered face with the image
                    mask = np.copy(renderedImg[:, :, 0])
                    renderedImg = ImageProcessing.colorTransfer(cameraImg, renderedImg, mask)
                    cameraImg = ImageProcessing.blendImages(renderedImg, cameraImg, mask)


                    #drawing of the mesh and keypoints
                    if drawOverlay:
                        drawPoints(cameraImg, shape2D.T)
                        drawProjectedShape(cameraImg, [mean3DShape, blendshapes], projectionModel, mesh, modelParams, lockedTranslation)
            #imgg=self.textureImg
            #imgg = cv2.resize(imgg,(100,100))
            #cameraImg = Add_image(cameraImg,imgg)
            
            if writer is not None:
                writer.write(cameraImg)
            
            #fig = plt.figure()
            #ax = fig.add_subplot(211)
            #ax.imshow(cameraImg)
            #plt.draw()
            self.image = cameraImg
            show = cv2.resize(self.image,(640,480))     #把读到的帧的大小重新设置为 640x480
            show = cv2.cvtColor(show,cv2.COLOR_BGR2RGB) #视频色彩转换回RGB，这样才是现实的颜色
            showImage = QtGui.QImage(show.data,show.shape[1],show.shape[0],QtGui.QImage.Format_RGB888) #把读取到的视频数据变成QImage形式
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage)) 
##################

    def show_camera(self):
        
        flag, bgr_image = self.cap.read()
        if flag:
            # bgr_image = img
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            faces = detect_faces(face_detection, gray_image)

            img_h, img_w, _ = np.shape(rgb_image)

            for face_coordinates in faces:
                x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
                rgb_face = rgb_image[y1:y2, x1:x2]

                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    rgb_face = cv2.resize(rgb_face, (gender_target_size))
                    gray_face = cv2.resize(gray_face, (emotion_target_size))
                except:
                    continue
                # run_thread(bgr_image)
        
                gray_face = preprocess_input(gray_face, False)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
                emotion_text = emotion_labels[emotion_label_arg]

                # emotion_window.append(English_2_chinese_emotion(emotion_text))

                rgb_face = np.expand_dims(rgb_face, 0)
                rgb_face = preprocess_input(rgb_face, False)
                gender_prediction = gender_classifier.predict(rgb_face)
                gender_label_arg = np.argmax(gender_prediction)
                gender_text = gender_labels[gender_label_arg]
                # gender_window.append(English_2_chinese_gender(gender_text))

                set_icon = emotion_text+"_"+gender_text
                print(set_icon)
                icon_img = icon_dict[set_icon]
                words_img = words_dict[set_icon]

                ###################
                if(self.frq % 60 == 0):
        
                    # detect faces using dlib detector
                    detected = detector(rgb_image, 1)
                    print(detected)
                    faces_age = np.empty((len(detected), img_size, img_size, 3))
        
                    if len(detected) > 0:
                        for i, d in enumerate(detected):
                            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                            xw1 = max(int(x1 - margin * w), 0)
                            yw1 = max(int(y1 - margin * h), 0)
                            xw2 = min(int(x2 + margin * w), img_w - 1)
                            yw2 = min(int(y2 + margin * h), img_h - 1)
                            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                            faces_age[i, :, :, :] = cv2.resize(rgb_image[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                            faces_age[i, :, :, :] = cv2.resize(rgb_image[y1:y2, x1:x2, :], (img_size, img_size))

                        # predict ages and genders of the detected faces
                        results = model.predict(faces_age)
                        ages = np.arange(0, 101).reshape(101, 1)
                        predicted_ages = results[1].dot(ages).flatten()
                        self.age_position = []
                        print(predicted_ages)
                        for i, d in enumerate(detected):
                            print(i,d)
                            self.age_position = str(int(predicted_ages[i]))


                if gender_text == gender_labels[0]:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)


                ###################
                if((face_coordinates[0] - face_coordinates[2]) > 50 and (face_coordinates[0] - face_coordinates[2]) < 180 and (face_coordinates[1]-80) > 20):
            
                    solid_box = draw_solid_box(face_coordinates, rgb_image)
                    draw_bounding_box(face_coordinates, rgb_image, color)
                    solid_box = Addemotion(face_coordinates,solid_box,icon_img)
                    solid_box = Addemotion_word(face_coordinates,solid_box,words_img)

                    print("-*---------")
                    print(face_coordinates)
                    print("----///////")
                    print(self.age_position)
                    print("----///////")
                    draw_text(face_coordinates, solid_box, self.age_position,
                        (255,255,255), 0, -20, 1, 1)
                    print("----------")
                print(self.frq)
                self.frq += 1
            
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            show = cv2.resize(rgb_image, (1080, 960))
            showImage = QtGui.QImage(show, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
    #bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    #cv2.imshow('window_frame', bgr_image)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #exit(0)

        
    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"關閉", u"是否關閉！")

        msg.addButton(ok,QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'確定')

        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            #             self.socket_client.send_command(self.socket_client.current_user_command)
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept() 
            



if __name__ == "__main__":
    App = QApplication(sys.argv)
    ex = Ui_MainWindow()
    ex.show()
    sys.exit(App.exec_())