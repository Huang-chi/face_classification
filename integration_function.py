#!/usr/bin/python
# -*- coding: UTF-8 -*-

# import from python library
import sys
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QBrush, QPixmap, QFont
import os
import dlib
import numpy as np
import random

# import from self define module

face_fusion_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'src')
print(face_fusion_path)
sys.path.append(face_fusion_path)

from statistics import mode

from keras.models import load_model

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

from PIL import Image, ImageTk
from utils.grad_cam import compile_gradient_function
from utils.grad_cam import compile_saliency_function
from utils.grad_cam import register_gradient
from utils.grad_cam import modify_backprop
from utils.grad_cam import calculate_guided_gradient_CAM
from utils.inference import detect_faces
from utils.inference import apply_offsets

from utils.datasets import get_class_to_arg

# import age function module
from test_function import get_args
from test_function import draw_label
from test_function import video_capture
from test_function import yield_images
from test_function import yield_images_from_dir

# import face fusion module
face_identification_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'fw\FaceSwap')
sys.path.append(face_identification_path)
import models_face_fusion
import utils_face_fusion
import FaceRendering
from ImportIamage import return_image_path
from add_image import Add_image
from ImportIamage import return_image_path
import NonLinearLeastSquares
import ImageProcessing

global flag
flag = True
global IsRunning
IsRunning = True

# parameters for loading data and images
predictor_path = "./fw/shape_predictor_68_face_landmarks.dat"
detection_model_path = './trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = './trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = './trained_models/gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

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
# cv2.namedWindow('window_frame')

# video_capture = cv2.VideoCapture(0)

# load in all emotion icon
words_dict = load_emotion_icon()

###########
args = get_args()
depth = args.depth
k = args.width
weight_file = args.weight_file
margin = args.margin
image_dir = args.image_dir
###########

if not weight_file:
    weight_file = get_file(
        "weights.28-3.73.hdf5",
        pretrained_model,
        cache_subdir="pretrained_models",
        file_hash=modhash,
        cache_dir=str(Path(__file__).resolve().parent))

# for face detection
detector = dlib.get_frontal_face_detector()

# load model and weights
img_size = 64
model = WideResNet(img_size, depth=depth, k=k)()
model.load_weights(weight_file)


class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        # self.face_recong = face.Recognition()
        self.timer_camera = QtCore.QTimer()
        self.timer_camera1 = QtCore.QTimer()
        self.timer_camera2 = QtCore.QTimer()
        self.timer_camera0 = QtCore.QTimer()
        self.status_dict = {
            "1": self.timer_camera,
            "2": self.timer_camera1,
            "3": self.timer_camera2
        }
        # self.button_status_dict = {} 160,160,160 -> 95,97,03
        self.cap = cv2.VideoCapture(0)
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x = 0
        self.count = 0
        self.frq = 0
        self.age_position = []
        self.image_name = None

        pe = QPalette()
        pe.setColor(QPalette.WindowText, Qt.white)
        self.label.setAutoFillBackground(True)
        pe.setColor(QPalette.Window, Qt.black)
        self.label.setPalette(pe)
        self.label.setFont(QFont("Roman times", 70, QFont.Bold))

        self.timeLine = QtCore.QTimeLine()
        self.timeLine.setCurveShape(3)  # linear Timeline
        self.timeLine.frameChanged.connect(self.setText)
        self.timeLine.finished.connect(self.nextNews)

        self.feed()

        # "1" is face idemtification, "2" is face fusion, "3" is AR
        self.status = "0"

    def set_ui(self):
        self.label = QtWidgets.QLabel('')  # label showing the news
        self.label.setAlignment(
            QtCore.Qt.AlignRight)  # text starts on the right

        self.label.setFixedWidth(1150)
        self.label.setFixedHeight(100)

        self.__layout_main = QtWidgets.QVBoxLayout()  # 垂直排版
        self.__layout_fun_button = QtWidgets.QHBoxLayout()  # 水平排版
        self.__layout_data_show = QtWidgets.QHBoxLayout()
        self.__layout_logo_show = QtWidgets.QHBoxLayout()

        self.__layout_main.setContentsMargins(0, 0, 0, 0)

        # Set image on the button  start
        ICON_HEIGHT = 300
        ICON_WIDTH = 200

        self.button_change_face = QtWidgets.QPushButton(u'')  # button
        self.button_change_face.setIcon(
            QtGui.QIcon('./fw/data/Tai-Tzu-ying.jpg'))
        self.button_change_face.setIconSize(
            QtCore.QSize(ICON_HEIGHT, ICON_WIDTH))

        self.button_AR_function = QtWidgets.QPushButton(u'')  # button
        self.button_AR_function.setIcon(QtGui.QIcon('./src/img/AR.png'))
        self.button_AR_function.setIconSize(
            QtCore.QSize(ICON_HEIGHT, ICON_WIDTH))

        self.button_open_camera = QtWidgets.QPushButton(u'')
        self.button_open_camera.setIcon(
            QtGui.QIcon('./src/img/age_gender_emotion.png'))
        self.button_open_camera.setIconSize(
            QtCore.QSize(ICON_HEIGHT, ICON_WIDTH))

        self.button_face_fusion = QtWidgets.QPushButton(u'')
        self.button_face_fusion.setIcon(
            QtGui.QIcon('./src/img/face_fustion.PNG'))
        self.button_face_fusion.setIconSize(
            QtCore.QSize(ICON_HEIGHT, ICON_WIDTH))

        # Set image on the button  end

        # Button 的顏色修改
        button_color = [
            self.button_open_camera, self.button_face_fusion,
            self.button_AR_function
        ]  # button
        for i in range(3):
            button_color[i].setStyleSheet(
                "QPushButton{color:black}"
                "QPushButton:hover{color:red}"
                "QPushButton{background-color:rgb(160, 160, 160)}"
                "QPushButton{border:2px}"
                "QPushButton{border-radius:10px}"
                "QPushButton{padding:2px 4px}")

        # set button size
        BUTTON_HEIGHT = 250
        BUTTON_WIDTH = 40
        self.button_open_camera.setMinimumHeight(BUTTON_HEIGHT)
        self.button_open_camera.setMinimumWidth(BUTTON_WIDTH)
        self.button_face_fusion.setMinimumHeight(BUTTON_HEIGHT)
        self.button_face_fusion.setMinimumWidth(BUTTON_WIDTH)
        self.button_AR_function.setMinimumHeight(BUTTON_HEIGHT)
        self.button_AR_function.setMinimumWidth(BUTTON_WIDTH)

        self.button_change_face.setMaximumHeight(BUTTON_HEIGHT)
        self.button_change_face.setMaximumWidth(BUTTON_HEIGHT)

        self.setGeometry(100, 100, 1217, 684)

        pix2 = QPixmap('')
        self.lb2 = QLabel()
        self.lb2.setFixedSize(175, 205)
        self.lb2.setPixmap(pix2)

        # 資訊顯示
        self.label_show_camera = QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(80, 100)  # Camera　frame　size

        self.label_show_camera.setFixedSize(1100, 1000)  # Main frame size
        self.label_show_camera.setAutoFillBackground(False)

        self.__layout_main.addWidget(self.label)
        self.__layout_main.addWidget(self.lb2)
        self.__layout_main.addWidget(self.label_show_camera)
        self.__layout_main.addLayout(self.__layout_data_show)
        self.__layout_main.addLayout(self.__layout_fun_button)

        self.__layout_data_show.addWidget(self.button_change_face)

        # layer button
        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_face_fusion)
        self.__layout_fun_button.addWidget(self.button_AR_function)

        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'攝像頭')

        # 設定背景圖片
        palette1 = QPalette()
        palette1.setBrush(self.backgroundRole(),
                          QBrush(QPixmap('./src/img/background.png')))
        self.setPalette(palette1)

    def feed(self):
        self.nl = int(self.label.width() / 70)
        news = [
            '嘉義市政府為全力配合108年全國大專院校運動會，嘉義市拚運動及觀光經濟，配合',
            '中正大學主辦今年全國大專校院運動會，全力支援賽事場地及交通接駁等周邊服務，市長黃敏惠與全大運吉祥物「阿里」、「阿桃」，到史蹟資料館前拍攝宣傳影片，歡迎各地選手及民眾來嘉義市小旅行。'
        ]
        appendix = '　' * self.nl
        news.append(appendix)
        delimiter = ''  # shown between the messages
        self.news = delimiter.join(news)
        newsLength = len(self.news)  # number of letters in news = frameRange
        lps = 3  # letters per second
        # duration until the whole string is shown in milliseconds
        dur = newsLength * 1000 / lps
        self.timeLine.setDuration(dur)
        self.timeLine.setFrameRange(0, newsLength)
        self.timeLine.start()

    def setText(self, number_of_frame):
        if number_of_frame < self.nl:
            start = 0
        else:
            start = number_of_frame - self.nl
        text = '{}'.format(self.news[start:number_of_frame])
        self.label.setText(text)

    def nextNews(self):
        self.feed()  # start again

    def slot_init(self):
        # Fumctial button clicked
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.button_face_fusion.clicked.connect(self.face_fusion_click)
        self.button_AR_function.clicked.connect(self.button_open_AR)

        # Timer connect
        self.timer_camera.timeout.connect(self.face_Identification)
        self.timer_camera1.timeout.connect(self.face_Fusion)
        self.timer_camera2.timeout.connect(self.AR_function)
        self.timer_camera0.timeout.connect(self.show_camera)

        # Change image_name button
        self.button_change_face.clicked.connect(
            self.change_button_icon_and_face_fusion_image)

    def button_open_AR(self):
        if self.status != "0":
            print("---- button_open_AR ---")
            self.status_dict[self.status].stop()
            self.cap.release()

        print("--- Timer camera2 status ----")
        print(self.timer_camera2.isActive())

        if self.timer_camera2.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)

            if flag == False:
                msg = QtWidgets.QMessageBox.warning(
                    self,
                    u"Warning",
                    u"請檢測相機與電腦是否連線正確",
                    buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                print("timer_camera2")
                # self.timer_camera.stop()
                self.status = "2"

                self.timer_camera2.start(30)
        else:
            self.status_dict[self.status].stop()
            self.cap.release()
            self.status = "0"
            self.label_show_camera.clear()

    def AR_function(self):
        # import AR accessories
        face_patterns = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        eye = cv2.CascadeClassifier('haarcascade_eye.xml')
        hats = []
        eyes = []
        for i in range(2):
            hats.append(cv2.imread('./AR_img/hat%d.png' % i, -1))
            eyes.append(cv2.imread('./AR_img/glass%d.png' % i, -1))

        print("------- AR function is running --------")
        ret, sample_image = self.cap.read()
        print(sample_image)
        centers = []
        while True:
            img_gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY, 1)
            faces = face_patterns.detectMultiScale(
                img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
            for face in faces:
                print(face)
                x, y, w, h = face
            # 取帽子
            hat = random.choice(hats)
            # 取眼鏡
            eye = random.choice(eyes)
            # 调整帽子尺寸
            scale = h / hat.shape[0] * 1.0
            hat = cv2.resize(hat, (0, 0), fx=scale, fy=scale)
            scale = h / eye.shape[0] * 0.4
            eye = cv2.resize(eye, (0, 0), fx=scale, fy=scale)

            # 根据人臉放帽子位置
            x_offset = int(x + w / 2 - hat.shape[1] / 2)
            y_offset = int(y - hat.shape[0] / 1.5)

            # 算位置
            x1, x2 = max(x_offset, 0), min(x_offset + hat.shape[1],
                                           sample_image.shape[1])
            y1, y2 = max(y_offset, 0), min(y_offset + hat.shape[0],
                                           sample_image.shape[0])
            hat_x1 = max(0, -x_offset)
            hat_x2 = hat_x1 + x2 - x1
            hat_y1 = max(0, -y_offset)
            hat_y2 = hat_y1 + y2 - y1

            # 根据人臉放眼鏡位置
            x_offset = int(x + w / 2 - eye.shape[1] / 2)
            y_offset = int(y + h / 2 - eye.shape[0] / 1.5)

            # 算位置
            x11, x22 = max(x_offset, 0), min(x_offset + eye.shape[1],
                                             sample_image.shape[1])
            y11, y22 = max(y_offset, 0), min(y_offset + eye.shape[0],
                                             sample_image.shape[0])
            eye_x1 = max(0, -x_offset)
            eye_x2 = eye_x1 + x22 - x11
            eye_y1 = max(0, -y_offset)
            eye_y2 = eye_y1 + y22 - y11

            # 透明部分的處理 eye
            alpha_h_eye = eye[eye_y1:eye_y2, eye_x1:eye_x2, 3] / 255
            alpha_eye = 1 - alpha_h_eye

            # 透明部分的處理
            alpha_h = hat[hat_y1:hat_y2, hat_x1:hat_x2, 3] / 255
            alpha = 1 - alpha_h

            # 按3个通道合并图片
            for c in range(0, 3):
                sample_image[y1:y2, x1:x2, c] = (
                    alpha_h * hat[hat_y1:hat_y2, hat_x1:hat_x2, c] +
                    alpha * sample_image[y1:y2, x1:x2, c])
                sample_image[y11:y22, x11:x22, c] = (
                    alpha_h_eye * eye[eye_y1:eye_y2, eye_x1:eye_x2, c] +
                    alpha_eye * sample_image[y11:y22, x11:x22, c])

            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_GRAY2BGR)
            show = cv2.resize(sample_image, (1080, 960))
            showImage = QtGui.QImage(show, show.shape[1], show.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label_show_camera.setPixmap(
                QtGui.QPixmap.fromImage(showImage))

    def change_button_icon_and_face_fusion_image(self):
        print("---- Change face -----")
        print(self.image_name)
        self.image_name = return_image_path()
        self.button_change_face.setIcon(QtGui.QIcon(self.image_name))
        self.textureImg = cv2.imread(self.image_name)

    def face_fusion_click(self, flag):
        if self.status != "0":
            print("---- button_open_camera_click ---")
            self.status_dict[self.status].stop()
            self.cap.release()

        self.image_name = return_image_path()
        self.button_change_face.setIcon(QtGui.QIcon(self.image_name))
        print("--- Timer camera1 status ----")
        print(self.timer_camera1.isActive())

        if self.timer_camera1.isActive() == False:

            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(
                    self,
                    u"Warning",
                    u"請檢測相機與電腦是否連線正確",
                    buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                print("timer_camera1")
                self.status = "2"
                self.timer_camera1.start(30)
        else:
            self.status_dict[self.status].stop()
            self.cap.release()
            self.status = "0"

            self.label_show_camera.clear()

    def show_camera(self):
        flag, self.image = self.cap.read()
        show = cv2.resize(self.image, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)

        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def button_open_camera_click(self):
        self.button_change_face.setVisible(False)
        if self.status != "0":
            print("---- button_open_camera_click ---")
            self.status_dict[self.status].stop()
            self.cap.release()

        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(
                    self,
                    u"Warning",
                    u"請檢測相機與電腦是否連線正確",
                    buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                print("timer_camera")
                self.status = "1"
                self.timer_camera.start(30)
        else:
            self.status_dict[self.status].stop()
            self.cap.release()
            self.status = "0"
            # self.timer_camera.stop()
            # self.cap.release()
            self.label_show_camera.clear()

    def face_Fusion(self):
        print("----- Face fusion function is start -----")
        print(self.status)

        # cameraImg = self.cap.read()[1]

        # import image path

        self.textureImg = cv2.imread(self.image_name)

        maxImageSizeForDetection = 320
        detector = dlib.get_frontal_face_detector()
        mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils_face_fusion.load3DFaceModel(
            "./fw/candide.npz")

        projectionModel = models_face_fusion.OrthographicProjectionBlendshapes(
            blendshapes.shape[0])

        modelParams = None
        lockedTranslation = False
        drawOverlay = False
        # cap = cv2.VideoCapture(cv2.CAP_DSHOW)

        writer = None
        cameraImg = self.cap.read()[1]
        self.textureImg = cv2.imread(self.image_name)

        # textureCoords = utils_face_fusion.getFaceTextureCoords(self.textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)
        # renderer = FaceRendering.FaceRenderer(cameraImg, self.textureImg, textureCoords, mesh)

        while True:
            print("----- Face fusion function is running -----")
            cameraImg = self.cap.read()[1]
            if (self.status != "2"):
                print("end2")
                break

            textureCoords = utils_face_fusion.getFaceTextureCoords(
                self.textureImg, mean3DShape, blendshapes, idxs2D, idxs3D,
                detector, predictor)
            renderer = FaceRendering.FaceRenderer(cameraImg, self.textureImg,
                                                  textureCoords, mesh)

            shapes2D = utils_face_fusion.getFaceKeypoints(
                cameraImg, detector, predictor, maxImageSizeForDetection)

            if shapes2D is not None:
                for shape2D in shapes2D:
                    # 3D model parameter initialization
                    modelParams = projectionModel.getInitialParameters(
                        mean3DShape[:, idxs3D], shape2D[:, idxs2D])
                    # cameraImg = self.cap.read()[1]
                    # 3D model parameter optimization
                    modelParams = NonLinearLeastSquares.GaussNewton(
                        modelParams,
                        projectionModel.residual,
                        projectionModel.jacobian,
                        ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]],
                         shape2D[:, idxs2D]),
                        verbose=0)
                    # rendering the model to an image
                    shape3D = utils_face_fusion.getShape3D(
                        mean3DShape, blendshapes, modelParams)
                    renderedImg = renderer.render(shape3D)

                    # blending of the rendered face with the image
                    mask = np.copy(renderedImg[:, :, 0])
                    renderedImg = ImageProcessing.colorTransfer(
                        cameraImg, renderedImg, mask)
                    cameraImg = ImageProcessing.blendImages(
                        renderedImg, cameraImg, mask)

                    # drawing of the mesh and keypoints
                    if drawOverlay:
                        drawPoints(cameraImg, shape2D.T)
                        drawProjectedShape(cameraImg,
                                           [mean3DShape, blendshapes],
                                           projectionModel, mesh, modelParams,
                                           lockedTranslation)

            imgg = self.textureImg
            imgg = cv2.resize(imgg, (100, 100))
            cameraImg = Add_image(cameraImg, imgg)

            if writer is not None:
                writer.write(cameraImg)

            self.image = cameraImg
            show = cv2.resize(self.image, (1080, 960))  # 把读到的帧的大小重新设置为 640x480
            # 视频色彩转换回RGB，这样才是现实的颜色
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(
                show.data, show.shape[1], show.shape[0],
                QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
            self.label_show_camera.setPixmap(
                QtGui.QPixmap.fromImage(showImage))
        if (self.status != "2"):
            self.label_show_camera.clear()
            print("end1")
            return None

    def face_Identification(self):

        flag, bgr_image = self.cap.read()
        if (self.status != "1"):
            return None
        if flag:
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            faces = detect_faces(face_detection, gray_image)

            img_h, img_w, _ = np.shape(rgb_image)

            a = 0
            for face_coordinates in faces:
                
                # Output face number
                print("---- Print face numbers ----")
                print(faces.size/4)

                x1, x2, y1, y2 = apply_offsets(face_coordinates,
                                               gender_offsets)
                rgb_face = rgb_image[y1:y2, x1:x2]

                x1, x2, y1, y2 = apply_offsets(face_coordinates,
                                               emotion_offsets)
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
                emotion_label_arg = np.argmax(
                    emotion_classifier.predict(gray_face))
                emotion_text = emotion_labels[emotion_label_arg]

                # emotion_window.append(English_2_chinese_emotion(emotion_text))

                rgb_face = np.expand_dims(rgb_face, 0)
                rgb_face = preprocess_input(rgb_face, False)
                gender_prediction = gender_classifier.predict(rgb_face)
                gender_label_arg = np.argmax(gender_prediction)
                gender_text = gender_labels[gender_label_arg]
                # gender_window.append(English_2_chinese_gender(gender_text))

                set_icon = gender_text + "_" + emotion_text
                print("----- emotion + gender -----")
                print(set_icon)
                words_img = words_dict[set_icon]

                ###################

                if gender_text == gender_labels[0]:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)

                if (self.frq % 15 == 0):
                    # detect faces using dlib detector
                    detected = detector(rgb_image, 1)

                    faces_age = np.empty(
                        (len(detected), img_size, img_size, 3))

                    if len(detected) > 0:
                        for i, d in enumerate(detected):
                            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right(
                            ) + 1, d.bottom() + 1, d.width(), d.height()
                            xw1 = max(int(x1 - margin * w), 0)
                            yw1 = max(int(y1 - margin * h), 0)
                            xw2 = min(int(x2 + margin * w), img_w - 1)
                            yw2 = min(int(y2 + margin * h), img_h - 1)
                            cv2.rectangle(rgb_image, (x1, y1), (x2, y2),
                                          (255, 0, 0), 2)
                            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                            faces_age[i, :, :, :] = cv2.resize(
                                rgb_image[yw1:yw2 + 1, xw1:xw2 + 1, :],
                                (img_size, img_size))
                            faces_age[i, :, :, :] = cv2.resize(
                                rgb_image[y1:y2, x1:x2, :], (img_size, img_size))

                        # predict ages and genders of the detected faces
                        results = model.predict(faces_age)
                        ages = np.arange(0, 101).reshape(101, 1)
                        predicted_ages = results[1].dot(ages).flatten()
                        self.age_position = []
                        print("----- age -----")
                        print(predicted_ages)
                        for i, d in enumerate(detected):
                            print("-----every ages -----")
                            print(str(int(predicted_ages[i])))
                            self.age_position.append(
                                str(int(predicted_ages[i])))

                # if((face_coordinates[0] - face_coordinates[2]) > 50 and (face_coordinates[0] - face_coordinates[2]) < 180 and (face_coordinates[1]-80) > 20):
                if (face_coordinates[1] - 80) > 20:
                    print("---- draw -----")
                    print(a)

                    solid_box = draw_solid_box(face_coordinates, rgb_image)
                    draw_bounding_box(face_coordinates, rgb_image, color)
                    words_img = cv2.resize(words_img,
                                           (180, 65))
                    solid_box = Addemotion_word(face_coordinates, solid_box,
                                                words_img)

                    draw_text(face_coordinates, solid_box,
                              self.age_position[a], (255, 255, 255), 0, -20, 1,
                              1)

                    a += 1
                    self.frq += 1

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            show = cv2.resize(rgb_image, (1080, 960))
            showImage = QtGui.QImage(show, show.shape[1], show.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label_show_camera.setPixmap(
                QtGui.QPixmap.fromImage(showImage))

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"關閉",
                                    u"是否關閉！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
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
            if self.timer_camera1.isActive():
                self.timer_camera1.stop()
            event.accept()


if __name__ == "__main__":
    App = QApplication(sys.argv)
    ex = Ui_MainWindow()
    ex.show()
    sys.exit(App.exec_())
