import sys

from statistics import mode
from pathlib import Path
import cv2
from keras.models import load_model
import numpy as np
import time
import dlib

from utils.datasets import get_labels
from utils.datasets import English_2_chinese_emotion
from utils.datasets import English_2_chinese_gender
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
from emotion_icon import return_finish

from keras.utils.data_utils import get_file
from contextlib import contextmanager
from wide_resnet import WideResNet
#################
from test_function import get_args
from test_function import draw_label
from test_function import video_capture
from test_function import yield_images
from test_function import yield_images_from_dir
#################

import threading

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
gender_window = []
emotion_window = []


class scanning_face():
    def __init__(self, flag, bgr_image):
        print(os.path.abspath(''))
        self.flag = flag
        self.bgr_image = bgr_image
        self.frq = 0
        self.icon_dict, self.words_dict = load_emotion_icon()
        ###########
        self.args = get_args()
        self.depth = self.args.depth
        self.k = self.args.width
        self.weight_file = self.args.weight_file
        self.margin = self.args.margin
        ###########

        if not self.weight_file:
            self.weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                    file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

        # for face detection
        self.detector = dlib.get_frontal_face_detector()

        # load model and weights
        img_size = 64
        self.model = WideResNet(img_size, depth=self.depth, k=self.k)()
        self.model.load_weights(self.weight_file)




    def show_face_information(self):
        # bgr_image = img
        gray_image = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2RGB)
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

            emotion_window.append(emotion_text)
            # emotion_window.append(English_2_chinese_emotion(emotion_text))

            rgb_face = np.expand_dims(rgb_face, 0)
            rgb_face = preprocess_input(rgb_face, False)
            gender_prediction = gender_classifier.predict(rgb_face)
            gender_label_arg = np.argmax(gender_prediction)
            gender_text = gender_labels[gender_label_arg]
            # gender_window.append(English_2_chinese_gender(gender_text))

            set_icon = emotion_text+"_"+gender_text
            print(set_icon)
            icon_img = self.icon_dict[set_icon]
            words_img = self.words_dict[set_icon]

            # if len(gender_window) > frame_window:
            #     emotion_window.pop(0)
            #     gender_window.pop(0)
            # try:
            #     emotion_mode = mode(emotion_window)
            #     gender_mode = mode(gender_window)
            # except:
            #     continue

            if gender_text == gender_labels[0]:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
        
            ###################
            if( self.frq % 60 == 0):
        
                # detect faces using dlib detector
                detected = self.detector(rgb_image, 1)
                print(detected)
                faces_age = np.empty((len(detected), img_size, img_size, 3))
        
                if len(detected) > 0:
                    for i, d in enumerate(detected):
                        x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                        xw1 = max(int(x1 - self.margin * w), 0)
                        yw1 = max(int(y1 - self.margin * h), 0)
                        xw2 = min(int(x2 + self.margin * w), img_w - 1)
                        yw2 = min(int(y2 + self.margin * h), img_h - 1)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                        faces_age[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

                    # predict ages and genders of the detected faces
                    results = self.model.predict(faces_age)
                    ages = np.arange(0, 101).reshape(101, 1)
                    predicted_ages = results[1].dot(ages).flatten()
                    print(predicted_ages)
            ###################

            self.frq += 1


            if((face_coordinates[0] - face_coordinates[2]) > 50 and (face_coordinates[0] - face_coordinates[2]) < 180 and (face_coordinates[1]-80) > 20):
                solid_box = draw_solid_box(face_coordinates, rgb_image)
                draw_bounding_box(face_coordinates, rgb_image, color)
                solid_box = Addemotion(face_coordinates, solid_box, icon_img)
                solid_box = Addemotion_word(face_coordinates, solid_box, words_img)
                draw_text(face_coordinates, rgb_image, str(int(predicted_ages))
                    ,(255, 255, 255), 0, -20, 1, 1)
            
            return rgb_image


if __name__ == '__main__':
    a = scanning_face()
    show_face_information()