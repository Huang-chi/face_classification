# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from PIL import ImageDraw, ImageFont, Image

def load_image(image_path, grayscale=False, target_size=None):
    pil_image = image.load_img(image_path, grayscale, target_size)
    return image.img_to_array(pil_image)

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def draw_solid_box(face_coordinates, image_array):
    x, y, w, h = face_coordinates
    solid_box = cv2.rectangle(image_array, (x, y-10), (x + w, y-80), (0, 0, 0), -1)
    return solid_box

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    # font = ImageFont.truetype('../models/simsun.ttc', 26)
    # print(font)
    # img_pil = Image.fromarray(image_array)
    # draw = ImageDraw.Draw(img_pil)
    # draw.text((x + x_offset, y + y_offset), text, font=font, fill=color)
    # font = ImageFont.truetype(".\simsun\simsun.ttc", 20, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
    
    # cv2.putText(image_array, text, (x + x_offset, y + y_offset),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             font_scale, color, thickness, cv2.LINE_AA)


    # pilimg = Image.fromarray(image_array)
 
    # # PIL图片上打印汉字
    # draw = ImageDraw.Draw(pilimg) # 图片上打印
    # font = ImageFont.truetype(".\simsun\simsun.ttc", 20, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
    # draw.text((x + x_offset, y + y_offset), text, (255, 0, 0), font=font) # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
 
    # # PIL图片转cv2 图片
    # cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    # cv2.imshow("图片", cv2charimg) # 汉字窗口标题显示乱码



def get_colors(num_classes):
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = np.asarray(colors) * 255
    return colors

