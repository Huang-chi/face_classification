import cv2 as cv
import matplotlib.pyplot as plt
import random as rd
import pandas as pd

from emotion_icon import load_emotion_icon

angry_man = None


all_image = load_emotion_icon()
# images = pd.DataFrame(all_image)
# print(all_image['angry_man'])

# load 
# angry_man = cv.imread("./img/angry_man.jpg",-1)
# print(angry_man.shape)
# plt.imshow(all_image['angry_man'],'gray')

# cv.imshow("123",all_image['angry_man'])
# cv.waitKey(0);
rgb_image = cv.imread("./img/123.jpg",-1)
print(rgb_image.shape)
s_img = all_image["happy_man"]
# print(face_coordinates[:2])
x,y = rgb_image[:2]
print(s_img.shape)

x = 221
y = 80
x_offset = 0 +x
y_offset = -45 +y
y1, y2 = y_offset, y_offset + s_img.shape[0]
x1, x2 = x_offset, x_offset + s_img.shape[1]
alpha_s = s_img[:, :, 2] / 255.0
cv.imshow('alpha_s',alpha_s)
    # cv.waitKey (0)
alpha_l = 1.0 - alpha_s
cv.imshow('alpha_l',alpha_l)
    # cv.waitKey (0)
print(alpha_l.shape)
print(x_offset,y_offset)

for c in range(0, 3):
    rgb_image[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + 
    alpha_l * rgb_image[y1:y2, x1:x2, c])
       
cv.imshow('rgb_image',rgb_image)
cv.waitKey (0)