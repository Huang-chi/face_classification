import cv2 as cv
import numpy as np

def Add_image(image_array, emotion_icon=None):
    
    x_offset = 520
    y_offset = 0
    y1, y2 = y_offset, y_offset + emotion_icon.shape[0]
    x1, x2 = x_offset, x_offset + emotion_icon.shape[1]
    
    cv.rectangle(image_array, (x1, y1), (x2, y2), (0, 0, 0), -1)
    
    width = emotion_icon.shape[0]
    height = emotion_icon.shape[1]
    alpha_value = np.ones((width, height, 1))*255
    emotion_icon1 = np.c_[emotion_icon, alpha_value]
    alpha_s = emotion_icon1[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    
    for c in range(0, 3):
        image_array[y1:y2, x1:x2, c] = (alpha_s * emotion_icon[:, :, c] + 
                                        alpha_l * image_array[y1:y2, x1:x2, c])
    
    return image_array
