import cv2 as cv
import numpy as np


# angry_man = None
# ICON_SIZE = (40,40)
# WORDS_SIZE = (160,90)

# ICON_IMG_PATH = "./img/emotion/"
# WORD_IMG_PATH = "./img/emotion_word/"
# IMG_PATH = "./img_1/"

# np.set_printoptions(threshold=np.NaN)

# EMOTION_ICON = [
#         "angry_man","angry_woman","disgust_man","disgust_woman",
#         "disgust_woman","fear_man","fear_woman","happy_man","happy_woman",
#         "neutral_man","neutral_woman","sad_man","sad_woman","surprise_man",
#         "surprise_woman"]



# EMOTION_WORD = [
#         "angry_man","angry_woman","disgust_man","disgust_woman",
#         "disgust_woman","fear_man","fear_woman","happy_man","happy_woman",
#         "neutral_man","neutral_woman","sad_man","sad_woman","surprise_man",
#         "surprise_woman"
# ]

def Add_image(image_array,emotion_icon=None):
    

    x_offset = 520
    y_offset = 0
    y1, y2 = y_offset, y_offset + emotion_icon.shape[0]
    x1, x2 = x_offset, x_offset + emotion_icon.shape[1]
    
    cv.rectangle(image_array, (x1, y1), (x2, y2), (0, 0, 0), -1)
    alpha_s = emotion_icon[:, :, 2] / 255.0
    alpha_l = 1.0 - alpha_s
    

    for c in range(0, 3):
        image_array[y1:y2, x1:x2, c] = (alpha_s * emotion_icon[:, :, c] + 
                                        alpha_l * image_array[y1:y2, x1:x2, c])
    
    return image_array
