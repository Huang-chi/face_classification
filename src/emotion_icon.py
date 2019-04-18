import cv2 as cv


angry_man = None
ICON_SIZE = (40, 40)
WORDS_SIZE = (160, 90)

ICON_IMG_PATH = "./img/emotion/"
WORD_IMG_PATH = "./img/emotion_word/"
IMG_PATH = "./img_1/"


EMOTION_ICON = [
        "angry_man", "angry_woman", "disgust_man", "disgust_woman",
        "fear_man", "fear_woman", "happy_man", "happy_woman",
        "neutral_man", "neutral_woman", "sad_man", "sad_woman", "surprise_man",
        "surprise_woman"]


EMOTION_WORD = [
        "angry_man", "angry_woman", "disgust_man", "disgust_woman",
        "fear_man", "fear_woman", "happy_man", "happy_woman",
        "neutral_man", "neutral_woman", "sad_man", "sad_woman", "surprise_man",
        "surprise_woman"]


def load_emotion_icon():

    icon_dict = {}
    words_dict = {}

    for icon in EMOTION_ICON:
        tmp_img = cv.imread(IMG_PATH+icon+".png", -1)
        tmp_img = cv.resize(tmp_img, ICON_SIZE)
        icon_dict[icon] = tmp_img

    for icon in EMOTION_WORD:
        tmp_img = cv.imread(WORD_IMG_PATH+icon+".png", -1)
        tmp_img = cv.resize(tmp_img, WORDS_SIZE)
        tmp_img = cv.cvtColor(tmp_img, cv.COLOR_RGB2BGR)
        words_dict[icon] = tmp_img

    return icon_dict, words_dict


def Addemotion(coordinates, image_array, emotion_icon=None):
    print(coordinates)
    x, y, w = coordinates[:3]
    x_offset = x + w - emotion_icon.shape[1]
    y_offset = -80 + y
    y1, y2 = y_offset, y_offset + emotion_icon.shape[0]
    x1, x2 = x_offset, x_offset + emotion_icon.shape[1]

    alpha_s = emotion_icon[:, :, 2] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        image_array[y1:y2, x1:x2, c] = (alpha_s * emotion_icon[:, :, c] + 
                                        alpha_l * image_array[y1:y2, x1:x2, c])

    return image_array


def Addemotion_word(coordinates, image_array, emotion_icon=None):
    x, y, w, h = coordinates
    x_offset = x 
    y_offset = -90 + y
    y1, y2 = y_offset, y_offset + emotion_icon.shape[0]
    x1, x2 = x_offset, x_offset + emotion_icon.shape[1]

    alpha_s = emotion_icon[:, :, 2] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        image_array[y1:y2, x1:x2, c] = (alpha_s * emotion_icon[:, :, c] + 
                                        alpha_l * image_array[y1:y2, x1:x2, c])
    
    return image_array


def return_finish():
    return "True"