from os import listdir
from os.path import isfile, isdir, join
import random
import os

scanning_face_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
# sys.path.append(scanning_face_path)
print("22222")
print(scanning_face_path)
print("22222")

IMAGE_PATH = scanning_face_path

def import_image():
    files = listdir(IMAGE_PATH)
    image_dict = {}
    index = 0
    for f in files:
        fullpath = join(IMAGE_PATH, f)
        if isfile(fullpath):
            image_dict[index] = f
            index += 1

    return image_dict

def random_image(image_dict):
    image_len = len(image_dict)-1
    index = random.randint(0,24)
    return image_dict[index]

def return_image_path():
    image_dict = import_image()
    result = random_image(image_dict)
    return IMAGE_PATH+result

# if __name__ == "__main__":
#     image_dict = import_image()
#     result = random_image(image_dict)
#     print(result)