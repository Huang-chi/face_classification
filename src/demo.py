########
import threading
import time
########

from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'


predicted_ages = 1

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))

def compute_to_ages(bgr_image,model,img_size,detector,margin):
    for img in bgr_image:
        # input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = np.shape(img)

        # detect faces using dlib detector
        detected = detector(img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))
        
        # print("-------------start if---------------")
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

            # predict ages and genders of the detected faces
            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            print(predicted_genders)
            set_predicted_ages(predicted_ages)
            # print(get_predicted_ages)
    
    
def main(bgr_image):
    print("-------------start---------------")
    args = get_args()
    depth = args.depth
    k = args.width
    weight_file = args.weight_file
    margin = args.margin
    image_dir = args.image_dir

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
    # print(image_generator)
    # print("-------------start for loop ---------------")

    # 创建两个线程
    t = threading.Thread(target = compute_to_ages(bgr_image,model,img_size,detector,margin))
    print("---------keeo going-----------")
    ########

def run_thread(bgr_image):
    # start thread
    try:
        t.start()
    except:
        print ("Error: unable to start thread")
    
    

    # for img in bgr_image:
    #     # input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img_h, img_w= np.shape(img)

    #     # detect faces using dlib detector
    #     detected = detector(img, 1)
    #     faces = np.empty((len(detected), img_size, img_size, 3))
        
    #     # print("-------------start if---------------")
    #     if len(detected) > 0:
    #         for i, d in enumerate(detected):
    #             x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
    #             xw1 = max(int(x1 - margin * w), 0)
    #             yw1 = max(int(y1 - margin * h), 0)
    #             xw2 = min(int(x2 + margin * w), img_w - 1)
    #             yw2 = min(int(y2 + margin * h), img_h - 1)
    #             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #             # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
    #             faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

    #         # predict ages and genders of the detected faces
    #         results = model.predict(faces)
    #         predicted_genders = results[0]
    #         ages = np.arange(0, 101).reshape(101, 1)
    #         predicted_ages = results[1].dot(ages).flatten()
    #         print(predicted_ages)
    #         set_predicted_ages(predicted_ages)
    #         print(get_predicted_ages)

def set_predicted_ages(new_ages):
    predicted_ages = new_ages

def get_predicted_ages():
    # print(predicted_ages)    
    return predicted_ages

# if __name__ == '__main__':
#     main()



    