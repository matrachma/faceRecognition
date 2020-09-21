import cv2
import os
import numpy as np
from .wide_resnet import WideResNet
from keras.utils.data_utils import get_file


class FaceCV(object):
    """
    Singleton class for face recongnition task
    """
    CASE_PATH = "recognizer/pretrained_models/default.xml"
    MODEL_PATH = os.path.join(os.getcwd(), "recognizer/pretrained_models")
    WRN_WEIGHTS_PATH = "https://github.com/matrachma/faceRecognition/releases/download/v1/weights.18-4.06.hdf5"

    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = self.MODEL_PATH
        fpath = get_file(os.path.join(model_dir, 'weights.18-4.06.hdf5'),
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir="",
                         cache_dir=model_dir)
        self.model.load_weights(fpath)

    @classmethod
    def crop_face(self, imgarray, section, margin=40, size=64):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w, h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w - 1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h - 1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def detect_face(self, image):
        result = ""
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)

        grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            grayImg,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(8, 8)
        )
        face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
        for i, face in enumerate(faces):
            face_img, cropped = self.crop_face(image, face, margin=40, size=self.face_size)
            (x, y, w, h) = cropped
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 200, 0), 2)
            face_imgs[i, :, :, :] = face_img
        if len(face_imgs) > 0:
            # predict ages and genders of the detected faces
            results = self.model.predict(face_imgs)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
        for i, face in enumerate(faces):
            label = "{}:{}".format("female" if predicted_genders[i][0] > 0.5 else "male", int(predicted_ages[i]))
            result += label + "|"

        return result
