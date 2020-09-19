from urllib.request import Request, urlopen

import cv2

import numpy as np


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    headers = {'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11'}
    req = Request(url=url, headers=headers)
    resp = urlopen(req)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return the image
    return image
