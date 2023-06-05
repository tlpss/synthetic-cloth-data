import cv2
import numpy as np


def sample_hsv_color():
    hue = np.random.uniform(0, 180)
    saturation = np.random.uniform(0.0, 1)
    value = np.random.uniform(0.0, 1)
    return np.array([hue, saturation, value])


def hsv_to_rgb(hsv: np.ndarray):
    assert hsv.shape == (3,)
    hsv = hsv.astype(np.float32)
    rgb = cv2.cvtColor(hsv[np.newaxis, np.newaxis, ...], cv2.COLOR_HSV2RGB)
    return rgb[0][0]
