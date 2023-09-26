import cv2
import numpy as np


def sample_hsv_color():
    """hsv color with h,s,v in range (0,1) as in blender"""
    hue = np.random.uniform(0, 1.0)
    saturation = np.random.uniform(0.0, 1)
    value = np.random.uniform(0.0, 1)
    return np.array([hue, saturation, value])


def hsv_to_rgb(hsv: np.ndarray):
    """converts hsv in range (0,1) to rgb in range (0,1)"""
    assert hsv.shape == (3,)
    assert np.all(hsv <= 1.0), "hsv values must be in range (0,1)"
    hsv = hsv.astype(np.float32)
    hsv[0] *= 360  # convert from (0,1) to degrees as in blender
    rgb = cv2.cvtColor(hsv[np.newaxis, np.newaxis, ...], cv2.COLOR_HSV2RGB)
    return rgb[0][0]


if __name__ == "__main__":
    hsv = np.array([0.8, 0.3, 0.3])
    rgb = hsv_to_rgb(hsv)
    print(rgb)
    assert np.allclose(rgb, np.array([0.28, 0.21, 0.3]), atol=0.01)
