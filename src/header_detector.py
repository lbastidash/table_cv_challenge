# src/header_detector.py
import cv2
import numpy as np


def detect_header(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 5
    )

    proj = bw.sum(axis=1)
    proj = proj / max(proj.max(), 1)

    for i, v in enumerate(proj):
        if v < 0.1 and i > 5:
            return (0, i)

    return None
