# src/row_structure.py
import cv2
import numpy as np


def detect_rows_by_structure(img, header_end):
    body = img[header_end:, :]

    gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)

    grad = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = abs(grad)

    proj = grad.sum(axis=1)
    proj = proj / max(proj.max(), 1)

    rows = []
    in_row = False
    start = 0

    for i, v in enumerate(proj):
        if v > 0.1 and not in_row:
            start = i
            in_row = True
        elif v <= 0.1 and in_row:
            rows.append((start + header_end, i + header_end))
            in_row = False

    return rows
