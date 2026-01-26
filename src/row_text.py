# src/row_text.py
import cv2
import numpy as np


def detect_rows_by_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 5
    )

    proj = bw.sum(axis=1)
    proj = proj / max(proj.max(), 1)

    rows = []
    in_row = False
    start = 0

    for i, v in enumerate(proj):
        if v > 0.15 and not in_row:
            start = i
            in_row = True
        elif v <= 0.15 and in_row:
            rows.append((start, i))
            in_row = False

    return rows
