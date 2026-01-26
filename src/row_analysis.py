# src/row_analysis.py
import cv2
import numpy as np


def extract_row_bands(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 5
    )

    projection = bw.sum(axis=1)
    projection = projection / projection.max()

    # Detectar picos (bandas de texto)
    bands = []
    in_band = False
    start = 0

    for i, v in enumerate(projection):
        if v > 0.15 and not in_band:
            in_band = True
            start = i
        elif v <= 0.15 and in_band:
            bands.append((start, i))
            in_band = False

    if in_band:
        bands.append((start, len(projection)))

    return bands
