# src/sap_row_structure.py
import cv2
import numpy as np


def detect_sap_rows(img, header_end_y):
    body = img[header_end_y:, :]

    gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)

    # Detectar cambios suaves (no texto)
    grad = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.abs(grad)

    proj = grad.sum(axis=1)
    proj = proj / max(proj.max(), 1)

    # Buscar picos periódicos
    peaks = proj > 0.1

    rows = []
    in_row = False
    start = 0

    for i, v in enumerate(peaks):
        if v and not in_row:
            start = i
            in_row = True
        elif not v and in_row:
            rows.append((start, i))
            in_row = False

    return rows
