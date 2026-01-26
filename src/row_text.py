# src/row_text.py
import cv2
import numpy as np

def detect_rows_by_text(crop_img, debug=False):
    """
    Detecta filas usando proyección vertical de texto (relativo al crop).
    Devuelve lista de (r0, r1) relativos al crop.
    """
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 5)
    proj = bw.sum(axis=1).astype(float)
    if proj.max() > 0:
        proj = proj / proj.max()
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
    if in_row:
        rows.append((start, len(proj)))
    if debug:
        print(f"[row_text] rows found: {len(rows)}")
    return rows
