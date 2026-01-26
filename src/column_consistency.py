# src/column_consistency.py
import cv2
import numpy as np

def detect_column_peaks(img, debug=False):
    """
    Detecta picos (posibles columnas) y devuelve lista de x positions relativos al img.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 5)
    col_proj = bw.sum(axis=0).astype(float)
    if col_proj.max() > 0:
        col_proj = col_proj / col_proj.max()

    # suavizar con gaussian
    k = max(3, img.shape[1] // 200)
    if k % 2 == 0:
        k += 1
    col_smooth = cv2.GaussianBlur(col_proj.reshape(1, -1), (1, k), 0).reshape(-1)

    peaks_mask = col_smooth > 0.18
    peaks_positions = []
    in_peak = False
    start = 0
    for i, v in enumerate(peaks_mask):
        if v and not in_peak:
            in_peak = True
            start = i
        elif not v and in_peak:
            peaks_positions.append(int((start + i) / 2))
            in_peak = False
    if in_peak:
        peaks_positions.append(int((start + len(peaks_mask)-1) / 2))

    if debug:
        print(f"[column_consistency] peaks detected: {peaks_positions}")
    return peaks_positions
