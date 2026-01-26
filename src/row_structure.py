# src/row_structure.py
import cv2
import numpy as np

def detect_rows_by_structure(crop_img, header_end_rel, debug=False):
    """
    Detecta filas por cambios geométricos (gradiente horizontal) en el body del crop.
    header_end_rel: y relativo (en píxeles) donde termina el header dentro del crop.
    Devuelve filas relativas al crop: [(r0, r1), ...] con coordenadas absolutas dentro del crop.
    """
    h, w = crop_img.shape[:2]
    body = crop_img[header_end_rel:, :]
    if body.size == 0:
        return []

    gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)
    grad = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.abs(grad)
    proj = grad.sum(axis=1)
    if proj.max() > 0:
        proj = proj / proj.max()

    rows = []
    in_row = False
    start = 0
    threshold = 0.08  # umbral para gradiente
    for i, v in enumerate(proj):
        if v > threshold and not in_row:
            start = i
            in_row = True
        elif v <= threshold and in_row:
            # devolver coordenadas relativas al crop (sumar header_end_rel)
            rows.append((header_end_rel + start, header_end_rel + i))
            in_row = False
    if in_row:
        rows.append((header_end_rel + start, header_end_rel + len(proj)))
    if debug:
        print(f"[row_structure] rows found: {len(rows)} header_end_rel={header_end_rel}")
    return rows
