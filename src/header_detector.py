# src/header_detector.py
import cv2
import numpy as np

def detect_header(crop_img, debug=False):
    """
    Detecta un bloque de header en la porción superior del crop.
    Devuelve bbox relativa al crop: (hx1, hy1, hx2, hy2) o None.
    """
    h, w = crop_img.shape[:2]
    top_frac = 0.35
    top_h = max(12, int(h * top_frac))
    top_region = crop_img[:top_h, :].copy()

    gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    bw = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w//12), 3))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    dil = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)

    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0

    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        if ww < w * 0.15 or hh < 8:
            continue
        area = ww * hh
        width_ratio = ww / w
        height_ratio = hh / top_h
        score = area * (0.6 * width_ratio + 0.4 * (1 - height_ratio))
        if score > best_score:
            best_score = score
            best = (x, y, x + ww, y + hh)

    if best is None:
        # fallback: buscar por proyección si no se detectó contorno significativo
        row_proj = bw.sum(axis=1)
        if row_proj.max() > 0:
            idxs = (row_proj > (0.2 * row_proj.max())).nonzero()[0]
            if len(idxs) > 0:
                y0 = int(max(0, idxs[0] - 2))
                y1 = int(min(top_h, idxs[-1] + 2))
                best = (0, y0, w, y1)
            else:
                return None
        else:
            return None

    if debug:
        print(f"[header_detector] header rel bbox: {best} top_h={top_h} best_score={best_score}")
    return best
