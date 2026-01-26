# src/prototype.py

import cv2
import numpy as np


def extract_prototype_features(proto_img, debug=False):
    gray = cv2.cvtColor(proto_img, cv2.COLOR_BGR2GRAY)

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 5
    )

    text_density = bw.sum() / bw.size

    # Heurística CLAVE
    mode = "WEB" if text_density > 0.05 else "SAP"

    if debug:
        print("[DEBUG] Prototype text density:", round(text_density, 4))
        print("[DEBUG] Detected mode:", mode)

    return {
        "mode": mode,
        "height": proto_img.shape[0]
    }
