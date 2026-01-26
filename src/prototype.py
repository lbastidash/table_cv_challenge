# src/prototype.py
import cv2
import numpy as np
from column_consistency import detect_column_peaks

def extract_prototype_features(proto_img, debug=False):
    """
    Extrae: mode (SAP/WEB), height, width, text_density, proto_col_count.
    mode:
      - SAP: si detect_column_peaks >= 3 (columnas claras)
      - WEB: si text density alto y proto_col_count bajo/indeterminado
    """
    gray = cv2.cvtColor(proto_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # binarizar suave para estimar presencia de texto
    bw = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 5)

    text_density = float(bw.sum()) / (255.0 * bw.size) * 100.0  # porcentaje aproximado

    # contar columnas en prototype
    cols = detect_column_peaks(proto_img, debug=debug)
    proto_col_count = len(cols)

    # heurística: si hay 3+ columnas visuales => SAP-like (column alignment strong)
    if proto_col_count >= 3:
        mode = "SAP"
    else:
        # si mucho texto => WEB
        mode = "WEB" if text_density > 5.0 else "SAP"

    if debug:
        print(f"[prototype] text_density={text_density:.3f} proto_col_count={proto_col_count} mode={mode}")

    return {
        "mode": mode,
        "height": h,
        "width": w,
        "text_density": text_density,
        "proto_col_count": proto_col_count
    }
