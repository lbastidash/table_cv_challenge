# src/structural_proposals.py
import cv2
import numpy as np

from column_consistency import detect_column_peaks
from row_structure import detect_rows_by_structure


def generate_structural_candidates(image, proto_features, debug=False):
    """
    Generates table candidates purely from structural cues.
    Designed mainly for SAP-like UIs.
    """
    img_h, img_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # edge detection
    edges = cv2.Canny(gray, 50, 150)

    # strong vertical emphasis
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    verticals = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v)

    contours, _ = cv2.findContours(
        verticals, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # --------------------------------------------------
        # Hard geometric filters
        # --------------------------------------------------
        if w / img_w < 0.25:
            continue

        if h / img_h < 0.08:
            # ❌ toolbars / menus
            continue

        if h / img_h > 0.9:
            continue

        # expand slightly
        pad_x = int(w * 0.05)
        pad_y = int(h * 0.05)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # --------------------------------------------------
        # Structural validation
        # --------------------------------------------------
        col_peaks = detect_column_peaks(crop, debug=False)
        if len(col_peaks) < 2:
            continue

        # try to detect rows structurally (SAP-safe)
        rows = detect_rows_by_structure(
            crop, header_end_rel=0, debug=False
        )
        if len(rows) < 1:
            continue

        # --------------------------------------------------
        # Toolbar / footer suppression
        # --------------------------------------------------
        top_bias = y1 / img_h
        bottom_bias = y2 / img_h

        score = 0.0
        score += len(col_peaks) * 3.0
        score += len(rows) * 2.0
        score += (w / img_w) * 5.0
        score += (h / img_h) * 3.0

        if top_bias < 0.05:
            score -= 6.0
        if bottom_bias > 0.95:
            score -= 6.0

        if debug:
            print(
                f"[structural] candidate {(x1,y1,x2,y2)} "
                f"cols={len(col_peaks)} rows={len(rows)} score={score:.2f}"
            )

        if score > 6.0:
            candidates.append((x1, y1, x2, y2))

    if debug:
        print(f"[structural] generated {len(candidates)} candidates")

    return candidates
