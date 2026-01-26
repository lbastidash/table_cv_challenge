# src/prototype.py
import cv2
import numpy as np
import os
from structural_proposals import (
    detect_columns_by_consistency,
    detect_header_by_projection,
    detect_rows_by_structure,
)

def extract_prototype_features(proto_img, proto_path=None, debug=False):
    """
    proto_img: numpy image (cv2.imread)
    proto_path: optional path or filename string used to set mode by name
    Returns a dict with geometry features and mode.
    """
    h, w = proto_img.shape[:2]
    # treat whole prototype as the table area
    table = proto_img.copy()

    # detect columns
    cols = detect_columns_by_consistency(table, debug=debug)
    col_widths = [c2 - c1 for (c1, c2) in cols] if cols else []

    # detect header
    header = detect_header_by_projection(table, debug=debug)
    header_end = header[3] if header else int(0.12 * h)

    # detect rows
    rows = detect_rows_by_structure(table, header_end_rel=header_end, debug=debug)
    row_heights = [r2 - r1 for (r1, r2) in rows] if rows else []

    proto_features = {
        "proto_width": w,
        "proto_height": h,
        "ncols": len(cols),
        "nrows": len(rows),
        "median_col_width": int(np.median(col_widths)) if col_widths else None,
        "median_row_height": int(np.median(row_heights)) if row_heights else None,
    }

    # determine mode by filename if given (SAP.png -> SAP, WEB.png -> WEB)
    mode = "WEB"
    if proto_path:
        base = os.path.basename(proto_path).lower()
        if "sap" in base:
            mode = "SAP"
        elif "web" in base:
            mode = "WEB"
    proto_features["mode"] = mode

    if debug:
        print(f"[prototype] mode={mode} ncols={proto_features['ncols']} nrows={proto_features['nrows']} median_col={proto_features['median_col_width']} median_row={proto_features['median_row_height']}")

    return proto_features