# src/prototype.py
import cv2
import numpy as np
from structural_proposals import detect_columns_by_consistency, detect_header_by_projection, detect_rows_by_structure

def extract_prototype_features(proto_img, proto_name="", debug=False):

    h, w = proto_img.shape[:2]

    # asumimos toda la imagen como tabla
    table = proto_img.copy()

    # columnas por proyección vertical
    cols = detect_columns_by_consistency(table, debug=debug)
    col_widths = [c2 - c1 for c1, c2 in cols]

    # header + filas
    header = detect_header_by_projection(table, debug=debug)
    header_end = header[3] if header else int(0.15 * h)

    rows = detect_rows_by_structure(table, header_end_rel=header_end, debug=debug)
    row_heights = [r2 - r1 for r1, r2 in rows]

    # estadísticas robustas
    proto_features = {
        "proto_width": w,
        "proto_height": h,
        "median_col_width": np.median(col_widths) if col_widths else None,
        "median_row_height": np.median(row_heights) if row_heights else None,
        "ncols": len(cols),
        "nrows": len(rows),
    }

    # modo por nombre (por ahora)
    name = proto_name.lower()
    if "sap" in name:
        proto_features["mode"] = "SAP"
    else:
        proto_features["mode"] = "WEB"

    return proto_features
