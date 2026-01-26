# src/detector.py
import os
import cv2
import numpy as np
from ultralytics import YOLO

from header_detector import detect_header
from row_text import detect_rows_by_text
from row_structure import detect_rows_by_structure
from column_consistency import detect_column_peaks
from structural_proposals import generate_structural_candidates

MODEL_PATH = "models/table-detection-and-extraction.pt"


# src/detector.py
DEBUG_DIR = "debug_candidates"

_model = None

def _load_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def detect_table_and_header(image, proto_features, conf=0.25, debug=False):
    """
    Returns:
        table_bbox: (x1,y1,x2,y2)
        header_bbox: (x1,y1,x2,y2)
        rows_abs: list[(x1,y1,x2,y2)]
        columns_abs: list[x positions]
    """
    model = _load_model()
    results = model(image, conf=conf)[0]

    img_h, img_w = image.shape[:2]
    candidates = []

    if debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Collect YOLO candidates
    # ------------------------------------------------------------
    yolo_boxes = []
    names = model.names if hasattr(model, "names") else {}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = (
            names.get(cls_id, str(cls_id))
            if isinstance(names, dict)
            else names[cls_id]
        )
        if "table" not in class_name.lower():
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        if (x2 - x1) > img_w * 0.2:
            yolo_boxes.append((x1, y1, x2, y2))

    # ------------------------------------------------------------
    # 2. Fallback: add structural proposals if YOLO is weak
    # ------------------------------------------------------------
    if len(yolo_boxes) < 2:
        if debug:
            print("[detector] YOLO returned few candidates, adding structural proposals")
        structural_boxes = generate_structural_candidates(
            image, proto_features, debug=debug
        )
        yolo_boxes.extend(structural_boxes)

    # ------------------------------------------------------------
    # 3. Evaluate each candidate
    # ------------------------------------------------------------
    for idx, (x1, y1, x2, y2) in enumerate(yolo_boxes):
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        mode = proto_features.get("mode", "WEB")

        # -------------------------
        # Header detection (mandatory)
        # -------------------------
        header_rel = detect_header(crop, debug=debug)
        if header_rel is None:
            if debug:
                print(f"[detector] candidate {idx} rejected: no header")
            continue

        hx1, hy1, hx2, hy2 = header_rel
        header_abs = (x1 + hx1, y1 + hy1, x1 + hx2, y1 + hy2)
        header_end_rel = hy2

        # -------------------------
        # Row detection
        # -------------------------
        if mode == "SAP":
            rows_rel = detect_rows_by_structure(crop, header_end_rel, debug=debug)
        else:
            rows_rel = detect_rows_by_text(crop, debug=debug)

        rows_abs = [
            (x1, y1 + r0, x2, y1 + r1)
            for (r0, r1) in rows_rel
        ]

        # -------------------------
        # Column detection
        # -------------------------
        col_peaks = detect_column_peaks(crop, debug=debug)
        columns_abs = [x1 + px for px in col_peaks]

        nrows = len(rows_abs)
        ncols = len(col_peaks)

        # -------------------------
        # Minimal validity rules
        # -------------------------
        min_rows = 1 if mode == "SAP" else 2
        if nrows < min_rows:
            if debug:
                print(
                    f"[detector] candidate {idx} rejected: too few rows ({nrows})"
                )
            continue

        if mode == "SAP" and ncols < 2:
            if debug:
                print(
                    f"[detector] candidate {idx} rejected: too few columns ({ncols})"
                )
            continue

        height = y2 - y1
        width = x2 - x1

        if mode == "SAP" and height < img_h * 0.05:
            if debug:
                print(
                    f"[detector] candidate {idx} rejected: extremely small height"
                )
            continue

        # ------------------------------------------------------------
        # 4. Scoring (SAP-biased toward structure)
        # ------------------------------------------------------------
        height_ratio = height / img_h
        width_ratio = width / img_w
        header_ratio = (hy2 - hy1) / max(1, height)

        score = 0.0

        # structure is king for SAP
        score += ncols * (3.0 if mode == "SAP" else 1.5)
        score += nrows * 2.0

        # prefer wide tables
        score += width_ratio * 5.0

        # penalize bottom UI bars
        bottom_penalty = max(0.0, (y2 / img_h) - 0.75)
        score -= bottom_penalty * 10.0

        # header must exist but not dominate
        score += min(header_ratio, 0.25) * 4.0

        candidates.append({
            "bbox": (x1, y1, x2, y2),
            "header": header_abs,
            "rows": rows_abs,
            "columns": columns_abs,
            "score": score,
            "index": idx
        })

        if debug:
            base = os.path.join(DEBUG_DIR, f"cand_{idx}_score_{score:.2f}")
            cv2.imwrite(base + ".png", crop)
            with open(base + ".txt", "w") as f:
                f.write(f"mode: {mode}\n")
                f.write(f"bbox: {(x1,y1,x2,y2)}\n")
                f.write(f"nrows: {nrows}\n")
                f.write(f"ncols: {ncols}\n")
                f.write(f"height_ratio: {height_ratio:.3f}\n")
                f.write(f"width_ratio: {width_ratio:.3f}\n")
                f.write(f"header_ratio: {header_ratio:.3f}\n")
                f.write(f"score: {score:.3f}\n")

    # ------------------------------------------------------------
    # 5. Select best candidate
    # ------------------------------------------------------------
    if not candidates:
        raise RuntimeError("No valid table found after candidate validation")

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]

    if debug:
        print(
            f"[detector] selected candidate {best['index']} "
            f"with score {best['score']:.2f}"
        )

    return best["bbox"], best["header"], best["rows"], best["columns"]