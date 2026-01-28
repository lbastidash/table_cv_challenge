# src/detector.py
import os
import cv2
import numpy as np
from ultralytics import YOLO

from structural_proposals import (
    detect_columns_by_consistency,
    detect_rows_by_structure,
    detect_header_by_projection,
)

MODEL_PATH = "models/table-detection-and-extraction.pt"
# confidence (float): Confidence threshold for detection
# iou_threshold (float): IoU threshold for NMS

CONF_DEFAULT = 0.40
IOU_DEFAULT = 0.45
DEBUG_DIR = "debug_candidates"

# ---------------------------
# YOLO proposals (multi-scale + sliding)
# ---------------------------
def yolo_propose_many(model, image, conf, iou, mode="WEB", debug=False):
    h, w = image.shape[:2]
    proposals = []

    scales = [1, 0.75, 1.25, 0.25]
    for s in scales:
        res = model.predict(
            image,
            imgsz=(int(w * s), int(h * s)),
            conf=conf,
            iou=iou,
            verbose=False,
        )
        if not res:
            continue

        r = res[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        scores = r.boxes.conf.cpu().numpy()

        if s != 1.0:
            boxes = boxes / s

        for b, cid, sc in zip(boxes, cls, scores):
            x1, y1, x2, y2 = map(int, b.tolist())
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue

            if mode == "SAP":
                pad_x, pad_y = 0.1, 0.10
            else:
                pad_x, pad_y = 0.0000001, 0.1

            nx1, ny1, nx2, ny2 = expand_bbox(
                (x1, y1, x2, y2), image.shape, pad_x, pad_y
            )

            proposals.append((nx1, ny1, nx2, ny2, cid, sc))

    # sliding windows
    if max(h, w) > 1200:
        if mode == "SAP":
            win = int(min(h, w) * 0.4)
            stride = int(win * 0.33)
        else:
            win = int(min(h, w) * 0.3)
            stride = int(win * 0.2)

        for y in range(0, h - win + 1, stride):
            for x in range(0, w - win + 1, stride):
                crop = image[y : y + win, x : x + win]
                res = model.predict(
                    crop,
                    imgsz=(win, win),
                    conf=max(conf * 0.7, 0.2),
                    iou=iou,
                    verbose=False,
                )
                if not res:
                    continue

                r = res[0]
                boxes = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                scores = r.boxes.conf.cpu().numpy()

                for b, cid, sc in zip(boxes, cls, scores):
                    bx1 = int(b[0]) + x
                    by1 = int(b[1]) + y
                    bx2 = int(b[2]) + x
                    by2 = int(b[3]) + y
                    if bx2 - bx1 < 8 or by2 - by1 < 8:
                        continue

                    if mode == "SAP":
                        pad_x, pad_y = 0.06, 0.10
                    else:
                        pad_x, pad_y = 0.03, 0.06

                    nx1, ny1, nx2, ny2 = expand_bbox(
                        (bx1, by1, bx2, by2), image.shape, pad_x, pad_y
                    )

                    proposals.append((nx1, ny1, nx2, ny2, cid, sc))

    # dedupe
    final = []
    for p in proposals:
        keep = True
        for i, q in enumerate(final):
            if box_overlap(p[:4], q[:4]) > 0.9:
                if p[5] > q[5]:
                    final[i] = p
                keep = False
                break
        if keep:
            final.append(p)

    return final

# ---------------------------
# Box overlap
# ---------------------------
def box_overlap(boxA, boxB):
    """
    Calculate the percentage overlap between two boxes.
    
    Args:
        boxA: First bounding box coordinates
        boxB: Second bounding box coordinates
            
    Returns:
        Percentage of overlap between the boxes
    """
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    min_area = min(boxA_area, boxB_area)
    if min_area == 0:
        return 0.0
            
    return (intersection_area / min_area)


def merge_boxes_iou(boxes, iou_thresh=0.10):
    merged = []
    for b in boxes:
        added = False
        for i, m in enumerate(merged):
            if box_overlap(b, m) > iou_thresh:
                merged[i] = (
                    min(b[0], m[0]),
                    min(b[1], m[1]),
                    max(b[2], m[2]),
                    max(b[3], m[3]),
                )
                added = True
                break
        if not added:
            merged.append(b)
    return merged
# ---------------------------
# BBox expansion
# ---------------------------
def expand_bbox(bbox, image_shape, pad_x_rel=0.02, pad_y_rel=0.04):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * pad_x_rel)
    pad_y = int(bh * pad_y_rel)

    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(w, x2 + pad_x)
    ny2 = min(h, y2 + pad_y)

    return (nx1, ny1, nx2, ny2)



# ---------------------------
# Merge vertically split tables
# ---------------------------
def merge_vertical_tables(cands, debug=False):
    if len(cands) <= 1:
        return cands

    cands = sorted(cands, key=lambda c: c["bbox"][1])
    merged = []

    for c in cands:
        if not merged:
            merged.append(c)
            continue

        prev = merged[-1]
        px1, py1, px2, py2 = prev["bbox"]
        cx1, cy1, cx2, cy2 = c["bbox"]

        hor_overlap = min(px2, cx2) - max(px1, cx1)
        min_w = min(px2 - px1, cx2 - cx1)
        hor_ratio = hor_overlap / max(min_w, 1)

        v_gap = cy1 - py2
        col_ratio = min(prev["ncols"], c["ncols"]) / max(prev["ncols"], c["ncols"])

        if hor_ratio > 0.8 and v_gap < 0.2 * (py2 - py1) and col_ratio > 0.7:
            prev["bbox"] = (px1, py1, px2, cy2)
            prev["score"] += c["score"] * 0.8
            prev["nrows"] += c["nrows"]
        else:
            merged.append(c)

    return merged
# ---------------------------
# MAIN entry
# ---------------------------
def detect_table_and_header(image, proto_features, conf=CONF_DEFAULT, iou=IOU_DEFAULT, debug=False):
    os.makedirs(DEBUG_DIR, exist_ok=True)

    mode = proto_features["mode"]  # SAP / WEB
    model = YOLO(MODEL_PATH)
    h, w = image.shape[:2]

    proposals = yolo_propose_many(model, image, conf, iou, mode, debug)

    candidates = merge_boxes_iou([p[:4] for p in proposals])

    scored = []
    for idx, (x1, y1, x2, y2) in enumerate(candidates):
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        header = detect_header_by_projection(crop, debug=debug)
        if header is None:
            continue

        rows = detect_rows_by_structure(crop, header_end_rel=header[3], debug=debug)
        cols = detect_columns_by_consistency(crop, debug=debug)
        
        score = len(rows) * 3 + len(cols) * 4
        score += (x2 - x1) / w * 4
        score += (y2 - y1) / h * 4

        scored.append(
            {
                "bbox": (x1, y1, x2, y2),
                "header": (x1 + header[0], y1 + header[1], x1 + header[2], y1 + header[3]),
                "rows": rows,
                "cols": cols,
                "nrows": len(rows),
                "ncols": len(cols),
                "score": score,
            }
        )

    if not scored:
        raise RuntimeError("No valid table found")

    scored = merge_vertical_tables(scored, debug)
    best = max(scored, key=lambda c: c["score"])

    rows_abs = [(best["bbox"][1] + r0, best["bbox"][1] + r1) for r0, r1 in best["rows"]]
    cols_abs = [(best["bbox"][0] + c0, best["bbox"][0] + c1) for c0, c1 in best["cols"]]

    return best["bbox"], best["header"], rows_abs, cols_abs


