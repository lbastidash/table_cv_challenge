# src/detector.py
from ultralytics import YOLO

from header_detector import detect_header
from row_text import detect_rows_by_text
from row_structure import detect_rows_by_structure


MODEL_PATH = "models/table-detection-and-extraction.pt"


model = YOLO(MODEL_PATH)



def detect_table_and_header(image, proto_features):
    results = model(image, conf=0.25)[0]

    best = None

    for box in results.boxes:
        if model.names[int(box.cls[0])] != "table":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = image[y1:y2, x1:x2]

        header = detect_header(crop)
        if not header:
            continue

        header_end = header[1]

        if proto_features["mode"] == "WEB":
            rows = detect_rows_by_text(crop)
        else:
            rows = detect_rows_by_structure(crop, header_end)

        if len(rows) < 2:
            continue

        score = len(rows)

        if not best or score > best["score"]:
            best = {
                "table": (x1, y1, x2, y2),
                "header": (x1, y1, x2, y1 + header_end),
                "rows": rows,
                "score": score
            }

    if not best:
        raise RuntimeError("No valid table found")

    return best["table"], best["header"]
