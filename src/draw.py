# src/draw.py
import cv2

def draw_results(image, table_bbox, header_bbox=None, rows=None, columns=None):
    """
    rows: list of (y1_abs, y2_abs)
    columns: list of (x1_abs, x2_abs)
    table_bbox: (x1,y1,x2,y2)
    """
    out = image.copy()
    x1, y1, x2, y2 = map(int, table_bbox)
    # header verde
    if header_bbox:
        hx1, hy1, hx2, hy2 = map(int, header_bbox)
        cv2.rectangle(out, (hx1, hy1), (hx2, hy2), (0,255,0), 2)

    # rows amarillo: rows are absolute y pairs
    if rows:
        for (ry1, ry2) in rows:
            cv2.rectangle(out, (x1, int(ry1)), (x2, int(ry2)), (0,255,255), 1)

    # columns negras (líneas)
    if columns:
        for (cx1, cx2) in columns:
            cx = int(cx1)
            cv2.line(out, (cx, y1), (cx, y2), (0,0,0), 2)
    # tabla completa rojo

    cv2.rectangle(out, (x1, y1), (x2, y2), (0,0,255), 2)

    return out
