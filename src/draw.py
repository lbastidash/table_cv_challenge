# src/draw.py
import cv2

def draw_results(image, table_bbox, header_bbox=None, rows=None, columns=None):
    out = image.copy()
    # tabla completa rojo
    x1, y1, x2, y2 = map(int, table_bbox)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0,0,255), 2)

    # header verde
    if header_bbox:
        hx1, hy1, hx2, hy2 = map(int, header_bbox)
        cv2.rectangle(out, (hx1, hy1), (hx2, hy2), (0,255,0), 2)

    # rows amarillo
    if rows:
        for r in rows:
            rx1, ry1, rx2, ry2 = map(int, r)
            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0,255,255), 1)

    # columns negras (líneas)
    if columns:
        for cx in columns:
            cx = int(cx)
            cv2.line(out, (cx, y1), (cx, y2), (0,0,0), 2)

    return out
