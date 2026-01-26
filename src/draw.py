# src/draw.py
import cv2


def draw_results(image, table_bbox, header_bbox=None):
    x1, y1, x2, y2 = table_bbox

    # Tabla en rojo
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Header en verde (si existe)
    if header_bbox:
        hx1, hy1, hx2, hy2 = header_bbox
        cv2.rectangle(image, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)

    return image
