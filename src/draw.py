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
        # rows: [(y1, y2), ...] RELATIVOS a la tabla
        tx1, ty1, tx2, ty2 = table_bbox

        for (ry1, ry2) in rows:
            y1 = int(ty1 + ry1)
            y2 = int(ty1 + ry2)
            cv2.rectangle(
                out,
                (int(tx1), y1),
                (int(tx2), y2),
                (0, 255, 255),
                2
            )

    # columns negras (líneas)
    if columns:
        for (cx1, cx2) in columns:
            x1 = int(tx1 + cx1)
            x2 = int(tx1 + cx2)
            cv2.line(
                out,
                (x1, int(ty1)),
                (x1, int(ty2)),
                (0, 0, 0),
                2
            )
    return out
