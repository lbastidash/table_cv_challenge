# src/visualize.py
import cv2

def _bound_box_to_points(bbox):
    x1, y1, x2, y2 = bbox
    return (int(x1), int(y1)), (int(x2), int(y2))

def draw_results(image, table_bbox, header_bbox, structure):
    out = image.copy()

    # tabla completa (rojo)
    p1, p2 = _bound_box_to_points(table_bbox)
    cv2.rectangle(out, p1, p2, (0,0,255), 2)

    # header (verde)
    if header_bbox is not None:
        hp1, hp2 = _bound_box_to_points(header_bbox)
        cv2.rectangle(out, hp1, hp2, (0,255,0), 2)

    # rows (amarillo)
    for r in structure.get("rows", []):
        r1, r2 = _bound_box_to_points(r)
        cv2.rectangle(out, r1, r2, (0,255,255), 1)

    # columns (negro - lineas)
    for c in structure.get("columns", []):
        (cx1, cy1, cx2, cy2) = c
        cv2.line(out, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (0,0,0), 2)

    # cells (delicadas en azul claro)
    for row_cells in structure.get("cells", []):
        for cell in row_cells:
            c1, c2 = _bound_box_to_points(cell)
            cv2.rectangle(out, c1, c2, (255,200,0), 1)

    return out
