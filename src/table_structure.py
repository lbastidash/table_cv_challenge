# src/table_structure.py
# src/table_structure.py
import cv2
import numpy as np

def _cluster_line_positions(positions, tol=8):
    """
    Agrupa posiciones de líneas (int) que están muy cerca (tolerance = tol).
    Devuelve lista ordenada de posiciones representativas.
    """
    if not positions:
        return []
    pos = sorted(positions)
    clusters = []
    current = [pos[0]]
    for p in pos[1:]:
        if p - current[-1] <= tol:
            current.append(p)
        else:
            clusters.append(int(np.median(current)))
            current = [p]
    clusters.append(int(np.median(current)))
    return clusters

def _detect_lines_by_morphology(gray):
    """
    Detecta líneas horizontales y verticales usando morfología, tolerante a bordes débiles.
    """
    h, w = gray.shape
    # binarizar adaptativamente
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 15, 8)
    # detectar horizontales
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//30 + 1, 1))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel)

    # detectar verticales
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h//30 + 1))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vert_kernel)

    # extraer coordenadas de líneas
    horiz_positions = []
    vert_positions = []

    # perfil horizontal
    if horiz is not None and horiz.any():
        row_sum = horiz.sum(axis=1)
        horiz_positions = list(np.where(row_sum > 0)[0])

    if vert is not None and vert.any():
        col_sum = vert.sum(axis=0)
        vert_positions = list(np.where(col_sum > 0)[0])

    # agrupar por proximidad
    horiz_positions = _cluster_line_positions(horiz_positions, tol=max(3, int(0.01*h)))
    vert_positions = _cluster_line_positions(vert_positions, tol=max(3, int(0.01*w)))

    return horiz_positions, vert_positions

def extract_structure(image, table_bbox, prototype_features, debug=False):
    """
    Dada la bbox de la tabla (x1,y1,x2,y2) devuelve:
      {"rows": [(x1,y1,x2,y2), ...], "columns": [(x1,y1,x2,y2), ...], "cells":[((x1,y1,x2,y2), ...)]}
    Si no se detectan líneas, se usa el avg_row_height del prototype para segmentar por filas.
    """
    x1, y1, x2, y2 = table_bbox
    table_img = image[y1:y2, x1:x2].copy()
    if table_img.size == 0:
        return {"rows": [], "columns": [], "cells": []}

    gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)

    # detectar líneas por morfología
    horiz_positions, vert_positions = _detect_lines_by_morphology(gray)

    if debug:
        print("Líneas horizontales detectadas (relativas a table):", horiz_positions)
        print("Líneas verticales detectadas (relativas a table):", vert_positions)

    h_table, w_table = gray.shape

    # Si no hay horizontales suficientes, fallback a proyección + avg_row_height
    rows = []
    if len(horiz_positions) >= 2:
        # convertir a rectángulos rows: entre líneas consecutivas
        # asegurar que incluimos borde superior e inferior si no presentes
        if horiz_positions[0] > 5:
            horiz_positions.insert(0, 0)
        if horiz_positions[-1] < h_table - 5:
            horiz_positions.append(h_table)
        for i in range(len(horiz_positions) - 1):
            ry1 = horiz_positions[i]
            ry2 = horiz_positions[i + 1]
            # map back to original image coords
            rows.append((x1, y1 + ry1, x2, y1 + ry2))
    else:
        # fallback: dividir por avg_row_height del prototype
        avg_h = prototype_features.get("avg_row_height", max(20, h_table // 6))
        nrows = max(1, h_table // avg_h)
        for i in range(nrows):
            ry1 = int(i * avg_h)
            ry2 = int(min(h_table, (i + 1) * avg_h))
            rows.append((x1, y1 + ry1, x2, y1 + ry2))

    # Column split: usar vert_positions si disponibles, sino intentar detectar por proyeccion vertical
    columns = []
    if len(vert_positions) >= 2:
        if vert_positions[0] > 5:
            vert_positions.insert(0, 0)
        if vert_positions[-1] < w_table - 5:
            vert_positions.append(w_table)
        for i in range(len(vert_positions) - 1):
            cx1 = vert_positions[i]
            cx2 = vert_positions[i+1]
            columns.append((x1 + cx1, y1, x1 + cx2, y2))
    else:
        # fallback: intentar detectar columnas por proyeccion vertical del texto/separadores
        col_profile = cv2.Canny(gray, 50, 150).sum(axis=0)
        thresh = col_profile.mean() * 0.5
        peaks = np.where(col_profile > thresh)[0]
        if len(peaks) > 2:
            groups = _cluster_line_positions(peaks.tolist(), tol=max(3, int(0.01*w_table)))
            if groups[0] > 5:
                groups.insert(0, 0)
            if groups[-1] < w_table - 5:
                groups.append(w_table)
            for i in range(len(groups)-1):
                cx1 = groups[i]
                cx2 = groups[i+1]
                columns.append((x1 + cx1, y1, x1 + cx2, y2))
        else:
            # si no detectamos columnas, crear una sola (full width)
            columns.append((x1, y1, x2, y2))

    # celdas: intersección rows x columns (opcional)
    cells = []
    for r in rows:
        rx1, ry1, rx2, ry2 = r
        row_cells = []
        for c in columns:
            cx1, cy1, cx2, cy2 = c
            # intersect
            cell_x1 = max(rx1, cx1)
            cell_y1 = max(ry1, cy1)
            cell_x2 = min(rx2, cx2)
            cell_y2 = min(ry2, cy2)
            if cell_x2 - cell_x1 > 5 and cell_y2 - cell_y1 > 5:
                row_cells.append((cell_x1, cell_y1, cell_x2, cell_y2))
        cells.append(row_cells)

    structure = {
        "rows": rows,
        "columns": columns,
        "cells": cells
    }

    return structure
