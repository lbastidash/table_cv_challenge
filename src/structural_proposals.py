# src/structural_proposals.py
import cv2
import numpy as np

# ------------------------------------------------------------
# Column detector (returns list of (x0,x1) relative to crop)
# ------------------------------------------------------------

def detect_columns_by_consistency(crop, debug=False):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    mag = np.abs(sobel)

    proj = mag.mean(axis=0)
    proj = cv2.GaussianBlur(proj, (9, 1), 0)

    thresh = proj.mean() + proj.std()
    xs = np.where(proj > thresh)[0]

    if len(xs) < 2:
        return []

    lines = []
    start = xs[0]
    for x in xs[1:]:
        if x > start + 3:
            lines.append(start)
        start = x
    lines.append(start)

    cols = []
    prev = 0
    for x in lines:
        if x - prev > 10:
            cols.append((prev, x))
        prev = x

    return cols


# ------------------------------------------------------------
# Header detection by vertical gradient projection
# returns (0,0,w,header_h) or None
# ------------------------------------------------------------
def detect_header_by_projection(crop, debug=False):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    grad = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.abs(grad)
    proj = mag.mean(axis=1)
    proj = cv2.GaussianBlur(proj, (1, 21), 0)

    h = crop.shape[0]
    search_h = min(80, max(10, h // 3))
    best_y = None
    best_score = 0
    for y in range(3, search_h - 3):
        s = proj[y : y + 5].sum()
        if s > best_score:
            best_score = s
            best_y = y
    if best_y is None or best_score <= 1e-3:
        return None
    header_bottom = min(h, best_y + 6)
    if debug:
        print(f"[header] bottom at {header_bottom} (score {best_score:.2f})")
    return (0, 0, crop.shape[1], header_bottom)


# ------------------------------------------------------------
# Row detection by horizontal gradient structure
# returns list of (r0,r1) relative to crop
# ------------------------------------------------------------

def detect_rows_by_structure(crop, header_end_rel=0, debug=False):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    sobel = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.abs(sobel)

    proj = mag.mean(axis=1)

    # suavizar
    proj = cv2.GaussianBlur(proj, (1, 9), 0)

    thresh = proj.mean() + 0.8 * proj.std()
    ys = np.where(proj > thresh)[0]

    if len(ys) < 2:
        return []

    # agrupar picos
    lines = []
    start = ys[0]
    for y in ys[1:]:
        if y > start + 3:
            lines.append(start)
        start = y
    lines.append(start)

    # generar filas
    rows = []
    prev = header_end_rel
    for y in lines:
        if y - prev > 8:
            rows.append((prev, y))
        prev = y

    return rows



def estimate_pitch(intervals, min_size=6):
    sizes = [b - a for a, b in intervals if (b - a) >= min_size]
    if len(sizes) < 2:
        return None
    return int(np.median(sizes))

