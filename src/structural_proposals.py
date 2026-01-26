# src/structural_proposals.py
import cv2
import numpy as np
from typing import List, Tuple
from math import ceil

# Reuse the column/row/header detectors (must be same as imports in detector)
def detect_columns_by_consistency(crop, debug=False):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    mag = np.abs(grad)
    proj = mag.mean(axis=0)
    if proj.max()>0:
        proj = proj / proj.max()
    k = max(3, crop.shape[1]//200)
    if k%2==0: k+=1
    proj_s = cv2.GaussianBlur(proj.reshape(1,-1),(1,k),0).reshape(-1)
    thresh = np.percentile(proj_s, 70)
    peaks = np.where(proj_s>thresh)[0]
    if len(peaks)==0: return []
    cols=[]
    s=peaks[0]
    for i in range(1,len(peaks)):
        if peaks[i] > peaks[i-1] + 4:
            cols.append((s, peaks[i-1]))
            s = peaks[i]
    cols.append((s, peaks[-1]))
    if debug: print("[columns] ", len(cols))
    return cols

def detect_header_by_projection(crop, debug=False):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    grad = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.abs(grad)
    proj = mag.mean(axis=1)
    proj = cv2.GaussianBlur(proj, (1,21), 0)
    h = crop.shape[0]
    search_h = min(80, max(10, h//3))
    best_y=None; best_s=0
    for y in range(3, search_h-3):
        s = proj[y:y+5].sum()
        if s>best_s:
            best_s=s; best_y=y
    if best_y is None or best_s<=1e-3: return None
    return (0,0,crop.shape[1], min(h, best_y+6))

def detect_rows_by_structure(crop, header_end_rel=0, debug=False):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    grad = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.abs(grad)
    body = mag[header_end_rel:]
    if body.size==0: return []
    proj = body.mean(axis=1)
    proj = cv2.GaussianBlur(proj, (1,21),0)
    if proj.max()==0: return []
    thresh = np.percentile(proj, 70)
    peaks = np.where(proj>thresh)[0]
    if peaks.size==0: return []
    rows=[]
    s=peaks[0]
    for i in range(1,len(peaks)):
        if peaks[i] > peaks[i-1]+3:
            rows.append((header_end_rel + s, header_end_rel + peaks[i-1]))
            s = peaks[i]
    rows.append((header_end_rel + s, header_end_rel + peaks[-1]))
    if debug: print("[rows] ", len(rows))
    return rows

def propose_structural_candidates(image, proto_features=None, debug=False) -> List[Tuple[int,int,int,int]]:
    """
    Conservative structural proposals:
      - use Sobel magnitude and morphological close to find blobs
      - filter by size and by presence of columns+rows in the crop
    """
    img_h, img_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.convertScaleAbs(np.abs(gx)+np.abs(gy))
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3,9))
    joined = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, se)
    # OTSU threshold
    _, th = cv2.threshold(joined,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    proposals=[]
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w < 0.2*img_w: continue
        if h < 0.045*img_h: continue
        if h > 0.95*img_h: continue
        if y < 0.02*img_h or (y+h) > 0.98*img_h: continue
        crop = image[y:y+h, x:x+w]
        cols = detect_columns_by_consistency(crop, debug=False)
        if len(cols)<2: continue
        header = detect_header_by_projection(crop, debug=False)
        h_end = header[3] if header is not None else 0
        rows = detect_rows_by_structure(crop, header_end_rel=h_end, debug=False)
        if len(rows) < 2: continue
        score = len(cols)*2.0 + len(rows)*3.0 + (w/img_w)*4.0 + (h/img_h)*2.0
        if score > 6.0:
            proposals.append((x,y,x+w,y+h))
            if debug:
                print("[structural_proposals] add", (x,y,x+w,y+h), "cols", len(cols), "rows", len(rows), "score", score)
    if debug:
        print("[structural_proposals] proposals:", len(proposals))
    return proposals
