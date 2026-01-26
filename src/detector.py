# src/detector.py
import os
import math
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

from structural_proposals import propose_structural_candidates, detect_columns_by_consistency, detect_rows_by_structure, detect_header_by_projection

MODEL_PATH = "models/table-detection-and-extraction.pt"  # tu modelo HF
DEBUG_DIR = "debug_candidates"
CONF_DEFAULT = 0.30
IOU_DEFAULT = 0.45

# ---- util IoU ----
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    if xB<=xA or yB<=yA:
        return 0.0
    inter = (xB-xA)*(yB-yA)
    union = (boxA[2]-boxA[0])*(boxA[3]-boxA[1]) + (boxB[2]-boxB[0])*(boxB[3]-boxB[1]) - inter
    return inter/union if union>0 else 0.0

# ---- merge IoU-based ----
def merge_boxes_iou(boxes, iou_thresh=0.3):
    if len(boxes)==0:
        return []
    used = [False]*len(boxes)
    merged = []
    for i, b in enumerate(boxes):
        if used[i]:
            continue
        bx = np.array(b, dtype=float)
        used[i]=True
        # grow by merging overlapping boxes
        changed = True
        while changed:
            changed=False
            for j, c in enumerate(boxes):
                if used[j]: continue
                if iou(bx, c) > iou_thresh:
                    # merge by taking min x1,y1 and max x2,y2
                    bx = np.array([min(bx[0], c[0]), min(bx[1], c[1]), max(bx[2], c[2]), max(bx[3], c[3])], dtype=float)
                    used[j]=True
                    changed=True
        merged.append(tuple(map(int,bx.tolist())))
    return merged

# ---- cluster cell boxes into table candidates ----
def cluster_cells_to_tables(cell_boxes, eps_pixels=80, min_samples=3):
    """
    cell_boxes: Nx4 array
    returns list of candidate bboxes (x1,y1,x2,y2)
    """
    if len(cell_boxes)==0:
        return []
    centers = np.array([[(b[0]+b[2])/2.0, (b[1]+b[3])/2.0] for b in cell_boxes])
    # eps in pixels; if image large, increase eps
    clustering = DBSCAN(eps=eps_pixels, min_samples=min_samples).fit(centers)
    labels = clustering.labels_
    candidates=[]
    for lab in set(labels):
        if lab==-1: continue
        group_idx = np.where(labels==lab)[0]
        group_boxes = np.array(cell_boxes)[group_idx]
        x1 = int(group_boxes[:,0].min())
        y1 = int(group_boxes[:,1].min())
        x2 = int(group_boxes[:,2].max())
        y2 = int(group_boxes[:,3].max())
        candidates.append((x1,y1,x2,y2))
    return candidates

# ---- run YOLO at multiple scales + sliding windows ----
def yolo_propose_many(
    model,
    image,
    conf=CONF_DEFAULT,
    iou=IOU_DEFAULT,
    mode="WEB",
    debug=False
):
    """
    Returns list of (x1,y1,x2,y2, class_id, conf)
    Strategy:
      - run once on full image (default)
      - run on resized scales (0.5, 1.0, 1.5)
      - run sliding windows (overlap 50%) at medium scale if image big
    """
    h,w = image.shape[:2]
    proposals=[]
    scales = [1.0, 0.75, 1.25]  # try a few
    for s in scales:
        new_h = int(h*s); new_w = int(w*s)
        try:
            res = model.predict(image, imgsz=(new_w,new_h), conf=conf, iou=iou, verbose=False)
        except Exception:
            # fallback to single call
            res = model(image, conf=conf, iou=iou)
        if len(res)==0:
            continue
        r = res[0]
        # boxes in res[0].boxes.xyxy
        try:
            boxes = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()
        except Exception:
            # older / different API shapes
            boxes = np.array(r.boxes.xyxy)
            cls = np.array(r.boxes.cls).astype(int)
            scores = np.array(r.boxes.conf)
        # map boxes back to original image if scale changed
        if s!=1.0:
            inv_s = 1.0/s
            boxes = boxes * inv_s
        for b, c_id, sc in zip(boxes, cls, scores):
            x1,y1,x2,y2 = map(int,b.tolist())
            # clamp
            x1 = max(0, min(x1,w-1)); x2 = max(0, min(x2,w-1))
            y1 = max(0, min(y1,h-1)); y2 = max(0, min(y2,h-1))
            if x2-x1<4 or y2-y1<4: 
                continue
            proposals.append((x1,y1,x2,y2,int(c_id), float(sc)))
    # sliding windows if image large
    if max(h,w) > 1200:
        if mode == "SAP":
            # SAP: 
            win = int(min(h, w) * 0.4)
            stride = int(win * 0.33)
        else:
            # WEB:
            win = int(min(h, w) * 0.5)
            stride = int(win * 0.5)
        for y in range(0, h-win+1, stride):
            for x in range(0, w-win+1, stride):
                crop = image[y:y+win, x:x+win]
                try:
                    res = model.predict(crop, imgsz=(win,win), conf=max(conf*0.7,0.2), iou=iou, verbose=False)
                except Exception:
                    res = model(crop, conf=max(conf*0.7,0.2), iou=iou)
                if len(res)==0: continue
                r = res[0]
                try:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    cls = r.boxes.cls.cpu().numpy().astype(int)
                    scores = r.boxes.conf.cpu().numpy()
                except Exception:
                    boxes = np.array(r.boxes.xyxy)
                    cls = np.array(r.boxes.cls).astype(int)
                    scores = np.array(r.boxes.conf)
                for b, c_id, sc in zip(boxes, cls, scores):
                    bx1,by1,bx2,by2 = 0,0,0,0  # placeholder
                    x1b = int(b[0])+x; y1b = int(b[1])+y; x2b = int(b[2])+x; y2b = int(b[3])+y
                    if x2b-x1b<4 or y2b-y1b<4: continue
                    proposals.append((x1b,y1b,x2b,y2b,int(c_id),float(sc)))
    # deduplicate by merging IoU>0.9 identicals
    final=[]
    for p in proposals:
        added=False
        for i,q in enumerate(final):
            if box_iou(p[:4], q[:4])>0.9:
                # keep higher score
                if p[5] > q[5]:
                    final[i]=p
                added=True
                break
        if not added:
            final.append(p)
    return final

def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if min(areaA, areaB) == 0:
        return 0.0

    return interArea / min(areaA, areaB)

# ---- main detection function ----
def detect_table_and_header(image, proto_features, conf=CONF_DEFAULT, iou=IOU_DEFAULT, debug=False):
    """
    New detector:
      - get many proposals via yolo_propose_many (tables, cells)
      - if cell/row boxes exist, cluster them to produce table candidates
      - merge all proposals and structural fallbacks, score, pick best
      - return (table_bbox, header_bbox, rows_abs, cols_abs)
    """
    os.makedirs(DEBUG_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)

    h,w = image.shape[:2]
    mode = proto_features.get("mode", "WEB") if proto_features else "WEB"
    proposals = yolo_propose_many(
        model, image, conf=conf, iou=iou, mode=mode, debug=debug
    )

    # separate by class name if possible
    names = model.names if hasattr(model, "names") else {}
    table_boxes=[]
    cell_boxes=[]
    row_boxes=[]
    for (x1,y1,x2,y2,cid,score) in proposals:
        cname = names.get(cid, str(cid)).lower() if names else str(cid)
        if "table" in cname and "cell" not in cname:
            table_boxes.append((x1,y1,x2,y2,score))
        elif "cell" in cname or "table cell" in cname or "table_cell" in cname:
            cell_boxes.append((x1,y1,x2,y2))
        elif "row" in cname:
            row_boxes.append((x1,y1,x2,y2))

    # if many cell boxes -> cluster them into candidate tables
    cluster_candidates=[]
    if len(cell_boxes)>0:
        cluster_candidates = cluster_cells_to_tables(cell_boxes, eps_pixels=max(30,int(min(h,w)*0.03)), min_samples=3)

    # Merge YOLO table boxes with cluster candidates
    merged_candidates = []
    # add raw table boxes (without score)
    merged_candidates.extend([ (b[0],b[1],b[2],b[3]) for b in table_boxes ])
    # add clusters
    merged_candidates.extend(cluster_candidates)

    # lastly, structural proposals fallback
    structural = propose_structural_candidates(image, proto_features, debug=debug)
    for s in structural:
        if s not in merged_candidates:
            merged_candidates.append(s)

    # if still empty, fallback to the largest YOLO detection area (if any)
    if len(merged_candidates)==0 and len(proposals)>0:
        sorted_by_area = sorted(proposals, key=lambda p: (p[2]-p[0])*(p[3]-p[1]), reverse=True)
        merged_candidates.append(tuple(sorted_by_area[0][:4]))

    # merge overlapping candidates (IoU merging)
    merged_candidates = merge_boxes_iou(merged_candidates, iou_thresh=0.15)

    # Score each candidate
    scored=[]
    for idx, (x1,y1,x2,y2) in enumerate(merged_candidates):
        crop = image[y1:y2, x1:x2]
        ch,cw = crop.shape[:2]
        if ch<4 or cw<4: continue

        # header detection
        header = detect_header_by_projection(crop, debug=debug)
        if header is None:
            continue
        hx1,hy1,hx2,hy2 = header
        header_abs = (x1+hx1, y1+hy1, x1+hx2, y1+hy2)
        header_h = hy2-hy1

        # rows by structure (works for SAP even without text)
        
        expected_h = proto_features["median_row_height"]
        rows = detect_rows_by_structure(crop, header_end_rel=hy2, debug=debug)

        rows = [
            r for r in rows
            if abs((r[1] - r[0]) - expected_h) < 0.5 * expected_h
        ]
        nrows = len(rows)

        # columns by consistency
        expected_w = proto_features["median_col_width"]
        cols = detect_columns_by_consistency(crop, debug=debug)

        cols = [
            c for c in cols
            if abs((c[1] - c[0]) - expected_w) < 0.6 * expected_w
        ]
        ncols = len(cols)
        # cells count intersection of detected cells (if any proposals present)
        # count how many proposal cell boxes fall in this candidate
        cell_count = sum(1 for cb in cell_boxes if cb[0]>=x1 and cb[2]<=x2 and cb[1]>=y1 and cb[3]<=y2)

        # scoring: favor many cells/rows/cols, favor width and height reasonable, penalize bottom
        score = 0.0
        score += cell_count * 4.0
        score += nrows * 3.0
        score += ncols * 2.0
        score += ( (x2-x1)/w ) * 4.0
        score += min(0.2, header_h/max(1,ch)) * 8.0

        # penalties
        height_ratio = ch/h
        if height_ratio < 0.045:
            score -= (0.05 - height_ratio) * 50.0
        bottom_pen = max(0.0, (y2)/h - 0.8)
        score -= bottom_pen * 60.0

        # slight penalty if header occupies too much of crop -> probably small crop
        if header_h / max(1,ch) > 0.3:
            score -= 12.0

        if debug:
            fn = os.path.join(DEBUG_DIR, f"cand_all_{idx}_score_{score:.2f}.png")
            cv2.imwrite(fn, crop)
            with open(fn.replace(".png",".txt"), "w") as f:
                f.write(f"bbox: {(x1,y1,x2,y2)}\n")
                f.write(f"cell_count: {cell_count}\n")
                f.write(f"nrows: {nrows}\n")
                f.write(f"ncols: {ncols}\n")
                f.write(f"height_ratio: {height_ratio:.3f}\n")
                f.write(f"bottom_pen: {bottom_pen:.3f}\n")
                f.write(f"score: {score:.3f}\n")

        scored.append(((x1,y1,x2,y2), header_abs, rows, cols, score))

    if not scored:
        raise RuntimeError("No valid table found after proposals")

    # pick best
    scored.sort(key=lambda x: x[4], reverse=True)
    best = scored[0]
    table_bbox, header_abs, rows, cols, best_score = best

    # convert rows_rel (relative to crop) to absolute rows_abs
    rows_abs = [ (table_bbox[1] + r0, table_bbox[1] + r1) for (r0,r1) in rows ]
    columns_abs = [ (table_bbox[0] + c0, table_bbox[0] + c1) for (c0,c1) in cols ]

    return table_bbox, header_abs, rows_abs, columns_abs
