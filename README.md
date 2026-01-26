# Table Detection — Computer Vision Challenge

**Repository:** [https://github.com/lbastidash/table_cv_challenge](https://github.com/lbastidash/table_cv_challenge)

---

## Overview

This project delivers a Python/OpenCV solution that detects and segments a complete table inside a larger screenshot using a *partial prototype image*. 
---

## What the solution delivers (explicit compliance)

The program outputs bounding boxes for:

* the **entire table**,
* the **header region**,
* **each row**,
* **each column**,

**Interface contract**

* **Inputs (positional):**

  1. Path to screenshot image
  2. Path to prototype image (header + ≥2 rows)
* **Stdout:** machine-readable, JSON-like structure containing keys:
  `table`, `header`, `rows`, `columns`, `cells`
* **Image artifact:** annotated output image (e.g. `output.png`) written to disk

## Quick start (evaluator friendly)

### Requirements

* Python 3.8+
* pip

Install dependencies:

```bash
pip install -r requiretments.txt
```

### Run

```bash
python src/main.py /path/to/screenshot.png /path/to/prototype.png
```

### Expected output

**Stdout (example):**

```json
{
  "table": [x, y, w, h],
  "header": [x, y, w, h],
  "rows": [[x, y, w, h], ...],
  "columns": [[x, y, w, h], ...],
  "cells": [
    {"row": 0, "col": 0, "bbox": [x, y, w, h]},
    ...
  ]
}
```

**File:** annotated image saved to the working directory (example included in the repo).


---

## Technical approach (concise)

* **Prototype analysis:** extracts relative header/row dimensions and separator cues.
* **Preprocessing:** scale normalization and contrast enhancement.
* **Output normalization:** consistent bounding-box format for automation use cases.

The approach is deterministic, explainable, and suitable for on-premise environments.

---

## Limitations

* Prototype must include header and at least two rows (challenge constraint).
* Highly stylized, non-orthogonal, or heavily distorted tables may require additional preprocessing.

---

## License


## Author

Luis Bastidas
Computer Vision