# src/main.py
import sys
import cv2
import os
import json 

from prototype import extract_prototype_features
from detector import detect_table_and_header
from draw import draw_results

def main(screenshot_path, prototype_path, output_path="output.png", debug=False):
    screenshot = cv2.imread(screenshot_path)
    prototype = cv2.imread(prototype_path)

    if screenshot is None:
        raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")
    if prototype is None:
        raise FileNotFoundError(f"Prototype not found: {prototype_path}")

    proto_features = extract_prototype_features(prototype, proto_path=prototype_path, debug=debug)

    # detect_table_and_header devuelve table_bbox, header_bbox, rows_abs, columns_abs
    table_bbox, header_bbox, rows_abs, columns_abs = detect_table_and_header(
        screenshot, proto_features, debug=debug
    )

    # Dictionary to save results
    data_to_save = {
        "table_boundaries": table_bbox,
        "header_boundaries": header_bbox,
        "rows": rows_abs,
        "columns": columns_abs
    }

    # imprimir resultados en stdout
    print(f"Table boundaries: {table_bbox}")
    print(f"Header boundaries: {header_bbox}")
    for i, r in enumerate(rows_abs):
        print(f"Row {i}: {r}")
    for i, c in enumerate(columns_abs):
        print(f"Column {i}: {c}")

    # 3. Guarda el archivo .json
    # Usamos 'indent=4' para que el archivo sea legible para humanos
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False, default=int)

    print("\nFile 'results.json' saved successfully.")


    # dibujar y guardar
    out = draw_results(
        screenshot.copy(),
        table_bbox=table_bbox,
        header_bbox=header_bbox,
        rows=rows_abs,
        columns=columns_abs
    )

    cv2.imwrite(output_path, out)
    print(f"Output image saved to {output_path}")

if __name__ == "__main__":
    debug = "--debug" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        print("Usage: python src/main.py <screenshot> <prototype> [--debug]")
        sys.exit(1)
    main(args[0], args[1], debug=debug)
