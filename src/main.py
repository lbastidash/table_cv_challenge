# src/main.py
import sys
import cv2

from prototype import extract_prototype_features
from detector import detect_table_and_header
from draw import draw_results


def main(screenshot_path, prototype_path, debug=False):
    screenshot = cv2.imread(screenshot_path)
    prototype = cv2.imread(prototype_path)

    if screenshot is None:
        raise FileNotFoundError(screenshot_path)
    if prototype is None:
        raise FileNotFoundError(prototype_path)

    proto_features = extract_prototype_features(prototype, debug=debug)

    table_bbox, header_bbox = detect_table_and_header(
        screenshot,
        proto_features
    )

    print("TABLE:", table_bbox)
    print("HEADER:", header_bbox)

    out = draw_results(screenshot.copy(), table_bbox, header_bbox)
    cv2.imwrite("output.png", out)

    print("Saved output.png")


if __name__ == "__main__":
    debug = "--debug" in sys.argv

    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if len(args) != 2:
        print("Usage: python main.py <screenshot> <prototype> [--debug]")
        sys.exit(1)

    main(args[0], args[1], debug=debug)
