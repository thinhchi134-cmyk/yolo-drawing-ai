from pathlib import Path
import json
import cv2
import pymupdf


# ============================================================
# CONFIG
# ============================================================

PDF_PATH = Path(
    r"D:\file thinh\Job data\machine vision test\data\test1.pdf"
)

PREDICT_DIR = Path(
    r"D:\file thinh\Job data\machine vision test\yolo pipeline\170_new\runs\detect\runs\predict_394_v1"
)

LABEL_DIR = PREDICT_DIR / "labels"

OUTPUT_DIR = PREDICT_DIR / "coordinate_verify_v0"

# PDF pages we want to test
TEST_PAGES = [24, 25, 26, 27]

CLASS_NAMES = {
    0: "titleblock",
    1: "revision",
    2: "notes",
    3: "legend",
    4: "plan",
    5: "section",
    6: "elevation",
    7: "detail",
    8: "schedule",
    9: "keyplan",
}

# Only visualization threshold.
# We do NOT discard detections below this.
CONF_THRESHOLD = 0.25

# Overlay thickness
BOX_THICKNESS = 4


# ============================================================
# PAGE / IMAGE MATCHING
# ============================================================

def find_image(page_number):
    """
    Find the rendered image corresponding to a PDF page.

    Expected examples:
        page_024.jpg
        page_024.png
        page_24.jpg
        page_24.png
    """

    candidates = [
        f"page_{page_number:03d}.jpg",
        f"page_{page_number:03d}.jpeg",
        f"page_{page_number:03d}.png",

        f"page_{page_number}.jpg",
        f"page_{page_number}.jpeg",
        f"page_{page_number}.png",
    ]

    for name in candidates:

        direct = PREDICT_DIR / name

        if direct.exists():
            return direct

    # Search recursively
    for name in candidates:

        matches = list(
            PREDICT_DIR.rglob(name)
        )

        if matches:
            return matches[0]

    return None


def find_label(page_number):
    """
    Find YOLO label corresponding to PDF page.
    """

    candidates = [
        f"page_{page_number:03d}.txt",
        f"page_{page_number}.txt",
    ]

    for name in candidates:

        direct = LABEL_DIR / name

        if direct.exists():
            return direct

    return None


# ============================================================
# YOLO LABEL READER
# ============================================================

def read_yolo_labels(label_path):

    detections = []

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                print(
                    f"[WARNING] Invalid line "
                    f"{label_path.name}:{line_number}"
                )
                continue

            try:

                class_id = int(parts[0])

                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                confidence = None

                if len(parts) >= 6:
                    confidence = float(parts[5])

            except ValueError:

                print(
                    f"[WARNING] Cannot parse "
                    f"{label_path.name}:{line_number}"
                )

                continue

            if (
                confidence is not None
                and confidence < CONF_THRESHOLD
            ):
                continue

            detections.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(
                    class_id,
                    f"class_{class_id}"
                ),
                "confidence": confidence,
                "xc": xc,
                "yc": yc,
                "w": w,
                "h": h,
            })

    return detections


# ============================================================
# YOLO NORMALIZED → IMAGE PIXELS
# ============================================================

def yolo_to_image_bbox(
    detection,
    image_width,
    image_height
):

    xc = detection["xc"] * image_width
    yc = detection["yc"] * image_height

    w = detection["w"] * image_width
    h = detection["h"] * image_height

    x1 = xc - w / 2
    y1 = yc - h / 2

    x2 = xc + w / 2
    y2 = yc + h / 2

    return [
        x1,
        y1,
        x2,
        y2,
    ]


# ============================================================
# IMAGE → PDF
# ============================================================

def image_bbox_to_pdf(
    image_bbox,
    image_width,
    image_height,
    pdf_width,
    pdf_height
):

    x1, y1, x2, y2 = image_bbox

    scale_x = image_width / pdf_width
    scale_y = image_height / pdf_height

    pdf_x1 = x1 / scale_x
    pdf_y1 = y1 / scale_y

    pdf_x2 = x2 / scale_x
    pdf_y2 = y2 / scale_y

    return [
        pdf_x1,
        pdf_y1,
        pdf_x2,
        pdf_y2,
    ]


# ============================================================
# PDF → IMAGE
# ============================================================

def pdf_bbox_to_image(
    pdf_bbox,
    image_width,
    image_height,
    pdf_width,
    pdf_height
):

    x1, y1, x2, y2 = pdf_bbox

    scale_x = image_width / pdf_width
    scale_y = image_height / pdf_height

    image_x1 = x1 * scale_x
    image_y1 = y1 * scale_y

    image_x2 = x2 * scale_x
    image_y2 = y2 * scale_y

    return [
        image_x1,
        image_y1,
        image_x2,
        image_y2,
    ]


# ============================================================
# ROUND TRIP ERROR
# ============================================================

def calculate_round_trip_error(
    original,
    reconstructed
):

    errors = [
        abs(original[i] - reconstructed[i])
        for i in range(4)
    ]

    return {
        "x1_error_px": errors[0],
        "y1_error_px": errors[1],
        "x2_error_px": errors[2],
        "y2_error_px": errors[3],
        "max_error_px": max(errors),
        "mean_error_px": sum(errors) / 4,
    }


# ============================================================
# DRAW BBOX
# ============================================================

def draw_bbox(
    image,
    bbox,
    label,
    confidence,
    color
):

    x1, y1, x2, y2 = [
        int(round(v))
        for v in bbox
    ]

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        BOX_THICKNESS
    )

    if confidence is not None:

        text = (
            f"{label} "
            f"{confidence:.2f}"
        )

    else:

        text = label

    text_y = max(
        30,
        y1 - 10
    )

    cv2.putText(
        image,
        text,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# PROCESS ONE PAGE
# ============================================================

def process_page(
    pdf_doc,
    page_number
):

    print()
    print("=" * 70)
    print(f"PAGE {page_number}")
    print("=" * 70)

    # --------------------------------------------------------
    # Find image
    # --------------------------------------------------------

    image_path = find_image(
        page_number
    )

    if image_path is None:

        print(
            "[ERROR] Rendered image not found."
        )

        return None

    # --------------------------------------------------------
    # Find label
    # --------------------------------------------------------

    label_path = find_label(
        page_number
    )

    if label_path is None:

        print(
            "[ERROR] YOLO label not found."
        )

        return None

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "[ERROR] Cannot read image:"
        )

        print(image_path)

        return None

    image_height, image_width = (
        image.shape[:2]
    )

    # --------------------------------------------------------
    # Open PDF page
    #
    # Important:
    # PDF page numbering is zero-based.
    # User page 24 -> index 23.
    # --------------------------------------------------------

    page = pdf_doc[
        page_number - 1
    ]

    pdf_rect = page.rect

    pdf_width = pdf_rect.width
    pdf_height = pdf_rect.height

    print(
        f"Image size : "
        f"{image_width} x {image_height} px"
    )

    print(
        f"PDF size   : "
        f"{pdf_width:.3f} x "
        f"{pdf_height:.3f} pt"
    )

    # --------------------------------------------------------
    # Read YOLO
    # --------------------------------------------------------

    detections = read_yolo_labels(
        label_path
    )

    print(
        f"Detections : "
        f"{len(detections)}"
    )

    # --------------------------------------------------------
    # Prepare output image
    # --------------------------------------------------------

    overlay = image.copy()

    page_results = []

    # --------------------------------------------------------
    # Process detections
    # --------------------------------------------------------

    for index, detection in enumerate(
        detections,
        start=1
    ):

        class_name = detection[
            "class_name"
        ]

        confidence = detection[
            "confidence"
        ]

        # YOLO → image
        image_bbox = yolo_to_image_bbox(
            detection,
            image_width,
            image_height
        )

        # Image → PDF
        pdf_bbox = image_bbox_to_pdf(
            image_bbox,
            image_width,
            image_height,
            pdf_width,
            pdf_height
        )

        # PDF → Image
        reconstructed_bbox = pdf_bbox_to_image(
            pdf_bbox,
            image_width,
            image_height,
            pdf_width,
            pdf_height
        )

        # Round-trip error
        error = calculate_round_trip_error(
            image_bbox,
            reconstructed_bbox
        )

        # ----------------------------------------------------
        # Draw original image bbox
        # ----------------------------------------------------

        draw_bbox(
            overlay,
            image_bbox,
            class_name,
            confidence,
            (0, 255, 0)
        )

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print()
        print(
            f"Region {index}: "
            f"{class_name}"
        )

        if confidence is not None:

            print(
                f"  confidence : "
                f"{confidence:.4f}"
            )

        print(
            "  image bbox :",
            [
                round(v, 3)
                for v in image_bbox
            ]
        )

        print(
            "  pdf bbox   :",
            [
                round(v, 3)
                for v in pdf_bbox
            ]
        )

        print(
            "  round trip :",
            [
                round(v, 3)
                for v in reconstructed_bbox
            ]
        )

        print(
            f"  max error  : "
            f"{error['max_error_px']:.8f} px"
        )

        page_results.append({

            "region_id": (
                f"page_{page_number:03d}"
                f"_r{index:03d}"
            ),

            "class_id": detection[
                "class_id"
            ],

            "class": class_name,

            "confidence": confidence,

            "bbox_image": [
                round(v, 6)
                for v in image_bbox
            ],

            "bbox_pdf": [
                round(v, 6)
                for v in pdf_bbox
            ],

            "bbox_image_round_trip": [
                round(v, 6)
                for v in reconstructed_bbox
            ],

            "round_trip_error": error,
        })

    # --------------------------------------------------------
    # Save overlay
    # --------------------------------------------------------

    overlay_dir = (
        OUTPUT_DIR
        / "overlays"
    )

    overlay_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    overlay_path = (
        overlay_dir
        / f"page_{page_number:03d}_verify.jpg"
    )

    cv2.imwrite(
        str(overlay_path),
        overlay
    )

    print()
    print(
        "Overlay saved:"
    )

    print(
        overlay_path
    )

    # --------------------------------------------------------
    # Page result
    # --------------------------------------------------------

    return {

        "page": page_number,

        "image": {
            "path": str(image_path),
            "width": image_width,
            "height": image_height,
        },

        "pdf": {
            "width": pdf_width,
            "height": pdf_height,
        },

        "detections": page_results,

        "overlay": str(
            overlay_path
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COORDINATE VERIFICATION V0")
    print("=" * 70)

    print()
    print("PDF:")
    print(PDF_PATH)

    print()
    print("Prediction:")
    print(PREDICT_DIR)

    print()
    print(
        "Test pages:",
        TEST_PAGES
    )

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not PDF_PATH.exists():

        print()
        print(
            "[ERROR] PDF does not exist:"
        )

        print(PDF_PATH)

        return

    if not PREDICT_DIR.exists():

        print()
        print(
            "[ERROR] Prediction directory "
            "does not exist:"
        )

        print(PREDICT_DIR)

        return

    if not LABEL_DIR.exists():

        print()
        print(
            "[ERROR] YOLO labels directory "
            "does not exist:"
        )

        print(LABEL_DIR)

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Open PDF
    # --------------------------------------------------------

    pdf_doc = pymupdf.open(
        str(PDF_PATH)
    )

    print()
    print(
        f"PDF pages: "
        f"{len(pdf_doc)}"
    )

    # --------------------------------------------------------
    # Process selected pages
    # --------------------------------------------------------

    results = []

    for page_number in TEST_PAGES:

        if (
            page_number < 1
            or page_number > len(pdf_doc)
        ):

            print(
                f"[ERROR] Invalid page: "
                f"{page_number}"
            )

            continue

        result = process_page(
            pdf_doc,
            page_number
        )

        if result is not None:

            results.append(result)

    # --------------------------------------------------------
    # Save JSON report
    # --------------------------------------------------------

    report_path = (
        OUTPUT_DIR
        / "coordinate_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Global error
    # --------------------------------------------------------

    all_errors = []

    for page in results:

        for detection in page[
            "detections"
        ]:

            all_errors.append(
                detection[
                    "round_trip_error"
                ]["max_error_px"]
            )

    print()
    print("=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

    print(
        f"Pages tested: "
        f"{len(results)}"
    )

    if all_errors:

        print(
            f"Maximum round-trip error: "
            f"{max(all_errors):.10f} px"
        )

        print(
            f"Mean max error: "
            f"{sum(all_errors) / len(all_errors):.10f} px"
        )

    print()
    print(
        "Report:"
    )

    print(
        report_path
    )

    print()
    print(
        "Overlay directory:"
    )

    print(
        OUTPUT_DIR / "overlays"
    )

    pdf_doc.close()


if __name__ == "__main__":
    main()