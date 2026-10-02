
"""
region_analyzer_v1.py

Pipeline:

YOLO detection
    ↓
YOLO normalized bbox
    ↓
Image pixel bbox
    ↓
PDF bbox
    ↓
Extract PDF text inside region
    ↓
Region JSON

V1:
- PDF text extraction ONLY
- No OCR
- No geometry
- No VLM
- Test pages 24-27 first

Author: Thinh
"""

from pathlib import Path
import json
import math

import pymupdf


# ============================================================
# CONFIG
# ============================================================

PDF_PATH = Path(
    r"D:\file thinh\Job data\machine vision test\data\test1.pdf"
)

PREDICT_DIR = Path(
    r"D:\file thinh\Job data\machine vision test\yolo pipeline"
    r"\170_new\runs\detect\runs\predict_394_v1"
)

LABEL_DIR = PREDICT_DIR / "labels"

OUTPUT_DIR = PREDICT_DIR / "region_analyzer_v1"

PAGES_TO_TEST = [24, 25, 26, 27]


# ------------------------------------------------------------
# Text filtering
# ------------------------------------------------------------

# Minimum fraction of text bbox that must overlap region bbox.
#
# Example:
#
#       REGION
#   ┌──────────────────┐
#   │     ┌──────────┐ │
#   │     │   TEXT   │ │
#   │     └──────────┘ │
#   └──────────────────┘
#
# If overlap_ratio >= threshold → keep text.
#
TEXT_OVERLAP_THRESHOLD = 0.30


# ------------------------------------------------------------
# Small padding around region when extracting text.
#
# PDF drawings often have labels slightly outside the visual
# region boundary.
#
# Start with 0 for strict testing.
# Later we can experiment with 2, 5, 10 PDF points.
# ------------------------------------------------------------

REGION_PADDING_PDF = 0.0


# ------------------------------------------------------------
# YOLO class names
# ------------------------------------------------------------

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


# ============================================================
# BASIC UTILITIES
# ============================================================

def ensure_output_dirs():
    """
    Create output directories.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "pages").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / "debug").mkdir(
        parents=True,
        exist_ok=True
    )


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def page_number_to_label_file(page_number):
    """
    Convert 1-based page number to YOLO label filename.

    IMPORTANT:
    This assumes the prediction image files are named in a way
    that corresponds to the PDF page order.

    For example:

        page_024.txt

    or:

        024.txt

    The function below tries several common patterns.
    """

    candidates = [
        LABEL_DIR / f"page_{page_number:03d}.txt",
        LABEL_DIR / f"{page_number:03d}.txt",
        LABEL_DIR / f"page_{page_number:04d}.txt",
        LABEL_DIR / f"{page_number:04d}.txt",
        LABEL_DIR / f"{page_number}.txt",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


# ============================================================
# YOLO LABEL PARSING
# ============================================================

def parse_yolo_label_file(label_path):
    """
    Parse YOLO txt file.

    Expected format:

        class_id
        x_center
        y_center
        width
        height
        confidence

    Example:

        4 0.475 0.521 0.721 0.631 0.986278

    Returns:

        [
            {
                "class_id": 4,
                "class_name": "plan",
                "confidence": 0.986278,
                "bbox_yolo": {
                    "x_center": ...,
                    "y_center": ...,
                    "width": ...,
                    "height": ...
                }
            }
        ]
    """

    detections = []

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            print(
                f"[WARNING] Invalid YOLO line "
                f"{label_path.name}:{line_number}"
            )
            continue

        try:
            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            confidence = (
                float(parts[5])
                if len(parts) >= 6
                else None
            )

        except ValueError:
            print(
                f"[WARNING] Cannot parse line "
                f"{label_path.name}:{line_number}"
            )
            continue

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}"
        )

        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,

                "bbox_yolo": {
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                }
            }
        )

    return detections


# ============================================================
# YOLO → IMAGE BBOX
# ============================================================

def yolo_to_image_bbox(
    x_center,
    y_center,
    width,
    height,
    image_width,
    image_height,
):
    """
    Convert normalized YOLO bbox to pixel bbox.

    Returns:

        [x1, y1, x2, y2]

    Image coordinate system:

        (0,0)
          ┌──────────────────► x
          │
          │
          │
          ▼
          y
    """

    x1 = (x_center - width / 2.0) * image_width
    y1 = (y_center - height / 2.0) * image_height

    x2 = (x_center + width / 2.0) * image_width
    y2 = (y_center + height / 2.0) * image_height

    x1 = clamp(x1, 0.0, image_width)
    y1 = clamp(y1, 0.0, image_height)

    x2 = clamp(x2, 0.0, image_width)
    y2 = clamp(y2, 0.0, image_height)

    return [
        x1,
        y1,
        x2,
        y2,
    ]


# ============================================================
# IMAGE BBOX → PDF BBOX
# ============================================================

def image_bbox_to_pdf_bbox(
    image_bbox,
    image_width,
    image_height,
    pdf_width,
    pdf_height,
):
    """
    Convert image pixel bbox → PDF point bbox.

    IMPORTANT:

    We assume the rendered YOLO image preserves the same
    aspect ratio and orientation as the PDF page.

    Formula:

        scale_x = image_width / pdf_width
        scale_y = image_height / pdf_height

        pdf_x = image_x / scale_x
        pdf_y = image_y / scale_y
    """

    x1, y1, x2, y2 = image_bbox

    scale_x = image_width / pdf_width
    scale_y = image_height / pdf_height

    pdf_x1 = x1 / scale_x
    pdf_y1 = y1 / scale_y

    pdf_x2 = x2 / scale_x
    pdf_y2 = y2 / scale_y

    pdf_x1 = clamp(pdf_x1, 0.0, pdf_width)
    pdf_y1 = clamp(pdf_y1, 0.0, pdf_height)

    pdf_x2 = clamp(pdf_x2, 0.0, pdf_width)
    pdf_y2 = clamp(pdf_y2, 0.0, pdf_height)

    return [
        pdf_x1,
        pdf_y1,
        pdf_x2,
        pdf_y2,
    ]


# ============================================================
# PDF BBOX HELPERS
# ============================================================

def bbox_area(bbox):
    """
    Calculate bbox area.
    """

    x1, y1, x2, y2 = bbox

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return width * height


def bbox_intersection(a, b):
    """
    Calculate intersection bbox of:

        a = [x1, y1, x2, y2]
        b = [x1, y1, x2, y2]

    Returns None if no intersection.
    """

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return None

    return [
        x1,
        y1,
        x2,
        y2,
    ]


def bbox_overlap_ratio(
    text_bbox,
    region_bbox,
):
    """
    Calculate:

        intersection_area / text_area

    This answers:

        "How much of the text bbox lies inside
         the region?"

    NOT:

        intersection / region area
    """

    text_area = bbox_area(text_bbox)

    if text_area <= 0:
        return 0.0

    intersection = bbox_intersection(
        text_bbox,
        region_bbox
    )

    if intersection is None:
        return 0.0

    intersection_area = bbox_area(intersection)

    return intersection_area / text_area


def pad_bbox(
    bbox,
    padding,
    page_width,
    page_height,
):
    """
    Expand bbox by padding in PDF points.
    """

    x1, y1, x2, y2 = bbox

    x1 -= padding
    y1 -= padding

    x2 += padding
    y2 += padding

    x1 = clamp(x1, 0.0, page_width)
    y1 = clamp(y1, 0.0, page_height)

    x2 = clamp(x2, 0.0, page_width)
    y2 = clamp(y2, 0.0, page_height)

    return [
        x1,
        y1,
        x2,
        y2,
    ]


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_spans(page):
    """
    Extract individual text spans from a PDF page.

    We use:

        page.get_text("dict")

    because it provides:

        blocks
        lines
        spans
        bbox
        text

    Returns a list of text spans.
    """

    text_dict = page.get_text("dict")

    spans = []

    for block_index, block in enumerate(
        text_dict.get("blocks", [])
    ):

        # Text block usually has type 0.
        # Image block usually has type 1.
        if block.get("type") != 0:
            continue

        for line_index, line in enumerate(
            block.get("lines", [])
        ):

            for span_index, span in enumerate(
                line.get("spans", [])
            ):

                text = span.get("text", "")

                if not text:
                    continue

                bbox = span.get("bbox")

                if not bbox or len(bbox) != 4:
                    continue

                spans.append(
                    {
                        "text": text,

                        "bbox_pdf": [
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                        ],

                        "block_index": block_index,
                        "line_index": line_index,
                        "span_index": span_index,

                        "font": span.get("font"),
                        "size": span.get("size"),
                        "flags": span.get("flags"),
                    }
                )

    return spans


def extract_region_text(
    page,
    region_bbox,
):
    """
    Extract text spans whose bbox overlaps the region
    sufficiently.

    Returns:

        [
            {
                "text": "...",
                "bbox_pdf": [...],
                "overlap_ratio": ...
            }
        ]
    """

    all_spans = extract_text_spans(page)

    selected = []

    for span in all_spans:

        text_bbox = span["bbox_pdf"]

        overlap_ratio = bbox_overlap_ratio(
            text_bbox,
            region_bbox
        )

        if overlap_ratio < TEXT_OVERLAP_THRESHOLD:
            continue

        selected.append(
            {
                "text": span["text"],

                "bbox_pdf": text_bbox,

                "overlap_ratio": round(
                    overlap_ratio,
                    6
                ),

                "block_index": span["block_index"],
                "line_index": span["line_index"],
                "span_index": span["span_index"],

                "font": span["font"],
                "size": span["size"],
                "flags": span["flags"],
            }
        )

    return selected


# ============================================================
# TEXT CLEANING / ORDERING
# ============================================================

def clean_text(text):
    """
    Light cleaning only.

    IMPORTANT:
    Do NOT aggressively modify text at this stage.

    The raw PDF text is valuable for later semantic analysis.
    """

    return text.replace(
        "\u00a0",
        " "
    )


def sort_region_text(text_items):
    """
    Sort approximately top-to-bottom, left-to-right.
    """

    return sorted(
        text_items,
        key=lambda item: (
            item["bbox_pdf"][1],
            item["bbox_pdf"][0],
        )
    )


def build_text_content(text_items):
    """
    Build a human-readable text representation.
    """

    lines = []

    for item in text_items:

        text = clean_text(
            item["text"]
        ).strip()

        if not text:
            continue

        lines.append(text)

    return "\n".join(lines)


# ============================================================
# PAGE IMAGE SIZE
# ============================================================

def find_page_image_size(
    page_number,
):
    """
    Find the rendered image corresponding to a PDF page.

    This function searches common filenames.

    NOTE:
    We need the actual YOLO rendered image size because
    YOLO coordinates are normalized against that image.

    Returns:

        (width, height, image_path)

    or None.
    """

    image_candidates = []

    stems = [
        f"page_{page_number:03d}",
        f"page_{page_number:04d}",
        f"{page_number:03d}",
        f"{page_number:04d}",
        f"{page_number}",
    ]

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    ]

    # Search recursively.
    for stem in stems:

        for ext in extensions:

            candidates = list(
                PREDICT_DIR.rglob(
                    stem + ext
                )
            )

            image_candidates.extend(
                candidates
            )

    # Remove duplicates while preserving order.
    unique_candidates = []

    seen = set()

    for path in image_candidates:

        path_key = str(path.resolve())

        if path_key in seen:
            continue

        seen.add(path_key)

        unique_candidates.append(path)

    if not unique_candidates:
        return None

    # Try PIL only for dimensions.
    try:
        from PIL import Image

        image_path = unique_candidates[0]

        with Image.open(image_path) as img:
            width, height = img.size

        return (
            width,
            height,
            image_path
        )

    except Exception as e:

        print(
            f"[WARNING] Could not read image size: "
            f"{unique_candidates[0]}"
        )

        print(
            f"          Error: {e}"
        )

        return None


# ============================================================
# PROCESS ONE REGION
# ============================================================

def process_region(
    page,
    page_number,
    detection,
    image_width,
    image_height,
):
    """
    Convert one YOLO detection into a Region record.
    """

    pdf_width = page.rect.width
    pdf_height = page.rect.height

    yolo_bbox = detection["bbox_yolo"]

    # --------------------------------------------------------
    # YOLO → image pixels
    # --------------------------------------------------------

    bbox_image = yolo_to_image_bbox(
        x_center=yolo_bbox["x_center"],
        y_center=yolo_bbox["y_center"],
        width=yolo_bbox["width"],
        height=yolo_bbox["height"],
        image_width=image_width,
        image_height=image_height,
    )

    # --------------------------------------------------------
    # image → PDF
    # --------------------------------------------------------

    bbox_pdf = image_bbox_to_pdf_bbox(
        image_bbox=bbox_image,
        image_width=image_width,
        image_height=image_height,
        pdf_width=pdf_width,
        pdf_height=pdf_height,
    )

    # --------------------------------------------------------
    # Optional PDF padding
    # --------------------------------------------------------

    extraction_bbox = pad_bbox(
        bbox_pdf,
        REGION_PADDING_PDF,
        pdf_width,
        pdf_height,
    )

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    text_items = extract_region_text(
        page,
        extraction_bbox
    )

    text_items = sort_region_text(
        text_items
    )

    text_content = build_text_content(
        text_items
    )

    # --------------------------------------------------------
    # Region ID
    # --------------------------------------------------------

    class_name = detection["class_name"]

    region_index = (
        f"{class_name}_"
        f"{page_number:03d}"
    )

    # --------------------------------------------------------
    # Final JSON object
    # --------------------------------------------------------

    region = {
        "region_id": region_index,

        "page": page_number,

        "class_id": detection["class_id"],
        "class_name": class_name,

        "confidence": detection["confidence"],

        "bbox_image": [
            round(v, 3)
            for v in bbox_image
        ],

        "bbox_pdf": [
            round(v, 3)
            for v in bbox_pdf
        ],

        "extraction_bbox_pdf": [
            round(v, 3)
            for v in extraction_bbox
        ],

        "page_size_pdf": {
            "width": round(
                pdf_width,
                3
            ),
            "height": round(
                pdf_height,
                3
            ),
        },

        "image_size": {
            "width": image_width,
            "height": image_height,
        },

        "text": text_items,

        "text_content": text_content,

        "geometry": None,

        "ocr": None,

        "vlm": None,

        "analysis": {
            "text_span_count": len(
                text_items
            ),

            "text_char_count": len(
                text_content
            ),

            "geometry_status": "not_processed",

            "ocr_status": "not_processed",

            "vlm_status": "not_processed",
        },
    }

    return region


# ============================================================
# PROCESS ONE PAGE
# ============================================================

def process_page(
    doc,
    page_number,
):
    """
    Process one PDF page.
    """

    print()
    print("=" * 70)
    print(
        f"PAGE {page_number}"
    )
    print("=" * 70)

    # PDF pages are zero-indexed internally.
    page_index = page_number - 1

    if page_index < 0 or page_index >= len(doc):
        print(
            f"[ERROR] Page {page_number} "
            f"is outside PDF."
        )

        return None

    page = doc[page_index]

    # --------------------------------------------------------
    # Find YOLO labels
    # --------------------------------------------------------

    label_path = page_number_to_label_file(
        page_number
    )

    if label_path is None:

        print(
            f"[ERROR] Could not find YOLO label "
            f"for page {page_number}"
        )

        return None

    print(
        f"[INFO] Label: {label_path}"
    )

    # --------------------------------------------------------
    # Find rendered image
    # --------------------------------------------------------

    image_info = find_page_image_size(
        page_number
    )

    if image_info is None:

        print(
            f"[ERROR] Could not find rendered "
            f"image for page {page_number}"
        )

        return None

    image_width, image_height, image_path = (
        image_info
    )

    print(
        f"[INFO] Image: {image_path}"
    )

    print(
        f"[INFO] Image size: "
        f"{image_width} x {image_height}"
    )

    print(
        f"[INFO] PDF size: "
        f"{page.rect.width:.3f} x "
        f"{page.rect.height:.3f}"
    )

    # --------------------------------------------------------
    # Parse YOLO
    # --------------------------------------------------------

    detections = parse_yolo_label_file(
        label_path
    )

    print(
        f"[INFO] YOLO detections: "
        f"{len(detections)}"
    )

    # --------------------------------------------------------
    # Process regions
    # --------------------------------------------------------

    regions = []

    class_counter = {}

    for detection in detections:

        region = process_region(
            page=page,
            page_number=page_number,
            detection=detection,
            image_width=image_width,
            image_height=image_height,
        )

        # Make ID unique even if there are multiple
        # regions of the same class.
        class_name = region["class_name"]

        class_counter[class_name] = (
            class_counter.get(
                class_name,
                0
            ) + 1
        )

        region["region_id"] = (
            f"page_{page_number:03d}_"
            f"{class_name}_"
            f"{class_counter[class_name]:02d}"
        )

        regions.append(region)

        print(
            f"  [{class_name:10s}] "
            f"conf="
            f"{detection['confidence']:.4f} "
            f"text_spans="
            f"{len(region['text'])}"
        )

    # --------------------------------------------------------
    # Page JSON
    # --------------------------------------------------------

    page_result = {
        "page": page_number,

        "pdf": {
            "width": round(
                page.rect.width,
                3
            ),
            "height": round(
                page.rect.height,
                3
            ),
        },

        "image": {
            "width": image_width,
            "height": image_height,
            "path": str(image_path),
        },

        "source": {
            "pdf": str(PDF_PATH),
            "yolo_labels": str(label_path),
        },

        "region_count": len(regions),

        "regions": regions,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "pages"
        / f"page_{page_number:03d}.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            page_result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[SAVED] {output_path}"
    )

    return page_result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("REGION ANALYZER V1")
    print("=" * 70)

    print(
        f"PDF: {PDF_PATH}"
    )

    print(
        f"YOLO labels: {LABEL_DIR}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print(
        f"Pages: {PAGES_TO_TEST}"
    )

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"PDF not found:\n{PDF_PATH}"
        )

    if not LABEL_DIR.exists():

        raise FileNotFoundError(
            f"YOLO label directory not found:\n"
            f"{LABEL_DIR}"
        )

    ensure_output_dirs()

    # --------------------------------------------------------
    # Open PDF
    # --------------------------------------------------------

    doc = pymupdf.open(
        PDF_PATH
    )

    print()
    print(
        f"[INFO] PDF pages: {len(doc)}"
    )

    # --------------------------------------------------------
    # Process pages
    # --------------------------------------------------------

    all_pages = []

    for page_number in PAGES_TO_TEST:

        result = process_page(
            doc,
            page_number
        )

        if result is not None:
            all_pages.append(result)

    # --------------------------------------------------------
    # Global JSON
    # --------------------------------------------------------

    all_regions = []

    for page_result in all_pages:

        all_regions.extend(
            page_result["regions"]
        )

    global_result = {

        "project": "machine_vision_drawing",

        "version": "region_analyzer_v1",

        "source": {
            "pdf": str(PDF_PATH),
            "predict_dir": str(PREDICT_DIR),
            "label_dir": str(LABEL_DIR),
        },

        "config": {
            "text_overlap_threshold":
                TEXT_OVERLAP_THRESHOLD,

            "region_padding_pdf":
                REGION_PADDING_PDF,
        },

        "pages_processed": [
            page_result["page"]
            for page_result in all_pages
        ],

        "page_count": len(
            all_pages
        ),

        "region_count": len(
            all_regions
        ),

        "regions": all_regions,
    }

    global_output = (
        OUTPUT_DIR
        / "all_regions_v1.json"
    )

    with open(
        global_output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            global_result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    class_counts = {}

    for region in all_regions:

        class_name = region[
            "class_name"
        ]

        class_counts[class_name] = (
            class_counts.get(
                class_name,
                0
            ) + 1
        )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Pages processed : "
        f"{len(all_pages)}"
    )

    print(
        f"Regions         : "
        f"{len(all_regions)}"
    )

    print()
    print("Regions by class:")

    for class_name, count in sorted(
        class_counts.items()
    ):

        print(
            f"  {class_name:12s}: "
            f"{count}"
        )

    print()
    print(
        f"[SAVED] {global_output}"
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    doc.close()


if __name__ == "__main__":
    main()