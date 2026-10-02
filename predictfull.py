from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "runs/detect/runs/yolo26s_fine_tune1/weights/best.pt"

SOURCE_DIR = "plan"

OUTPUT_PROJECT = "runs"

OUTPUT_NAME = "predict_plan"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("YOLO ALL TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = YOLO(MODEL_PATH)

    print(f"Model : {MODEL_PATH}")
    print(f"Source: {SOURCE_DIR}")
    print()

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    results = model.predict(

        # Folder containing the 394 pages
        source=SOURCE_DIR,

        # YOLO inference image size
        imgsz=1024,

        # Keep low-confidence detections for inspection
        conf=0.01,

        # Use NVIDIA GPU
        device=0,

        # Process one image at a time
        batch=1,

        # Do not keep all prediction results in RAM
        stream=True,

        # Save images with bounding boxes
        save=True,

        # Save YOLO-format prediction labels
        save_txt=True,

        # Save confidence together with labels
        save_conf=True,

        # Output folder
        project=OUTPUT_PROJECT,

        # Output experiment name
        name=OUTPUT_NAME,

        # Allow existing folder
        exist_ok=True,

        # Show progress
        verbose=True,
    )

    # --------------------------------------------------------
    # Consume streamed results
    # --------------------------------------------------------

    count = 0

    for result in results:

        count += 1

        print(
            f"Processed: "
            f"{count}/394"
        )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)

    print(
        f"Results: "
        f"{OUTPUT_PROJECT}/{OUTPUT_NAME}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()